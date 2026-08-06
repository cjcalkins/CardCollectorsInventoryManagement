"""
shipping_routes.py
==================

The Shipping tab: orders in, labels out, tracking back.

Packaged as a Blueprint so app.py needs three lines to adopt it rather than
another 700 in an already-14k-line module. It depends only on Flask, the
models, and shipping_providers — never on app.py — so there's no import cycle.

Flow
----
    sale email ──> SaleEvent ──> Order(needs_address)
    CSV / manual ─────────────>  Order(ready)
                                    │  address filled in
                                    ▼
                             quote → buy label → Shipment(+PDF on disk)
                                    │
                                    ▼
                        print (single or merged batch)
                                    │
                                    ▼
                     poll tracking → Shipment.events → Order(shipped/delivered)

Label PDFs live under UPLOAD_FOLDER/shipping_labels/<order_id>/, which keeps
them inside the storage root the Storage settings page already moves and backs
up. Paths are stored upload-relative, matching ScanRecord.image_path.

Routes are mounted under /shipping. app.py's permission gate maps a path's
first segment to a resource, so add {"shipping": "shops"} to that map (see
INTEGRATION.md) and these inherit the existing Shops permission.
"""

import csv
import io
import os
import re
from datetime import datetime, timedelta

from flask import Blueprint, current_app, jsonify, render_template, request, send_file

from models import db, utcnow, ShopConnection, ScanRecord, SaleEvent
from shipping_models import Order, OrderItem, Shipment
from shipping_providers import (
    SHIPPING_PROVIDERS, SHIPPING_SECRET_FIELDS, ENVELOPE_SIZES,
    get_shipping_provider, _validate_destination, normalize_state, normalize_country,
)

shipping_bp = Blueprint("shipping", __name__)

LABEL_SUBDIR = "shipping_labels"

# Statuses still worth asking the carrier about. Delivered/cancelled are final.
_ACTIVE_STATUSES = ("labeled", "shipped")


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _persist():
    db.session.commit()


def _get_connection(provider_key, create=False):
    """
    Shipping credentials reuse the shop_connections table. The Shops page only
    iterates over shop_providers.MARKETPLACES, so these rows stay invisible
    there while inheriting its storage and secret handling.
    """
    conn = ShopConnection.query.filter_by(marketplace=provider_key).first()
    if conn is None and create:
        conn = ShopConnection(marketplace=provider_key, enabled=False, config={},
                              status="disconnected")
        db.session.add(conn)
        db.session.commit()
    return conn


def _provider(provider_key):
    conn = _get_connection(provider_key, create=True)
    return get_shipping_provider(provider_key, conn, persist=_persist), conn


def _connected_provider_keys():
    """Providers whose connection is enabled — the only shipments worth the
    bounded tracking budget. _refresh_shipment early-returns on a disconnected
    provider WITHOUT stamping last_tracked_at (a disconnect must not fake a
    poll), so queries that select "due" shipments have to exclude them here or
    those rows match forever and starve the working provider's shipments out
    of the LIMIT."""
    keys = []
    for k in SHIPPING_PROVIDERS:
        conn = _get_connection(k)
        if conn is not None and conn.enabled:
            keys.append(k)
    return keys


def _connection_public_view(conn, provider_key):
    """Config safe for the browser: secrets become a set/unset flag."""
    meta = SHIPPING_PROVIDERS[provider_key]
    cfg = (conn.config if conn else {}) or {}
    public, secrets_set = {}, {}
    for field in meta["fields"]:
        k = field["key"]
        if field.get("secret") or k in SHIPPING_SECRET_FIELDS.get(provider_key, set()):
            secrets_set[k] = bool(str(cfg.get(k, "")).strip())
        else:
            public[k] = cfg.get(k, "")
    return {
        "enabled": bool(conn.enabled) if conn else False,
        "status": conn.status if conn else "disconnected",
        "status_detail": conn.status_detail if conn else "",
        "connected_at": conn.connected_at.strftime("%Y-%m-%d %H:%M") if (conn and conn.connected_at) else "",
        "config": public,
        "secrets_set": secrets_set,
    }


def _label_dir(order_id):
    root = os.path.join(current_app.config["UPLOAD_FOLDER"], LABEL_SUBDIR, str(order_id))
    os.makedirs(root, exist_ok=True)
    return root


def _save_label(order_id, shipment_id, data, ext="pdf"):
    """Write the label and return its upload-relative path."""
    fname = f"shipment_{shipment_id}.{ext.lower()}"
    abs_path = os.path.join(_label_dir(order_id), fname)
    with open(abs_path, "wb") as fh:
        fh.write(data)
    return f"{LABEL_SUBDIR}/{order_id}/{fname}"


def _label_abs(shipment):
    if not shipment.label_path:
        return None
    p = os.path.join(current_app.config["UPLOAD_FOLDER"], shipment.label_path)
    return p if os.path.exists(p) else None


def _order_json(o, full=False):
    ship = o.latest_shipment
    out = {
        "id": o.id,
        "source": o.source,
        "external_order_id": o.external_order_id or "",
        "status": o.status,
        "buyer_name": o.buyer_name or "",
        "buyer_company": o.buyer_company or "",
        "city": o.city or "",
        "state": o.state or "",
        "zipcode": o.zipcode or "",
        "country": o.country or "US",
        "item_count": sum(i.qty or 0 for i in o.items),
        "value": o.declared_value,
        "address_complete": o.address_complete,
        "created_at": o.created_at.strftime("%Y-%m-%d %H:%M") if o.created_at else "",
        "shipment": None,
    }
    if ship:
        out["shipment"] = {
            "id": ship.id,
            "provider": ship.provider,
            "tracking_code": ship.tracking_code or "",
            "tracking_url": ship.tracking_url or "",
            "carrier": ship.carrier or "",
            "service": ship.service or "",
            "rate_amount": ship.rate_amount,
            "insured": bool(ship.insured),
            "tracking_status": ship.tracking_status or "",
            "tracking_percent": ship.tracking_percent,
            "delivery_date": ship.delivery_date or "",
            "has_label": bool(ship.label_path),
            "last_tracked_at": ship.last_tracked_at.strftime("%Y-%m-%d %H:%M") if ship.last_tracked_at else "",
        }
    if full:
        out.update({
            "address1": o.address1 or "", "address2": o.address2 or "",
            "email": o.email or "", "phone": o.phone or "",
            "shipping_paid": o.shipping_paid or 0.0,
            "notes": o.notes or "",
            "items": [{
                "id": i.id, "name": i.name, "set_name": i.set_name or "",
                "condition": i.condition or "", "foil": bool(i.foil),
                "qty": i.qty, "price": i.price, "record_id": i.record_id,
            } for i in o.items],
            "shipments": [{
                "id": s.id, "provider": s.provider, "status": s.status,
                "tracking_code": s.tracking_code or "", "tracking_url": s.tracking_url or "",
                "carrier": s.carrier or "", "service": s.service or "",
                "rate_amount": s.rate_amount, "envelope_size": s.envelope_size or "",
                "insured": bool(s.insured), "insurance_amount": s.insurance_amount,
                "tracking_status": s.tracking_status or "",
                "tracking_percent": s.tracking_percent,
                "delivery_date": s.delivery_date or "",
                "has_label": bool(s.label_path),
                "error": s.error or "",
                "events": s.events or [],
                "created_at": s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "",
            } for s in o.shipments],
        })
    return out


def _sync_order_status(order):
    """Derive the order's status from its newest shipment. Never moves backwards."""
    ship = order.latest_shipment
    if not ship:
        order.status = "ready" if order.address_complete else "needs_address"
        return
    if ship.is_delivered:
        order.status = "delivered"
    elif (ship.events or []) and order.status in ("labeled", "ready", "needs_address"):
        order.status = "shipped"
        order.shipped_at = order.shipped_at or utcnow()
    elif order.status in ("ready", "needs_address"):
        order.status = "labeled"


# ──────────────────────────────────────────────────────────────────────────────
# Order creation from a parsed sale email — called by app.py's sale pipeline
# ──────────────────────────────────────────────────────────────────────────────
def ensure_order_from_sale(parsed, source="tcgplayer", sale_events=None):
    """
    Turn one parsed sale email into an Order, idempotently.

    Called from app.py's _process_sale_email. Sale emails carry the items but
    seldom a usable postal address, so the order lands in `needs_address` and
    waits for a CSV import or manual entry rather than pretending it's ready.

    Re-running on the same email adds nothing: the (source, external_order_id)
    unique constraint means an existing order is returned untouched.
    """
    order_id = (parsed.get("order_id") or "").strip()
    if not order_id:
        return None   # nothing to key on; don't create an unmatchable order

    existing = Order.query.filter_by(source=source, external_order_id=order_id).first()
    if existing:
        return existing

    items = parsed.get("items") or []
    if not items:
        return None   # unparsed email — SaleEvent already records it for review

    order = Order(source=source, external_order_id=order_id, status="needs_address",
                  notes=(parsed.get("subject") or "")[:200])

    # Some senders include a shipping block; take it when the parser found one.
    addr = parsed.get("address") or {}
    if addr:
        order.buyer_name = addr.get("name") or None
        order.address1 = addr.get("address1") or None
        order.address2 = addr.get("address2") or None
        order.city = addr.get("city") or None
        order.state = normalize_state(addr.get("state")) or None
        order.zipcode = addr.get("zip") or None
        order.country = normalize_country(addr.get("country"))

    db.session.add(order)
    db.session.flush()

    ev_by_title = {}
    for ev in (sale_events or []):
        ev_by_title[(ev.item_title or "").strip()] = ev

    for it in items:
        title = (it.get("name") or "").strip() or "(unknown)"
        lookup = title + (f" ({it['set']})" if it.get("set") else "")
        ev = ev_by_title.get(lookup)
        db.session.add(OrderItem(
            order_id=order.id,
            record_id=ev.record_id if ev else None,
            sale_event_id=ev.id if ev else None,
            name=title[:300],
            set_name=(it.get("set") or "")[:200] or None,
            condition=(it.get("condition") or "")[:60] or None,
            foil=bool(it.get("foil")),
            qty=max(int(it.get("qty") or 1), 1),
            price=float(it.get("price") or 0.0),
        ))

    order.item_total = order.declared_value
    if order.address_complete:
        order.status = "ready"
    db.session.commit()
    return order


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────
@shipping_bp.route("/shipping")
def shipping_page():
    connections = {k: _connection_public_view(_get_connection(k), k) for k in SHIPPING_PROVIDERS}
    counts = {}
    for st in ("needs_address", "ready", "labeled", "shipped", "delivered"):
        counts[st] = Order.query.filter_by(status=st).count()
    return render_template(
        "shipping.html",
        providers=SHIPPING_PROVIDERS,
        connections=connections,
        envelope_sizes=ENVELOPE_SIZES,
        counts=counts,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Provider config
# ──────────────────────────────────────────────────────────────────────────────
@shipping_bp.route("/shipping/save/<provider_key>", methods=["POST"])
def shipping_save(provider_key):
    if provider_key not in SHIPPING_PROVIDERS:
        return jsonify({"status": "error", "message": "Unknown shipping provider"}), 404
    conn = _get_connection(provider_key, create=True)
    cfg = dict(conn.config or {})

    for field in SHIPPING_PROVIDERS[provider_key]["fields"]:
        k = field["key"]
        if field.get("type") == "checkbox":
            cfg[k] = "1" if request.form.get(k) in ("1", "true", "on", "yes") else ""
            continue
        submitted = request.form.get(k, None)
        if submitted is None:
            continue
        submitted = submitted.strip()
        is_secret = field.get("secret") or k in SHIPPING_SECRET_FIELDS.get(provider_key, set())
        if is_secret and submitted == "":
            continue   # blank means "keep the stored secret"
        cfg[k] = submitted

    conn.config = cfg
    conn.updated_at = utcnow()
    db.session.commit()
    return jsonify({"status": "success",
                    "message": f"{SHIPPING_PROVIDERS[provider_key]['label']} settings saved."})


@shipping_bp.route("/shipping/test/<provider_key>", methods=["POST"])
def shipping_test(provider_key):
    if provider_key not in SHIPPING_PROVIDERS:
        return jsonify({"status": "error", "message": "Unknown shipping provider"}), 404
    provider, conn = _provider(provider_key)
    result = provider.test_connection()

    conn.status = "connected" if result.get("ok") else "error"
    conn.status_detail = result.get("message", "")
    if result.get("ok"):
        conn.enabled = True
        conn.connected_at = conn.connected_at or utcnow()
    conn.updated_at = utcnow()
    db.session.commit()
    return jsonify({"status": "success" if result.get("ok") else "error",
                    "message": result.get("message", ""),
                    "connected": bool(result.get("ok"))})


@shipping_bp.route("/shipping/disconnect/<provider_key>", methods=["POST"])
def shipping_disconnect(provider_key):
    if provider_key not in SHIPPING_PROVIDERS:
        return jsonify({"status": "error", "message": "Unknown shipping provider"}), 404
    conn = _get_connection(provider_key)
    if conn:
        cfg = dict(conn.config or {})
        cfg.pop("api_key", None)   # drop the secret, keep the addresses/preferences
        conn.config = cfg
        conn.enabled = False
        conn.status = "disconnected"
        conn.status_detail = "Disconnected."
        conn.connected_at = None
        db.session.commit()
    return jsonify({"status": "success", "message": "Disconnected."})


# ──────────────────────────────────────────────────────────────────────────────
# Orders
# ──────────────────────────────────────────────────────────────────────────────
@shipping_bp.route("/shipping/orders")
def shipping_orders():
    page = max(int(request.args.get("page", 1) or 1), 1)
    per_page = min(max(int(request.args.get("per_page", 25) or 25), 1), 100)
    status = (request.args.get("status") or "").strip()
    q = (request.args.get("q") or "").strip()

    query = Order.query
    if status and status != "all":
        query = query.filter(Order.status == status)
    if q:
        like = f"%{q}%"
        query = query.filter(db.or_(Order.external_order_id.ilike(like),
                                    Order.buyer_name.ilike(like),
                                    Order.buyer_company.ilike(like)))
    total = query.count()
    rows = (query.order_by(Order.created_at.desc(), Order.id.desc())
            .offset((page - 1) * per_page).limit(per_page).all())
    return jsonify({
        "status": "success",
        "orders": [_order_json(o) for o in rows],
        "page": page, "per_page": per_page, "total": total,
        "pages": max((total + per_page - 1) // per_page, 1),
    })


@shipping_bp.route("/shipping/orders/<int:order_id>")
def shipping_order_detail(order_id):
    o = Order.query.get(order_id)
    if not o:
        return jsonify({"status": "error", "message": "Order not found"}), 404
    return jsonify({"status": "success", "order": _order_json(o, full=True)})


def _apply_order_form(o, f):
    """Copy submitted address/value fields onto an order."""
    o.buyer_name = (f.get("buyer_name", o.buyer_name or "") or "").strip() or None
    o.buyer_company = (f.get("buyer_company", o.buyer_company or "") or "").strip() or None
    o.address1 = (f.get("address1", o.address1 or "") or "").strip() or None
    o.address2 = (f.get("address2", o.address2 or "") or "").strip() or None
    o.city = (f.get("city", o.city or "") or "").strip() or None
    o.state = normalize_state(f.get("state", o.state or "")) or None
    o.zipcode = (f.get("zipcode", o.zipcode or "") or "").strip() or None
    o.country = normalize_country(f.get("country", o.country or "US"))
    o.email = (f.get("email", o.email or "") or "").strip() or None
    o.phone = (f.get("phone", o.phone or "") or "").strip() or None
    if f.get("notes") is not None:
        o.notes = (f.get("notes") or "").strip() or None
    for key, attr in (("shipping_paid", "shipping_paid"), ("item_total", "item_total")):
        if f.get(key) not in (None, ""):
            try:
                setattr(o, attr, round(float(f.get(key)), 2))
            except (TypeError, ValueError):
                pass


@shipping_bp.route("/shipping/orders/create", methods=["POST"])
def shipping_order_create():
    f = request.form
    external = (f.get("external_order_id") or "").strip() or None
    source = (f.get("source") or "manual").strip() or "manual"

    if external and Order.query.filter_by(source=source, external_order_id=external).first():
        return jsonify({"status": "error",
                        "message": f"Order {external} already exists for {source}."}), 409

    o = Order(source=source, external_order_id=external)
    _apply_order_form(o, f)
    db.session.add(o)
    db.session.flush()

    # Items arrive as parallel arrays from the form.
    names = f.getlist("item_name")
    qtys = f.getlist("item_qty")
    prices = f.getlist("item_price")
    for i, name in enumerate(names):
        name = (name or "").strip()
        if not name:
            continue
        try:
            qty = max(int(qtys[i]), 1) if i < len(qtys) and qtys[i] else 1
        except (TypeError, ValueError):
            qty = 1
        try:
            price = round(float(prices[i]), 2) if i < len(prices) and prices[i] else 0.0
        except (TypeError, ValueError):
            price = 0.0
        db.session.add(OrderItem(order_id=o.id, name=name[:300], qty=qty, price=price))

    db.session.flush()
    o.item_total = o.declared_value
    o.status = "ready" if o.address_complete else "needs_address"
    db.session.commit()
    return jsonify({"status": "success", "message": f"Order #{o.id} created.",
                    "order": _order_json(o, full=True)})


@shipping_bp.route("/shipping/orders/<int:order_id>/update", methods=["POST"])
def shipping_order_update(order_id):
    o = Order.query.get(order_id)
    if not o:
        return jsonify({"status": "error", "message": "Order not found"}), 404
    _apply_order_form(o, request.form)
    if o.status in ("needs_address", "ready"):
        o.status = "ready" if o.address_complete else "needs_address"
    db.session.commit()
    return jsonify({"status": "success", "message": "Order saved.",
                    "order": _order_json(o, full=True)})


@shipping_bp.route("/shipping/orders/<int:order_id>/delete", methods=["POST"])
def shipping_order_delete(order_id):
    o = Order.query.get(order_id)
    if not o:
        return jsonify({"status": "error", "message": "Order not found"}), 404
    if any(s.status == "purchased" and s.provider == "easypost" for s in o.shipments):
        return jsonify({"status": "error",
                        "message": "This order has purchased postage. Refund or void it in EasyPost "
                                   "first — deleting it here won't get your money back. Then mark "
                                   "the shipment voided here to unlock deletion."}), 400
    # Clean up label files; leaving orphans behind would quietly grow storage.
    label_root = os.path.join(current_app.config["UPLOAD_FOLDER"], LABEL_SUBDIR, str(o.id))
    if os.path.isdir(label_root):
        for name in os.listdir(label_root):
            try:
                os.remove(os.path.join(label_root, name))
            except OSError:
                pass
        try:
            os.rmdir(label_root)
        except OSError:
            pass
    db.session.delete(o)
    db.session.commit()
    return jsonify({"status": "success", "message": f"Order #{order_id} deleted."})


# ──────────────────────────────────────────────────────────────────────────────
# CSV import (TCGplayer / ManaPool / eBay shipping exports)
# ──────────────────────────────────────────────────────────────────────────────
# Marketplaces all export "an order per row" but agree on nothing else, so map
# a generous set of header spellings onto our fields. Everything is matched
# case- and punctuation-insensitively.
_CSV_MAP = {
    "external_order_id": ("order #", "order number", "order id", "ordernumber", "orderid", "order"),
    "buyer_name": ("name", "full name", "customer name", "buyer name", "ship to name", "recipient"),
    "first_name": ("firstname", "first name"),
    "last_name": ("lastname", "last name"),
    "buyer_company": ("company", "company name", "ship to company"),
    "address1": ("address1", "address line 1", "address", "street", "street 1", "ship to address 1"),
    "address2": ("address2", "address line 2", "street 2", "ship to address 2", "apartment"),
    "city": ("city", "ship to city"),
    "state": ("state", "province", "ship to state", "state/province"),
    "zipcode": ("postalcode", "postal code", "zip", "zipcode", "zip code", "ship to zip"),
    "country": ("country", "ship to country"),
    "email": ("email", "email address", "buyer email"),
    "phone": ("phone", "phone number"),
    "item_name": ("product name", "item", "item name", "product", "card name", "description"),
    "item_qty": ("quantity", "qty", "item quantity"),
    "item_price": ("price", "unit price", "item price", "product price"),
    "item_set": ("set", "set name", "edition", "expansion"),
    "item_condition": ("condition",),
    "value": ("value of products", "order total", "total", "item total", "product value"),
    "shipping_paid": ("shipping", "shipping fee", "shipping paid", "shipping price"),
}


def _norm_header(h):
    return re.sub(r"[^a-z0-9 ]+", "", str(h or "").strip().lower()).strip()


def _build_header_index(fieldnames):
    """Map our field -> actual CSV column name."""
    norm_to_actual = {_norm_header(h): h for h in (fieldnames or []) if h}
    index = {}
    for field, aliases in _CSV_MAP.items():
        for alias in aliases:
            if alias in norm_to_actual:
                index[field] = norm_to_actual[alias]
                break
    return index


def _row_get(row, index, field, default=""):
    col = index.get(field)
    if not col:
        return default
    return (row.get(col) or "").strip()


@shipping_bp.route("/shipping/orders/import_csv", methods=["POST"])
def shipping_import_csv():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "Choose a CSV file to import."}), 400
    source = (request.form.get("source") or "csv").strip() or "csv"

    try:
        raw = file.read().decode("utf-8-sig", errors="replace")
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Couldn't read that file — {exc}"}), 400

    try:
        reader = csv.DictReader(io.StringIO(raw))
        rows = list(reader)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Couldn't parse that CSV — {exc}"}), 400
    if not rows:
        return jsonify({"status": "error", "message": "That CSV has no rows."}), 400

    index = _build_header_index(reader.fieldnames)
    if "external_order_id" not in index:
        cols = ", ".join(reader.fieldnames or [])
        return jsonify({"status": "error",
                        "message": f"No order-number column found. Looked for 'Order #' or similar; "
                                   f"the file has: {cols[:200]}"}), 400

    created, updated, skipped, line_items = 0, 0, 0, 0
    # One order can span several rows (one per item), so group by order number.
    grouped = {}
    for row in rows:
        oid = _row_get(row, index, "external_order_id")
        if not oid:
            skipped += 1
            continue
        grouped.setdefault(oid, []).append(row)

    for oid, group in grouped.items():
        head = group[0]
        order = Order.query.filter_by(source=source, external_order_id=oid).first()
        is_new = order is None
        if is_new:
            order = Order(source=source, external_order_id=oid)
            db.session.add(order)
            db.session.flush()

        name = _row_get(head, index, "buyer_name")
        if not name:
            first = _row_get(head, index, "first_name")
            last = _row_get(head, index, "last_name")
            name = " ".join(p for p in (first, last) if p)
        if name:
            order.buyer_name = name[:200]
        for field, attr, cap in (
            ("buyer_company", "buyer_company", 200), ("address1", "address1", 200),
            ("address2", "address2", 200), ("city", "city", 120),
            ("zipcode", "zipcode", 12), ("email", "email", 200), ("phone", "phone", 40),
        ):
            v = _row_get(head, index, field)
            if v:
                setattr(order, attr, v[:cap])
        st = _row_get(head, index, "state")
        if st:
            order.state = normalize_state(st)
        country = _row_get(head, index, "country")
        if country:
            # Exports write "US" or "United States" interchangeably.
            order.country = normalize_country(country)
        sp = _row_get(head, index, "shipping_paid")
        if sp:
            try:
                order.shipping_paid = round(float(re.sub(r"[^0-9.\-]", "", sp) or 0), 2)
            except ValueError:
                pass

        # Items: only rebuild them for new orders, so a re-import never
        # duplicates lines on an order someone has already edited.
        if is_new:
            for row in group:
                iname = _row_get(row, index, "item_name")
                if not iname:
                    continue
                try:
                    qty = max(int(float(_row_get(row, index, "item_qty", "1") or 1)), 1)
                except ValueError:
                    qty = 1
                try:
                    price = round(float(re.sub(r"[^0-9.\-]", "",
                                               _row_get(row, index, "item_price", "0")) or 0), 2)
                except ValueError:
                    price = 0.0
                db.session.add(OrderItem(
                    order_id=order.id, name=iname[:300], qty=qty, price=price,
                    set_name=(_row_get(row, index, "item_set") or None or "")[:200] or None,
                    condition=(_row_get(row, index, "item_condition") or "")[:60] or None,
                ))
                line_items += 1
            db.session.flush()

            if not order.items:
                # No item columns — fall back to a declared order value.
                val = _row_get(head, index, "value")
                if val:
                    try:
                        order.item_total = round(float(re.sub(r"[^0-9.\-]", "", val) or 0), 2)
                    except ValueError:
                        order.item_total = 0.0
            else:
                order.item_total = order.declared_value

        if order.status in ("needs_address", "ready"):
            order.status = "ready" if order.address_complete else "needs_address"
        created += 1 if is_new else 0
        updated += 0 if is_new else 1

    db.session.commit()
    msg = f"Imported {created} new order(s)"
    if updated:
        msg += f", updated {updated}"
    if line_items:
        msg += f", {line_items} line item(s)"
    if skipped:
        msg += f", skipped {skipped} row(s) with no order number"
    return jsonify({"status": "success", "message": msg + ".",
                    "created": created, "updated": updated, "skipped": skipped})


# ──────────────────────────────────────────────────────────────────────────────
# Rates + labels
# ──────────────────────────────────────────────────────────────────────────────
@shipping_bp.route("/shipping/quote/<int:order_id>", methods=["POST"])
def shipping_quote(order_id):
    o = Order.query.get(order_id)
    if not o:
        return jsonify({"status": "error", "message": "Order not found"}), 404
    key = (request.form.get("provider") or "").strip()
    if key not in SHIPPING_PROVIDERS:
        return jsonify({"status": "error", "message": "Pick a shipping provider."}), 400

    problems = _validate_destination(o)
    if problems:
        return jsonify({"status": "error",
                        "message": "This order still needs " + ", ".join(problems) + "."}), 400

    provider, _ = _provider(key)
    opts = {k: v for k, v in request.form.items() if k != "provider"}
    result = provider.quote(o, opts)
    return jsonify({"status": "success" if result.get("ok") else "error",
                    "message": result.get("message", ""),
                    "rates": result.get("rates", []),
                    "shipment_id": result.get("shipment_id", "")})


@shipping_bp.route("/shipping/label/<int:order_id>", methods=["POST"])
def shipping_create_label(order_id):
    o = Order.query.get(order_id)
    if not o:
        return jsonify({"status": "error", "message": "Order not found"}), 404
    key = (request.form.get("provider") or "").strip()
    if key not in SHIPPING_PROVIDERS:
        return jsonify({"status": "error", "message": "Pick a shipping provider."}), 400

    conn = _get_connection(key)
    if not conn or not conn.enabled:
        return jsonify({"status": "error",
                        "message": f"Connect {SHIPPING_PROVIDERS[key]['label']} first."}), 400

    opts = {}
    for k in ("envelope_size", "rate_id", "shipment_id", "weight_oz", "length", "width", "height"):
        v = (request.form.get(k) or "").strip()
        if v:
            opts[k] = v
    if request.form.get("insured") is not None:
        opts["insured"] = request.form.get("insured") in ("1", "true", "on", "yes")

    provider, _ = _provider(key)
    result = provider.create_label(o, opts)

    if not result.get("ok"):
        # Record the failure against the order so it's visible later, not just
        # in a toast the user already dismissed.
        s = Shipment(order_id=o.id, provider=key, status="error",
                     error=result.get("message", "")[:500])
        db.session.add(s)
        db.session.commit()
        return jsonify({"status": "error", "message": result.get("message", "Label failed.")}), 502

    s = Shipment(
        order_id=o.id, provider=key,
        external_id=result.get("external_id") or None,
        tracking_code=result.get("tracking_code") or None,
        tracking_url=result.get("tracking_url") or None,
        carrier=result.get("carrier") or None,
        service=result.get("service") or None,
        envelope_size=result.get("envelope_size") or None,
        rate_amount=result.get("rate_amount"),
        currency=result.get("currency", "USD"),
        insured=bool(result.get("insured")),
        insurance_amount=result.get("insurance_amount"),
        label_format=result.get("label_format", "PDF"),
        status="purchased" if SHIPPING_PROVIDERS[key]["buys_postage"] else "created",
        events=[],
    )
    db.session.add(s)
    db.session.flush()

    label_bytes = result.get("label_bytes")
    if label_bytes:
        try:
            s.label_path = _save_label(o.id, s.id, label_bytes,
                                       ext=result.get("label_format", "PDF").lower())
        except OSError as exc:
            s.error = f"Label bought but couldn't be saved to disk — {exc}"

    _sync_order_status(o)
    db.session.commit()
    return jsonify({"status": "success", "message": result.get("message", "Label created."),
                    "order": _order_json(o, full=True), "shipment_id": s.id,
                    "duplicate": bool(result.get("duplicate"))})


@shipping_bp.route("/shipping/shipments/<int:shipment_id>/void", methods=["POST"])
def shipping_shipment_void(shipment_id):
    """Locally mark a shipment voided after it was refunded/voided with the
    provider. Nothing here talks to the provider — this is the bookkeeping
    exit off "purchased" that order deletion checks; without it, nothing ever
    transitioned a shipment off that status and any order with bought postage
    was permanently undeletable (and Shipment's documented "voided" state was
    unreachable)."""
    s = Shipment.query.get(shipment_id)
    if not s:
        return jsonify({"status": "error", "message": "Shipment not found"}), 404
    if s.status not in ("purchased", "created", "error"):
        return jsonify({"status": "error",
                        "message": f"A shipment in status '{s.status}' can't be voided."}), 400
    # A delivered/shipped package still has status "purchased" (tracking_status
    # is a separate field) — voiding it would reset a live order backwards.
    if s.is_delivered or (s.order and s.order.status in ("shipped", "delivered")):
        return jsonify({"status": "error",
                        "message": "This shipment already moved — carriers won't refund it and "
                                   "voiding here would corrupt the order's history."}), 400
    s.status = "voided"
    if s.order:
        # latest_shipment only counts purchased/created, so the order falls
        # back to ready/needs_address and can be re-labeled or deleted.
        _sync_order_status(s.order)
    db.session.commit()
    return jsonify({"status": "success",
                    "message": "Shipment marked voided. If postage was bought, make sure the "
                               "refund was requested in the provider's dashboard.",
                    "order": _order_json(s.order, full=True) if s.order else None})


@shipping_bp.route("/shipping/label/<int:shipment_id>/reprint", methods=["POST"])
def shipping_reprint(shipment_id):
    s = Shipment.query.get(shipment_id)
    if not s:
        return jsonify({"status": "error", "message": "Shipment not found"}), 404
    provider, _ = _provider(s.provider)
    size = (request.form.get("envelope_size") or "").strip() or None

    result = provider.relabel(s, size) if s.provider == "tcgtracking" else provider.relabel(s)
    if not result.get("ok"):
        return jsonify({"status": "error", "message": result.get("message", "Reprint failed.")}), 502

    if size:
        s.envelope_size = size
    try:
        s.label_path = _save_label(s.order_id, s.id, result["label_bytes"],
                                   ext=result.get("label_format", "PDF").lower())
    except OSError as exc:
        return jsonify({"status": "error", "message": f"Couldn't save the label — {exc}"}), 500
    db.session.commit()
    return jsonify({"status": "success", "message": "Label regenerated.", "shipment_id": s.id})


# ──────────────────────────────────────────────────────────────────────────────
# Serving + printing labels
# ──────────────────────────────────────────────────────────────────────────────
@shipping_bp.route("/shipping/label/<int:shipment_id>/view")
def shipping_label_view(shipment_id):
    s = Shipment.query.get(shipment_id)
    if not s:
        return jsonify({"status": "error", "message": "Shipment not found"}), 404
    path = _label_abs(s)
    if not path:
        return jsonify({"status": "error",
                        "message": "No label file on disk. Use Reprint to fetch it again."}), 404
    mimetype = "application/pdf" if path.lower().endswith(".pdf") else "image/png"
    return send_file(path, mimetype=mimetype, as_attachment=False,
                     download_name=os.path.basename(path))


@shipping_bp.route("/shipping/label/<int:shipment_id>/download")
def shipping_label_download(shipment_id):
    s = Shipment.query.get(shipment_id)
    if not s:
        return jsonify({"status": "error", "message": "Shipment not found"}), 404
    path = _label_abs(s)
    if not path:
        return jsonify({"status": "error", "message": "No label file on disk."}), 404
    order = s.order
    stem = (order.external_order_id or f"order-{order.id}") if order else f"shipment-{s.id}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(stem))
    ext = os.path.splitext(path)[1] or ".pdf"
    return send_file(path, mimetype="application/pdf", as_attachment=True,
                     download_name=f"label-{safe}{ext}")


@shipping_bp.route("/shipping/print")
def shipping_print():
    """
    Merge the labels for several orders into one PDF for a single print job.

    Uses PyMuPDF, which app.py already depends on for PDF import. If it isn't
    installed, say so plainly rather than 500 — the per-order label still
    prints fine on its own.
    """
    ids = [i for i in (request.args.get("ids") or "").split(",") if i.strip().isdigit()]
    if not ids:
        return jsonify({"status": "error", "message": "No orders selected."}), 400

    try:
        import fitz   # PyMuPDF
    except ImportError:
        return jsonify({"status": "error",
                        "message": "Batch printing needs PyMuPDF (pip install PyMuPDF). "
                                   "You can still print each label individually."}), 501

    paths, missing = [], []
    for oid in ids:
        o = Order.query.get(int(oid))
        if not o:
            continue
        s = o.latest_shipment
        p = _label_abs(s) if s else None
        if p and p.lower().endswith(".pdf"):
            paths.append(p)
        else:
            missing.append(o.external_order_id or f"#{o.id}")

    if not paths:
        return jsonify({"status": "error",
                        "message": "None of those orders have a label PDF yet."}), 400

    merged = fitz.open()
    try:
        for p in paths:
            with fitz.open(p) as src:
                merged.insert_pdf(src)
        buf = merged.tobytes()
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Couldn't merge the labels — {exc}"}), 500
    finally:
        merged.close()

    resp = send_file(io.BytesIO(buf), mimetype="application/pdf", as_attachment=False,
                     download_name=f"labels-{utcnow():%Y%m%d-%H%M}.pdf")
    if missing:
        resp.headers["X-Labels-Missing"] = ",".join(missing[:20])
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# Tracking
# ──────────────────────────────────────────────────────────────────────────────
def _refresh_shipment(s):
    """Poll one shipment and fold the result into the row. Returns the result dict."""
    provider, conn = _provider(s.provider)
    if provider is None or not conn or not conn.enabled:
        return {"ok": False, "message": f"{s.provider} isn't connected."}
    result = provider.track(s)
    s.last_tracked_at = utcnow()
    if not result.get("ok"):
        return result
    s.tracking_status = result.get("status") or s.tracking_status
    s.tracking_percent = result.get("percent")
    s.delivery_date = result.get("delivery_date") or s.delivery_date
    if result.get("tracking_code") and not s.tracking_code:
        s.tracking_code = result["tracking_code"]
    if result.get("tracking_url") and not s.tracking_url:
        s.tracking_url = result["tracking_url"]
    s.events = result.get("events") or []
    if s.order:
        _sync_order_status(s.order)
    return result


@shipping_bp.route("/shipping/track/<int:shipment_id>", methods=["POST"])
def shipping_track_one(shipment_id):
    s = Shipment.query.get(shipment_id)
    if not s:
        return jsonify({"status": "error", "message": "Shipment not found"}), 404
    result = _refresh_shipment(s)
    db.session.commit()
    return jsonify({"status": "success" if result.get("ok") else "error",
                    "message": result.get("message", ""),
                    "shipment": {
                        "id": s.id,
                        "tracking_status": s.tracking_status or "",
                        "tracking_percent": s.tracking_percent,
                        "delivery_date": s.delivery_date or "",
                        "events": s.events or [],
                        # None until the first successful poll — e.g. a refresh
                        # clicked while the provider is disconnected.
                        "last_tracked_at": (s.last_tracked_at.strftime("%Y-%m-%d %H:%M")
                                            if s.last_tracked_at else ""),
                    },
                    "order_status": s.order.status if s.order else ""})


@shipping_bp.route("/shipping/track/refresh", methods=["POST"])
def shipping_track_refresh():
    """
    Poll every shipment that's still moving.

    Deliberately bounded: only orders in flight, only shipments not polled in
    the last `min_age_minutes`, newest first, capped at `limit`. USPS scans
    land a few times a day, so hammering the API on every page load would spend
    rate limit for nothing.
    """
    try:
        limit = min(max(int(request.form.get("limit") or 50), 1), 200)
    except ValueError:
        limit = 50
    try:
        min_age = max(int(request.form.get("min_age_minutes") or 30), 0)
    except ValueError:
        min_age = 30

    enabled = _connected_provider_keys()
    if not enabled:
        return jsonify({"status": "success",
                        "message": "No shipping provider is connected.",
                        "checked": 0, "moved": 0, "failed": 0})

    cutoff = utcnow() - timedelta(minutes=min_age)
    q = (Shipment.query.join(Order, Shipment.order_id == Order.id)
         .filter(Shipment.provider.in_(enabled))
         .filter(Shipment.status.in_(("purchased", "created")))
         .filter(Order.status.in_(_ACTIVE_STATUSES))
         .filter(db.or_(Shipment.last_tracked_at.is_(None), Shipment.last_tracked_at <= cutoff))
         .order_by(Shipment.created_at.desc())
         .limit(limit))

    checked, moved, failed = 0, 0, 0
    for s in q.all():
        before = s.tracking_status
        r = _refresh_shipment(s)
        checked += 1
        if not r.get("ok"):
            failed += 1
        elif s.tracking_status != before:
            moved += 1
    db.session.commit()

    if not checked:
        msg = "Nothing to check — every shipment is delivered or was polled recently."
    else:
        msg = f"Checked {checked} shipment(s); {moved} changed status."
        if failed:
            msg += f" {failed} couldn't be reached."
    return jsonify({"status": "success", "message": msg,
                    "checked": checked, "moved": moved, "failed": failed})


# ──────────────────────────────────────────────────────────────────────────────
# Optional background tracking poller (opt-in via SHIPPING_POLL_BACKGROUND=1).
# The "Refresh tracking" button is the primary, always-available path — same
# pattern as the email monitor's poller.
# ──────────────────────────────────────────────────────────────────────────────
_poller_started = False


def start_tracking_poller(app, interval_minutes=180):
    global _poller_started
    if _poller_started:
        return
    if os.environ.get("SHIPPING_POLL_BACKGROUND", "") != "1":
        return
    _poller_started = True

    import threading

    def _loop():
        import time
        while True:
            time.sleep(max(interval_minutes, 15) * 60)
            try:
                with app.app_context():
                    enabled = _connected_provider_keys()
                    if not enabled:
                        continue
                    cutoff = utcnow() - timedelta(minutes=60)
                    rows = (Shipment.query.join(Order, Shipment.order_id == Order.id)
                            .filter(Shipment.provider.in_(enabled))
                            .filter(Shipment.status.in_(("purchased", "created")))
                            .filter(Order.status.in_(_ACTIVE_STATUSES))
                            .filter(db.or_(Shipment.last_tracked_at.is_(None),
                                           Shipment.last_tracked_at <= cutoff))
                            .limit(100).all())
                    for s in rows:
                        _refresh_shipment(s)
                    db.session.commit()
            except Exception:
                # A poller must never take the app down; the manual button
                # remains available and will surface any real error.
                pass

    t = threading.Thread(target=_loop, daemon=True, name="shipping-tracking-poller")
    t.start()
