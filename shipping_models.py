"""
shipping_models.py
==================

Order / shipment tables for the Shipping tab.

Kept in their own module (rather than appended to models.py) so the shipping
feature is a drop-in: importing this module registers the tables on the same
SQLAlchemy metadata, and the existing `db.create_all()` at startup creates them.
Nothing in models.py needs to change.

Shape
-----
Order      — one buyer-facing order, from a marketplace sale email, a CSV, or
             typed in by hand. Holds the ship-to address.
OrderItem  — one sold line on that order, optionally linked back to the
             ScanRecord it came from (so the Sold page and analytics still line
             up) and to the SaleEvent that created it.
Shipment   — one label bought for an order. An order can have several (a
             reprint in a different size, or a replacement after a void), but
             only the newest non-error row is treated as "the" label.

A note on money: rate_amount / insurance_amount are what the *carrier or
service* charged, recorded at purchase time. They are never recomputed later —
a rate quoted today is not the rate you paid last week.
"""

from models import db, utcnow


# Order.status lifecycle:
#   needs_address -> ready -> labeled -> shipped -> delivered
# `needs_address` is the landing state for orders created from sale emails,
# which carry the items but rarely a usable postal address.
ORDER_STATUSES = ("needs_address", "ready", "labeled", "shipped", "delivered", "cancelled")


class Order(db.Model):
    """
    A single order to fulfil. `external_order_id` is the marketplace's own order
    number (a TCGplayer order #, eBay order id, ...); it doubles as the
    idempotency key with `source`, and is what gets sent to a shipping provider
    as its `source_order_id` so re-posting the same order never buys a second
    label.
    """
    __tablename__ = "orders"
    id                = db.Column(db.Integer, primary_key=True)
    source            = db.Column(db.String(30), default="manual", index=True)  # tcgplayer|ebay|shopify|manual|csv
    external_order_id = db.Column(db.String(120), nullable=True, index=True)
    status            = db.Column(db.String(20), default="needs_address", index=True)

    # Ship-to. Deliberately flat/denormalized: this is a shipping snapshot, not
    # a customer record, and it must stay exactly as it was when the label was
    # bought even if the buyer later changes their address.
    buyer_name    = db.Column(db.String(200), nullable=True)
    buyer_company = db.Column(db.String(200), nullable=True)
    address1      = db.Column(db.String(200), nullable=True)
    address2      = db.Column(db.String(200), nullable=True)
    city          = db.Column(db.String(120), nullable=True)
    state         = db.Column(db.String(10), nullable=True)   # 2-letter US state / CA province
    zipcode       = db.Column(db.String(12), nullable=True)
    country       = db.Column(db.String(2), default="US")
    email         = db.Column(db.String(200), nullable=True)
    phone         = db.Column(db.String(40), nullable=True)

    item_total    = db.Column(db.Float, default=0.0)   # sum(qty * price) of items
    shipping_paid = db.Column(db.Float, default=0.0)   # what the buyer paid for shipping
    notes         = db.Column(db.Text, nullable=True)

    created_at    = db.Column(db.DateTime, default=utcnow, index=True)
    updated_at    = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)
    shipped_at    = db.Column(db.DateTime, nullable=True)

    items = db.relationship("OrderItem", backref="order", cascade="all, delete-orphan",
                            order_by="OrderItem.id")
    shipments = db.relationship("Shipment", backref="order", cascade="all, delete-orphan",
                                order_by="Shipment.id")

    __table_args__ = (
        db.UniqueConstraint("source", "external_order_id", name="uq_order_source_external"),
    )

    # -- derived -------------------------------------------------------------
    @property
    def declared_value(self):
        """Order value used for insurance + customs. Items win over any stored total."""
        if self.items:
            return round(sum((i.qty or 0) * (i.price or 0.0) for i in self.items), 2)
        return round(self.item_total or 0.0, 2)

    @property
    def latest_shipment(self):
        """Newest shipment that actually produced a label, else None."""
        good = [s for s in self.shipments if s.status in ("purchased", "created")]
        return good[-1] if good else None

    @property
    def address_complete(self):
        """True when there's enough here for a provider to accept the order."""
        need = [self.address1, self.city, self.state, self.zipcode]
        named = self.buyer_name or self.buyer_company
        return bool(named and all(str(v or "").strip() for v in need))

    def __repr__(self):
        return f"<Order {self.source}:{self.external_order_id or self.id} {self.status}>"


class OrderItem(db.Model):
    """
    One sold line. `record_id` is a soft link back to inventory — it stays
    nullable because a sale email line doesn't always resolve to a scan, and an
    order still needs to ship whether or not the match succeeded.
    """
    __tablename__ = "order_items"
    id            = db.Column(db.Integer, primary_key=True)
    order_id      = db.Column(db.Integer, db.ForeignKey("orders.id"), nullable=False, index=True)
    record_id     = db.Column(db.Integer, db.ForeignKey("scan_records.id"), nullable=True)
    sale_event_id = db.Column(db.Integer, db.ForeignKey("sale_events.id"), nullable=True)

    name      = db.Column(db.String(300), nullable=False)
    set_name  = db.Column(db.String(200), nullable=True)
    condition = db.Column(db.String(60), nullable=True)
    foil      = db.Column(db.Boolean, default=False)
    qty       = db.Column(db.Integer, default=1)
    price     = db.Column(db.Float, default=0.0)   # unit price

    record = db.relationship("ScanRecord")

    def __repr__(self):
        return f"<OrderItem {self.name} x{self.qty}>"


class Shipment(db.Model):
    """
    One purchased/created label.

    `label_path` is upload-relative (same convention as ScanRecord.image_path)
    and always points at a PDF — EasyPost is explicitly asked for PDF output so
    that batch printing can merge labels from either provider into one sheet.

    `events` caches the provider's scan history as a list of dicts so the
    tracking timeline renders without a network round-trip on every page load.
    """
    __tablename__ = "shipments"
    id          = db.Column(db.Integer, primary_key=True)
    order_id    = db.Column(db.Integer, db.ForeignKey("orders.id"), nullable=False, index=True)
    provider    = db.Column(db.String(30), nullable=False)   # tcgtracking|easypost

    external_id   = db.Column(db.String(120), nullable=True)  # tcgtracking order_id / easypost shp_...
    tracking_code = db.Column(db.String(120), nullable=True, index=True)
    tracking_url  = db.Column(db.String(500), nullable=True)

    carrier     = db.Column(db.String(60), nullable=True)     # USPS, UPS, ...
    service     = db.Column(db.String(80), nullable=True)     # GroundAdvantage, IMB envelope, ...
    envelope_size = db.Column(db.String(30), nullable=True)   # tcgtracking size id (10, L64, ...)

    rate_amount      = db.Column(db.Float, nullable=True)     # postage actually charged
    currency         = db.Column(db.String(10), default="USD")
    insured          = db.Column(db.Boolean, default=False)
    insurance_amount = db.Column(db.Float, nullable=True)

    label_path   = db.Column(db.String(255), nullable=True)   # upload-relative PDF
    label_format = db.Column(db.String(10), default="PDF")

    status = db.Column(db.String(20), default="created")      # created|purchased|error|voided
    error  = db.Column(db.Text, nullable=True)

    # Tracking cache
    tracking_status  = db.Column(db.String(60), nullable=True)   # provider's own status string
    tracking_percent = db.Column(db.Integer, nullable=True)      # 0-100 progress, when offered
    delivery_date    = db.Column(db.String(30), nullable=True)
    last_tracked_at  = db.Column(db.DateTime, nullable=True)
    events           = db.Column(db.JSON, default=list)

    created_at = db.Column(db.DateTime, default=utcnow)

    @property
    def is_delivered(self):
        return (self.tracking_status or "").strip().lower() == "delivered"

    def __repr__(self):
        return f"<Shipment {self.provider} {self.tracking_code or '-'} {self.status}>"
