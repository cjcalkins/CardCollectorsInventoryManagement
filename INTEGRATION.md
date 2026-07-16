# Shipping integration — orders, labels, tracking

Adds a **Shipping** tab: marketplace orders in, shipping labels out, USPS tracking back.

---

## Two things I found before writing any code

### 1. `tcgtracker.com` is not the site you want

`tcgtracker.com` is a **Pokémon collection tracker** — cards you own, set completion, price
snapshots. It has no orders, no labels, no postage.

The service that does everything you described — import orders from TCGplayer/ManaPool/eBay,
print envelopes, USPS tracking, EasyPost postage, PIP insurance — is **`tcgtracking.com`**
(with the *-ing*). That's what this integration targets. Worth a look before you wire it to a
live account, in case you did mean a third site.

### 2. TCGTracking's API can't buy EasyPost postage

TCGTracking *does* integrate EasyPost — but only inside their dashboard. Their public API is
exactly three endpoints:

| Endpoint | Does |
|---|---|
| `POST /api/v1/orders` | Create order → IMB envelope + PDF label |
| `GET /api/v1/orders/track` | USPS Informed Visibility scan events |
| `GET /api/v1/orders/{id}/pdf` | Regenerate the PDF |

There is no rate endpoint, no postage endpoint, no EasyPost endpoint. So
"TCGTracking → EasyPost postage purchase" as a single API integration isn't buildable — the
capability isn't exposed.

Also worth being clear on, because it's easy to miss: **an IMB is tracking, not postage.**
A TCGTracking envelope still needs a stamp on it. It's not a cheaper way to buy postage;
it's free tracking on mail you're franking yourself.

**So I built both**, behind one interface, because they do different jobs:

| | TCGTracking | EasyPost |
|---|---|---|
| Buys postage | ✗ (you stamp it) | ✓ real postage |
| Tracking | ✓ free USPS IV | ✓ carrier tracking |
| Best for | plain envelope, few cards | bubble mailer, parcels |
| Cost | per-label fee after 100 free, PIP insurance | postage + EasyPost fees |
| Insurance | PIP, capped at $50 | full declared value |

Per order you pick a provider. Both produce a PDF, both store it, both print, both track.
If you only ever ship stamped envelopes, connect TCGTracking alone and ignore EasyPost.

---

## Files

| File | New? | What |
|---|---|---|
| `shipping_models.py` | new | `Order`, `OrderItem`, `Shipment` |
| `shipping_providers.py` | new | TCGTracking + EasyPost connectors |
| `shipping_routes.py` | new | Blueprint: all `/shipping/*` routes |
| `templates/shipping.html` | new | The tab |
| `email_monitor.py` | **replaces yours** | adds `parse_shipping_address()` |
| `app.py` | 4 small patches | see below |

`models.py` is **untouched** — the new tables live in `shipping_models.py` on the same
metadata, so your existing `db.create_all()` creates them on next start. No migration needed.

Credentials reuse the `shop_connections` table (rows `tcgtracking` / `easypost`). The Shops
page only iterates `MARKETPLACES`, so they never show up there — but they inherit its secret
masking and storage. **`inventory.db` now also holds shipping API keys.**

---

## app.py patches

### 1 — Register the blueprint

After line ~12253 (`from shop_providers import MARKETPLACES, ...`):

```python
# ====================== SHIPPING (labels + tracking) ======================
# Imported at module level so shipping_models' tables are registered on
# db.metadata before the db.create_all() in __main__ runs.
import shipping_models  # noqa: F401  (registers Order/OrderItem/Shipment)
from shipping_routes import shipping_bp, ensure_order_from_sale, start_tracking_poller

app.register_blueprint(shipping_bp)
```

### 2 — Let the permission gate see `/shipping`

In `_resource_for_path`, in the dict at line ~2496, add one entry:

```python
        "shops": "shops", "shop": "shops",
        "shipping": "shops",          # <-- add: Shipping reuses the Shops permission
```

Without this, `/shipping` falls through to `None` and any signed-in user can reach it.
If you'd rather gate it separately, add `("shipping", "Shipping / labels")` to
`PROTECTED_RESOURCES` and map `"shipping": "shipping"` instead.

### 3 — Turn sales into orders

In `_process_sale_email` (~line 13719), collect the events and hand them over:

```python
    items = parsed.get("items", [])
    ...
    created_events = []                     # <-- add
    for item in items:
        ...
        db.session.add(ev)
        db.session.flush()
        created_events.append(ev)           # <-- add
        ...

    # ---- add this block just before the closing commit ----
    # Fan the sale out into a shippable order. Idempotent per (source, order_id);
    # lands in "needs address" unless the email carried a usable address block.
    try:
        ensure_order_from_sale(parsed, source=source, sale_events=created_events)
    except Exception as exc:
        app.logger.warning("Could not create an order from %s: %s", parsed.get("order_id"), exc)

    db.session.commit()
    return out
```

Wrapped in `try` on purpose: a shipping problem must never break the sale pipeline that
decrements inventory and delists on other marketplaces.

### 4 — Optional background tracking poller

In the `__main__` block, next to `start_email_poller()` (~line 14311):

```python
        start_email_poller()
        start_tracking_poller(app)     # <-- add; no-op unless SHIPPING_POLL_BACKGROUND=1
```

Off by default. The **Refresh tracking** button is the primary path.

### 5 — Optional: a card on the Settings page

In `settings.html`, next to the Shops card:

```html
{% if can_view('shops') %}
<div class="col-12 col-md-6 col-xl-4">
    <a href="/shipping" class="setting-card">
        <div class="sc-icon"><i class="fas fa-box-open"></i></div>
        <div>
            <h5>Shipping</h5>
            <p>Buy postage, print labels per order, and follow USPS tracking to delivery.</p>
        </div>
    </a>
</div>
{% endif %}
```

---

## Setup

**TCGTracking** — sign in at tcgtracking.com → API tab → add a payment method → generate a
key. Paste the shipper number + key, set a default envelope size, press **Test connection**.
(There's no ping endpoint, so the test probes `track` with an impossible tracking number:
401 means bad key, 404 means you're in. Nothing is created.)

**EasyPost** — sign up → Account Settings → API Keys. **Start with the test key** (`EZTK…`,
mode `test`): it makes fake labels and costs nothing. Fill in the ship-from address — rates
fail without it. Switch to the production key (`EZAK…`) when the flow looks right. The
connector warns you if the key and the mode disagree, which is the mistake that costs money.

Then: `pip install PyMuPDF` if it isn't already there — batch printing merges label PDFs with
it. Individual labels print fine without it.

---

## How it flows

```
sale email ──> SaleEvent ──> Order (needs address)
CSV / manual ───────────────> Order (ready)
                                 │  address filled in
                                 ▼
                         quote → buy → Shipment + PDF on disk
                                 │
                                 ▼
                      print (one, or merged batch)
                                 │
                                 ▼
                 poll tracking → events → shipped → delivered
```

Sale emails carry the items but almost never a postal address, so those orders land in
**needs address** and wait. Fill it in by hand, or import the marketplace's shipping export —
the CSV importer matches column names loosely (`Order #`, `order_number`, `OrderID`…), groups
rows sharing an order number into one order, and on re-import updates addresses without
duplicating line items.

Labels are stored at `UPLOAD_FOLDER/shipping_labels/<order_id>/` — inside the storage root
that Settings → Storage already relocates and backs up. Paths are stored upload-relative,
matching `ScanRecord.image_path`.

---

## Decisions worth knowing about

**EasyPost labels are requested as PDF**, not the default PNG, so batch printing can merge
either provider's output into one job.

**Insurance is never guessed.** TCGTracking caps PIP at $50 and computes the premium
server-side, so `insure_amount` is deliberately not sent. EasyPost insures full declared
value. Both are off unless you tick the box or set a per-provider default.

**Rates are re-quoted at purchase** rather than trusting a rate id from an earlier click —
EasyPost rates go stale, and a stale id fails at the worst moment.

**Deleting an order with purchased EasyPost postage is blocked.** Deleting the row doesn't
refund the label; void it in EasyPost first.

**Tracking polling is bounded** — only orders in flight, only shipments not checked in the
last 30 minutes, capped at 50 per press. USPS posts a few scans a day; polling harder just
burns the 250/min rate limit.

**Failed label attempts are recorded** as an error `Shipment` row rather than only a toast,
so "why didn't this print?" is answerable tomorrow.

---

## What I tested

Ran against a stubbed HTTP layer (no live calls, no money spent):

- provider config save / test / disconnect, both providers
- secret masking — a stored `EZAK…` key never reaches the page HTML
- order create, address validation (bad state, missing street, zero value)
- rate lookup, both providers
- label purchase, PDF written to disk, insurance capped at $50 correctly
- **duplicate order** → API returns no PDF → recovered via the regenerate endpoint
- **`generate_pdf` failure** → order exists with tracking → PDF recovered, not lost
- provider validation rejection → error row recorded against the order
- label view / download / reprint
- tracking refresh, single and batch; order auto-advances to `shipped` on first scan
- batch print merges N labels into one PDF
- CSV import: multi-row grouping, `United States` → `US`, re-import doesn't duplicate items
- sale email → order, idempotent on re-read, no phantom order from an unparsed email
- address parser against 6 layouts including adversarial ones

Untested against the live APIs — I have no account for either. The first real
`Create label` is the moment to use an EasyPost **test** key.
