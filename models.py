from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Product(db.Model):
    __tablename__ = 'products'
    id = db.Column(db.Integer, primary_key=True)
    brand = db.Column(db.String(100), nullable=False)
    product_name = db.Column(db.String(200), nullable=False)
    sku = db.Column(db.String(50), unique=True, nullable=True)
    price = db.Column(db.Float, nullable=True)
    stock = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f"<Product {self.brand} {self.product_name}>"

class ScanRecord(db.Model):
    __tablename__ = 'scan_records'
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    image_path_back = db.Column(db.String(255), nullable=True)  # back-of-card image, when scanned
    display_image_path = db.Column(db.String(255), nullable=True)
    # Optional override used ONLY for the stacked/grouped thumbnail on the
    # Inventory page when this record is chosen as a duplicate group's
    # representative. Set via /duplicates/resolve. Never affects this
    # record's own image_path, so its individual inventory_detail page
    # always keeps showing the photo it was actually scanned with.
    scan_date = db.Column(db.DateTime, default=datetime.utcnow)
    template_used = db.Column(db.String(100))  # e.g. "product_label_v1"
    extracted_data = db.Column(db.JSON)        # stores all OCR segments
    matched_product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=True)
    matched_product = db.relationship('Product')

    # ------------------------------------------------------------------ #
    # Denormalized "hot" columns, derived from extracted_data at write time
    # (see the SQLAlchemy before_insert/before_update events in app.py). They
    # are pure cache — extracted_data remains the source of truth — but they let
    # filtering, sorting, de-duplication, and pagination happen in indexed SQL
    # instead of by loading and looping over every row in Python, which is what
    # keeps queries fast into the millions of records.
    #   *_key      : normalized (lower/stripped) copies of common fields
    #   dup_hash   : identity of a duplicate group (name|serial|edition|holo) for
    #                FINALIZED records; NULL for unfinalized (each is its own group)
    #   is_*       : booleans promoted out of the JSON for fast WHERE clauses
    # ------------------------------------------------------------------ #
    game_key      = db.Column(db.String(120), index=True)
    album_key     = db.Column(db.String(200), index=True)
    name_key      = db.Column(db.String(300), index=True)
    card_type_key = db.Column(db.String(80),  index=True)
    dup_hash      = db.Column(db.String(64),  index=True)
    is_finalized  = db.Column(db.Boolean, default=False, index=True)
    is_catalog    = db.Column(db.Boolean, default=False, index=True)
    is_archived   = db.Column(db.Boolean, default=False, index=True)
    # "Held" = still in your possession (default). Cleared to False when an entry
    # is sold, which moves it off the Inventory list and onto the Sold page.
    # Mirrors extracted_data["held"]; NULL is treated as held (True) everywhere.
    is_held       = db.Column(db.Boolean, default=True, index=True)

    __table_args__ = (
        # Serves the hot Inventory query: filter by game + owned/active, order by recency.
        db.Index('idx_scan_hot', 'game_key', 'is_catalog', 'is_archived', 'scan_date'),
        db.Index('idx_scan_album_hot', 'album_key', 'is_catalog', 'is_archived'),
    )


class ShopConnection(db.Model):
    """
    One row per marketplace connector (tcgplayer / ebay / shopify).

    `config` holds all connection settings and secrets as JSON — e.g. the
    Shopify store domain + access token, the eBay client id/secret/RuName plus
    OAuth tokens, or the TCGplayer public/private keys plus a cached bearer
    token. Kept in the local SQLite DB so the Shops tab is self-contained; this
    is a personal, locally-run tool, but the file still holds live credentials,
    so treat inventory.db as a secret.
    """
    __tablename__ = 'shop_connections'
    id           = db.Column(db.Integer, primary_key=True)
    marketplace  = db.Column(db.String(30), unique=True, nullable=False)  # tcgplayer|ebay|shopify
    enabled      = db.Column(db.Boolean, default=False)
    config       = db.Column(db.JSON, default=dict)   # settings + secrets + cached tokens
    status       = db.Column(db.String(20), default='disconnected')  # connected|disconnected|error
    status_detail = db.Column(db.Text, nullable=True)                 # last test/connect message
    connected_at = db.Column(db.DateTime, nullable=True)
    updated_at   = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ShopConnection {self.marketplace} {self.status}>"


class Listing(db.Model):
    """
    Tracks the state of a single inventory record's listing on one marketplace.

    Unique per (record_id, marketplace): a record has at most one listing per
    shop. `external_id` is the marketplace's own identifier (Shopify product id,
    eBay offer/listing id, TCGplayer SKU id), and `sku` is the stable
    cross-marketplace SKU we assign (CCIM-<record_id>) so listings can be pulled
    back and matched to records.
    """
    __tablename__ = 'listings'
    id            = db.Column(db.Integer, primary_key=True)
    record_id     = db.Column(db.Integer, db.ForeignKey('scan_records.id'), nullable=False)
    marketplace   = db.Column(db.String(30), nullable=False)  # tcgplayer|ebay|shopify
    sku           = db.Column(db.String(80), nullable=True)
    external_id   = db.Column(db.String(120), nullable=True)  # marketplace product/offer/listing id
    external_url  = db.Column(db.String(500), nullable=True)
    title         = db.Column(db.String(300), nullable=True)
    price         = db.Column(db.Float, nullable=True)
    currency      = db.Column(db.String(10), default='USD')
    quantity      = db.Column(db.Integer, default=0)
    status        = db.Column(db.String(20), default='not_listed')  # not_listed|draft|active|ended|error
    last_error    = db.Column(db.Text, nullable=True)
    last_synced   = db.Column(db.DateTime, nullable=True)
    extra         = db.Column(db.JSON, default=dict)  # marketplace-specific ids (variant/inventory item, etc.)

    record = db.relationship('ScanRecord', backref=db.backref('listings', cascade='all, delete-orphan'))

    __table_args__ = (
        db.UniqueConstraint('record_id', 'marketplace', name='uq_listing_record_marketplace'),
    )

    def __repr__(self):
        return f"<Listing r{self.record_id} {self.marketplace} {self.status}>"


class EmailMonitor(db.Model):
    """
    IMAP mailbox watched for marketplace sale-notification emails (e.g. TCGplayer
    "you made a sale"). A single row is expected. The password is stored here so
    the monitor is self-contained; like the shop tokens, treat inventory.db as a
    secret. `last_uid` lets polling skip already-seen messages; idempotency of
    actual sale processing is enforced separately via SaleEvent.
    """
    __tablename__ = 'email_monitors'
    id             = db.Column(db.Integer, primary_key=True)
    enabled        = db.Column(db.Boolean, default=False)
    host           = db.Column(db.String(200), nullable=True)
    port           = db.Column(db.Integer, default=993)
    use_ssl        = db.Column(db.Boolean, default=True)
    username       = db.Column(db.String(200), nullable=True)
    password       = db.Column(db.String(300), nullable=True)   # app-password / mailbox password
    folder         = db.Column(db.String(120), default='INBOX')
    sender_filter  = db.Column(db.String(200), default='tcgplayer.com')
    subject_filter = db.Column(db.String(200), default='sold')
    source         = db.Column(db.String(30), default='tcgplayer')  # which marketplace these sales are from
    mark_seen      = db.Column(db.Boolean, default=True)
    poll_interval  = db.Column(db.Integer, default=0)   # seconds; 0 = manual only
    last_uid       = db.Column(db.Integer, default=0)
    last_checked   = db.Column(db.DateTime, nullable=True)
    status         = db.Column(db.String(20), default='disconnected')
    status_detail  = db.Column(db.Text, nullable=True)
    updated_at     = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<EmailMonitor {self.username or '-'} {self.status}>"


class SaleEvent(db.Model):
    """
    One parsed sale line from a notification email. Provides an audit trail and
    idempotency: (source, order_id, item_title) is unique so re-reading the same
    email never double-processes a sale.
    """
    __tablename__ = 'sale_events'
    id           = db.Column(db.Integer, primary_key=True)
    source       = db.Column(db.String(30), default='tcgplayer')
    order_id     = db.Column(db.String(120), nullable=True)
    item_title   = db.Column(db.String(400), nullable=True)
    qty          = db.Column(db.Integer, default=1)
    price        = db.Column(db.Float, nullable=True)
    record_id    = db.Column(db.Integer, db.ForeignKey('scan_records.id'), nullable=True)
    status       = db.Column(db.String(20), default='unmatched')  # unmatched|matched|processed|error|unparsed
    detail       = db.Column(db.Text, nullable=True)
    email_subject = db.Column(db.String(400), nullable=True)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    record = db.relationship('ScanRecord')

    __table_args__ = (
        db.UniqueConstraint('source', 'order_id', 'item_title', name='uq_saleevent_order_item'),
    )

    def __repr__(self):
        return f"<SaleEvent {self.source} {self.order_id} {self.status}>"


class ReferenceCard(db.Model):
    """
    Local cache of a single TCGplayer product pulled from tcgcsv.com, used as a
    reference catalog to identify scanned cards and auto-fill entry fields from
    OCR results. One row per TCGplayer productId. `extended` keeps the full
    extendedData key/value map (Number, Rarity, HP, Stage, ...) so game-specific
    fields stay available without needing a column each.
    """
    __tablename__ = 'reference_cards'
    id           = db.Column(db.Integer, primary_key=True)
    category_id  = db.Column(db.Integer, nullable=False, index=True)   # tcgplayer categoryId (the "game")
    group_id     = db.Column(db.Integer, nullable=False, index=True)   # tcgplayer groupId  (the "set")
    product_id   = db.Column(db.Integer, unique=True, nullable=False)  # tcgplayer productId (primary key upstream)
    game         = db.Column(db.String(120), index=True)              # category display name, e.g. "Pokemon"
    set_name     = db.Column(db.String(200))                          # group name, e.g. "SWSH12: Silver Tempest"
    name         = db.Column(db.String(300), index=True)
    clean_name   = db.Column(db.String(300))
    number       = db.Column(db.String(40), index=True)              # extendedData "Number", e.g. "139/195"
    rarity       = db.Column(db.String(80))
    image_url    = db.Column(db.String(500))
    url          = db.Column(db.String(500))
    market_price = db.Column(db.Float, nullable=True)
    extended     = db.Column(db.JSON, default=dict)                  # full extendedData as {name: value}
    updated_at   = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ReferenceCard {self.game} {self.name} {self.number}>"


class ReferenceSync(db.Model):
    """
    Bookkeeping for reference-catalog downloads from tcgcsv.com. One row per
    synced category (game): how many products are cached and when, plus the
    tcgcsv last-updated stamp seen at sync time so we can skip redundant pulls
    (tcgcsv only refreshes once daily).
    """
    __tablename__ = 'reference_syncs'
    id             = db.Column(db.Integer, primary_key=True)
    category_id    = db.Column(db.Integer, unique=True, nullable=False)
    game           = db.Column(db.String(120))
    product_count  = db.Column(db.Integer, default=0)
    group_count    = db.Column(db.Integer, default=0)
    remote_updated = db.Column(db.String(60), nullable=True)   # value of last-updated.txt at sync time
    last_synced    = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status         = db.Column(db.String(20), default='idle')  # idle|syncing|ok|error
    status_detail  = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<ReferenceSync cat={self.category_id} {self.game} n={self.product_count}>"


class TypeReference(db.Model):
    """
    A labelled reference image of a card "type" icon (e.g. a Pokemon energy
    symbol, a Yu-Gi-Oh attribute, ...), used to identify a scanned card's type
    by template matching rather than colour alone.

    One row per exemplar image; a type can have several exemplars, and each game
    keeps its own set. `game` is stored lower-cased for matching against a
    record's game. `image_path` is an upload-relative PNG of the (tight) icon.
    `source` records how it was added: 'upload' (user supplied an icon image) or
    'capture' (cropped from one of the user's own scanned cards).
    """
    __tablename__ = 'type_references'
    id          = db.Column(db.Integer, primary_key=True)
    game        = db.Column(db.String(120), index=True, nullable=False)  # lower-cased game key
    type_name   = db.Column(db.String(80), nullable=False)               # "Fire", "Water", ...
    region      = db.Column(db.String(20), default='top_right')          # top_left | top_right
    image_path  = db.Column(db.String(255), nullable=False)              # upload-relative PNG of the icon
    source      = db.Column(db.String(30), default='upload')            # upload | capture
    note        = db.Column(db.String(200), nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<TypeReference {self.game} {self.type_name} [{self.region}] #{self.id}>"


class AppSetting(db.Model):
    """
    Simple key/value store for application settings and secrets — most notably
    API keys that used to live in a root .env file. Kept in the local DB so they
    are editable at runtime from Settings → API Keys, take effect immediately,
    and are backed up / migrated with the rest of the data. Like the shop tokens
    and mailbox password, this means inventory.db holds live credentials, so
    treat the database file as a secret.
    """
    __tablename__ = 'app_settings'
    id         = db.Column(db.Integer, primary_key=True)
    key        = db.Column(db.String(120), unique=True, nullable=False)
    value      = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AppSetting {self.key}>"
