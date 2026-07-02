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
