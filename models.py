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
    scan_date = db.Column(db.DateTime, default=datetime.utcnow)
    template_used = db.Column(db.String(100))  # e.g. "product_label_v1"
    extracted_data = db.Column(db.JSON)        # stores all OCR segments
    matched_product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=True)
    matched_product = db.relationship('Product')
