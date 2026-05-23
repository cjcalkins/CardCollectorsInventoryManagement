# 📦 Card Collector Inventory Manager

A self-hosted web application for scanning, organizing, and browsing a physical card collection — without ever opening a binder. Built with Flask and OpenCV, it uses OCR to read card data directly from photos of binder pages, building a searchable inventory you can browse from any device.

---

## ✨ Features

### 📷 Full-Page Scanning (Import)
Photograph an entire 3×3 binder page and let the app do the rest:

- Upload a page photo and define your cut lines to split it into 9 individual card slots
- Automatic perspective correction and card alignment using OpenCV edge detection
- Manual corner-pin override for cards that weren't auto-detected cleanly
- Batch OCR runs across all 9 cards simultaneously with live per-card progress streaming
- Uses configurable ROI (Region of Interest) templates so OCR targets the right part of each card — name, set, number, edition, etc.
- Extracted fields are saved to the database along with the card image

### 🔍 Single Card Scanning
Scan or photograph an individual card for quick addition to inventory:

- Draw ROI boxes directly on the card preview in-browser
- Build and save reusable field templates (percentage-based, so they work at any resolution)
- Instant OCR result preview before confirming the record

### 📋 Inventory Management
A full-featured table view of everything you own:

- Filter by game, album, set, edition, holographic status, and more
- Full-text search across all card fields
- Sortable columns with server-side pagination
- Thumbnail previews inline in the table
- Click through to a detail page to edit any field, attach a TCGPlayer/JustTCG price link, or copy data from an existing record

### 🔗 Duplicate Detection
- Dedicated duplicates view identifies cards sharing the same name, serial number, and edition
- Helps you spot accidental double-scans or genuine duplicates in your collection

### 📚 Album Browser
Browse your collection as it looks in the binder — without touching the binder:

- Albums are automatically organized from the `album` field on each scan record
- Each album shows a paginated 3×3 grid matching physical binder page layout
- Click any card slot to jump to its full detail view
- Set a custom cover image for each album and game tile
- Game tiles on the inventory landing page can also have custom cover art

### 💰 Pricing Integration
- Fetch live market prices via the JustTCG API directly from a card's detail page
- Save a TCGPlayer URL per record for one-click price lookups

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| Database | SQLite via SQLAlchemy |
| Image Processing | OpenCV, Pillow |
| OCR | Tesseract (via pytesseract) |
| Frontend | Bootstrap 5, Font Awesome |
| Streaming | Server-Sent Events (SSE) for live OCR progress |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Tesseract OCR installed and on your PATH
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - Windows: [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/card-collector-inventory.git
cd card-collector-inventory

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
```

### Running the App

```bash
python app.py
```

Then open [http://localhost:5005](http://localhost:5005) in your browser.

The database and all upload folders are created automatically on first run.

---

## 📁 Project Structure

```
├── app.py                  # Main Flask application and all routes
├── models.py               # SQLAlchemy models (Product, ScanRecord)
├── templates/
│   ├── index.html          # Single card scan + ROI template builder
│   ├── import.html         # Full binder page import workflow
│   ├── inventory.html      # Inventory table with filters and search
│   ├── inventory_detail.html   # Individual card detail / edit view
│   ├── inventory_game_select.html  # Game selection landing page
│   ├── albums.html         # Album list with cover images
│   ├── album_detail.html   # 3x3 binder page grid view
│   └── duplicates.html     # Duplicate card manager
│   └── roi/                # Saved ROI field templates (JSON)
├── static/
│   └── favicon.jpg
└── uploads/
    ├── inventory_cards/    # Permanent card images
    ├── import_pages/       # Uploaded binder page photos
    ├── temp_split/         # Intermediate split card images
    ├── temp_cards/         # Aligned card images awaiting OCR confirmation
    ├── album_covers/       # Custom album cover images
    └── game_covers/        # Custom game tile cover images
```

---

## 🗺 Routes Reference

| Route | Description |
|---|---|
| `/` | Single card scan + ROI template builder |
| `/import` | Full 3×3 binder page import |
| `/inventory` | Game selection, then filtered inventory table |
| `/inventory/<id>` | Card detail: edit, price, copy-from |
| `/albums` | Album list |
| `/albums/<name>` | Paginated binder page grid for one album |
| `/duplicates` | Duplicate card manager |

---

## 🔧 ROI Templates

ROI templates define which regions of a card image Tesseract should read for each field (name, set, card number, etc.). Templates are stored as JSON in `templates/roi/` and use percentage-based coordinates so they work regardless of image resolution.

You can create and save new templates directly in the browser on the Scan page. One template can serve an entire game or set of similarly-formatted cards.

---

## 📝 Import Workflow (Step by Step)

1. **Photograph** a 3×3 binder page as flat and evenly lit as possible
2. **Upload** the photo on the Import page and enter the game, album, and page number
3. **Adjust cut lines** to align with the pocket borders on the page
4. **Auto-process** — the app splits, perspective-corrects, and sharpens all 9 cards
5. **Review** each card; use manual corner-pin for any that need adjustment
6. **Run OCR** — all 9 cards are OCR'd using your selected template with live progress
7. **Confirm** — records and images are committed to the inventory database

---

## 🤝 Contributing

Pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
