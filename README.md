# Card Collector Inventory Manager

A self-hosted web app for cataloguing, valuing, and selling trading-card-game (TCG)
collections. Point a scanner or camera at your cards, let it detect and OCR each one,
match it against downloadable price catalogs, and manage the whole collection — from
intake and storage to analytics, financial reports, and eBay listings — from any device
on your network.

> Built to run on a single machine (desktop, mini-PC, or Raspberry Pi) and be reached
> from phones, tablets, and laptops on the same LAN over a stable `https://` name.

---

## Table of contents

- [Highlights](#highlights)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [First-run setup](#first-run-setup)
- [Configuration](#configuration)
- [Usage guide](#usage-guide)
- [Roles & permissions](#roles--permissions)
- [Security](#security)
- [Project structure](#project-structure)
- [Notes & limitations](#notes--limitations)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Highlights

- **Scan & identify** cards from images, multi-card grid sheets, PDFs, or a **live camera**, with automatic card detection, cropping, and OCR of name/number.
- **Catalog matching** against downloadable reference data (per game), with market prices.
- **Full inventory** with intake price, market value, sold price, condition, storage, albums, duplicates, and held/sold tracking.
- **Analytics & financial reports** — weekly / monthly / quarterly / annual PDF reports covering acquisitions, sales by shop, strategies, and profit.
- **eBay listing tools** — a drag-and-drop (WYSIWYG) HTML description template that auto-fills from each item, plus bulk-listing CSV export.
- **Quick Scan** — identify cards live to a CSV for lookups without touching inventory.
- **Multi-user** with roles and per-tab/tool view/edit permissions.
- **Secure by default** — HTTPS out of the box, CSRF protection, upload safeguards, and a hardened auth system.
- **Zero-config LAN access** via an mDNS `.local` name — no router setup or per-device host edits.

---

## Features

### Scanning & import
- Upload single cards, **3×3 grid sheets**, or **PDFs**; automatic card **detection and cropping** (OpenCV).
- **Background PDF auto-import** with automatically detected cut lines, processed page-by-page.
- **OCR** of card name and number, with a reference-catalog scorer to suggest the best match.
- Optional cloud identification providers as a fallback.

### Reference data (price catalogs)
- Download per-game catalogs from providers (e.g. tcgcsv / PokémonTCG), with **incremental, background sync** and progress reporting.
- **Upload your own CSV** for custom games.
- Per-game entry fields derived from the catalog columns.

### Inventory management
- Rich records: name, set, number, rarity, game, edition, condition, **intake price**, **current/market value**, **sold price**, collection, album, and storage location.
- **Database match / wrong match** controls to correct identifications.
- **Duplicates** detection and grouping.
- **Held vs. sold** tracking, with sale dates stamped for reporting.
- Per-record **net** (sold − intake) display and filtering/sorting.

### Albums & storage
- **Binder view** with 3×3 album pages and custom album cover images.
- Named **storage locations** to track where physical cards live.

### Analytics & financial reports
- Analytics dashboards with group-by metrics, collection pricing, and CSV export.
- **PDF (and on-screen) reports** for any **weekly / monthly / quarterly / annual** period, including:
  - Acquisitions grouped by collection (purchase price + market value + sold value).
  - **Sales split by shop** (eBay, TCGplayer, manual/direct) with revenue and profit.
  - Derived **strategies** (flip vs. hold vs. buy-and-hold).
  - Overall financial outcome: cost, revenue, realized profit, net cash flow, unrealized gain.

### Shops & listings (eBay)
- **WYSIWYG eBay description template editor** (drag-and-drop text, images, columns) that fills `{{tokens}}` from each item (name, set, number, rarity, condition, price, photos…).
- Upload a custom HTML template or edit raw HTML; live "preview with data".
- Applies automatically to items listed from inventory and from the Builder.
- **eBay bulk-listing CSV export** (File-Exchange style) with the rendered HTML description per row.

### Quick Scan (camera)
- Live camera feed → detect → OCR → catalog match, streamed to an on-screen table.
- **Nothing is saved to inventory** — export the results to CSV when done.
- A 2-second overlay tells you whether the scanned card is **already in your held inventory**.

### Pricing & search
- On-demand price lookups (JustTCG / TCGplayer).
- Search your inventory by image.

### Platform
- **Authentication & roles** with granular permissions and an admin UI.
- **HTTPS by default** with an auto-generated self-signed certificate.
- **mDNS** advertising for a stable `https://<name>.local` address; the name is configurable in Settings.

---

## Tech stack

- **Backend:** Python + [Flask](https://flask.palletsprojects.com/), [SQLAlchemy](https://www.sqlalchemy.org/) (SQLite by default)
- **Imaging / OCR:** [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/), [NumPy](https://numpy.org/), [PyMuPDF](https://pymupdf.readthedocs.io/) (PDF), [pytesseract](https://github.com/madmaze/pytesseract) + the Tesseract engine
- **Reports:** [ReportLab](https://www.reportlab.com/) (PDF generation)
- **Security / networking:** [cryptography](https://cryptography.io/) (self-signed TLS), [zeroconf](https://python-zeroconf.readthedocs.io/) (mDNS, optional)
- **Frontend:** server-rendered HTML + vanilla JS, [Bootstrap](https://getbootstrap.com/), [Font Awesome](https://fontawesome.com/), and [GrapesJS](https://grapesjs.com/) (the eBay template editor, via CDN)

---

## Requirements

- **Python 3.10+**
- **Tesseract OCR** system binary (for card OCR):
  - Debian/Ubuntu: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: install from the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)
- Python packages (see `requirements.txt`), typically:

```text
Flask
SQLAlchemy
Flask-SQLAlchemy
python-dotenv
opencv-python-headless
Pillow
numpy
PyMuPDF
pytesseract
reportlab
cryptography
zeroconf        # optional, enables the .local name
requests
```

> `PyMuPDF`, `pytesseract`/Tesseract, and `zeroconf` are optional at boot — the app
> starts without them and simply disables the related feature with a clear message.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-username>/card-collector-inventory-manager.git
cd card-collector-inventory-manager

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.lock   # exact versions this app is tested against
# or:  pip install -r requirements.txt   # latest compatible, may drift

# 4. (Optional) create a .env file for configuration — see below
```

**Which file to use.** `requirements.lock` pins every package to a version this app has
actually been run against, so two installs get the same app. `requirements.txt` states
only minimum versions, and minimums drift: resolving it today already brings in **seven
packages a major version above** what those minimums describe (numpy 1→2, opencv 4→5,
reportlab 4→5, cryptography 42→50, Pillow 10→12, networkx 2→3, plus rapidocr and
onnxruntime, which until recently had no constraint at all).

Use the lock for anything you depend on. Use `requirements.txt` when you deliberately
want current versions — and if that combination works, refresh the lock from it (the
command is in the lock file's header) so the next person inherits a tested set.

---

## Running the app

```bash
python app.py
```

By default the app serves over **HTTPS** on port **443** and advertises itself on the
local network. On first launch it generates a self-signed certificate (stored in
`certs/`) and prints the address to visit, for example:

```
 • Visit: https://127.0.0.1  (or  https://cardcollector.local  on this LAN)
 [mDNS] Also reachable on this network at:  https://cardcollector.local   (IP 192.168.1.42)
```

Notes on access:

- **Port 443 is privileged.** Run with elevated privileges to bind it, or the app
  automatically falls back to `https://cardcollector.local:8443`. On Linux you can grant
  the capability once instead of using `sudo`:
  `sudo setcap 'cap_net_bind_service=+ep' $(readlink -f $(which python3))`.
- **Self-signed certificate warning:** each device shows a one-time "not secure" prompt —
  accept it once to enable full functionality (including the live camera). To remove the
  warning, drop a locally-trusted cert (e.g. via [mkcert](https://github.com/FiloSottile/mkcert))
  at `certs/cardcollector.crt` / `certs/cardcollector.key`, or front the app with
  Tailscale / Cloudflare for a publicly-trusted certificate.
- **HTTPS is required for the live camera** (browsers only allow camera access on secure
  origins), which is why it's the default. Set `USE_HTTPS=0` to serve plain HTTP instead.

---

## First-run setup

On a fresh install the app redirects to a one-time setup page to create the first
**administrator** account. Creating it also seeds three starter roles:

| Role          | Access                                   |
|---------------|-------------------------------------------|
| Administrator | Full access; manages users and roles      |
| Editor        | Edit everything (no user/role management)  |
| Viewer        | View everything                            |

After that, sign in at `/auth/login`. Manage users and roles from **Settings → Users** and
**Settings → Roles**.

---

## Configuration

All configuration is via environment variables (a `.env` file is supported). Common options:

| Variable                  | Default            | Description |
|---------------------------|--------------------|-------------|
| `USE_HTTPS`               | `1`                | Serve over HTTPS with a self-signed cert. `0` for plain HTTP. |
| `PORT`                    | `443` / `80`       | Listening port (HTTPS / HTTP default). |
| `PORT_FALLBACK`           | `8443` / `5005`    | Port used if the primary port can't bind. |
| `SECRET_KEY`              | *(auto-persisted)* | Flask session secret. Auto-generated and stored if unset. |
| `SESSION_COOKIE_SECURE`   | *(unset)*          | Mark the session cookie `Secure`. **Set this to `1` when running behind a TLS-terminating proxy** (gunicorn/uWSGI + nginx): the app sees plain HTTP there and cannot tell. Running `python app.py` over HTTPS turns it on by itself. Leave unset for plain HTTP — a `Secure` cookie on an `http://` origin is never sent back, which looks like "login does nothing". |
| `IMAP_ALLOW_INSECURE_TLS` | *(unset)*          | Accept an IMAP server certificate that isn't trusted or doesn't match the hostname. **Only set this for your own mail server with a self-signed certificate** — it disables the check that stops anything on the network path from impersonating your mail provider and collecting the mailbox password. It does **not** allow an unencrypted connection: a server that won't start TLS is still refused. |
| `DISABLE_AUTH`            | *(unset)*          | Kill-switch to disable authentication entirely. |
| `FLASK_DEBUG`             | `0`                | Enable the Werkzeug debugger/reloader (local dev only — unsafe on a LAN). |
| `DATABASE_URL`            | SQLite file        | SQLAlchemy database URL. |
| `MAX_UPLOAD_MB`           | `1024`             | Max upload size (MB). |
| `MAX_IMAGE_MEGAPIXELS`    | `2000`             | Decoded-image pixel ceiling (decompression-bomb guard). |
| `MAX_IMAGE_DECODE_RATIO`  | `300`              | Max decoded-size / file-size ratio before an image is rejected as a bomb. |
| `MAX_PDF_PAGES`           | `5000`             | Max pages accepted from a single PDF. |
| `PDF_CAPPED_DPI`          | `600`              | Cap for PDF page rasterization DPI. |
| `REFERENCE_PROVIDER`      | provider default   | Reference-catalog data provider. |
| `IDENTIFY_PROVIDER`       | provider default   | Card identification provider. |
| `INVENTORY_MAX_RECORDS`   | high default       | Safety cap on total inventory records. |

The advertised network name (`<name>.local`) is set in **Settings → Network Name**.

---

## Usage guide

1. **Download a catalog** — Settings → Reference Data. Pick a game and sync its catalog so
   cards can be matched and priced.
2. **Import cards** — Import page. Upload a photo, a 3×3 grid sheet, or a PDF; the app
   detects, crops, OCRs, and suggests matches. Confirm to add them to inventory.
3. **Manage inventory** — browse, filter, and sort; open a record to edit fields, fix a
   match, set prices, and mark items sold.
4. **Organize** — group cards into **albums** (binder pages) and **storage** locations.
5. **Analyze** — Analytics for dashboards; **Reports** for downloadable PDF period reports.
6. **Sell** — design your **eBay listing template** (Settings → eBay Listing Template), then
   export a bulk-listing CSV or list items with the rendered HTML description.
7. **Quick Scan** — identify cards live from the camera into a CSV without saving them, with
   an instant "already in inventory" indicator.

---

## Roles & permissions

Each role defines, per tab/tool, whether members can **view**, **edit**, or have **no access**.
Enforcement is centralized server-side: read (GET) requests need *view*; changes need *edit*.
Administrators bypass all checks and manage accounts. The UI also hides tabs a user can't view.

Permissioned areas include: Game Templates, Inventory, Albums, Import, Duplicates, Analytics,
Financial Reports, Quick Scan, Search by Image, Reference Data, Shops/Listings, Pricing,
Identification, API Keys, Storage, Network, General Settings, and Upgrade/Export.

---

## Security

Security-conscious defaults are built in:

- **Authentication** with hashed passwords (Werkzeug), session cookies set `HttpOnly` +
  `SameSite=Lax`, a persistent signed-session secret, and a login **brute-force throttle**.
- **CSRF protection** via a per-session synchronizer token, attached transparently to all
  same-origin state-changing requests (no per-form wiring needed).
- **Role-based authorization** enforced centrally on every request.
- **Upload safeguards:** a global size cap plus a **ratio-aware decompression-bomb guard**
  that allows genuine high-resolution scans while rejecting tiny files that expand to
  enormous images; PDFs are bounded by a render-DPI cap, page-count cap, and disk-spill.
- **CSV formula-injection** neutralization on all exports.
- **Security headers** (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`),
  path-traversal guards on file access, a sandboxed template preview, and the interactive
  debugger disabled by default.
- **HTTPS** by default with a self-signed certificate whose private key is stored `0600`.

For internet exposure (beyond a trusted LAN), place the app behind a reverse proxy or tunnel
that terminates a publicly-trusted certificate (e.g. Tailscale, Cloudflare Tunnel, or Nginx +
Let's Encrypt).

---

## Project structure

```
.
├── app.py               # Main Flask application: routes, imaging/OCR, auth, reports, shops…
├── models.py            # SQLAlchemy models (ScanRecord, Product, SaleEvent, Reference*, …)
├── builder.py           # "Builder" tool for assembling sellable lots/products
├── shop_providers.py    # Marketplace/shop integrations (e.g. eBay)
├── email_monitor.py     # Optional sale/email monitoring
├── card_ocr.py          # OCR + catalog-matching helpers
├── templates/           # Jinja templates (inventory, import, albums, analytics, settings…)
├── static/              # CSS, images, and other static assets
├── certs/               # Auto-generated self-signed TLS cert/key (gitignored)
└── requirements.txt
```

> The main application logic lives in `app.py`; several inline admin/utility pages (auth,
> users, roles, network, reports, quick-scan, eBay template) are served directly from it.

---

## Notes & limitations

- Designed for **single-process, LAN/desktop** deployment. It is not hardened for hostile
  public internet exposure without a proxy in front.
- The **live camera** requires a secure context (HTTPS or `localhost`); on plain HTTP the
  Quick Scan page falls back to a device "Photo" capture button.
- OCR accuracy depends on lighting, framing, and catalog coverage — always review matches.
- Catalog matching requires the relevant **game catalog to be downloaded** first.
- Some features are optional and degrade gracefully if their library isn't installed
  (PDF import → PyMuPDF, OCR → Tesseract, `.local` name → zeroconf).

---

## Contributing

Contributions are welcome. A typical workflow:

1. Fork the repository and create a feature branch.
2. Make your change with clear, focused commits.
3. Verify the app still boots and the affected pages work.
4. Open a pull request describing the change and motivation.

Please avoid committing local data, uploads, the SQLite database, or generated
certificates. A suggested `.gitignore`:

```gitignore
.venv/
__pycache__/
*.pyc
.env
certs/
*.sqlite3
*.db
uploads/
instance/
```

---

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE).
*(Update this section if you choose a different license.)*

---

## Disclaimer

This is an independent tool and is **not affiliated with, endorsed by, or sponsored by**
eBay, TCGplayer, The Pokémon Company, Wizards of the Coast, or any other rights holder.
All product names, logos, and brands are property of their respective owners and are used
for identification purposes only. You are responsible for complying with the terms of
service of any marketplace or data provider you connect to.
