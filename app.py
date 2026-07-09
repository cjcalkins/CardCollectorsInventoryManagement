from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, redirect, Response
import os
import re as _re
# OpenCV enforces a max decoded-image size (OPENCV_IO_MAX_IMAGE_PIXELS, default
# ~1.07 G px) at import time. Very high-DPI card scans (a 2400-DPI letter page is
# ~0.54 G px, larger for bigger pages) can approach it, so raise the ceiling
# before cv2 is imported. Tunable via env; the memory budget below is the real
# safety limit.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(1 << 40))
import cv2
import json
import shutil
import tempfile
import numpy as np
from datetime import datetime
from PIL import Image
# Pillow raises DecompressionBombError above ~89 MP by default; legitimate
# high-DPI card scans exceed that, so lift the ceiling. None disables the check;
# a finite value can be set via MAX_IMAGE_MEGAPIXELS. The per-page memory budget
# (PDF_MAX_MEGAPIXELS) is what actually protects RAM.
_max_img_mp = os.environ.get("MAX_IMAGE_MEGAPIXELS", "").strip()
Image.MAX_IMAGE_PIXELS = int(float(_max_img_mp) * 1_000_000) if _max_img_mp else None
from werkzeug.utils import secure_filename
from models import (db, Product, ScanRecord, ShopConnection, Listing, EmailMonitor,
                    SaleEvent, ReferenceCard, ReferenceSync, TypeReference, AppSetting)
from dotenv import load_dotenv
load_dotenv()

# PyMuPDF is used to rasterize uploaded PDF pages (see /pdf_extract_pages).
# Import is optional at module load time so the rest of the app keeps working
# even on installs that haven't run `pip install PyMuPDF` yet — the PDF
# import route will just report a clear error instead of crashing the app.
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# card_ocr provides front-image OCR (top/bottom band -> name + N/M collector
# number) and a matcher against existing records. Imported optionally so the
# app still boots on installs without pytesseract / the tesseract-ocr binary;
# the /ocr_identify route reports a clear error instead of crashing.
try:
    import card_ocr
except Exception:
    card_ocr = None

# Reference-catalog provider: downloads a game's card catalog into ReferenceCard
# rows so OCR results can be matched to a real card and auto-fill entry data. The
# active source is pokemontcg.io (v2) — it covers the vintage WOTC sets tcgcsv is
# missing (Base, Jungle, Fossil, Gym, Neo, ...). tcgcsv_sync is kept as a fallback
# if the pokemontcg adapter isn't importable. Imported optionally so the app still
# boots without network/module access; the /reference routes report a clear error.
ref_sync = None
try:
    import pokemontcg_sync as ref_sync
except Exception:
    try:
        import tcgcsv_sync as ref_sync
    except Exception:
        ref_sync = None

app = Flask(__name__, template_folder="templates")

# ====================== CONFIG ======================
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ---------------------------------------------------------------------------- #
# Dynamic storage locations
# ----------------------------------------------------------------------------
# All sizable file/folder locations (image store, temp working area, ROI
# templates, and the SQLite DB) are user-relocatable at runtime from
# Settings → Storage. The chosen roots are persisted in storage_config.json
# next to this file — the ONE bootstrap anchor — and every other path in the
# program is derived from them here and read from app.config at call time, so
# nothing else hardcodes a location. See the storage routes below for the
# move-and-cleanup logic.
# ---------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_CONFIG_PATH = os.path.join(BASE_DIR, "storage_config.json")

# Subfolders each movable root "owns" — used both to derive app.config paths and
# to know exactly what to migrate when a root is relocated.
STORAGE_UPLOAD_SUBDIRS = ["inventory_cards", "type_refs", "debug", "orb_cache"]
STORAGE_TEMP_SUBDIRS   = ["import_pages", "temp_split", "temp_cards", "temp_pdf_pages"]

# Defaults reproduce the original on-disk layout exactly: images and temp both
# live under ./uploads, ROI under ./templates/roi, DB at ./inventory.db. `temp`
# and `uploads` may share a directory (the default) because each move only
# touches its own owned subfolders, so they never collide.
DEFAULT_STORAGE = {
    "uploads": "uploads",
    "temp":    "uploads",
    "roi":     os.path.join("templates", "roi"),
    "db":      "inventory.db",
}
STORAGE_SLOTS = ("uploads", "temp", "roi", "db")


def _resolve_storage_path(p):
    """Resolve a stored location. Relative paths are anchored to the program
    directory (BASE_DIR), so behaviour doesn't depend on the current working
    directory. Absolute paths are used as-is."""
    p = str(p or "").strip()
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


def _path_to_sqlite_uri(path):
    return "sqlite:///" + os.path.abspath(path)


def _sqlite_uri_to_path(uri):
    uri = str(uri or "")
    if uri.startswith("sqlite:///"):
        return uri[len("sqlite:///"):]
    return uri


def load_storage_config():
    """Read storage_config.json, falling back to defaults for any missing key."""
    cfg = dict(DEFAULT_STORAGE)
    pending = []
    try:
        with open(STORAGE_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        for k in DEFAULT_STORAGE:
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                cfg[k] = v.strip()
        pending = [p for p in (data.get("pending_deletions") or []) if isinstance(p, str)]
    except (OSError, ValueError):
        pass
    cfg["pending_deletions"] = pending
    return cfg


def save_storage_config(cfg):
    """Persist the storage config atomically."""
    out = {k: cfg.get(k, DEFAULT_STORAGE[k]) for k in DEFAULT_STORAGE}
    out["pending_deletions"] = list(cfg.get("pending_deletions") or [])
    tmp = STORAGE_CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, STORAGE_CONFIG_PATH)


def apply_storage_config(cfg):
    """Derive every concrete path in the program from the storage roots and push
    them into app.config. Called once at startup and again after any move."""
    uploads = _resolve_storage_path(cfg["uploads"])
    temp    = _resolve_storage_path(cfg["temp"])
    roi     = _resolve_storage_path(cfg["roi"])
    db_path = _resolve_storage_path(cfg["db"])

    app.config["UPLOAD_FOLDER"]           = uploads
    app.config["INVENTORY_IMAGE_FOLDER"]  = os.path.join(uploads, "inventory_cards")
    app.config["TYPE_REF_FOLDER"]         = os.path.join(uploads, "type_refs")
    app.config["ORB_CACHE_FOLDER"]        = os.path.join(uploads, "orb_cache")
    app.config["ROI_TEMPLATE_FOLDER"]     = roi
    app.config["TEMP_IMPORT_FOLDER"]      = os.path.join(temp, "import_pages")
    app.config["TEMP_SPLIT_FOLDER"]       = os.path.join(temp, "temp_split")
    app.config["TEMP_CARD_FOLDER"]        = os.path.join(temp, "temp_cards")
    app.config["TEMP_PDF_FOLDER"]         = os.path.join(temp, "temp_pdf_pages")
    app.config["SQLALCHEMY_DATABASE_URI"] = _path_to_sqlite_uri(db_path)
    # Resolved roots, handy for the Storage settings UI.
    app.config["STORAGE_ROOTS"] = {"uploads": uploads, "temp": temp, "roi": roi, "db": db_path}


# Module-level handle to the active storage config (routes read/update this).
STORAGE = load_storage_config()
apply_storage_config(STORAGE)


# ---------------------------------------------------------------------------- #
# System mode: "Sorting Machine" vs "Dedicated Server"
# ----------------------------------------------------------------------------
# Chosen once on first run (or after a Reset). "sorting_machine" enforces the
# 1M-entry cap and is the only choice on a Raspberry Pi. "dedicated_server"
# lifts the cap and is intended for the higher-capability PostgreSQL backend;
# when DATABASE_URL is set it is used as the database (SQLAlchemy handles the
# rest, and the SQLite-only tuning/indexes below are guarded).
# ---------------------------------------------------------------------------- #
SYSTEM_CONFIG_PATH = os.path.join(BASE_DIR, "system_config.json")
VALID_MODES = ("sorting_machine", "dedicated_server")
SYSTEM = {"mode": None, "unlimited_native_import": False}

# Minimum swap required before the operator may enable unlimited-native import
# (rendering full-resolution scans can need multiple GB per page).
REQUIRED_SWAP_BYTES = 8 * 1000 ** 3   # ~8 GB


def load_system_config():
    try:
        with open(SYSTEM_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        mode = data.get("mode")
        return {
            "mode": mode if mode in VALID_MODES else None,
            "unlimited_native_import": bool(data.get("unlimited_native_import", False)),
        }
    except (OSError, ValueError):
        return {"mode": None, "unlimited_native_import": False}


def save_system_config():
    tmp = SYSTEM_CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({
            "mode": SYSTEM["mode"],
            "unlimited_native_import": bool(SYSTEM.get("unlimited_native_import", False)),
        }, fh, indent=2)
    os.replace(tmp, SYSTEM_CONFIG_PATH)


def _system_mode():
    return SYSTEM["mode"]


def set_system_mode(mode):
    if mode not in VALID_MODES:
        raise ValueError("invalid mode")
    SYSTEM["mode"] = mode
    save_system_config()


def _system_swap_bytes():
    """Total configured swap in bytes (0 if unknown/none). Reads /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("SwapTotal:"):
                    return int(line.split()[1]) * 1024   # value is in kB
    except Exception:
        pass
    return 0


def _native_import_unlimited():
    # An explicit env override wins (headless / Dedicated Server deployments).
    env = os.environ.get("PDF_UNLIMITED_NATIVE")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(SYSTEM.get("unlimited_native_import", False))


def set_native_import_unlimited(enabled):
    """Enable/disable unlimited native-resolution import. Enabling REQUIRES at
    least REQUIRED_SWAP_BYTES of swap; raises ValueError otherwise."""
    enabled = bool(enabled)
    if enabled:
        swap = _system_swap_bytes()
        if swap < REQUIRED_SWAP_BYTES:
            raise ValueError(
                f"At least {REQUIRED_SWAP_BYTES / 1000**3:.0f} GB of swap is required to enable "
                f"unlimited native-resolution import; this system has "
                f"{swap / 1000**3:.1f} GB. Add swap (e.g. an 8 GB swapfile on the NVMe) and retry."
            )
    SYSTEM["unlimited_native_import"] = enabled
    save_system_config()


_pi_cache = {"v": None}


def _is_raspberry_pi():
    """Best-effort Raspberry Pi detection. Env overrides FORCE_PI / FORCE_NOT_PI
    make this testable and let operators correct a misdetection."""
    if os.environ.get("FORCE_NOT_PI") == "1":
        return False
    if os.environ.get("FORCE_PI") == "1":
        return True
    if _pi_cache["v"] is not None:
        return _pi_cache["v"]
    found = False
    try:
        for path in ("/sys/firmware/devicetree/base/model", "/proc/device-tree/model"):
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    if b"raspberry pi" in fh.read().lower():
                        found = True
                        break
        if not found and os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", errors="ignore") as fh:
                txt = fh.read().lower()
            if "raspberry pi" in txt or "bcm2" in txt:
                found = True
    except Exception:
        found = False
    _pi_cache["v"] = found
    return found


SYSTEM = load_system_config()

# In Dedicated Server mode, use PostgreSQL if a connection URL is provided.
_DATABASE_URL = os.environ.get("DATABASE_URL")
if SYSTEM["mode"] == "dedicated_server" and _DATABASE_URL:
    app.config["SQLALCHEMY_DATABASE_URI"] = _DATABASE_URL

db.init_app(app)


# SQLite performance tuning: applied to every new connection. WAL gives far
# better read/write concurrency and speed at scale; the rest trade a little
# durability margin for throughput (safe for a personal, single-writer tool).
# Guarded so it only ever touches SQLite connections.
import sqlite3
from sqlalchemy import event as _sa_event
from sqlalchemy.engine import Engine as _SA_Engine


@_sa_event.listens_for(_SA_Engine, "connect")
def _apply_sqlite_pragmas(dbapi_conn, _conn_record):
    if not isinstance(dbapi_conn, sqlite3.Connection):
        return
    cur = dbapi_conn.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL")       # concurrent reads while writing
        cur.execute("PRAGMA synchronous=NORMAL")     # safe with WAL, much faster writes
        cur.execute("PRAGMA busy_timeout=5000")      # wait instead of erroring on locks
        cur.execute("PRAGMA temp_store=MEMORY")      # temp b-trees in RAM
        cur.execute("PRAGMA cache_size=-16000")      # ~16 MB page cache per connection
        cur.execute("PRAGMA mmap_size=134217728")    # 128 MB memory-mapped I/O
        cur.execute("PRAGMA foreign_keys=ON")
    except Exception:
        pass
    finally:
        cur.close()


# ====================== DIRECTORY SETUP ======================
def ensure_dirs():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["ROI_TEMPLATE_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"], "debug"), exist_ok=True)
    os.makedirs(app.config["ORB_CACHE_FOLDER"], exist_ok=True)
    os.makedirs(app.config["TEMP_IMPORT_FOLDER"], exist_ok=True)
    os.makedirs(app.config["TEMP_SPLIT_FOLDER"], exist_ok=True)
    os.makedirs(app.config["TEMP_CARD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["INVENTORY_IMAGE_FOLDER"], exist_ok=True)
    os.makedirs(app.config["TEMP_PDF_FOLDER"], exist_ok=True)
    os.makedirs(app.config["TYPE_REF_FOLDER"], exist_ok=True)


# ====================== APP SETTINGS / API KEYS ======================
# API keys (and similar secrets) live in the app_settings DB table instead of a
# root .env file: editable at runtime from Settings → API Keys, effective
# immediately, and backed up / migrated with the database. get_api_key() reads
# the DB-backed cache first and falls back to the environment (.env) so nothing
# breaks before the one-time seed runs.
KNOWN_API_KEYS = [
    {
        "key": "JUSTTCG_API_KEY",
        "label": "JustTCG API Key",
        "description": "Card price and manual search lookups via JustTCG.",
        "docs": "https://justtcg.com",
    },
    {
        "key": "XIMILAR_API_TOKEN",
        "label": "Ximilar API Token",
        "description": "Image-based card recognition (Search by Image / photo ID).",
        "docs": "https://www.ximilar.com",
    },
    {
        "key": "POKEMONTCG_API_KEY",
        "label": "Pokémon TCG API Key",
        "description": "Reference catalog sync from pokemontcg.io (optional — raises rate limits).",
        "docs": "https://dev.pokemontcg.io",
    },
]
_KNOWN_KEY_NAMES = {k["key"] for k in KNOWN_API_KEYS}

_settings_cache = {}   # key -> value (populated at startup and on save)


def get_setting(key, default=""):
    if key in _settings_cache:
        v = _settings_cache[key]
        return v if v is not None else default
    return os.environ.get(key, default)


# API keys are just settings; a clearer name for call sites.
get_api_key = get_setting


def set_setting(key, value):
    key = (key or "").strip()
    if not key:
        return
    row = AppSetting.query.filter_by(key=key).first()
    if row is None:
        db.session.add(AppSetting(key=key, value=value))
    else:
        row.value = value
    db.session.commit()
    _settings_cache[key] = value
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def delete_setting(key):
    row = AppSetting.query.filter_by(key=key).first()
    if row:
        db.session.delete(row)
        db.session.commit()
    _settings_cache.pop(key, None)
    os.environ.pop(key, None)


def load_settings():
    """Load settings into the in-memory cache and mirror them into the
    environment (DB wins). One-time: seed known keys from an existing .env so
    upgrades keep working, after which the DB is the source of truth."""
    try:
        rows = AppSetting.query.all()
    except Exception:
        return
    for r in rows:
        _settings_cache[r.key] = r.value
        if r.value is not None:
            os.environ[r.key] = r.value
    for name in _KNOWN_KEY_NAMES:
        if name not in _settings_cache:
            envv = os.environ.get(name)
            if envv:
                try:
                    set_setting(name, envv)
                except Exception:
                    _settings_cache[name] = envv



# ====================== PATH HELPERS ======================
def normalize_to_upload_relative(path_value):
    if not path_value:
        return ""

    normalized = str(path_value).replace("\\", "/")

    # External URLs (e.g. Photo URL column from a CSV import) are stored and
    # returned as-is — they don't live under UPLOAD_FOLDER and shouldn't be
    # mangled by the prefix-stripping below.
    if normalized.startswith("http://") or normalized.startswith("https://"):
        return normalized

    upload_prefix = app.config["UPLOAD_FOLDER"].replace("\\", "/").rstrip("/") + "/"

    if normalized.startswith(upload_prefix):
        normalized = normalized[len(upload_prefix):]

    return normalized.lstrip("/")


def build_uploaded_file_url(path_value):
    relative_path = normalize_to_upload_relative(path_value)
    if not relative_path:
        return None
    if relative_path == "__blank__":
        return url_for("static", filename="blank.jpg")
    if relative_path.startswith("http://") or relative_path.startswith("https://"):
        return relative_path
    return url_for("uploaded_file", filename=relative_path)


def move_temp_card_to_inventory(filename):
    ensure_dirs()

    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("Invalid temp card filename")

    src = os.path.join(app.config["TEMP_CARD_FOLDER"], safe_name)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Temp card image not found: {safe_name}")

    stem, ext = os.path.splitext(safe_name)
    if not ext:
        ext = ".png"

    final_name = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
    dst = os.path.join(app.config["INVENTORY_IMAGE_FOLDER"], final_name)

    shutil.move(src, dst)
    return normalize_to_upload_relative(dst)


def parse_card_filename(filename):
    safe_name = secure_filename(filename)
    base_name = os.path.splitext(os.path.basename(safe_name))[0]
    parts = base_name.split("-")

    if len(parts) < 4:
        return {}

    # Current format embeds the page side as the final segment:
    #   game-album-page-slot-side.png
    # Older imports (from before front/back support) omit it:
    #   game-album-page-slot.png
    if len(parts) >= 5 and parts[-1] in ("front", "back"):
        side  = parts[-1]
        slot  = parts[-2]
        page  = parts[-3]
        album = "-".join(parts[1:-3])
    else:
        side  = "front"
        slot  = parts[-1]
        page  = parts[-2]
        album = "-".join(parts[1:-2])

    return {
        "game":  parts[0].replace("_", " "),
        "album": album.replace("_", " "),
        "page":  page,
        "slot":  slot,
        "side":  side,
    }


def get_record_value(record, key, default=""):
    extracted = record.extracted_data or {}
    value = extracted.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def find_saved_image(subfolder, name):
    """
    Return the URL for a previously uploaded cover image, or None.
    Scans uploads/<subfolder>/ for any file whose stem matches secure_filename(name).
    """
    folder = os.path.join(app.config["UPLOAD_FOLDER"], subfolder)
    if not os.path.isdir(folder):
        return None
    safe_stem = os.path.splitext(secure_filename(name))[0]
    for fname in os.listdir(folder):
        stem, _ = os.path.splitext(fname)
        if stem == safe_stem:
            return url_for("uploaded_file", filename=f"{subfolder}/{fname}")
    return None


def build_album_index():
    records = ScanRecord.query.order_by(ScanRecord.scan_date.desc()).all()
    album_map = {}

    for record in records:
        data = record.extracted_data or {}
        if _is_catalog_only(data):
            continue

        album_name = get_record_value(record, "album")
        if not album_name:
            continue

        game_name = get_record_value(record, "game")

        info = album_map.setdefault(
            album_name,
            {
                "name": album_name,
                "count": 0,
                "latest_scan": record.scan_date,
                "records": [],
                "games": set(),
            },
        )
        info["count"] += 1
        info["records"].append(record)

        if game_name:
            info["games"].add(game_name)

        if record.scan_date and (info["latest_scan"] is None or record.scan_date > info["latest_scan"]):
            info["latest_scan"] = record.scan_date

    albums = []
    for info in album_map.values():
        albums.append(
            {
                "name": info["name"],
                "count": info["count"],
                "latest_scan": info["latest_scan"],
                "records": info["records"],
                "games": sorted(info["games"]),
            }
        )

    return sorted(
        albums,
        key=lambda item: item["latest_scan"] or datetime.min,
        reverse=True,
    )


# ====================== GAME TEMPLATES ======================
# A "template" file (templates/roi/<name>.json) now represents a Game
# definition: just a name plus a flat list of fields that belong to every
# entry for that game. There are no OCR zones / ROI coordinates anymore —
# imported cards are simply created with these fields blank, ready to be
# filled in by hand from the Inventory / Inventory Detail pages.
def load_template(template_name="product_label"):
    template_path = os.path.join(app.config["ROI_TEMPLATE_FOLDER"], f"{template_name}.json")
    if not os.path.exists(template_path):
        template_path = os.path.join(app.config["ROI_TEMPLATE_FOLDER"], "product_label.json")

    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ====================== CARD ALIGNMENT HELPERS ======================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height), flags=cv2.INTER_CUBIC)
    return warped


def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def process_card_image(image, canny_low=50, canny_high=200, approx_eps=0.02, min_area_pct=0.05):
    orig = image.copy()
    ratio = 1.0

    if max(image.shape[:2]) > 1000:
        ratio = 1000.0 / max(image.shape[:2])
        image = cv2.resize(
            image,
            (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    card_contour = None
    image_area = image.shape[0] * image.shape[1]
    eps_values = [approx_eps * 0.75, approx_eps, approx_eps * 1.5, approx_eps * 2.5]

    for eps_factor in eps_values:
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_pct * image_area:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps_factor * peri, True)
            if len(approx) == 4:
                card_contour = approx
                break

        if card_contour is not None:
            break

    if card_contour is None:
        raise ValueError("Could not find a rectangular card")

    card_contour = card_contour.reshape(4, 2) / ratio
    warped = four_point_transform(orig, card_contour)
    warped = sharpen_image(warped)
    return warped


# Card edge types the detector understands (front-end selector values).
CARD_EDGE_TYPES = ("rounded", "square")
CARD_EDGE_DEFAULT = "rounded"


def normalize_card_edge_type(value):
    """Coerce a form value to a supported edge type, defaulting to 'rounded'."""
    v = (value or "").strip().lower()
    return v if v in CARD_EDGE_TYPES else CARD_EDGE_DEFAULT


def _expand_quad(quad, center, margin):
    """Push each corner of `quad` outward from `center` by `margin` pixels, so
    the crop includes a thin sliver of background beyond the card edge (ensures
    the complete card is captured and never clipped)."""
    cx, cy = center
    out = []
    for (x, y) in quad:
        vx, vy = float(x) - cx, float(y) - cy
        n = (vx * vx + vy * vy) ** 0.5 or 1.0
        out.append([x + margin * vx / n, y + margin * vy / n])
    return np.array(out, dtype="float32")


def _card_foreground_rect(bgr, scale_max=1400, min_area_pct=0.10):
    """
    Locate the card as the largest foreground region that differs from the
    background. The background colour is sampled from the four corners of the
    frame — almost always background for a 3x3-cut tile or a single-card photo —
    and Otsu picks the foreground/background split on the colour-distance map.
    This adapts to any background (white binder page, coloured mat, ...) and to
    cards lighter OR darker than their surroundings, and is far more reliable
    than edge-following when the card fills most of the frame.

    Returns (contour, minAreaRect, area_frac) in FULL-resolution coordinates,
    or None if nothing card-like is found.
    """
    H, W = bgr.shape[:2]
    sf = scale_max / max(H, W) if max(H, W) > scale_max else 1.0
    small = (cv2.resize(bgr, (max(1, int(W * sf)), max(1, int(H * sf))),
                        interpolation=cv2.INTER_AREA) if sf < 1 else bgr)
    sh, sw = small.shape[:2]

    # Estimate background colour from the four corner patches.
    k = max(6, int(0.04 * min(sh, sw)))
    corners = np.concatenate([
        small[:k, :k].reshape(-1, 3),  small[:k, -k:].reshape(-1, 3),
        small[-k:, :k].reshape(-1, 3), small[-k:, -k:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)

    dist = np.linalg.norm(small.astype(np.int16) - bg, axis=2)
    dist = np.clip(dist, 0, 255).astype(np.uint8)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kc = max(9, int(0.03 * min(sh, sw)))
    kc += (kc % 2 == 0)            # force odd kernel size
    ker = np.ones((kc, kc), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)   # fill holes in the card
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)    # drop small background specks

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area_frac = cv2.contourArea(c) / float(sh * sw)
    if area_frac < min_area_pct:
        return None
    if sf < 1:                    # scale the contour back up to full-res coords
        c = (c.astype("float32") / sf).astype("int32")
    return c, cv2.minAreaRect(c), area_frac


def detect_and_crop_card(bgr, edge_type=CARD_EDGE_DEFAULT, margin_frac=0.008, min_margin=2):
    """
    Detect the single card in `bgr`, correct its slight rotation, and crop
    tightly to it — including a 1-3px sliver of background beyond the card edge
    so the complete card is always captured (never clipped).

    `edge_type` controls how the card's corners are taken from the detected
    outline:

      "rounded" — modern rounded-corner cards: the minimum-area rotated
                  rectangle is used, which *encloses* the curved corners.

      "square"  — vintage sharp-corner cards: a 4-corner polygon is fit to the
                  outline so the crop follows the true corners (falling back to
                  the rotated rectangle if a clean quad can't be found).

    Returns (cropped_bgr, True) on success, or (bgr, False) when no card-shaped
    region is found — conservative, so it can be applied unconditionally without
    risking a good capture.
    """
    try:
        edge_type = normalize_card_edge_type(edge_type)
        found = _card_foreground_rect(bgr)
        if found is None:
            return bgr, False

        contour, rect, area_frac = found
        (cx, cy), (rw, rh), ang = rect
        if rw < 1 or rh < 1:
            return bgr, False
        if area_frac > 0.985:                       # card fills frame: nothing to crop
            return bgr, False
        if not (1.15 <= (max(rw, rh) / min(rw, rh)) <= 1.85):  # ~2.5x3.5 card shape
            return bgr, False

        # A few pixels of outward margin so the whole card (including its edge)
        # is captured. Scales gently with card size but stays small.
        margin = max(min_margin, int(round(margin_frac * min(rw, rh))))

        quad = None
        if edge_type == "square":
            peri = cv2.arcLength(contour, True)
            for eps in (0.02, 0.03, 0.05, 0.08):
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) == 4:
                    quad = _expand_quad(approx.reshape(4, 2).astype("float32"),
                                        (cx, cy), margin)
                    break

        # Rounded cards (and square cards whose quad fit failed) use the
        # minimum-area rectangle, inflated by the margin.
        if quad is None:
            inflated = ((cx, cy), (rw + 2 * margin, rh + 2 * margin), ang)
            quad = cv2.boxPoints(inflated).astype("float32")

        warped = four_point_transform(bgr, quad)
        # A card is always portrait on import (taller than it is wide). If the
        # detected region is landscape, it is NOT the card outline — almost always
        # an internal horizontal rectangle such as the artwork window or the
        # "NO. / Pokémon" info bar — so reject it rather than accept or rotate a
        # wrong crop. The caller then falls back to a plain deskew of the whole
        # tile, which is safe. (We never rotate a landscape result into portrait:
        # that would turn a wrong horizontal detection into a convincing-looking
        # but incorrect card.)
        if warped.shape[1] >= warped.shape[0]:
            return bgr, False
        warped = sharpen_image(warped)
        return warped, True
    except Exception:
        return bgr, False


def straighten_split_image(image, max_angle=15.0):
    """Slightly rotate a single split-page tile so the card sits upright.

    This intentionally does NOT try to locate the four corners of the card
    or run a perspective ("four point") warp — the 3x3 split already frames
    each card closely enough. All this does is estimate how far the card is
    tilted inside its tile and apply a small corrective rotation so it reads
    as upright. The crop/framing from the split is otherwise left alone.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Separate the card from whatever background is visible in this tile.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    image_area = image.shape[0] * image.shape[1]
    if cv2.contourArea(largest) < 0.05 * image_area:
        # Nothing large enough found — leave the tile as-is rather than guessing.
        return image

    angle = cv2.minAreaRect(largest)[-1]

    # cv2.minAreaRect reports an angle in [-90, 0); convert that into a small
    # +/- rotation relative to upright instead of a raw box angle.
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Ignore tiny noise and anything implausibly large for a "slight" fix.
    if abs(angle) < 0.3 or abs(angle) > max_angle:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    straightened = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return straightened


def split_image_3x3(pil_image, v1_pct, v2_pct, h1_pct, h2_pct):
    w, h = pil_image.size

    v_pos = [0, int(w * float(v1_pct) / 100.0), int(w * float(v2_pct) / 100.0), w]
    h_pos = [0, int(h * float(h1_pct) / 100.0), int(h * float(h2_pct) / 100.0), h]

    pieces = []
    for row in range(3):
        for col in range(3):
            box = (v_pos[col], h_pos[row], v_pos[col + 1], h_pos[row + 1])
            pieces.append(pil_image.crop(box))

    return pieces


# Maps a piece's position in the photographed image (1-9, left-to-right /
# top-to-bottom, i.e. the order returned by split_image_3x3) to the physical
# pocket number it should be filed under when that photo is of the BACK of a
# 9-pocket page. Flipping a page over left-to-right (as when turning it to
# photograph the back) mirrors the pocket order horizontally within each row,
# so a back photo's leftmost pocket in a row is actually the row's rightmost
# physical pocket, and vice versa. This keeps a card's front and back slot
# numbers pointing at the same physical pocket. Front photos need no
# remapping since they're numbered directly as photographed.
BACK_SLOT_MAP = {1: 3, 2: 2, 3: 1, 4: 6, 5: 5, 6: 4, 7: 9, 8: 8, 9: 7}


def resolve_slot_number(photographed_index, side):
    """
    Given a piece's 1-9 index in photographed (crop) order, return the slot
    number it should be filed under for the given page side ("front"/"back").
    """
    if side == "back":
        return BACK_SLOT_MAP[photographed_index]
    return photographed_index


def get_template_names():
    ensure_dirs()
    templates = [
        f.replace(".json", "")
        for f in os.listdir(app.config["ROI_TEMPLATE_FOLDER"])
        if f.endswith(".json")
    ]
    if not templates:
        templates = ["product_label"]
    return templates


def match_product_from_extracted(extracted):
    brand = extracted.get("brand", "").lower().strip()
    product_name = extracted.get("product_name", "").lower().strip()

    matched_product = None
    if brand or product_name:
        query = Product.query
        if brand:
            query = query.filter(Product.brand.like(f"%{brand}%"))
        if product_name:
            query = query.filter(Product.product_name.like(f"%{product_name}%"))
        matched_product = query.first()

    return matched_product


def find_existing_record_for_key(game, album, page, slot):
    """
    Look up a ScanRecord already occupying the same (game, album, page, slot)
    identity. Used to merge a front and back photo of the same physical
    pocket into a single record instead of creating two.
    """
    if not (game and album and page and slot):
        return None

    for record in ScanRecord.query.all():
        data = record.extracted_data or {}
        if (
            str(data.get("game",  "")).strip() == game  and
            str(data.get("album", "")).strip() == album and
            str(data.get("page",  "")).strip() == page  and
            str(data.get("slot",  "")).strip() == slot
        ):
            return record
    return None


# ====================== INVENTORY CAP ======================
# This SQLite-backed build is capped so it never grows past what a small machine
# (e.g. a Raspberry Pi) can serve well. Beyond this, the data belongs on the
# desktop build (PostgreSQL + partitioning + object storage); Settings → Upgrade
# exports a migration bundle for it. Cap is overridable via env for testing.
INVENTORY_MAX_RECORDS = int(os.environ.get("INVENTORY_MAX_RECORDS", "1000000"))

UPGRADE_MESSAGE = (
    f"Inventory cap reached ({INVENTORY_MAX_RECORDS:,} entries). To grow further, connect a "
    "dedicated computer and migrate to the desktop version, which uses PostgreSQL, table "
    "partitioning, and object storage for images. Open Settings \u2192 Upgrade to export your "
    "migration bundle."
)


class InventoryCapError(Exception):
    """Raised by create_scan_record when the hard record cap is reached."""
    def __init__(self, count):
        self.count = count
        self.limit = INVENTORY_MAX_RECORDS
        super().__init__(UPGRADE_MESSAGE)


# Cached row count so the per-record cap check during large imports is O(1)
# instead of a COUNT(*) per insert. Refreshed from the DB at the start of each
# import operation (which also picks up any deletions since last time).
_inv_count_cache = {"n": None}


def _inventory_count(refresh=False):
    if refresh or _inv_count_cache["n"] is None:
        _inv_count_cache["n"] = ScanRecord.query.count()
    return _inv_count_cache["n"]


def _inventory_count_bump(delta):
    if _inv_count_cache["n"] is not None:
        _inv_count_cache["n"] = max(0, _inv_count_cache["n"] + delta)


def _effective_cap():
    """The active record cap: the 1M limit in Sorting Machine mode, or None
    (uncapped) in Dedicated Server mode. Unset mode defaults to the cap so a
    fresh, not-yet-configured system can never be pushed past it before setup."""
    if _system_mode() == "dedicated_server":
        return None
    return INVENTORY_MAX_RECORDS


def _inventory_remaining():
    cap = _effective_cap()
    if cap is None:
        return float("inf")
    return max(0, cap - _inventory_count(refresh=True))


def create_scan_record(image_path, template_name, extracted, image_path_back=None):
    # Hard cap — the single choke point every import path passes through, so no
    # route can ever push the database past the limit (Sorting Machine mode only;
    # Dedicated Server is uncapped).
    cap = _effective_cap()
    if cap is not None and _inventory_count() >= cap:
        raise InventoryCapError(_inventory_count())

    matched_product = match_product_from_extracted(extracted)

    record = ScanRecord(
        image_path=normalize_to_upload_relative(image_path),
        image_path_back=normalize_to_upload_relative(image_path_back) if image_path_back else None,
        template_used=template_name,
        extracted_data=dict(extracted),
        matched_product_id=matched_product.id if matched_product else None,
    )
    db.session.add(record)
    db.session.commit()
    _inventory_count_bump(1)

    return matched_product, record


# ====================== OCR IDENTIFICATION HELPERS ======================
def _abs_record_image_path(path_value):
    """Absolute on-disk path for a record's stored image, or None if it has no
    real front image (blank sentinel / empty)."""
    if not path_value or str(path_value).strip() in ("", "__blank__"):
        return None
    relative = normalize_to_upload_relative(path_value)
    return os.path.join(app.config["UPLOAD_FOLDER"], relative)


def _build_ocr_candidates(exclude_record_id=None, catalog_only=False):
    """
    Build the plain candidate list card_ocr.match_ocr_to_records expects from
    existing ScanRecords. Each candidate carries the identity fields used for
    scoring (name/serial) plus display fields (set/game/thumbnail) so the UI can
    render the picker directly from the match result.

    catalog_only=True restricts matching to imported reference rows (the CSV
    "Imported Catalog"), which is usually the cleanest identification target.
    """
    candidates = []
    for r in ScanRecord.query.all():
        if exclude_record_id is not None and r.id == exclude_record_id:
            continue
        data = r.extracted_data or {}
        if catalog_only and not _is_catalog_only(data):
            continue

        name = _get_name(data)
        serial = _get_serial(data)
        if not name and not serial:
            continue  # nothing to match on

        candidates.append({
            "record_id": r.id,
            "name":      name,
            "serial":    serial,
            "set":       str(data.get("set") or data.get("set_name") or "").strip(),
            "game":      str(data.get("game") or "").strip(),
            "thumbnail": build_uploaded_file_url(r.image_path) if r.image_path else None,
        })
    return candidates


# Identity fields copied onto a record when the user accepts a matched source
# record via /ocr_apply. Only non-empty values are copied.
_OCR_COPY_KEYS = (
    "name", "product_name", "card_name", "title",
    "serial", "number", "set_number", "collector_number", "card_number",
    "set", "set_name", "rarity", "game",
)


# ====================== TCGCSV REFERENCE-CATALOG HELPERS ======================
# Fields copied onto a scan record when the user accepts a tcgcsv reference card.
_REFERENCE_APPLY_MAP = {
    # extracted_data key : ReferenceCard attribute
    "name":       "name",
    "set_number": "number",   # the N/M collector number -> the "Set Number" field
    "set":        "set_name",
    "rarity":     "rarity",
    "game":       "game",
}


def _reference_upsert(rec):
    """Insert or update a ReferenceCard from a normalized tcgcsv product dict
    (see ref_sync.normalize_product). Keyed on the upstream productId."""
    existing = ReferenceCard.query.filter_by(product_id=rec["product_id"]).first()
    if existing is None:
        existing = ReferenceCard(product_id=rec["product_id"])
        db.session.add(existing)
    existing.category_id  = rec["category_id"]
    existing.group_id     = rec["group_id"]
    existing.game         = rec["game"]
    existing.set_name     = rec["set_name"]
    existing.name         = rec["name"]
    existing.clean_name   = rec["clean_name"]
    existing.number       = rec["number"]
    existing.rarity       = rec["rarity"]
    existing.image_url    = rec["image_url"]
    existing.url          = rec["url"]
    existing.market_price = rec["market_price"]
    existing.extended     = rec["extended"]
    return existing


def _resolve_category_for_game(game_str):
    """
    Map a record's game name (e.g. "Pokemon", "Pokémon TCG") to a downloaded
    tcgcsv category. Returns (category_id, game_name) or (None, None). Only games
    that have actually been synced (a ReferenceSync row exists) can resolve —
    that row carries the authoritative tcgplayer categoryId.

    Matching is accent- and punctuation-insensitive: records commonly store
    "Pokémon" while tcgcsv's category is "Pokemon", and a naive compare would
    never match those, producing zero reference matches despite a good name read.
    """
    import unicodedata

    def _key(s):
        s = unicodedata.normalize("NFKD", str(s or ""))
        s = "".join(ch for ch in s if not unicodedata.combining(ch))  # drop accents
        return "".join(ch for ch in s.lower() if ch.isalnum())        # alnum only

    g = _key(game_str)
    if not g:
        return None, None
    syncs = ReferenceSync.query.all()
    for rs in syncs:                       # exact, accent/punctuation-insensitive
        if _key(rs.game) == g:
            return rs.category_id, rs.game
    for rs in syncs:                       # loose contains match either direction
        rg = _key(rs.game)
        if rg and (rg in g or g in rg):
            return rs.category_id, rs.game
    return None, None


def _collector_number_variants(num_str):
    """
    All plausible string forms of an N/M collector number, so an exact-string DB
    lookup matches regardless of zero-padding. tcgcsv pads the N to the digit
    width of the M (M=112 -> "024/112"; M=64 -> "05/64"; M=9 -> "3/9"), so the
    primary variant follows that rule; the bare and fixed-3-digit forms are added
    for safety against other conventions. Returns a set (empty if not an N/M).
    """
    m = _re.search(r"(\d{1,4})\s*/\s*(\d{1,4})", str(num_str or ""))
    if not m:
        return set()
    n, tot = int(m.group(1)), int(m.group(2))
    width = len(str(tot))                       # M's digit count drives N's padding
    return {
        f"{n}/{tot}",                           # stripped
        f"{n:0{width}d}/{tot}",                 # pad N to M's width  (CSV convention)
        f"{n:03d}/{tot:03d}",                   # legacy fixed 3-digit
        f"{n:03d}/{tot}",
    }


def _canonical_collector_number(num_str):
    """Return an N/M number padded to the CSV convention (N padded to M's digit
    width, e.g. 24/112 -> 024/112, 5/64 -> 05/64). Returns the input unchanged if
    it isn't a recognisable N/M number."""
    m = _re.search(r"(\d{1,4})\s*/\s*(\d{1,4})", str(num_str or ""))
    if not m:
        return str(num_str or "").strip()
    n, tot = int(m.group(1)), int(m.group(2))
    return f"{n:0{len(str(tot))}d}/{tot}"


# Evolution-stage / rarity / UI tokens that show up in the name zone but are
# never the card name. The evolution badge sits just left of the name and OCRs
# before it (often mangled, e.g. "BASIC" -> "SIC"), so we strip these — with a
# little fuzz for OCR errors — before using the name to narrow/score candidates.
_OCR_NAME_NOISE = frozenset({
    "hp", "stage", "stage1", "stage2", "basic", "evolves", "from", "pokemon",
    "pokmon", "illus", "no", "the", "ex", "gx", "restored", "mega", "break",
    "vmax", "vstar", "vunion", "tag", "team", "lv", "item", "supporter",
    "stadium", "energy", "legend",
})


def _strip_ocr_name_noise(name):
    """Drop stage/rarity/UI tokens (incl. lightly OCR-mangled ones) from an OCR'd
    name so 'BASIC Panpour' / 'sic Panpour' narrows and scores as 'Panpour'."""
    import difflib
    out = []
    for tok in _re.split(r"\s+", str(name or "").strip()):
        t = _re.sub(r"[^a-z0-9]", "", tok.lower())
        if not t or t.isdigit() or t in _OCR_NAME_NOISE:
            continue
        if len(t) <= 6 and any(
            len(w) >= 3 and difflib.SequenceMatcher(None, t, w).ratio() >= 0.7
            for w in _OCR_NAME_NOISE
        ):
            continue
        out.append(tok)
    return " ".join(out).strip()


def _reference_candidates_for_ocr(category_id, ocr_result, limit=8):
    """
    Build scored reference-card candidates for an OCR result within one game.
    Pre-narrows by collector number (then name) so we never fuzzy-score an entire
    game, then reuses card_ocr's scorer. Each candidate is tagged source
    "reference" and carries product_id + rich fields for auto-fill.

    Identification is NUMBER-FIRST: the collector number (N/M) is the reliable
    key, so we narrow by it before falling back to the name. The name is also
    stripped of stage/rarity noise so a stray badge token never derails the match.
    """
    if not category_id or card_ocr is None:
        return []

    number   = (ocr_result.get("number_guess") or "").strip()
    raw_name = (ocr_result.get("name_guess") or "").strip()
    name     = _strip_ocr_name_noise(raw_name) or raw_name
    base     = ReferenceCard.query.filter(ReferenceCard.category_id == category_id)

    narrowed = None
    if number:
        variants = _collector_number_variants(number) or {number}
        narrowed = base.filter(ReferenceCard.number.in_(list(variants))).limit(300).all()
    if not narrowed and name:
        first = (name.split() or [name])[0]
        narrowed = base.filter(ReferenceCard.name.ilike(f"%{first}%")).limit(500).all()
    if not narrowed:
        narrowed = base.limit(500).all()

    # Score with the cleaned name so the badge scrap doesn't drag the ratio down.
    ocr_result = {**ocr_result, "name_guess": name}

    candidates = [{
        "source":       "reference",
        "product_id":   rc.product_id,
        "name":         rc.name or "",
        "serial":       rc.number or "",
        "set":          rc.set_name or "",
        "game":         rc.game or "",
        "rarity":       rc.rarity or "",
        "url":          rc.url or "",
        "thumbnail":    rc.image_url or "",
        "market_price": rc.market_price,
    } for rc in narrowed]

    return card_ocr.match_ocr_to_records(ocr_result, candidates, limit=limit)


def remove_file_if_exists(path_value):
    """Delete an uploaded image file, with safety guards:
    - Ignores the '__blank__' sentinel (points to static/blank.jpg, never touched).
    - Only deletes files inside the inventory_cards subdirectory so nothing
      outside that folder can be accidentally removed.
    """
    if not path_value:
        return

    # Never delete the blank-slot sentinel — it maps to a static asset
    if str(path_value).strip() == "__blank__":
        return

    relative_path = normalize_to_upload_relative(path_value)

    # Safety: only allow deletion of files inside inventory_cards/
    normalised = relative_path.replace("\\", "/")
    if not normalised.startswith("inventory_cards/"):
        return

    absolute_path = os.path.join(app.config["UPLOAD_FOLDER"], relative_path)

    if os.path.exists(absolute_path) and os.path.isfile(absolute_path):
        os.remove(absolute_path)


# ====================== DUPLICATE HELPERS ======================
_NAME_KEYS   = ("product_name", "name", "card_name", "title")
_SERIAL_KEYS = ("serial", "number", "collector_number", "set_number", "card_number")


def _get_name(data: dict) -> str:
    for k in _NAME_KEYS:
        v = str(data.get(k, "")).strip()
        if v:
            return v.lower()
    return ""


def _get_serial(data: dict) -> str:
    for k in _SERIAL_KEYS:
        v = str(data.get(k, "")).strip()
        if v:
            return v.lower()
    return ""


# ====================== AUTO-IDENTIFY (used at end of import) ======================
# Minimum match score to auto-apply the top OCR identification on import. Scores
# come from match_ocr_to_records (capped at 1.0): an exact collector-number (N/M)
# match contributes +0.5 and the card-name similarity contributes up to +0.65, so
# 0.60 means "a confident combined name + number match" — e.g. an exact number
# plus even a partial name read, or a near-exact name on its own. Below this the
# entry is left blank for the person to check by hand. Overridable via env.
AUTO_IDENTIFY_MIN_SCORE = float(os.environ.get("AUTO_IDENTIFY_MIN_SCORE", "0.60"))

# ── Card "type" field (e.g. Pokemon energy type) ──
# Minimum confidence for a VISUAL type guess to auto-fill a record's type field.
# Colour-distinct types clear this; ambiguous red-orange/neutral guesses (kept at
# lower confidence in card_ocr) fall below it and are left blank.
TYPE_MIN_CONFIDENCE = 0.6
# extendedData keys that carry a card's type/colour when the reference catalog
# has it (authoritative). "Card Type" is intentionally excluded — for Pokemon it
# means Pokemon/Trainer/Energy, not the energy colour.
_TYPE_EXTENDED_KEYS = ("energy type", "color", "colors", "type", "attribute", "element")


def _template_type_field_key(template):
    """Return the template's type-like field key (matches type/energy/element/
    colour/attribute), or None if the game doesn't define one."""
    import re as _re
    for key in (template.get("fields", {}) or {}).keys():
        if _re.search(r"type|energy|element|colou?r|attribute", str(key), _re.I):
            return key
    return None


def _reference_type_value(product_id):
    """The card's type/colour from the matched reference card's extendedData,
    if present (authoritative). '' when unknown."""
    try:
        ref = ReferenceCard.query.filter_by(product_id=int(product_id)).first()
    except (TypeError, ValueError):
        ref = None
    if ref is None:
        return ""
    ext = ref.extended or {}
    lowered = {str(k).strip().lower(): v for k, v in ext.items()}
    for k in _TYPE_EXTENDED_KEYS:
        v = lowered.get(k)
        if str(v or "").strip():
            return str(v).strip()
    return ""


def _fill_type_field(record, type_value):
    """Set the record's type field to `type_value` if its Game defines one and it
    isn't already filled. Returns (field_key, value) on success, else None.
    Mutates record.extracted_data in memory; the caller commits."""
    if not type_value:
        return None
    try:
        template = load_template(record.template_used)
    except Exception:
        return None
    key = _template_type_field_key(template)
    if not key:
        return None
    ext = record.extracted_data or {}
    if str(ext.get(key) or "").strip():
        return None                       # already has a type — don't overwrite
    record.extracted_data = {**ext, key: type_value}
    return key, type_value


# ── Reference-icon library (template matching source) ──
def _type_game_key(game):
    """Normalise a game name to the key used for type-reference lookup."""
    return (game or "").strip().lower()


# Small cache of prepared (feature-computed) references per game key, invalidated
# whenever the library changes. Avoids re-reading/preparing icons for every card
# in a batch.
_TYPE_REF_CACHE = {}
_TYPE_REF_VERSION = 0


def _bump_type_refs():
    """Invalidate the prepared-reference cache after any add/delete."""
    global _TYPE_REF_VERSION
    _TYPE_REF_VERSION += 1
    _TYPE_REF_CACHE.clear()


def _load_prepared_type_refs(game):
    """
    Return prepared reference icons (features cached) for a game, or [] if the
    game has no library yet. Used to drive template-matching type detection.
    """
    if card_ocr is None:
        return []
    key = _type_game_key(game)
    if not key:
        return []
    cached = _TYPE_REF_CACHE.get(key)
    if cached is not None and cached[0] == _TYPE_REF_VERSION:
        return cached[1]

    rows = TypeReference.query.filter(db.func.lower(TypeReference.game) == key).all()
    raw = []
    for r in rows:
        abs_path = _abs_record_image_path(r.image_path)
        if not abs_path or not os.path.exists(abs_path):
            continue
        img = cv2.imread(abs_path)
        if img is None:
            continue
        raw.append({"type_name": r.type_name, "region": r.region, "image": img})
    prepared = card_ocr.prepare_type_references(raw)
    _TYPE_REF_CACHE[key] = (_TYPE_REF_VERSION, prepared)
    return prepared


def _apply_ocr_candidate(record, cand):
    """
    Merge identity fields from a matched candidate onto `record` (in memory; the
    caller commits). `cand` is one entry produced by the OCR matcher, either a
    tcgcsv reference card (source="reference", carries product_id + rich fields)
    or another existing record (source="record"). Returns the dict of applied
    updates, or {} if nothing could be applied.
    """
    updates = {}

    if cand.get("source") == "reference":
        try:
            ref = ReferenceCard.query.filter_by(product_id=int(cand.get("product_id"))).first()
        except (TypeError, ValueError):
            ref = None
        if ref is None:
            return {}
        for key, attr in _REFERENCE_APPLY_MAP.items():
            val = getattr(ref, attr, None)
            if str(val or "").strip():
                updates[key] = val
        if ref.url:
            updates["tcgplayer"] = {
                "url":          ref.url,
                "full_url":     ref.url,
                "source":       "tcgcsv",
                "saved_at":     datetime.utcnow().isoformat(),
                "product_id":   str(ref.product_id),
                "product_name": ref.name or "",
                "set_name":     ref.set_name or "",
                "set_number":   ref.number or "",
                "prices":       {"market": ref.market_price} if ref.market_price is not None else {},
            }
        # Current market value on identification — fill only if not already set,
        # so a value the user entered by hand is never overwritten.
        if getattr(ref, "market_price", None) is not None and \
           not str((record.extracted_data or {}).get("current_value") or "").strip():
            updates["current_value"] = ref.market_price
        # Catalog type/colour (authoritative), if the game has a type field and
        # the reference data carries it.
        cat_type = _reference_type_value(ref.product_id)
        if cat_type:
            type_key = _template_type_field_key(load_template(record.template_used))
            if type_key and not str((record.extracted_data or {}).get(type_key) or "").strip():
                updates[type_key] = cat_type
    else:  # an existing record
        try:
            source = ScanRecord.query.get(int(cand.get("record_id")))
        except (TypeError, ValueError):
            source = None
        if source is None:
            return {}
        sdata = source.extracted_data or {}
        for k in _OCR_COPY_KEYS:
            val = sdata.get(k)
            if str(val or "").strip():
                updates[k] = val

    if not updates:
        return {}

    merged = {**(record.extracted_data or {}), **updates}
    record.extracted_data = merged
    matched = match_product_from_extracted(merged)
    if matched:
        record.matched_product_id = matched.id
    return updates


def auto_identify_record(record, min_score=AUTO_IDENTIFY_MIN_SCORE):
    """
    OCR a record's front image and, if a candidate matches with sufficient
    confidence (score >= min_score, default 60% — a strong combined name + N/M
    match), apply that identity to the record. Otherwise the record is left
    untouched (blank) for manual entry.

    This is the same identification the inventory-detail page runs, invoked
    automatically at the end of an import. It never raises and never commits: any
    OCR/matching problem simply yields identified=False, and the caller decides
    when to persist.

    Separately from the name/serial identity, it also fills the Game's "type"
    field (e.g. Pokemon energy type) when the card provides one — from the
    matched reference card if identified, otherwise from a confident VISUAL
    reading of the type icon. This runs even when there's no identity match,
    so cards can still be sorted by type.

    Returns: { identified, reason, score, name, applied: {..},
               type_guess, type_confidence, type_applied: {field, value}|None }
    """
    out = {"identified": False, "reason": "", "score": None, "name": "",
           "applied": {}, "type_guess": "", "type_confidence": 0.0, "type_applied": None}

    if card_ocr is None:
        out["reason"] = "ocr_unavailable"
        return out

    abs_path = _abs_record_image_path(record.image_path)
    if not abs_path or not os.path.exists(abs_path):
        out["reason"] = "no_front_image"
        return out

    ext = record.extracted_data or {}
    game = ext.get("game", "")

    try:
        ocr = card_ocr.ocr_card_front(abs_path, game=game,
                                      type_refs=_load_prepared_type_refs(game))
    except Exception:
        out["reason"] = "ocr_error"
        return out
    if not ocr.get("ocr_available"):
        out["reason"] = "ocr_unavailable"
        return out

    out["type_guess"] = ocr.get("type_guess", "")
    out["type_confidence"] = ocr.get("type_confidence", 0.0)

    # Reference catalog (rich auto-fill) for this record's game, if synced, plus
    # any already-identified existing records — same sources as /ocr_identify.
    category_id, _ = _resolve_category_for_game(game)
    ref_matches = _reference_candidates_for_ocr(category_id, ocr) if category_id else []
    rec_matches = card_ocr.match_ocr_to_records(
        ocr, _build_ocr_candidates(exclude_record_id=record.id))
    for m in rec_matches:
        m["source"] = "record"

    # Best first; reference matches precede record matches on ties (listed first
    # and Python's sort is stable), so a confident catalog hit wins — it carries
    # set/rarity/TCGplayer data a bare record match can't.
    combined = sorted(ref_matches + rec_matches, key=lambda c: c.get("score", 0), reverse=True)
    top = combined[0] if combined else None

    if top is not None:
        out["score"] = top.get("score")
        out["name"] = top.get("name", "")

    # 1) Identity fields — applied when the top match clears min_score (60%).
    if top is not None and float(top.get("score", 0) or 0) >= float(min_score) - 1e-9:
        applied = _apply_ocr_candidate(record, top)
        if applied:
            out["identified"] = True
            out["reason"] = "applied"
            out["applied"] = applied
        else:
            out["reason"] = "apply_failed"
    else:
        out["reason"] = "below_threshold" if top is not None else "no_candidates"

    # 2) Type field — independent of the identity match. Prefer the catalog value
    #    (authoritative) when we identified a reference card; otherwise use a
    #    confident visual guess. _apply_ocr_candidate may already have filled it
    #    from the catalog, in which case _fill_type_field is a no-op.
    type_value = ""
    if out["identified"] and top is not None and top.get("source") == "reference":
        type_value = _reference_type_value(top.get("product_id"))
    if not type_value and out["type_confidence"] >= TYPE_MIN_CONFIDENCE:
        type_value = out["type_guess"]
    if type_value:
        filled = _fill_type_field(record, type_value)
        if filled:
            out["type_applied"] = {"field": filled[0], "value": filled[1]}

    # Reflect a type that was already filled from the catalog inside
    # _apply_ocr_candidate, so callers/UI can report it consistently.
    if out["type_applied"] is None:
        try:
            tkey = _template_type_field_key(load_template(record.template_used))
            tval = str((record.extracted_data or {}).get(tkey) or "").strip() if tkey else ""
            if tval:
                out["type_applied"] = {"field": tkey, "value": tval}
        except Exception:
            pass

    return out


EDITION_OPTIONS = ("Standard Edition", "First Edition", "Limited Edition")
EDITION_DEFAULT = "Standard Edition"

_HOLOGRAPHIC_OPTIONS = ("None", "Regular", "Reverse", "Shiny Text", "Special")


def _get_edition(data: dict) -> str:
    """Normalise edition to one of the known option strings.

    Handles legacy boolean fields so old records migrate gracefully:
      - first_edition == True   -> 'First Edition'
      - limited_edition == True -> 'Limited Edition'
      - otherwise               -> 'Standard Edition'
    """
    # New-style string field takes precedence
    raw = str(data.get("edition", "")).strip()
    if raw in EDITION_OPTIONS:
        return raw

    # Legacy boolean migration
    fe = data.get("first_edition", False)
    if fe is True or str(fe).strip().lower() == "true":
        return "First Edition"
    le = data.get("limited_edition", False)
    if le is True or str(le).strip().lower() == "true":
        return "Limited Edition"

    return EDITION_DEFAULT


# ====================== GROUPING HELPER ======================
def build_group_info(records):
    """
    Group a list of ScanRecord objects by (name, serial, edition, holographic).
    Only records whose extracted_data has finalized == True (or 'True') are grouped;
    unfinalized records are treated as singletons.

    Returns:
        group_info  – dict mapping representative record.id -> {
                          'count':     int,
                          'all_ids':   [int, ...],   # all IDs in the group
                          'locations': [str, ...],   # "page/slot" strings for duplicates
                      }
        rep_records – list of the one representative ScanRecord per group,
                      in the same relative order as the input list.
    """
    from collections import OrderedDict

    # key -> list of records
    groups: dict = OrderedDict()
    singletons = []

    for record in records:
        data = record.extracted_data or {}

        finalized = data.get("finalized", False)
        is_final = finalized is True or str(finalized).strip().lower() == "true"

        if not is_final:
            singletons.append(record)
            continue

        name  = _get_name(data)
        serial = _get_serial(data)
        edition = _get_edition(data)
        holo = str(data.get("holographic", "")).strip().lower()

        group_key = (name, serial, edition, holo)
        groups.setdefault(group_key, []).append(record)

    group_info = {}
    rep_records = []

    for group_key, members in groups.items():
        rep = members[0]  # representative = first encountered (most recent due to DESC ordering)
        all_ids = [r.id for r in members]
        locations = [
            "{}/{}".format(
                (r.extracted_data or {}).get("page", "?"),
                (r.extracted_data or {}).get("slot", "?"),
            )
            for r in members[1:]  # skip the rep itself
        ]
        group_info[rep.id] = {
            "count":     len(members),
            "all_ids":   all_ids,
            "locations": locations,
        }
        rep_records.append(rep)

    for record in singletons:
        group_info[record.id] = {
            "count":     1,
            "all_ids":   [record.id],
            "locations": [],
        }
        rep_records.append(record)

    return group_info, rep_records


# ====================== CONTEXT PROCESSOR ======================
@app.context_processor
def utility_functions():
    def build_inventory_url(page_num):
        return url_for(
            "inventory",
            page=page_num,
            search=request.args.get("search", ""),
            game=request.args.get("game", ""),
            album=request.args.get("album", ""),
            template=request.args.get("template", ""),
            per_page=request.args.get("per_page", 50),
            sort=request.args.get("sort", ""),
            sort_dir=request.args.get("sort_dir", "asc"),
            catalog=request.args.get("catalog", ""),
        )
    return dict(
        build_inventory_url=build_inventory_url,
        build_uploaded_file_url=build_uploaded_file_url,
    )


# ====================== FIRST-RUN SETUP GATE ======================
# Until a system mode is chosen (first run, or after a Reset), every page is
# redirected to /setup so the operator picks Sorting Machine vs Dedicated Server.
_SETUP_ALLOWED_PREFIXES = ("/setup", "/static")


@app.before_request
def _require_system_mode():
    if _system_mode() is not None:
        return  # already configured
    path = request.path or "/"
    if path == "/favicon.ico" or any(path.startswith(p) for p in _SETUP_ALLOWED_PREFIXES):
        return
    # Non-page (API/fetch) callers get a clear JSON signal; pages get redirected.
    if request.method != "GET" or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"status": "error", "needs_setup": True,
                        "message": "System not configured. Open the app to choose an implementation."}), 409
    return redirect(url_for("setup_page"))


@app.route("/setup")
def setup_page():
    # Already configured? Send them home.
    if _system_mode() is not None:
        return redirect(url_for("index"))
    return render_template("setup.html", is_pi=_is_raspberry_pi(),
                           pi_model=_pi_model_string())


@app.route("/setup/select", methods=["POST"])
def setup_select():
    mode = (request.form.get("mode") or (request.get_json(silent=True) or {}).get("mode") or "").strip()
    if mode not in VALID_MODES:
        return jsonify({"status": "error", "message": "Choose Sorting Machine or Dedicated Server."}), 400
    # A Raspberry Pi can only run the Sorting Machine build.
    if mode == "dedicated_server" and _is_raspberry_pi():
        return jsonify({"status": "error",
                        "message": "This device is a Raspberry Pi, which can only run the Sorting Machine "
                                   "implementation (1,000,000-entry cap). Use a more capable server or PC "
                                   "for the Dedicated Server implementation."}), 400
    set_system_mode(mode)
    return jsonify({"status": "success", "mode": mode, "redirect": url_for("index")})


def _pi_model_string():
    for path in ("/sys/firmware/devicetree/base/model", "/proc/device-tree/model"):
        try:
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    return fh.read().decode("utf-8", "ignore").replace("\x00", "").strip()
        except Exception:
            pass
    return "Raspberry Pi" if _is_raspberry_pi() else ""


# ====================== PAGE ROUTES ======================
def _welcome_stats():
    """Headline logistics for the welcome page. Uses the denormalized columns so
    these counts stay cheap as the collection grows."""
    from sqlalchemy import func, distinct
    coalesce = db.func.coalesce

    total = _inventory_count(refresh=True)
    catalog = ScanRecord.query.filter(coalesce(ScanRecord.is_catalog, False) == True).count()   # noqa: E712
    archived = ScanRecord.query.filter(coalesce(ScanRecord.is_archived, False) == True).count()  # noqa: E712
    in_inventory = max(0, total - catalog)

    games = (db.session.query(func.count(distinct(ScanRecord.game_key)))
             .filter(ScanRecord.game_key.isnot(None),
                     coalesce(ScanRecord.is_catalog, False) == False).scalar()) or 0     # noqa: E712
    albums = (db.session.query(func.count(distinct(ScanRecord.album_key)))
              .filter(ScanRecord.album_key.isnot(None),
                      coalesce(ScanRecord.is_catalog, False) == False).scalar()) or 0    # noqa: E712

    return {
        "cards_in_inventory": in_inventory,
        "archived": archived,
        "catalog": catalog,
        "games": int(games),
        "albums": int(albums),
    }


@app.route("/")
def index():
    """Welcome / landing page — the logo and the root URL both come here."""
    ensure_dirs()
    return render_template("welcome.html", stats=_welcome_stats(),
                           capacity=_capacity_status())


@app.route("/templates")
def templates_page():
    ensure_dirs()
    games = []
    for name in get_template_names():
        try:
            tpl = load_template(name)
        except Exception:
            continue
        fields = tpl.get("fields", {}) or {}
        games.append({
            "name": name,
            "fields": [
                {
                    "key": field_key,
                    "field_type": (field_cfg or {}).get("field_type", "text"),
                    "dropdown_options": (field_cfg or {}).get("dropdown_options", []),
                    "hidden": bool((field_cfg or {}).get("hidden", False)),
                }
                for field_key, field_cfg in fields.items()
            ],
        })
    games.sort(key=lambda g: g["name"])
    return render_template("index.html", games=games)


# Keys that are rendered as dedicated static columns or are internal/OCR metadata.
# They are excluded from the dynamic entry-field columns.
_STATIC_ENTRY_KEYS = frozenset({
    "game", "album", "collection", "page", "slot",
    "intake_price", "current_value", "sold_price",
    "__roi_fields_used",
    # Dedicated static columns and legacy/superseded keys that must never
    # surface as dynamic ad-hoc text columns.
    "edition", "holographic", "finalized", "tcgplayer", "grading",
    "first_edition", "limited_edition",
    "empty", "catalog_only",
})
_INTERNAL_KEY_PREFIXES = ("__ocr_", "__")
_INTERNAL_KEY_SUFFIXES = ("__ocr_conf", "__ocr_variant")


def _is_catalog_only(data: dict) -> bool:
    """
    Catalog-only records (created by CSV import) are reference/lookup rows —
    not owned inventory. They exist so their fields (name, set, rarity,
    TCGplayer URL/price, etc.) can be pulled into a real inventory entry via
    "Copy from Entry", but they must never appear in the Inventory list,
    Album view, CSV export, or duplicate-resolution tools themselves.
    """
    val = (data or {}).get("catalog_only", False)
    return val is True or str(val).strip().lower() == "true"


# ====================== DENORMALIZED COLUMN SYNC ======================
# The ScanRecord.*_key / dup_hash / is_* columns are a write-time cache of values
# that otherwise live inside the extracted_data JSON. They exist so filtering,
# sorting, de-duplication, and pagination can run in indexed SQL instead of by
# loading every row into Python — the key to staying fast into the millions.
#
# extracted_data stays the single source of truth: these columns are ALWAYS
# recomputed from it by the before_insert/before_update mapper events below, so
# every existing write path (which reassigns extracted_data) keeps them correct
# without needing to be touched individually.
import hashlib as _hashlib
from sqlalchemy import event as _sa_event2


def _bool_from(data, key):
    v = (data or {}).get(key, False)
    return v is True or str(v).strip().lower() == "true"


def _derive_card_type(data):
    """Best-effort type/colour value for a card, for optional fast filtering.
    Picks the first type-ish field present, ignoring the generic 'Card Type'."""
    for k, v in (data or {}).items():
        kl = str(k).strip().lower()
        if kl in ("card type", "card_type"):
            continue
        if _re.search(r"type|energy|element|colou?r|attribute", kl, _re.I):
            sv = str(v or "").strip()
            if sv:
                return sv.lower()
    return ""


def _compute_dup_hash(data):
    """Stable hash of a FINALIZED record's duplicate identity
    (name|serial|edition|holographic). Returns None for unfinalized records so
    each stays its own group (never merged)."""
    if not _bool_from(data, "finalized"):
        return None
    parts = (
        _get_name(data),
        _get_serial(data),
        _get_edition(data),
        str((data or {}).get("holographic", "")).strip().lower(),
    )
    return _hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def _derive_scan_columns(record):
    """Populate a ScanRecord's denormalized columns from its extracted_data."""
    data = record.extracted_data or {}
    record.game_key      = (str(data.get("game", "")).strip().lower() or None)
    record.album_key     = (str(data.get("album", "")).strip().lower() or None)
    record.name_key      = (_get_name(data) or None)
    record.card_type_key = (_derive_card_type(data) or None)
    record.dup_hash      = _compute_dup_hash(data)
    record.is_finalized  = _bool_from(data, "finalized")
    record.is_catalog    = _is_catalog_only(data)
    record.is_archived   = _bool_from(data, "archived")


@_sa_event2.listens_for(ScanRecord, "before_insert")
def _scan_before_insert(_mapper, _conn, target):
    _derive_scan_columns(target)


@_sa_event2.listens_for(ScanRecord, "before_update")
def _scan_before_update(_mapper, _conn, target):
    _derive_scan_columns(target)


def _is_entry_field(key: str) -> bool:
    """Return True if key should appear as a dynamic entry column."""
    if key in _STATIC_ENTRY_KEYS:
        return False
    for prefix in _INTERNAL_KEY_PREFIXES:
        if key.startswith(prefix):
            return False
    for suffix in _INTERNAL_KEY_SUFFIXES:
        if key.endswith(suffix):
            return False
    return True


def discover_entry_fields(records) -> list:
    """
    Collect all distinct extracted_data keys from a list of ScanRecord objects
    that are not already static columns or internal/OCR metadata keys.
    Returns a sorted list of field names.
    """
    seen = set()
    for record in records:
        data = record.extracted_data or {}
        for key in data.keys():
            if _is_entry_field(key):
                seen.add(key)
    # Sort: put well-known name-like keys first, then alphabetically
    priority = ("name", "product_name", "card_name", "title")
    def _sort_key(k):
        try:
            return (0, priority.index(k))
        except ValueError:
            return (1, k)
    return sorted(seen, key=_sort_key)


_EDITION_SORT_ORDER = {"Standard Edition": 0, "First Edition": 1, "Limited Edition": 2}


def _rep_sort_key(record, sort_col, group_info):
    """Return a comparable sort key for a representative ScanRecord."""
    data = record.extracted_data or {}

    if sort_col == "date":
        return record.scan_date or datetime.min
    if sort_col == "game":
        return (data.get("game") or "").lower()
    if sort_col == "album":
        return (data.get("album") or "").lower()
    if sort_col == "page":
        try:
            return int(data.get("page") or 0)
        except (ValueError, TypeError):
            return 0
    if sort_col == "slot":
        try:
            return int(data.get("slot") or 0)
        except (ValueError, TypeError):
            return 0
    if sort_col == "template":
        return (record.template_used or "").lower()
    if sort_col == "tcg_url":
        return 0 if bool((data.get("tcgplayer") or {}).get("url")) else 1
    if sort_col == "market_price":
        try:
            return float((data.get("tcgplayer") or {}).get("prices", {}).get("market") or 0)
        except (ValueError, TypeError):
            return 0.0
    if sort_col == "finalized":
        fin = data.get("finalized", False)
        return 0 if (fin is True or str(fin).strip().lower() == "true") else 1
    if sort_col == "edition":
        ed = data.get("edition", "Standard Edition") or "Standard Edition"
        return _EDITION_SORT_ORDER.get(ed, 99)
    if sort_col == "qty":
        return group_info.get(record.id, {}).get("count", 1)
    # Dynamic entry field (sort_col is the raw extracted_data key, e.g. "name", "atk")
    return (data.get(sort_col) or "").lower()


# ====================== GAME SELECTION HELPER ======================
def _inventory_game_select():
    """Render the game selection landing page for /inventory."""
    rows = (
        ScanRecord.query
        .with_entities(ScanRecord.extracted_data)
        .all()
    )

    game_map = {}
    catalog_count = 0
    for (extracted_data,) in rows:
        if isinstance(extracted_data, dict):
            data = extracted_data
        else:
            try:
                data = json.loads(extracted_data or "{}")
            except (ValueError, TypeError):
                data = {}

        if _is_catalog_only(data):
            catalog_count += 1
            continue

        game_name = str(data.get("game", "")).strip()
        if not game_name:
            game_name = "(Unknown Game)"

        if game_name not in game_map:
            game_map[game_name] = {"name": game_name, "count": 0, "albums": set(), "all_data": []}
        game_map[game_name]["count"] += 1
        album = str(data.get("album", "")).strip()
        if album:
            game_map[game_name]["albums"].add(album)
        game_map[game_name]["all_data"].append(data)

    games = []
    for info in game_map.values():
        # Discover fields for this game using a sample
        sample_records_fake = [type("R", (), {"extracted_data": d})() for d in info["all_data"][:200]]
        fields = discover_entry_fields(sample_records_fake)
        games.append({
            "name":        info["name"],
            "count":       info["count"],
            "album_count": len(info["albums"]),
            "fields":      fields,
        })

    games.sort(key=lambda g: g["name"].lower())
    for game in games:
        game["image_url"] = find_saved_image("game_icons", game["name"])
    return render_template("inventory_game_select.html", games=games, catalog_count=catalog_count)


class _InvPagination:
    """Template-compatible pagination object shared by both inventory paths."""
    def __init__(self, items, total, cur_page, pp, start):
        self.items    = items
        self.total    = total
        self.page     = cur_page
        self.per_page = pp
        self.pages    = max(1, -(-total // pp))  # ceiling division
        self.has_prev = cur_page > 1
        self.has_next = cur_page < self.pages
        self.prev_num = cur_page - 1
        self.next_num = cur_page + 1
        self.first    = start + 1 if items else 0
        self.last     = min(start + len(items), total)

    def iter_pages(self, left_edge=1, right_edge=1, left_current=2, right_current=2):
        last = self.pages
        for num in range(1, last + 1):
            if (num <= left_edge or num > last - right_edge
                    or (self.page - left_current <= num <= self.page + right_current)):
                yield num
            else:
                yield None


def _inventory_base_conditions(f_game, f_album, f_template, view_catalog):
    """WHERE conditions (on denormalized columns) shared by the fast-path
    representative query, count, member lookup, and field sampling."""
    from sqlalchemy import func as _f
    conds = [
        _f.coalesce(ScanRecord.is_catalog, False) == bool(view_catalog),
    ]
    if not view_catalog:
        conds.append(_f.coalesce(ScanRecord.is_archived, False) == False)  # noqa: E712
    if f_template:
        conds.append(ScanRecord.template_used == f_template)
    if f_game:
        conds.append(ScanRecord.game_key == f_game.strip().lower())
    if f_album:
        conds.append(ScanRecord.album_key == f_album.strip().lower())
    return conds


def _render_inventory_fast(f_game, f_album, f_template, view_catalog,
                           page, per_page, sort_col, sort_dir):
    """
    Fast Inventory path: de-duplicate and paginate entirely in SQL using the
    dup_hash column and window functions, loading only the page's ~50 rows
    instead of the whole filtered set. Used for the default (recency) view;
    the caller falls back to the Python path for free-text search and arbitrary
    field sorts. Raises on any DB error so the caller can fall back safely.
    """
    from sqlalchemy import select, func, cast, String, and_

    conds = _inventory_base_conditions(f_game, f_album, f_template, view_catalog)

    # Group key: dup_hash for finalized rows; a per-row token for the rest so
    # unfinalized records never merge together.
    grp = func.coalesce(ScanRecord.dup_hash,
                        "solo:" + cast(ScanRecord.id, String)).label("grp")

    windowed = (
        select(
            ScanRecord.id.label("id"),
            ScanRecord.scan_date.label("scan_date"),
            func.count().over(partition_by=grp).label("qty"),
            func.row_number().over(
                partition_by=grp,
                order_by=(ScanRecord.scan_date.desc(), ScanRecord.id.desc()),
            ).label("rn"),
        )
        .where(and_(*conds))
        .subquery()
    )

    reps = (
        select(windowed.c.id, windowed.c.qty)
        .where(windowed.c.rn == 1)
        .order_by(windowed.c.scan_date.desc(), windowed.c.id.desc())
    )

    total_groups = db.session.execute(
        select(func.count()).select_from(reps.subquery())
    ).scalar() or 0

    start = (page - 1) * per_page
    page_rows = db.session.execute(reps.limit(per_page).offset(start)).all()
    rep_ids  = [r.id for r in page_rows]
    qty_by_id = {r.id: r.qty for r in page_rows}

    # Load the representative ORM objects, preserving page order.
    rep_records = []
    if rep_ids:
        by_id = {r.id: r for r in ScanRecord.query.filter(ScanRecord.id.in_(rep_ids)).all()}
        rep_records = [by_id[i] for i in rep_ids if i in by_id]

    # Build group_info only for this page's reps (count + all_ids + duplicate
    # locations), fetching members of finalized groups in one query.
    from collections import defaultdict
    members_by_hash = defaultdict(list)
    hashes = [r.dup_hash for r in rep_records if r.dup_hash]
    if hashes:
        member_rows = (ScanRecord.query
                       .filter(ScanRecord.dup_hash.in_(hashes), and_(*conds))
                       .all())
        for m in member_rows:
            members_by_hash[m.dup_hash].append(m)

    group_info = {}
    for rep in rep_records:
        if rep.dup_hash and members_by_hash.get(rep.dup_hash):
            members = sorted(members_by_hash[rep.dup_hash],
                             key=lambda r: (r.scan_date or datetime.min, r.id), reverse=True)
            all_ids = [m.id for m in members]
            locations = [
                "{}/{}".format((m.extracted_data or {}).get("page", "?"),
                               (m.extracted_data or {}).get("slot", "?"))
                for m in members if m.id != rep.id
            ]
            group_info[rep.id] = {"count": qty_by_id.get(rep.id, len(members)),
                                  "all_ids": all_ids, "locations": locations}
        else:
            group_info[rep.id] = {"count": 1, "all_ids": [rep.id], "locations": []}

    pagination = _InvPagination(rep_records, total_groups, page, per_page, start)

    # Entry-field discovery from a bounded sample of the filtered set.
    sample = ScanRecord.query.filter(and_(*conds)).limit(500).all()
    entry_fields = discover_entry_fields(sample)

    template_fields_config = _template_fields_config()

    return render_template(
        "inventory.html",
        records=pagination.items,
        pagination=pagination,
        search="",
        f_game=f_game,
        f_album=f_album,
        f_template=f_template,
        per_page=per_page,
        entry_fields=entry_fields,
        group_info=group_info,
        sort_col=sort_col,
        sort_dir=sort_dir,
        template_fields_config=template_fields_config,
        catalog_view=view_catalog,
    )


def _template_fields_config():
    """Aggregate field-type config across all templates (shared by both paths)."""
    cfg = {}
    for tpl_name in get_template_names():
        try:
            tpl = load_template(tpl_name)
            for fk, fv in (tpl.get("fields") or {}).items():
                if fk not in cfg and isinstance(fv, dict):
                    cfg[fk] = {
                        "field_type":       fv.get("field_type", "text"),
                        "dropdown_options": fv.get("dropdown_options", []),
                        "hidden":           bool(fv.get("hidden", False)),
                    }
        except Exception:
            pass
    return cfg


# ============================================================================ #
# Builder — assemble packs / sets / sets-of-packs from a game's inventory
# ============================================================================ #
import builder as _builder


def _bf_field(data, keys):
    """Read the first non-empty value among candidate keys (case-insensitive)."""
    low = {str(k).strip().lower(): v for k, v in (data or {}).items()}
    for k in keys:
        v = low.get(k)
        if v not in (None, ""):
            return str(v).strip()
    return ""


def _builder_card_attrs(rec):
    data = rec.extracted_data or {}
    name = _get_name(data) or "Unnamed"
    cset = _bf_field(data, ["set", "set_name", "setname", "expansion", "series"])
    rarity = _bf_field(data, ["rarity"])
    holo = _bf_field(data, ["holographic", "holo", "foil", "finish"])
    ctype = _derive_card_type(data) or _bf_field(data, ["type", "card_type", "supertype"])
    number = _bf_field(data, ["number", "collector_number", "card_number", "set_number", "serial"])
    identity = (name.lower(), cset.lower(), number.lower())
    return {"id": rec.id, "identity": identity, "name": name, "set": cset,
            "rarity": rarity, "holo": holo, "type": ctype, "number": number}


def _builder_load(game):
    """Owned, active records for a game -> (attr pool, {id: record})."""
    recs = (_active_inventory_query()
            .filter(ScanRecord.game_key == (game or "").strip().lower())
            .all())
    pool = [_builder_card_attrs(r) for r in recs]
    return pool, {r.id: r for r in recs}


def _builder_options(pool):
    def distinct(field):
        return sorted({c[field] for c in pool if c.get(field)}, key=str.lower)
    return {
        "sets": distinct("set"),
        "rarities": distinct("rarity"),
        "types": distinct("type"),
        "holos": distinct("holo") or list(_HOLOGRAPHIC_OPTIONS),
        "total": len(pool),
    }


def _builder_result_card(attr, by_id):
    rec = by_id.get(attr["id"])
    return {
        "id": attr["id"], "name": attr["name"], "set": attr["set"],
        "rarity": attr["rarity"], "holo": attr["holo"], "type": attr["type"],
        "number": attr["number"],
        "image_url": build_uploaded_file_url(rec.image_path) if rec else "",
        "detail_url": url_for("inventory_detail", record_id=attr["id"]) if rec else "",
    }


def _card_identity(c):
    return ((c.get("name") or "").strip().lower(),
            (c.get("set") or "").strip().lower(),
            (c.get("number") or "").strip().lower())


def _dup_stats_flat(cards):
    """Duplicate identities within a single flat list of picked cards."""
    from collections import defaultdict
    groups = defaultdict(list)
    for c in cards:
        groups[_card_identity(c)].append(c)
    items = []
    for _ident, group in groups.items():
        if len(group) > 1:
            rep = group[0]
            items.append({"name": rep["name"], "set": rep["set"],
                          "number": rep["number"], "count": len(group)})
    items.sort(key=lambda x: (-x["count"], x["name"].lower()))
    return {
        "total_duplicate_cards": len(items),
        "total_extra_copies": sum(i["count"] - 1 for i in items),
        "items": items,
    }


def _dup_stats_packs(packs):
    """Cards that appear across more than one pack (within a pack they're already
    distinct), with which packs each shows up in."""
    from collections import defaultdict
    where = defaultdict(list)
    rep = {}
    for pi, pk in enumerate(packs):
        for c in pk["cards"]:
            ident = _card_identity(c)
            where[ident].append(pi + 1)
            rep.setdefault(ident, c)
    items = []
    for ident, pack_nums in where.items():
        if len(pack_nums) > 1:
            r = rep[ident]
            items.append({"name": r["name"], "set": r["set"], "number": r["number"],
                          "count": len(pack_nums), "packs": pack_nums})
    items.sort(key=lambda x: (-x["count"], x["name"].lower()))
    return {
        "total_duplicate_cards": len(items),
        "total_extra_copies": sum(i["count"] - 1 for i in items),
        "items": items,
    }


def _builder_games():
    """Games that have owned inventory, with counts (for the picker)."""
    from sqlalchemy import func
    rows = (db.session.query(ScanRecord.game_key, func.count())
            .filter(ScanRecord.game_key.isnot(None),
                    db.func.coalesce(ScanRecord.is_catalog, False) == False,   # noqa: E712
                    db.func.coalesce(ScanRecord.is_archived, False) == False)  # noqa: E712
            .group_by(ScanRecord.game_key).all())
    return sorted(({"name": g, "count": n} for g, n in rows if g), key=lambda x: x["name"])


@app.route("/inventory/builder")
def builder_page():
    game = (request.args.get("game") or "").strip()
    if not game:
        return render_template("builder.html", game="", options=None, games=_builder_games())
    pool, _ = _builder_load(game)
    return render_template("builder.html", game=game, options=_builder_options(pool), games=None)


@app.route("/inventory/builder/build", methods=["POST"])
def builder_build():
    body = request.get_json(silent=True) or {}
    game = (body.get("game") or "").strip()
    mode = (body.get("mode") or "").strip()
    if not game:
        return jsonify({"status": "error", "message": "No game specified."}), 400

    pool, by_id = _builder_load(game)
    if not pool:
        return jsonify({"status": "error", "message": "This game has no owned inventory to build from."}), 400

    def _counts(raw):
        out = {}
        for row in raw or []:
            key = str(row.get("value", "")).strip()
            try:
                n = int(row.get("count", 0))
            except (TypeError, ValueError):
                n = 0
            if key and n > 0:
                out[key] = out.get(key, 0) + n
        return out

    try:
        if mode == "pack":
            spec = {
                "size": int(body.get("size") or 0),
                "rarities": _counts(body.get("rarities")),
                "holos": _counts(body.get("holos")),
                "sets": body.get("sets") or [],
            }
            res = _builder.build_pack(pool, spec)
            cards = [_builder_result_card(c, by_id) for c in res["selected"]]
            out = {
                "cards": cards,
                "filled": res["filled"], "size": res["size"],
                "shortfalls": res["shortfalls"], "complete": res["complete"],
                "over_specified": res["over_specified"],
                "duplicates": _dup_stats_flat(cards),
            }
        elif mode == "set":
            spec = {
                "size": int(body.get("size") or 0),
                "allow_duplicates": bool(body.get("allow_duplicates")),
                "types": body.get("types") or [],
                "rarities": body.get("rarities") or [],
                "sets": body.get("sets") or [],
            }
            res = _builder.build_set(pool, spec)
            cards = [_builder_result_card(c, by_id) for c in res["selected"]]
            out = {
                "cards": cards,
                "filled": res["filled"], "size": res["size"],
                "shortfall": res["shortfall"], "complete": res["complete"],
                "duplicates": _dup_stats_flat(cards),
            }
        elif mode == "set_of_packs":
            pack = body.get("pack") or {}
            spec = {
                "size": int(pack.get("size") or 0),
                "rarities": _counts(pack.get("rarities")),
                "holos": _counts(pack.get("holos")),
                "sets": pack.get("sets") or [],
            }
            count = max(1, int(body.get("count") or 1))
            res = _builder.build_set_of_packs(pool, spec, count)
            packs_out = [
                {
                    "cards": [_builder_result_card(c, by_id) for c in pk["selected"]],
                    "filled": pk["filled"], "size": pk["size"],
                    "shortfalls": pk["shortfalls"], "complete": pk["complete"],
                } for pk in res["packs"]
            ]
            out = {
                "packs": packs_out,
                "count": res["count"],
                "duplicate_identities": res["duplicate_identities"],
                "duplicate_slots": res["duplicate_slots"],
                "all_complete": res["all_complete"],
                "duplicates": _dup_stats_packs(packs_out),
            }
        else:
            return jsonify({"status": "error", "message": "Unknown build mode."}), 400
    except (TypeError, ValueError) as exc:
        return jsonify({"status": "error", "message": f"Invalid criteria: {exc}"}), 400

    return jsonify({"status": "success", "mode": mode, "result": out})


# --- Builder: export picks to CSV, and "pull" (export + delete) --------------
def _builder_flatten(groups):
    """[(label, record_id), ...] from the client's group structure, in order."""
    flat = []
    for g in groups or []:
        label = str(g.get("label", "")).strip()
        for i in g.get("ids", []):
            try:
                flat.append((label, int(i)))
            except (TypeError, ValueError):
                pass
    return flat


def _builder_csv(groups):
    """Full per-card detail CSV (all inventory_detail fields), grouped by
    pack/set. Returns (csv_text, ids)."""
    import csv as _csv
    import io as _io

    flat = _builder_flatten(groups)
    ids = [i for _label, i in flat]
    recs = {r.id: r for r in ScanRecord.query.filter(ScanRecord.id.in_(ids)).all()}

    # Union of visible (non-internal) extracted_data fields, common ones first.
    common = ["name", "game", "set", "number", "rarity", "holographic", "type",
              "edition", "condition", "intake_price", "current_value", "sold_price",
              "price", "collection", "album", "page", "slot"]
    present = set()
    for _label, i in flat:
        r = recs.get(i)
        if r:
            for k in (r.extracted_data or {}).keys():
                if not str(k).startswith("__"):
                    present.add(k)
    ordered = [k for k in common if k in present] + sorted(k for k in present if k not in common)
    header = ["group", "record_id"] + ordered + ["template_used", "scan_date",
                                                 "matched_product", "matched_sku", "image_path"]

    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(header)
    for label, i in flat:
        r = recs.get(i)
        if not r:
            w.writerow([label, i] + [""] * (len(header) - 2))
            continue
        data = r.extracted_data or {}
        mp = r.matched_product
        row = [label, r.id] + [data.get(k, "") for k in ordered] + [
            r.template_used or "",
            r.scan_date.isoformat() if r.scan_date else "",
            (mp.product_name if mp else ""),
            (mp.sku if mp else ""),
            r.image_path or "",
        ]
        w.writerow(row)
    return buf.getvalue(), ids


def _delete_record_files(rec):
    """Best-effort removal of a record's image files + descriptor caches."""
    for rel in (rec.image_path, rec.image_path_back, rec.display_image_path):
        if not rel or rel == "__blank__" or str(rel).startswith(("http://", "https://")):
            continue
        p = os.path.join(app.config["UPLOAD_FOLDER"], normalize_to_upload_relative(rel))
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    for suffix in (".npy", ".gvec.npy"):
        c = os.path.join(app.config["ORB_CACHE_FOLDER"], f"{rec.id}{suffix}")
        try:
            if os.path.exists(c):
                os.remove(c)
        except OSError:
            pass


@app.route("/inventory/builder/export", methods=["POST"])
def builder_export():
    body = request.get_json(silent=True) or {}
    groups = body.get("groups") or []
    game = (body.get("game") or "build").strip() or "build"
    mode = (body.get("mode") or "build").strip()
    if not groups:
        return jsonify({"status": "error", "message": "Nothing to export."}), 400
    csv_text, _ids = _builder_csv(groups)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_game = _re.sub(r"[^A-Za-z0-9_-]+", "_", game)
    fname = f"{safe_game}_{mode}_{stamp}.csv"
    return Response(csv_text, mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.route("/inventory/builder/pull", methods=["POST"])
def builder_pull():
    """Delete the picked records (they've been physically pulled). Idempotent-ish:
    only records that still exist are removed."""
    body = request.get_json(silent=True) or {}
    groups = body.get("groups") or []
    ids = [i for _l, i in _builder_flatten(groups)]
    if not ids:
        return jsonify({"status": "error", "message": "No cards to pull."}), 400

    recs = ScanRecord.query.filter(ScanRecord.id.in_(ids)).all()
    if not recs:
        return jsonify({"status": "success", "deleted": 0, "message": "Nothing to remove."})

    del_ids = [r.id for r in recs]
    # Detach any sale-event references so FK constraints don't block deletion.
    SaleEvent.query.filter(SaleEvent.record_id.in_(del_ids)).update(
        {SaleEvent.record_id: None}, synchronize_session=False)
    for r in recs:
        _delete_record_files(r)
        db.session.delete(r)   # Listings cascade-delete with the record
    db.session.commit()
    _inventory_count_bump(-len(recs))

    return jsonify({"status": "success", "deleted": len(recs),
                    "message": f"Pulled {len(recs)} card(s) — removed from inventory."})


@app.route("/inventory")
def inventory():
    page       = request.args.get("page", 1, type=int)
    per_page   = request.args.get("per_page", 50, type=int)
    search     = request.args.get("search", "").strip()
    f_game     = request.args.get("game", "").strip()
    f_album    = request.args.get("album", "").strip()
    f_template = request.args.get("template", "").strip()
    sort_col   = request.args.get("sort", "").strip()
    sort_dir   = request.args.get("sort_dir", "asc").strip()
    # "Imported Catalog" view: CSV imports create hidden catalog_only
    # records (reference rows, not owned inventory) that never show up in
    # the normal Inventory list. ?catalog=1 flips this page to show ONLY
    # those hidden rows instead, so they can be found, edited, and deleted.
    view_catalog = request.args.get("catalog", "").strip() in ("1", "true", "yes")

    # If no filter is active, show the game selection landing page.
    if not f_game and not f_album and not f_template and not search and not view_catalog:
        return _inventory_game_select()

    if sort_dir not in ("asc", "desc"):
        sort_dir = "asc"

    per_page = min(per_page, 200)

    # ---- Fast path: default (recency) view with no free-text search ----------
    # De-dup + paginate in SQL so only the page's rows load. Any arbitrary field
    # sort or search falls through to the Python path below (which handles the
    # full range of sort keys). The fast path is guarded: if the denormalized
    # columns aren't present yet (e.g. mid-upgrade), we fall back transparently.
    effective_sort_early = sort_col[len("entry_"):] if sort_col.startswith("entry_") else sort_col
    if not search and not effective_sort_early:
        try:
            return _render_inventory_fast(
                f_game, f_album, f_template, view_catalog, page, per_page, sort_col, sort_dir)
        except Exception:
            db.session.rollback()  # fall back to the proven Python grouping path

    # Base query. game/album live inside the extracted_data JSON; we filter them
    # in SQL with json_extract so the matching expression indexes are used and we
    # only load the relevant rows (huge win once one game has tens of thousands
    # of records). template_used and the JSON text search are handled in SQL as
    # before.
    query = ScanRecord.query.order_by(ScanRecord.scan_date.desc())

    if f_template:
        query = query.filter(ScanRecord.template_used == f_template)
    if search:
        query = query.filter(ScanRecord.extracted_data.cast(db.Text).ilike(f"%{search}%"))
    if f_game:
        query = query.filter(db.func.json_extract(ScanRecord.extracted_data, "$.game") == f_game)
    if f_album:
        query = query.filter(db.func.json_extract(ScanRecord.extracted_data, "$.album") == f_album)

    all_records_raw = query.all()

    # Catalog-only records (from CSV import) are lookup/reference rows, not
    # owned inventory. Normally they're excluded from the Inventory list;
    # in catalog view, the filter is flipped so ONLY they show.
    if view_catalog:
        all_records_raw = [
            r for r in all_records_raw
            if _is_catalog_only(r.extracted_data or {})
        ]
    else:
        all_records_raw = [
            r for r in all_records_raw
            if not _is_catalog_only(r.extracted_data or {})
        ]

    # game/album already filtered in SQL above.
    all_records = all_records_raw

    # Archived rows (cold storage) are hidden from the normal Inventory list.
    if not view_catalog:
        all_records = [r for r in all_records if not _bool_from(r.extracted_data or {}, "archived")]

    # Build groups across the full filtered set so duplicates on other pages
    # are still counted in the quantity badge.
    group_info, rep_records = build_group_info(all_records)

    # Apply server-side sort across ALL representative records before paginating.
    # Columns named "entry_<field>" (matching the JS data-col convention) have their
    # prefix stripped so we look up the raw extracted_data key.
    effective_sort = sort_col
    if effective_sort.startswith("entry_"):
        effective_sort = effective_sort[len("entry_"):]

    if effective_sort:
        try:
            rep_records.sort(
                key=lambda r: _rep_sort_key(r, effective_sort, group_info),
                reverse=(sort_dir == "desc"),
            )
        except Exception:
            pass  # fall back to default ordering if sort fails

    # Paginate over the deduplicated (and now sorted) representative list
    total_reps = len(rep_records)
    start      = (page - 1) * per_page
    end        = start + per_page
    page_reps  = rep_records[start:end]

    # Lightweight pagination object compatible with the template
    class _Pagination:
        def __init__(self, items, total, cur_page, pp):
            self.items    = items
            self.total    = total
            self.page     = cur_page
            self.per_page = pp
            self.pages    = max(1, -(-total // pp))  # ceiling division
            self.has_prev = cur_page > 1
            self.has_next = cur_page < self.pages
            self.prev_num = cur_page - 1
            self.next_num = cur_page + 1
            self.first    = start + 1 if items else 0
            self.last     = min(start + len(items), total)

        def iter_pages(self, left_edge=1, right_edge=1, left_current=2, right_current=2):
            last = self.pages
            for num in range(1, last + 1):
                if (
                    num <= left_edge
                    or num > last - right_edge
                    or (self.page - left_current <= num <= self.page + right_current)
                ):
                    yield num
                else:
                    yield None

    pagination = _Pagination(page_reps, total_reps, page, per_page)

    # Discover dynamic entry fields from a sample of all filtered records
    all_sample = all_records[:500]
    entry_fields = discover_entry_fields(all_sample)

    # Aggregate field-type config from all known templates so the inventory
    # list can render boolean toggles, respect dropdown options, and know
    # which fields are marked "hidden" (kept off the table/detail page by
    # default, revealed only via the "Show Hidden Fields" switch).
    template_fields_config: dict = {}
    for tpl_name in get_template_names():
        try:
            tpl = load_template(tpl_name)
            for fk, fv in (tpl.get("fields") or {}).items():
                if fk not in template_fields_config and isinstance(fv, dict):
                    template_fields_config[fk] = {
                        "field_type":       fv.get("field_type", "text"),
                        "dropdown_options": fv.get("dropdown_options", []),
                        "hidden":           bool(fv.get("hidden", False)),
                    }
        except Exception:
            pass

    return render_template(
        "inventory.html",
        records=pagination.items,
        pagination=pagination,
        search=search,
        f_game=f_game,
        f_album=f_album,
        f_template=f_template,
        per_page=per_page,
        entry_fields=entry_fields,
        group_info=group_info,
        sort_col=sort_col,
        sort_dir=sort_dir,
        template_fields_config=template_fields_config,
        catalog_view=view_catalog,
    )


@app.route("/inventory/export_csv")
def inventory_export_csv():
    """
    Export the currently filtered inventory to a CSV file.
    Accepts the same filter params as /inventory (search, game, album, template)
    plus a `columns` param (comma-separated list of column keys to include).
    Column keys follow the same convention as the HTML: static keys like
    'date', 'game', 'album', 'page', 'slot', 'template', and dynamic keys
    prefixed with 'entry_' (e.g. 'entry_name', 'entry_atk').
    """
    import csv
    import io
    from flask import Response

    search     = request.args.get("search", "").strip()
    f_game     = request.args.get("game", "").strip()
    f_album    = request.args.get("album", "").strip()
    f_template = request.args.get("template", "").strip()
    columns_param = request.args.get("columns", "").strip()
    view_catalog = request.args.get("catalog", "").strip() in ("1", "true", "yes")

    # Build query — same logic as /inventory, but no pagination
    # Python-side filtering for game/album for SQLite compatibility (.astext is PostgreSQL-only)
    query = ScanRecord.query.order_by(ScanRecord.scan_date.desc())
    if f_template:
        query = query.filter(ScanRecord.template_used == f_template)
    if search:
        query = query.filter(ScanRecord.extracted_data.cast(db.Text).ilike(f"%{search}%"))

    records = query.all()
    records = [r for r in records if _is_catalog_only(r.extracted_data or {}) == view_catalog]
    if f_game:
        records = [
            r for r in records
            if str((r.extracted_data or {}).get("game", "")).strip() == f_game
        ]
    if f_album:
        records = [
            r for r in records
            if str((r.extracted_data or {}).get("album", "")).strip() == f_album
        ]

    # Determine which columns to export
    # Static column key → (header label, value extractor)
    STATIC_COL_MAP = {
        "date":     ("Date",      lambda r: r.scan_date.strftime("%Y-%m-%d %H:%M") if r.scan_date else ""),
        "game":     ("Game",      lambda r: str((r.extracted_data or {}).get("game", ""))),
        "album":    ("Album",     lambda r: str((r.extracted_data or {}).get("album", ""))),
        "page":     ("Page",      lambda r: str((r.extracted_data or {}).get("page", ""))),
        "slot":     ("Slot",      lambda r: str((r.extracted_data or {}).get("slot", ""))),
        "template": ("Template",  lambda r: str(r.template_used or "")),
        "tcg_url":      ("Price URL",    lambda r: str(((r.extracted_data or {}).get("tcgplayer") or {}).get("url", ""))),
        "market_price": ("Market Price", lambda r: str(((r.extracted_data or {}).get("tcgplayer") or {}).get("prices", {}).get("market", "") or "")),
    }

    # Parse the requested columns; fall back to all non-image/action static cols + all entry fields
    if columns_param:
        requested = [c.strip() for c in columns_param.split(",") if c.strip()]
    else:
        # Default: all static cols (except image/select/actions) + all discovered entry fields
        entry_fields = discover_entry_fields(records)
        requested = list(STATIC_COL_MAP.keys()) + [f"entry_{f}" for f in entry_fields]

    # Build ordered list of (header, extractor) for columns that are valid
    columns = []
    for col_key in requested:
        if col_key in STATIC_COL_MAP:
            label, extractor = STATIC_COL_MAP[col_key]
            columns.append((label, extractor))
        elif col_key.startswith("entry_"):
            field_key = col_key[len("entry_"):]
            label = field_key.replace("_", " ").title()
            extractor = (lambda fk: lambda r: str((r.extracted_data or {}).get(fk, "")))(field_key)
            columns.append((label, extractor))
        # 'image', 'select', 'actions' are silently skipped — not meaningful in CSV

    if not columns:
        return jsonify({"status": "error", "message": "No exportable columns selected"}), 400

    # Build CSV in memory
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\r\n")

    # Header row
    writer.writerow([label for label, _ in columns])

    # Data rows
    for record in records:
        writer.writerow([extractor(record) for _, extractor in columns])

    csv_bytes = output.getvalue().encode("utf-8-sig")  # utf-8-sig adds BOM for Excel

    filename = "inventory_export.csv"
    if f_game:
        filename = f"inventory_{f_game}_export.csv"
    elif f_album:
        filename = f"inventory_{f_album}_export.csv"

    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.route("/inventory/filter_options")
def inventory_filter_options():
    """
    Returns distinct Game, Album, and Template values for the inventory
    filter dropdowns. Uses with_entities for efficiency at large scale.
    Works whether extracted_data is db.JSON or db.Text.
    Pass catalog=1 to scope the values to hidden catalog-only rows (used by
    the Imported Catalog view) instead of normal owned-inventory rows.
    """
    view_catalog = request.args.get("catalog", "").strip() in ("1", "true", "yes")

    rows = (
        ScanRecord.query
        .with_entities(ScanRecord.extracted_data, ScanRecord.template_used)
        .all()
    )

    games = set()
    albums = set()
    templates = set()

    for extracted_data, template_used in rows:
        if isinstance(extracted_data, dict):
            data = extracted_data
        else:
            try:
                data = json.loads(extracted_data or "{}")
            except (ValueError, TypeError):
                data = {}

        if _is_catalog_only(data) != view_catalog:
            continue

        if data.get("game"):
            games.add(str(data["game"]).strip())
        if data.get("album"):
            albums.add(str(data["album"]).strip())
        if template_used:
            templates.add(str(template_used).strip())

    return jsonify({
        "games":     sorted(g for g in games     if g),
        "albums":    sorted(a for a in albums    if a),
        "templates": sorted(t for t in templates if t),
    })


@app.route("/inventory/all_ids")
def inventory_all_ids():
    """
    Return all ScanRecord IDs that match the current filter params
    (search, game, album, template).  Used by the "Select All in Filter"
    button to collect IDs across every page before a bulk operation.
    Respects the same catalog=1 flag as /inventory so "Select All" on the
    Imported Catalog view only ever selects catalog rows, and — just as
    important — "Select All" on the normal Inventory view can never quietly
    pull in hidden catalog rows that were never shown on the page.
    """
    search       = request.args.get("search",   "").strip()
    f_game       = request.args.get("game",     "").strip()
    f_album      = request.args.get("album",    "").strip()
    f_template   = request.args.get("template", "").strip()
    view_catalog = request.args.get("catalog", "").strip() in ("1", "true", "yes")

    query = ScanRecord.query.with_entities(ScanRecord.id, ScanRecord.extracted_data)

    if f_template:
        query = query.filter(ScanRecord.template_used == f_template)
    if search:
        query = query.filter(ScanRecord.extracted_data.cast(db.Text).ilike(f"%{search}%"))

    rows = query.all()

    # Python-side filtering for game / album (SQLite-safe)
    ids = []
    for row_id, extracted_data in rows:
        if isinstance(extracted_data, dict):
            data = extracted_data
        else:
            try:
                data = json.loads(extracted_data or "{}")
            except (ValueError, TypeError):
                data = {}
        if _is_catalog_only(data) != view_catalog:
            continue
        if f_game and str(data.get("game", "")).strip() != f_game:
            continue
        if f_album and str(data.get("album", "")).strip() != f_album:
            continue
        ids.append(row_id)

    return jsonify({"ids": ids, "count": len(ids)})


@app.route("/game_fields")
def game_fields():
    """
    Return the distinct entry-field keys used by existing records for a given game,
    plus albums that exist for that game.
    Used by the scan page to pre-populate a new entry with the correct field set.
    """
    game = request.args.get("game", "").strip()
    if not game:
        return jsonify({"fields": [], "albums": []})

    # Filter in Python — .astext requires PostgreSQL JSONB and is not available here
    all_records = ScanRecord.query.all()
    records = [
        r for r in all_records
        if str((r.extracted_data or {}).get("game", "")).strip() == game
    ]

    fields = discover_entry_fields(records)
    albums = sorted({
        str((r.extracted_data or {}).get("album", "")).strip()
        for r in records
        if (r.extracted_data or {}).get("album")
    })
    return jsonify({"fields": fields, "albums": albums})


@app.route("/inventory/<int:record_id>")
def inventory_detail(record_id):
    record = ScanRecord.query.get_or_404(record_id)

    # Determine prev/next IDs scoped to the same game as the current record.
    # Navigation wraps around: after the last entry of a game, loop back to the
    # first; before the first, loop to the last.  This prevents jumping into a
    # different game's records.
    #
    # Ordering follows the inventory list: scan_date DESC, id DESC.

    current_game = (record.extracted_data or {}).get("game", "")

    # Fetch all records that share the same game value, ordered DESC.
    # We work in Python so we avoid JSON-in-SQL portability issues with SQLite.
    game_records = (
        ScanRecord.query
        .order_by(ScanRecord.scan_date.desc(), ScanRecord.id.desc())
        .all()
    )
    game_records = [
        r for r in game_records
        if (r.extracted_data or {}).get("game", "") == current_game
    ]

    # Find the position of the current record in this ordered list
    current_index = next(
        (i for i, r in enumerate(game_records) if r.id == record.id), None
    )

    if current_index is not None and len(game_records) > 1:
        # Previous = one step earlier in DESC order (index - 1), wrap to last
        prev_record = game_records[(current_index - 1) % len(game_records)]
        # Next = one step later in DESC order (index + 1), wrap to first
        next_record = game_records[(current_index + 1) % len(game_records)]
    else:
        prev_record = None
        next_record = None

    # ── Find all duplicate records that belong to the same stack ──
    # A stack is defined by matching: name, serial, edition, holographic, finalized==True.
    # Only groups where finalized is True are stacked (same logic as build_group_info).
    data = record.extracted_data or {}
    finalized = data.get("finalized", False)
    is_final = finalized is True or str(finalized).strip().lower() == "true"

    stack_locations = []  # list of dicts: {id, album, page, slot, record_id}
    if is_final:
        name   = _get_name(data)
        serial = _get_serial(data)
        edition = _get_edition(data)
        holo   = str(data.get("holographic", "")).strip().lower()

        # Fetch all finalized records and filter to the same group key
        all_records = ScanRecord.query.order_by(ScanRecord.scan_date.asc(), ScanRecord.id.asc()).all()
        for r in all_records:
            if r.id == record.id:
                continue
            rdata = r.extracted_data or {}
            rfin = rdata.get("finalized", False)
            if not (rfin is True or str(rfin).strip().lower() == "true"):
                continue
            if (
                _get_name(rdata)   == name
                and _get_serial(rdata) == serial
                and _get_edition(rdata) == edition
                and str(rdata.get("holographic", "")).strip().lower() == holo
            ):
                stack_locations.append({
                    "record_id": r.id,
                    "album": rdata.get("album", ""),
                    "page":  rdata.get("page",  ""),
                    "slot":  rdata.get("slot",  ""),
                })

    # Load the template that was used for this record so we can render
    # fields with the correct input type (text / dropdown / boolean).
    try:
        tpl = load_template(record.template_used or "product_label")
        template_fields_config = tpl.get("fields", {})
    except Exception:
        template_fields_config = {}

    return render_template(
        "inventory_detail.html",
        record=record,
        prev_id=prev_record.id if prev_record else None,
        next_id=next_record.id if next_record else None,
        stack_locations=stack_locations,
        template_fields_config=template_fields_config,
    )


@app.route("/albums")
def albums():
    album_list = build_album_index()
    if album_list:
        return redirect(url_for("album_detail", album_name=album_list[0]["name"]))
    return render_template("albums.html", albums=[])


@app.route("/albums/list")
def albums_list():
    album_list = build_album_index()
    for album in album_list:
        album["image_url"] = find_saved_image("albums", album["name"])
    return render_template("albums.html", albums=album_list)


@app.route("/albums/upload_image", methods=["POST"])
def album_upload_image():
    album_name = request.form.get("album_name", "").strip()
    file = request.files.get("image")

    if not album_name or not file or not file.filename:
        return jsonify({"status": "error", "message": "Album name and image file are required"}), 400

    album_img_folder = os.path.join(app.config["UPLOAD_FOLDER"], "albums")
    os.makedirs(album_img_folder, exist_ok=True)

    # Preserve original extension; fall back to .jpg
    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".jpg"

    safe_album = secure_filename(album_name)
    filename = f"{safe_album}{ext}"
    save_path = os.path.join(album_img_folder, filename)
    file.save(save_path)

    relative_path = f"albums/{filename}"
    image_url = url_for("uploaded_file", filename=relative_path)
    return jsonify({"status": "success", "url": image_url})


@app.route("/inventory/upload_game_image", methods=["POST"])
def inventory_upload_game_image():
    game_name = request.form.get("game_name", "").strip()
    file = request.files.get("image")

    if not game_name or not file or not file.filename:
        return jsonify({"status": "error", "message": "Game name and image file are required"}), 400

    game_img_folder = os.path.join(app.config["UPLOAD_FOLDER"], "game_icons")
    os.makedirs(game_img_folder, exist_ok=True)

    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".jpg"

    safe_game = secure_filename(game_name)
    filename = f"{safe_game}{ext}"
    save_path = os.path.join(game_img_folder, filename)
    file.save(save_path)

    relative_path = f"game_icons/{filename}"
    image_url = url_for("uploaded_file", filename=relative_path)
    return jsonify({"status": "success", "url": image_url})


@app.route("/albums/<path:album_name>")
def album_detail(album_name):
    album_list = build_album_index()
    selected = next(
        (a for a in album_list if a["name"].lower() == album_name.strip().lower()),
        None,
    )

    if not selected:
        return redirect(url_for("albums_list"))

    def record_sort_key(record):
        page = get_record_value(record, "page")
        slot = get_record_value(record, "slot")
        try:
            page_num = int(page)
        except ValueError:
            page_num = 10 ** 9
        try:
            slot_num = int(slot)
        except ValueError:
            slot_num = 10 ** 9
        return (page_num, slot_num, record.scan_date or datetime.min)

    records = sorted(selected["records"], key=record_sort_key)
    page_groups = [records[i:i + 9] for i in range(0, len(records), 9)] or [[]]
    current_page = request.args.get("page", 1, type=int)
    current_page = max(1, min(current_page, len(page_groups)))
    current_records = page_groups[current_page - 1]

    grid = {}
    for record in current_records:
        slot = get_record_value(record, "slot")
        try:
            slot_num = int(slot)
        except ValueError:
            continue
        if 1 <= slot_num <= 9 and slot_num not in grid:
            grid[slot_num] = {
                "record": record,
                "name": (
                    get_record_value(record, "product_name")
                    or get_record_value(record, "name")
                    or f"Slot {slot_num}"
                ),
            }

    return render_template(
        "album_detail.html",
        album_name=selected["name"],
        currentpage=current_page,
        maxpage=len(page_groups),
        grid=grid,
    )


@app.route("/import")
def import_page():
    return render_template("import.html", templates=get_template_names())


# ====================== SINGLE CARD IMPORT ======================
# Above this uploaded-PDF size, don't keep the document buffered in RAM: spill
# it to a temp directory on disk and rasterize a single page at a time (see
# _iter_pdf_bgr_pages). Small PDFs are still handled entirely in memory.
PDF_SPILL_THRESHOLD = 500 * 1024 * 1024  # 500 MB


# PDF rasterization resolution for card imports. PyMuPDF zoom multiplies the
# PDF's native 72 DPI. Scanned PDFs embed the scan at its own resolution, so we
# render each page at that native pixel density instead of a fixed DPI — this
# keeps a scanned card at full scanner resolution instead of down-sampling it.
# PDF_RASTER_ZOOM is the FLOOR (used for vector/low-DPI pages, ≈288 DPI) and
# PDF_RASTER_ZOOM_MAX is the CEILING that protects memory on small machines
# (≈576 DPI). Both are tunable via env.
PDF_RASTER_ZOOM  = float(os.environ.get("PDF_RASTER_ZOOM", "4.0"))    # floor ≈288 DPI
PDF_CAPPED_DPI   = float(os.environ.get("PDF_CAPPED_DPI", "600"))     # cap when unlimited is OFF
PDF_SANITY_DPI   = float(os.environ.get("PDF_SANITY_DPI", "4800"))    # absolute guard even when ON


def _pdf_native_zoom(page):
    """Zoom that renders `page` at its embedded image's native pixel density (so
    a scan is captured at full scanner resolution). Policy:
      • unlimited native import OFF -> capped at PDF_CAPPED_DPI (600 DPI);
      • unlimited native import ON  -> true native (only a high sanity cap).
    The toggle lives in Settings → General and requires ≥8 GB swap to enable.
    Pages with no raster image fall back to the floor."""
    try:
        rect = page.rect
        page_w_pt = float(getattr(rect, "width", 0) or 612)   # noqa: F841 (kept for clarity)
        page_h_pt = float(getattr(rect, "height", 0) or 792)  # noqa: F841

        best_dpi = 0.0
        for info in page.get_image_info():
            iw = info.get("width") or 0
            ih = info.get("height") or 0
            bbox = info.get("bbox")
            if iw and ih and bbox:
                bw_in = max(1e-6, (bbox[2] - bbox[0]) / 72.0)
                bh_in = max(1e-6, (bbox[3] - bbox[1]) / 72.0)
                best_dpi = max(best_dpi, iw / bw_in, ih / bh_in)

        native = max(PDF_RASTER_ZOOM, (best_dpi / 72.0) if best_dpi > 0 else PDF_RASTER_ZOOM)
        ceiling = (PDF_SANITY_DPI if _native_import_unlimited() else PDF_CAPPED_DPI) / 72.0
        return min(native, ceiling)
    except Exception:
        return PDF_RASTER_ZOOM


def _pdf_render_matrix(page, zoom=None):
    """fitz.Matrix for a page. zoom=None -> render at native resolution
    (clamped); a explicit number forces that fixed zoom."""
    z = float(zoom) if zoom else _pdf_native_zoom(page)
    return fitz.Matrix(z, z)


def _pdf_page_to_bgr(page, matrix):
    """Rasterize one already-loaded PyMuPDF page into a standalone BGR array.

    cv2.cvtColor allocates a fresh array, so the result does not reference the
    pixmap's buffer — once the caller drops `pix`/`page`, that memory is freed."""
    pix = page.get_pixmap(matrix=matrix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def _iter_pdf_bgr_pages(pdf_bytes, zoom=None):
    """
    Yield each PDF page as a BGR image, in page order, one at a time.

    Only a single page is materialized in RAM at any moment — pages are never
    accumulated into a list — so peak memory stays flat regardless of page count.

    For PDFs larger than PDF_SPILL_THRESHOLD (500 MB) the file is first written
    to a private temp directory and opened from disk, so PyMuPDF reads pages
    lazily off the filesystem instead of us holding the whole document in RAM.
    That temp file and its directory are removed as soon as the final page has
    been consumed — or immediately if iteration is stopped early or errors out
    (the generator's finally block runs on .close()/GC too).

    Raises RuntimeError if PyMuPDF isn't installed.
    """
    if fitz is None:
        raise RuntimeError("PDF support isn't installed on the server. Run: pip install PyMuPDF")

    spill = len(pdf_bytes) > PDF_SPILL_THRESHOLD

    doc = None
    tmp_dir = None
    tmp_path = None
    try:
        if spill:
            ensure_dirs()
            tmp_dir = tempfile.mkdtemp(prefix="pdf_spill_", dir=app.config["TEMP_PDF_FOLDER"])
            tmp_path = os.path.join(tmp_dir, "source.pdf")
            with open(tmp_path, "wb") as fh:
                fh.write(pdf_bytes)
            # Drop this function's reference to the large in-RAM buffer now that
            # it lives on disk; the file is read a page at a time from here on.
            pdf_bytes = None
            doc = fitz.open(tmp_path)
        else:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for i in range(doc.page_count):
            page = doc.load_page(i)
            bgr = _pdf_page_to_bgr(page, _pdf_render_matrix(page, zoom))
            yield bgr
            # Release this page before loading the next one.
            del bgr, page
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
        # Clean up the spilled file + its temp directory once the last page has
        # been processed (or on early close / error).
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if tmp_dir:
            try:
                os.rmdir(tmp_dir)
            except OSError:
                shutil.rmtree(tmp_dir, ignore_errors=True)


def _iter_pdf_bgr_pages_from_path(pdf_path, zoom=None):
    """
    Yield each page of an on-disk PDF as a BGR image, one at a time.

    The PDF is opened directly from `pdf_path`, so PyMuPDF reads pages lazily
    off the filesystem and the whole document is never buffered in RAM — peak
    memory is a single rasterized page regardless of file size or page count.
    The caller owns `pdf_path` and is responsible for deleting it afterwards.

    Raises RuntimeError if PyMuPDF isn't installed.
    """
    if fitz is None:
        raise RuntimeError("PDF support isn't installed on the server. Run: pip install PyMuPDF")

    doc = None
    try:
        doc = fitz.open(pdf_path)
        for i in range(doc.page_count):
            page = doc.load_page(i)
            bgr = _pdf_page_to_bgr(page, _pdf_render_matrix(page, zoom))
            yield bgr
            del bgr, page
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def _save_single_card_image(bgr, suffix, edge_type):
    """Align/crop a card image (best-effort) by edge type and save it into the
    inventory folder. Returns (relative_path, was_cropped)."""
    img, cropped = detect_and_crop_card(bgr, edge_type)   # falls back to raw
    final_name = f"single_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    relative_path = normalize_to_upload_relative(os.path.join("inventory_cards", final_name))
    absolute_path = os.path.join(app.config["UPLOAD_FOLDER"], relative_path)
    cv2.imwrite(absolute_path, img)
    return relative_path, cropped


def _clean_collection(v):
    """Normalize the optional Collection tag: alphanumeric plus space/dash/
    underscore, trimmed, capped. Blank stays blank."""
    v = str(v or "").strip()
    v = _re.sub(r"[^A-Za-z0-9 _-]", "", v)
    return v[:100].strip()


def _normalize_game_name(g):
    """Store game names with a capitalized first character (e.g. 'pokemon' ->
    'Pokemon'). Underscores become spaces; the rest of the string is preserved so
    multi-word or specially-cased names (e.g. 'Magic the Gathering') aren't
    mangled. Used only for the stored/displayed value — the template key that the
    game maps to is left untouched."""
    g = str(g or "").replace("_", " ").strip()
    return (g[:1].upper() + g[1:]) if g else g


def _create_single_card(front_path, back_path, game, album, template, collection=""):
    """
    Create a blank inventory record for `game` (+optional album/collection) with
    the given front (required) and back (optional) image paths, then OCR-identify
    the FRONT ONLY — the back is never name/serial-checked. A match at or above the
    auto-identify threshold (60%) is applied and saved; anything less leaves the
    entry blank for manual entry.

    Returns (record, ident_dict).
    """
    blank_fields = {k: "" for k in (template.get("fields", {}) or {}).keys()}
    extracted = {**blank_fields, "game": _normalize_game_name(game)}
    if album:
        extracted["album"] = album
    if collection:
        extracted["collection"] = collection

    _, record = create_scan_record(
        image_path=front_path,
        template_name=game,
        extracted=extracted,
        image_path_back=back_path,
    )

    ident = {"identified": False, "reason": "skipped"}
    try:
        ident = auto_identify_record(record)   # uses the FRONT image only
        if ident.get("identified") or ident.get("type_applied"):
            db.session.commit()
    except Exception:
        db.session.rollback()
        ident = {"identified": False, "reason": "error"}
    return record, ident


def _finalize_pdf_pair(front_path, back_path, front_page, back_page, game, album, template, collection=""):
    """Create one inventory record from an already-saved front (+ optional back)
    image pair pulled from a PDF, and return the card dict the UI expects."""
    record, ident = _create_single_card(front_path, back_path, game, album, template, collection)
    return {
        "record_id":       record.id,
        "front_page":      front_page,
        "back_page":       back_page,
        "identified":      bool(ident.get("identified")),
        "identified_name": (ident.get("applied") or {}).get("name", "") if ident.get("identified") else "",
        "card_type":       (ident.get("type_applied") or {}).get("value", ""),
        "image_url":       build_uploaded_file_url(record.image_path),
        "detail_url":      url_for("inventory_detail", record_id=record.id),
    }


def _import_single_card_pdf(pdf_bytes, game, album, edge_type, template, collection=""):
    """
    Import a PDF (given as raw bytes) as front/back pairs. Suitable for smaller
    uploads; inputs over 500 MB are spilled to a temp file automatically (see
    _iter_pdf_bgr_pages). Prefer _import_single_card_pdf_path when the upload has
    already been streamed to disk, to avoid ever holding the file in RAM.
    """
    return _import_pdf_pages(_iter_pdf_bgr_pages(pdf_bytes), game, album, edge_type, template, collection)


def _import_single_card_pdf_path(pdf_path, game, album, edge_type, template, collection=""):
    """
    Import an already-on-disk PDF as front/back pairs, rasterizing one page at a
    time straight from the file. The document is never loaded into RAM, so this
    keeps the process's memory well under the 500 MB cap no matter how large the
    PDF is. The caller owns and cleans up `pdf_path`.
    """
    return _import_pdf_pages(_iter_pdf_bgr_pages_from_path(pdf_path), game, album, edge_type, template, collection)


def _import_pdf_pages(page_iter, game, album, edge_type, template, collection=""):
    """
    Shared front/back-pair importer: consume a stream of BGR pages (odd pages
    are FRONTS, even pages BACKS), build one inventory record per pair, and
    return the JSON response. Only the front of each pair is OCR-checked.

    Pages are pulled one at a time; the first page of a pair is saved to disk
    immediately so we never hold more than a single decoded page in RAM while
    waiting for its partner. `page_iter` is always closed, so any temp file the
    generator spilled (or opened) is cleaned up even on early exit or error.
    """
    cards = []
    cap_hit = False
    pending_front = None  # dict: {path, page_no} for a front awaiting its back
    try:
        page_no = 0
        for bgr in page_iter:
            page_no += 1
            if pending_front is None:
                # Front of a new pair — persist it now and drop the pixels.
                front_path, _ = _save_single_card_image(bgr, "front", edge_type)
                pending_front = {"path": front_path, "page_no": page_no}
                del bgr
                continue

            # We now have the back for the pending front.
            back_path, _ = _save_single_card_image(bgr, "back", edge_type)
            del bgr
            cards.append(_finalize_pdf_pair(
                pending_front["path"], back_path, pending_front["page_no"],
                page_no, game, album, template, collection))
            pending_front = None

        # A trailing odd page: a front with no back.
        if pending_front is not None:
            cards.append(_finalize_pdf_pair(
                pending_front["path"], None, pending_front["page_no"],
                None, game, album, template, collection))
    except InventoryCapError:
        cap_hit = True   # keep what imported; report the cap below
    except RuntimeError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Could not read PDF: {exc}"}), 500
    finally:
        # Closes the generator, running its cleanup (closing the PDF doc and
        # deleting any spilled temp file/dir) even if we stopped early or errored.
        page_iter.close()

    if not cards:
        if cap_hit:
            return jsonify({"status": "error", "cap_reached": True, "message": UPGRADE_MESSAGE}), 507
        return jsonify({"status": "error", "message": "PDF has no pages"}), 400

    id_count = sum(1 for c in cards if c["identified"])
    message = (f"Imported {len(cards)} card(s) from PDF — odd pages as fronts, even pages as backs. "
               f"{id_count} auto-identified, {len(cards) - id_count} left blank for manual entry.")
    if cap_hit:
        message += " " + UPGRADE_MESSAGE
    return jsonify({
        "status":           "success",
        "mode":             "pdf",
        "message":          message,
        "count":            len(cards),
        "identified_count": id_count,
        "cap_reached":      cap_hit,
        "cards":            cards,
    })


@app.route("/import_single_card", methods=["POST"])
def import_single_card():
    """
    Add card(s) to inventory from the single-card importer.

    Two upload shapes are accepted in `front_image`:
      • an image  — one card: `front_image` (required) + `back_image` (optional).
      • a PDF     — a front/back batch: odd pages (1,3,5,...) are fronts, even
                    pages (2,4,6,...) are backs; each pair becomes one card.

    Every image is best-effort auto-aligned/cropped by the selected edge type
    (falling back to the raw photo if no card outline is found). Only the FRONT
    of each card is OCR name/serial-checked; a 100% match is applied, otherwise
    the entry is left blank for manual entry.

    Form fields: game (required), album (optional), card_edge_type,
                 front_image (image or PDF, required), back_image (optional).
    """
    ensure_dirs()

    game  = (request.form.get("game")  or "").strip()
    album = (request.form.get("album") or "").strip()
    collection = _clean_collection(request.form.get("collection"))
    edge_type = normalize_card_edge_type(request.form.get("card_edge_type"))
    if not game:
        return jsonify({"status": "error", "message": "Game is required"}), 400

    front = request.files.get("front_image")
    if not front or not front.filename:
        return jsonify({"status": "error", "message": "A front image or PDF is required"}), 400
    back = request.files.get("back_image")

    # Refuse up front if the inventory is already full. (PDF batches can still
    # partially import up to the cap and report it — see _import_pdf_pages.)
    if _inventory_remaining() <= 0:
        return jsonify({"status": "error", "cap_reached": True, "message": UPGRADE_MESSAGE}), 507

    try:
        template = load_template(game)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Could not load game '{game}': {exc}"}), 400

    # Stream the primary upload straight to a temp file instead of front.read().
    # A PDF may be very large, and reading it into memory here would blow the
    # RAM budget before the page-at-a-time importer ever runs. FileStorage.save
    # copies in chunks (and big uploads are already spooled to disk by Werkzeug),
    # so the whole file never sits in RAM. We then sniff the type from the header.
    upload_dir = tempfile.mkdtemp(prefix="upload_", dir=app.config["TEMP_PDF_FOLDER"])
    front_tmp = os.path.join(upload_dir, "front_upload")
    try:
        front.save(front_tmp)

        head = b""
        try:
            with open(front_tmp, "rb") as fh:
                head = fh.read(5)
        except OSError:
            head = b""
        is_pdf = (
            (getattr(front, "mimetype", "") or "").lower() == "application/pdf"
            or os.path.splitext(front.filename)[1].lower() == ".pdf"
            or head == b"%PDF-"
        )

        if is_pdf:
            if fitz is None:
                return jsonify({"status": "error",
                                "message": "PDF support isn't installed on the server. Run: pip install PyMuPDF"}), 500
            # Rasterized one page at a time straight from disk — RAM stays capped
            # regardless of file size. Returns before the finally cleans up the
            # temp PDF (all pages are read by then).
            return _import_single_card_pdf_path(front_tmp, game, album, edge_type, template, collection)

        # ---- Single image: front (required) + optional back ----
        # Card photos are small, so decoding one from disk is well within budget.
        aligned_flags = {}
        try:
            front_img = cv2.imread(front_tmp, cv2.IMREAD_COLOR)
            if front_img is None:
                return jsonify({"status": "error", "message": "Could not read the front image"}), 400
            front_path, front_cropped = _save_single_card_image(front_img, "front", edge_type)
            aligned_flags["front"] = front_cropped
            del front_img

            back_path = None
            if back and back.filename:
                back_tmp = os.path.join(upload_dir, "back_upload")
                back.save(back_tmp)
                back_img = cv2.imread(back_tmp, cv2.IMREAD_COLOR)
                if back_img is not None:
                    back_path, back_cropped = _save_single_card_image(back_img, "back", edge_type)
                    aligned_flags["back"] = back_cropped
                    del back_img
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Could not process image: {exc}"}), 500
    finally:
        # The inventory copies are already written by _save_single_card_image,
        # so the raw uploads can go regardless of which branch ran.
        shutil.rmtree(upload_dir, ignore_errors=True)

    record, ident = _create_single_card(front_path, back_path, game, album, template, collection)

    # Let the UI mention when a card outline couldn't be found and the raw
    # photo was kept instead (so the user can retry or crop manually later).
    not_detected = [side for side, ok in aligned_flags.items() if not ok]
    if not_detected:
        message = ("Card added to inventory. Couldn't detect a "
                   f"{edge_type}-edged card in the {', '.join(not_detected)} "
                   "image, so the original photo was kept for that side.")
    else:
        message = "Card added to inventory — detected and cropped to the card."

    if ident.get("identified"):
        ident_name = (ident.get("applied") or {}).get("name", "")
        message += (f" Identified as “{ident_name}” and filled in automatically."
                    if ident_name else " Identified and filled in automatically.")
    else:
        message += " Left blank for manual entry (no confident match)."

    card_type = (ident.get("type_applied") or {}).get("value", "")
    if card_type:
        message += f" Type detected: {card_type}."

    return jsonify({
        "status":          "success",
        "mode":            "single",
        "message":         message,
        "record_id":       record.id,
        "edge_type":       edge_type,
        "front_aligned":   aligned_flags.get("front", False),
        "back_aligned":    aligned_flags.get("back", False) if back_path else None,
        "identified":      bool(ident.get("identified")),
        "identified_name": (ident.get("applied") or {}).get("name", "") if ident.get("identified") else "",
        "card_type":       card_type,
        "image_url":       build_uploaded_file_url(record.image_path),
        "image_url_back":  build_uploaded_file_url(record.image_path_back) if record.image_path_back else None,
        "detail_url":      url_for("inventory_detail", record_id=record.id),
    })



# ====================== FILE ROUTES ======================
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/temp_split/<path:filename>")
def temp_split_file(filename):
    return send_from_directory(app.config["TEMP_SPLIT_FOLDER"], filename)


@app.route("/temp_cards/<path:filename>")
def temp_card_file(filename):
    return send_from_directory(app.config["TEMP_CARD_FOLDER"], filename)


@app.route("/temp_pdf/<path:filename>")
def temp_pdf_file(filename):
    return send_from_directory(app.config["TEMP_PDF_FOLDER"], filename)


# ====================== INVENTORY UPDATE ROUTES ======================
@app.route("/update_scan/<int:record_id>", methods=["POST"])
def update_scan(record_id):
    record = ScanRecord.query.get_or_404(record_id)
    data = request.get_json() or {}
    new_data = data.get("extracted_data", {})
    # Strip legacy/internal keys that should never be stored as editable fields
    for hidden in ("card_lookup", "__roi_fields_used"):
        new_data.pop(hidden, None)
    record.extracted_data = {**(record.extracted_data or {}), **new_data}
    db.session.commit()
    return jsonify({"status": "success", "message": "Entry updated"})


@app.route("/update_scan_image/<int:record_id>", methods=["POST"])
def update_scan_image(record_id):
    record = ScanRecord.query.get_or_404(record_id)
    file = request.files.get("image")
    side = request.form.get("side", "front").strip().lower()
    if side not in ("front", "back"):
        side = "front"

    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    ensure_dirs()

    safe_name = secure_filename(file.filename)
    stem, ext = os.path.splitext(safe_name)
    if not ext:
        ext = ".png"

    suffix = "back" if side == "back" else "front"
    final_name = f"record_{record_id}_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
    relative_path = normalize_to_upload_relative(os.path.join("inventory_cards", final_name))
    absolute_path = os.path.join(app.config["UPLOAD_FOLDER"], relative_path)
    file.save(absolute_path)

    if side == "back":
        old_path = record.image_path_back
        record.image_path_back = relative_path
    else:
        old_path = record.image_path
        record.image_path = relative_path
    db.session.commit()

    if old_path and old_path != "__blank__" and normalize_to_upload_relative(old_path) != relative_path:
        remove_file_if_exists(old_path)

    return jsonify({
        "status":    "success",
        "message":   "Image updated successfully",
        "side":      side,
        "image_url": build_uploaded_file_url(record.image_path_back if side == "back" else record.image_path),
    })


@app.route("/delete_scan/<int:record_id>", methods=["POST"])
def delete_scan(record_id):
    record = ScanRecord.query.get_or_404(record_id)
    image_path      = record.image_path
    image_path_back = record.image_path_back
    db.session.delete(record)
    db.session.commit()
    remove_file_if_exists(image_path)
    remove_file_if_exists(image_path_back)
    return jsonify({"status": "success", "message": "Inventory item deleted"})


@app.route("/delete_scans", methods=["POST"])
def delete_scans():
    data = request.get_json() or {}
    record_ids = data.get("record_ids", [])

    if not isinstance(record_ids, list) or not record_ids:
        return jsonify({"status": "error", "message": "No record IDs provided"}), 400

    records = ScanRecord.query.filter(ScanRecord.id.in_(record_ids)).all()
    if not records:
        return jsonify({"status": "error", "message": "No matching records found"}), 404

    image_paths = [r.image_path for r in records] + [r.image_path_back for r in records]
    for record in records:
        db.session.delete(record)
    db.session.commit()

    for image_path in image_paths:
        remove_file_if_exists(image_path)

    return jsonify({"status": "success", "message": f"Deleted {len(records)} record(s)"})


@app.route("/migrate_clean_legacy_fields", methods=["POST"])
def migrate_clean_legacy_fields():
    """
    One-shot migration that scrubs superseded/legacy fields from every record:

      - first_edition   (boolean) -> promoted to edition='First Edition' if no
                                      edition key exists, then deleted
      - limited_edition (boolean) -> promoted to edition='Limited Edition' if no
                                      edition key exists, then deleted
      - holographic as a boolean  -> replaced with 'None' (enum string)
      - empty           (boolean) -> deleted entirely

    Safe to run multiple times — already-migrated records are left unchanged.
    """
    records = ScanRecord.query.all()
    updated = 0

    for record in records:
        data = dict(record.extracted_data or {})
        changed = False

        # ── Edition: promote legacy booleans, then delete them ────────────
        if data.get("edition", "") not in EDITION_OPTIONS:
            fe = data.get("first_edition", False)
            le = data.get("limited_edition", False)
            if fe is True or str(fe).strip().lower() == "true":
                data["edition"] = "First Edition"
            elif le is True or str(le).strip().lower() == "true":
                data["edition"] = "Limited Edition"
            else:
                data["edition"] = EDITION_DEFAULT
            changed = True

        for legacy_key in ("first_edition", "limited_edition"):
            if legacy_key in data:
                del data[legacy_key]
                changed = True

        # ── Holographic: replace any non-enum value with 'None' ───────────
        holo_raw = data.get("holographic")
        if holo_raw is not None and holo_raw not in _HOLOGRAPHIC_OPTIONS:
            data["holographic"] = "None"
            changed = True

        # ── Empty: delete entirely ─────────────────────────────────────────
        if "empty" in data:
            del data["empty"]
            changed = True

        if changed:
            record.extracted_data = data
            updated += 1

    if updated:
        db.session.commit()

    return jsonify({
        "status":  "success",
        "message": f"Migration complete. {updated} record(s) updated.",
        "updated": updated,
    })


@app.route("/add_custom_field", methods=["POST"])
def add_custom_field():
    """
    Add a custom key/value field to inventory records.
    Accepts either:
      { game, key, value }        — applies to ALL records matching the game name
      { record_ids, key, value }  — applies to explicit list of record IDs
    When matching by game name, an optional { catalog_only: true|false } keeps
    the match scoped to the same kind of row the person was looking at (hidden
    CSV-catalog rows vs. normal owned-inventory rows) so applying a field from
    the Imported Catalog view can never spill onto real inventory for the same
    game, and vice versa.
    """
    data  = request.get_json() or {}
    key   = data.get("key",   "").strip()
    value = data.get("value", "").strip()

    if not key or not value:
        return jsonify({"status": "error", "message": "Key and value are required"}), 400

    game         = data.get("game", "").strip()
    record_ids   = data.get("record_ids", [])
    scope_passed = "catalog_only" in data
    want_catalog = bool(data.get("catalog_only", False))

    if game:
        all_rows = (
            ScanRecord.query
            .with_entities(ScanRecord.id, ScanRecord.extracted_data)
            .all()
        )
        matching_ids = []
        for row_id, extracted_data in all_rows:
            if isinstance(extracted_data, dict):
                row_data = extracted_data
            else:
                try:
                    row_data = json.loads(extracted_data or "{}")
                except (ValueError, TypeError):
                    row_data = {}
            if str(row_data.get("game", "")).strip() != game:
                continue
            if scope_passed and _is_catalog_only(row_data) != want_catalog:
                continue
            matching_ids.append(row_id)

        if not matching_ids:
            return jsonify({"status": "error", "message": f"No records found for game '{game}'"}), 404

        records = ScanRecord.query.filter(ScanRecord.id.in_(matching_ids)).all()

    elif isinstance(record_ids, list) and record_ids:
        records = ScanRecord.query.filter(ScanRecord.id.in_(record_ids)).all()
        if not records:
            return jsonify({"status": "error", "message": "No matching records found"}), 404
    else:
        return jsonify({"status": "error", "message": "Provide either 'game' or 'record_ids'"}), 400

    for record in records:
        updated = dict(record.extracted_data or {})
        updated[key] = value
        record.extracted_data = updated

    db.session.commit()

    return jsonify({
        "status":  "success",
        "message": f"Added '{key}: {value}' to {len(records)} record(s)",
    })


# ====================== TEMPLATE (GAME) SAVE ROUTE ======================
def _slugify_template_name(name):
    return _re.sub(r"[^a-z0-9_]", "", str(name or "").strip().lower().replace(" ", "_"))


def _clean_template_fields(fields):
    """
    Normalize a raw {field_name: {field_type, dropdown_options}} payload into
    the on-disk template shape. Returns {} if nothing valid was provided.
    Shared by /template_save and the CSV import's "create new template" path.
    """
    if not isinstance(fields, dict):
        return {}

    cleaned_fields = {}
    for field_name, cfg in fields.items():
        if not isinstance(cfg, dict):
            continue

        field_key = _slugify_template_name(field_name)
        if not field_key:
            continue

        raw_field_type = str(cfg.get("field_type", "text")).strip().lower()
        if raw_field_type not in ("text", "dropdown", "boolean"):
            raw_field_type = "text"

        raw_opts = cfg.get("dropdown_options", [])
        if isinstance(raw_opts, list):
            dropdown_options = [str(o).strip() for o in raw_opts if str(o).strip()]
        else:
            dropdown_options = []

        if raw_field_type == "dropdown" and not dropdown_options:
            # Fall back to a plain text field rather than reject the whole
            # save — an empty-options dropdown isn't useful anyway.
            raw_field_type = "text"

        # A hidden field is still stored on every record (CSV import, manual
        # entry, etc.) and remains referenceable — e.g. via "Copy from Entry"
        # or a future export — but is left out of the editable field grid on
        # the Inventory Detail page.
        is_hidden = bool(cfg.get("hidden", False))

        entry = {"field_type": raw_field_type}
        if raw_field_type == "dropdown":
            entry["dropdown_options"] = dropdown_options
        if is_hidden:
            entry["hidden"] = True

        cleaned_fields[field_key] = entry

    return cleaned_fields


def _write_template_file(clean_name, cleaned_fields, csv_column_mapping=None):
    """
    Write a template's JSON file. Field definitions are always replaced with
    `cleaned_fields`. Any legacy `csv_column_mapping` already saved on the
    template is preserved across ordinary field edits — pass
    `csv_column_mapping` to explicitly set/merge it, or leave it None to just
    carry forward whatever was already saved. Mapping entries for fields that
    no longer exist are dropped automatically.
    """
    ensure_dirs()
    template_path = os.path.join(app.config["ROI_TEMPLATE_FOLDER"], f"{clean_name}.json")

    existing_mapping = {}
    if os.path.exists(template_path):
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                existing_mapping = (json.load(f) or {}).get("csv_column_mapping", {}) or {}
        except Exception:
            existing_mapping = {}

    if csv_column_mapping is not None:
        merged_mapping = {**existing_mapping, **csv_column_mapping}
    else:
        merged_mapping = existing_mapping

    # Drop stale entries for fields that no longer exist on this template.
    merged_mapping = {k: v for k, v in merged_mapping.items() if k in cleaned_fields and v}

    payload = {"name": clean_name, "fields": cleaned_fields}
    if merged_mapping:
        payload["csv_column_mapping"] = merged_mapping

    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@app.route("/template_save", methods=["POST"])
def template_save():
    """
    Save a Game definition: a name plus a flat list of fields that every
    entry for that game should have. Each field just needs a key and a
    field_type ("text" | "dropdown" | "boolean"); dropdown fields also carry
    a list of options. There are no image zones / ROI coordinates involved —
    imported cards for this game are simply created with these fields blank.
    """
    data   = request.get_json() or {}
    name   = data.get("name", "").strip()
    fields = data.get("fields", {})

    if not name:
        return jsonify({"status": "error", "message": "Game name is required"}), 400

    clean_name = _slugify_template_name(name)
    if not clean_name:
        return jsonify({"status": "error", "message": "Game name contains no valid characters"}), 400

    if not isinstance(fields, dict) or not fields:
        return jsonify({"status": "error", "message": "At least one field is required"}), 400

    cleaned_fields = _clean_template_fields(fields)
    if not cleaned_fields:
        return jsonify({"status": "error", "message": "No valid fields provided"}), 400

    try:
        _write_template_file(clean_name, cleaned_fields)
    except OSError as exc:
        return jsonify({"status": "error", "message": f"Could not write template file: {exc}"}), 500

    return jsonify({
        "status":  "success",
        "message": f"Game '{clean_name}' saved with {len(cleaned_fields)} field(s)",
        "name":    clean_name,
        "fields":  cleaned_fields,
    })


@app.route("/template_delete", methods=["POST"])
def template_delete():
    """Delete a Game definition file. Existing inventory records that used
    this game/template are left untouched — only the field definition goes
    away, so imports can no longer use it as a source of blank fields."""
    data = request.get_json() or {}
    name = str(data.get("name", "")).strip()

    clean_name = _slugify_template_name(name)
    if not clean_name:
        return jsonify({"status": "error", "message": "Game name is required"}), 400

    template_path = os.path.join(app.config["ROI_TEMPLATE_FOLDER"], f"{clean_name}.json")
    if not os.path.exists(template_path):
        return jsonify({"status": "error", "message": f"Game '{clean_name}' not found"}), 404

    try:
        os.remove(template_path)
    except OSError as exc:
        return jsonify({"status": "error", "message": f"Could not delete game: {exc}"}), 500

    return jsonify({"status": "success", "message": f"Game '{clean_name}' deleted"})


# ====================== TEMPLATE CONFIG (live field types) ======================
@app.route("/template_config/<template_name>")
def template_config(template_name):
    """
    Return the live field config for a given template as JSON.
    Used by the inventory detail page to detect field-type changes
    made after the page was server-rendered.
    Shape: { fieldKey: { field_type, dropdown_options?, hidden? }, … }
    """
    try:
        tpl = load_template(template_name or "product_label")
        fields = tpl.get("fields", {})
        # Return only field_type / dropdown_options / hidden — omit ROI coords
        slim = {
            k: {
                "field_type":       v.get("field_type", "text"),
                "dropdown_options": v.get("dropdown_options", []),
                "hidden":           bool(v.get("hidden", False)),
            }
            for k, v in fields.items()
        }
        return jsonify(slim)
    except Exception as exc:
        return jsonify({}), 200  # graceful fallback


# ====================== UPDATE FIELD TYPE ======================
@app.route("/update_field_type", methods=["POST"])
def update_field_type():
    """
    Patch the field_type (and optionally dropdown_options) for a single field
    across every template that contains that field key.

    Request JSON:
        {
            "field_key":        "my_field",
            "field_type":       "text" | "dropdown" | "boolean",
            "dropdown_options": ["Opt A", "Opt B"]   // required when field_type == "dropdown"
        }

    Response JSON:
        { "status": "success"|"error", "message": "...", "updated_templates": [...] }
    """
    data = request.get_json() or {}

    field_key  = str(data.get("field_key",  "")).strip()
    field_type = str(data.get("field_type", "")).strip().lower()
    raw_opts   = data.get("dropdown_options", [])

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not field_key:
        return jsonify({"status": "error", "message": "field_key is required"}), 400

    if field_type not in ("text", "dropdown", "boolean"):
        return jsonify({"status": "error",
                        "message": f"Invalid field_type '{field_type}'. Must be text, dropdown, or boolean."}), 400

    if isinstance(raw_opts, list):
        dropdown_options = [str(o).strip() for o in raw_opts if str(o).strip()]
    else:
        dropdown_options = []

    if field_type == "dropdown" and not dropdown_options:
        return jsonify({"status": "error",
                        "message": "dropdown_options must contain at least one option for dropdown fields"}), 400

    # ── Patch every template that contains this field key ────────────────────
    updated_templates = []
    errors            = []

    for tpl_name in get_template_names():
        tpl_path = os.path.join(app.config["ROI_TEMPLATE_FOLDER"], f"{tpl_name}.json")
        try:
            with open(tpl_path, "r", encoding="utf-8") as f:
                tpl = json.load(f)

            fields = tpl.get("fields", {})
            if field_key not in fields:
                continue  # this template doesn't have the field — skip

            # Patch in-place, preserving all existing ROI coordinates / config
            fields[field_key]["field_type"] = field_type
            if field_type == "dropdown":
                fields[field_key]["dropdown_options"] = dropdown_options
            else:
                # Remove stale dropdown_options for non-dropdown types
                fields[field_key].pop("dropdown_options", None)

            with open(tpl_path, "w", encoding="utf-8") as f:
                json.dump(tpl, f, indent=2)

            updated_templates.append(tpl_name)

        except Exception as exc:
            errors.append(f"{tpl_name}: {exc}")

    if errors and not updated_templates:
        return jsonify({"status": "error",
                        "message": "Failed to update templates: " + "; ".join(errors)}), 500

    if not updated_templates:
        return jsonify({"status": "error",
                        "message": f"Field '{field_key}' not found in any template"}), 404

    msg = f"'{field_key}' updated to {field_type} in template(s): {', '.join(updated_templates)}"
    if errors:
        msg += f" (errors on: {'; '.join(errors)})"

    return jsonify({
        "status":            "success",
        "message":           msg,
        "updated_templates": updated_templates,
    })


@app.route("/update_field_hidden", methods=["POST"])
def update_field_hidden():
    """
    Toggle the `hidden` flag for a single field key across every template that
    contains it (same cross-template scope as /update_field_type).

    Hidden fields are still stored on every record, but are kept off the
    Inventory table and the Inventory Detail edit form. They can be revealed and
    toggled back via the detail page's Layout editor, or shown in the Inventory
    table via its "Show Hidden Fields" switch.

    Request JSON:
        { "field_key": "my_field", "hidden": true | false }

    Response JSON:
        { "status": ..., "message": ..., "updated_templates": [...], "hidden": bool }
    """
    data = request.get_json() or {}

    field_key = str(data.get("field_key", "")).strip()
    hidden    = bool(data.get("hidden", False))

    if not field_key:
        return jsonify({"status": "error", "message": "field_key is required"}), 400

    # 'game' drives per-game grouping / navigation, so it must always stay visible.
    if field_key == "game":
        return jsonify({
            "status":  "error",
            "message": "The 'game' field is required for navigation and can't be hidden.",
        }), 400

    updated_templates = []
    errors            = []

    for tpl_name in get_template_names():
        tpl_path = os.path.join(app.config["ROI_TEMPLATE_FOLDER"], f"{tpl_name}.json")
        try:
            with open(tpl_path, "r", encoding="utf-8") as f:
                tpl = json.load(f)

            fields = tpl.get("fields", {})
            if field_key not in fields:
                continue  # this template doesn't have the field — skip

            if hidden:
                fields[field_key]["hidden"] = True
            else:
                # Clearing the flag entirely keeps template files tidy.
                fields[field_key].pop("hidden", None)

            with open(tpl_path, "w", encoding="utf-8") as f:
                json.dump(tpl, f, indent=2)

            updated_templates.append(tpl_name)

        except Exception as exc:
            errors.append(f"{tpl_name}: {exc}")

    if errors and not updated_templates:
        return jsonify({"status": "error",
                        "message": "Failed to update templates: " + "; ".join(errors)}), 500

    if not updated_templates:
        return jsonify({
            "status":  "error",
            "message": f"Field '{field_key}' isn't part of any saved template, so its "
                       f"hidden state can't be stored.",
        }), 404

    state = "hidden" if hidden else "visible"
    msg = f"'{field_key}' set to {state} in template(s): {', '.join(updated_templates)}"
    if errors:
        msg += f" (errors on: {'; '.join(errors)})"

    return jsonify({
        "status":            "success",
        "message":           msg,
        "updated_templates": updated_templates,
        "hidden":            hidden,
    })


# ====================== RECORDS SUMMARY ROUTE ======================
@app.route("/records_summary")
def records_summary():
    """
    Lightweight summary of all records for the copy-from dropdown
    on the inventory detail page. Strips internal OCR keys.
    """
    rows = (
        ScanRecord.query
        .with_entities(ScanRecord.id, ScanRecord.extracted_data)
        .order_by(ScanRecord.scan_date.desc())
        .all()
    )

    records = []
    for row_id, extracted_data in rows:
        if isinstance(extracted_data, dict):
            data = extracted_data
        else:
            try:
                data = json.loads(extracted_data or "{}")
            except (ValueError, TypeError):
                data = {}

        label = (
            data.get("product_name") or data.get("name") or
            data.get("card_name")   or data.get("title") or
            f"Record #{row_id}"
        )

        parts = [p for p in [
            "Catalog" if _is_catalog_only(data) else "",
            data.get("game", ""),
            data.get("album", ""),
            f"p{data.get('page','')}" if data.get("page") else "",
            f"s{data.get('slot','')}" if data.get("slot") else "",
        ] if p]

        clean_data = {
            k: v for k, v in data.items()
            if not k.startswith("__ocr_") and k != "__roi_fields_used"
        }

        records.append({
            "id":    row_id,
            "label": str(label).strip(),
            "sub":   " · ".join(parts),
            "data":  clean_data,
        })

    return jsonify({"records": records})


# ====================== JUSTTCG ROUTES ======================
# JustTCG public API — https://justtcg.com
# Search endpoint: GET https://justtcg.com/api/products/search
#   ?name=<card name>          required
#   &number=<set number>       optional but improves match accuracy
#   &game=<game name>          optional
# Returns JSON with a `products` array; each product has:
#   id, name, set_name, number, prices (market, low, mid, high), url

import urllib.request
import urllib.parse
import urllib.error

JUSTTCG_SEARCH_URL = "https://api.justtcg.com/v1/cards"
JUSTTCG_TIMEOUT    = 10  # seconds
# JUSTTCG_API_KEY is read at call time via get_api_key() so edits in
# Settings → API Keys take effect immediately (no restart).

# Maps human-readable / OCR'd game names to the slug values accepted by the
# JustTCG API.  An unrecognised value causes HTTP 400, so anything not listed
# here is dropped (empty string = no game filter, broader search).
_JUSTTCG_GAME_MAP = {
    "magic the gathering":    "magic-the-gathering",
    "magic: the gathering":   "magic-the-gathering",
    "magic":                  "magic-the-gathering",
    "mtg":                    "magic-the-gathering",
    "pokemon":                "pokemon",
    "pokémon":                "pokemon",
    "yugioh":                 "yugioh",
    "yu-gi-oh":               "yugioh",
    "yu-gi-oh!":              "yugioh",
    "disney lorcana":         "disney-lorcana",
    "lorcana":                "disney-lorcana",
    "one piece":              "one-piece-card-game",
    "one piece card game":    "one-piece-card-game",
    "digimon":                "digimon-card-game",
    "digimon card game":      "digimon-card-game",
    "flesh and blood":        "flesh-and-blood-tcg",
    "flesh and blood tcg":    "flesh-and-blood-tcg",
    "union arena":            "union-arena",
    "age of sigmar":          "age-of-sigmar",
    "warhammer 40000":        "warhammer-40000",
    "warhammer 40k":          "warhammer-40000",
    "warhammer40000":         "warhammer-40000",
}


def _justtcg_search(name: str, number: str = "", game: str = "") -> dict:
    """
    Call the JustTCG /v1/cards search endpoint and return the parsed JSON response.
    Uses `q` for card name search and `game` for game filtering.
    Raises urllib.error.URLError / ValueError on failure.
    """
    params = {"q": name.strip(), "include_price_history": "false"}
    if game:
        params["game"] = game.strip()

    url = JUSTTCG_SEARCH_URL + "?" + urllib.parse.urlencode(params)
    headers = {
        "Accept":     "application/json",
        "User-Agent": "CardCollectorInventoryManager/1.0",
        "x-api-key":  get_api_key("JUSTTCG_API_KEY"),
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=JUSTTCG_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


@app.route("/justtcg_missing_ids")
def justtcg_missing_ids():
    """
    Return a list of all ScanRecord IDs that do not yet have a market price
    stored under extracted_data['tcgplayer']['prices']['market'].
    Used by the bulk-fetch UI to know which records still need pricing.
    Only includes records that have a card name, so the fetch is likely to succeed.
    """
    rows = (
        ScanRecord.query
        .with_entities(ScanRecord.id, ScanRecord.extracted_data)
        .order_by(ScanRecord.id.asc())
        .all()
    )

    missing = []
    for row_id, extracted_data in rows:
        if isinstance(extracted_data, dict):
            data = extracted_data
        else:
            try:
                data = json.loads(extracted_data or "{}")
            except (ValueError, TypeError):
                data = {}

        # Catalog-only records (CSV import) aren't owned inventory — skip them
        if _is_catalog_only(data):
            continue

        # Must have a usable card name or the fetch will fail immediately
        card_name = (
            data.get("product_name") or data.get("name") or
            data.get("card_name")    or data.get("title") or ""
        ).strip()
        if not card_name:
            continue

        # Check whether a market price is already present
        tcg = data.get("tcgplayer") or {}
        prices = tcg.get("prices") or {}
        market = prices.get("market")
        if market is None or market == "":
            missing.append(row_id)

    return jsonify({"ids": missing, "count": len(missing)})


def _resolve_card_name_and_game(ext: dict):
    """
    Shared helper used by both justtcg_search_candidates and justtcg_fetch.
    Returns (card_name, set_number, game_slug).
    """
    card_name = (
        ext.get("product_name") or
        ext.get("name")         or
        ext.get("card_name")    or
        ext.get("title")        or
        ""
    ).strip()

    set_number = (
        ext.get("serial")     or
        ext.get("set_number") or
        ext.get("number")     or
        ""
    ).strip()

    raw_game = ext.get("game", "").strip()
    game = _JUSTTCG_GAME_MAP.get(raw_game.lower(), "")

    return card_name, set_number, game


def _build_entry_from_hit(hit: dict, card_name: str, set_number: str) -> dict:
    """
    Convert a single JustTCG Card object into the tcgplayer entry dict that
    gets stored in extracted_data['tcgplayer'].
    """
    variants = hit.get("variants") or []
    nm_normal = next(
        (v for v in variants
         if v.get("condition") in ("Near Mint", "NM") and v.get("printing") == "Normal"),
        variants[0] if variants else {},
    )
    market = nm_normal.get("price")
    low    = nm_normal.get("low_price")
    mid    = nm_normal.get("mid_price")
    high   = nm_normal.get("high_price")

    card_id      = hit.get("id", "")
    tcgplayer_id = hit.get("tcgplayerId", "")
    product_url  = (
        hit.get("url") or
        (f"https://justtcg.com/cards/{card_id}" if card_id else "https://justtcg.com")
    )

    return {
        "url":          product_url,
        "full_url":     product_url,
        "saved_at":     datetime.utcnow().isoformat(),
        "source":       "justtcg",
        "product_id":   str(card_id),
        "tcgplayer_id": str(tcgplayer_id),
        "product_name": hit.get("name", card_name),
        "set_name":     hit.get("set_name") or hit.get("set") or "",
        "set_number":   hit.get("number") or set_number,
        "prices": {
            "market": market,
            "low":    low,
            "mid":    mid,
            "high":   high,
        },
    }


@app.route("/justtcg_search/<int:record_id>", methods=["GET"])
def justtcg_search_candidates(record_id):
    """
    Returns all JustTCG search results for the card on this record without
    saving anything.  The UI uses this to present a picker so the user can
    choose the correct printing/set before committing.

    Response shape:
      { status, candidates: [ { card_id, name, set_name, set_number, game,
                                 rarity, nm_price, url } ] }
    """
    record = ScanRecord.query.get_or_404(record_id)
    ext    = record.extracted_data or {}

    card_name, set_number, game = _resolve_card_name_and_game(ext)

    if not card_name:
        return jsonify({
            "status":  "error",
            "message": "No card name found on this record (product_name / name / card_name).",
        }), 400

    try:
        api_data = _justtcg_search(card_name, number=set_number, game=game)
    except urllib.error.HTTPError as exc:
        return jsonify({
            "status":  "error",
            "message": f"JustTCG API returned HTTP {exc.code}: {exc.reason}",
        }), 502
    except urllib.error.URLError as exc:
        return jsonify({
            "status":  "error",
            "message": f"Could not reach JustTCG: {exc.reason}",
        }), 502
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({
            "status":  "error",
            "message": f"Unexpected response from JustTCG: {exc}",
        }), 502

    products = api_data.get("data") or []

    if not products:
        return jsonify({
            "status":   "not_found",
            "message":  f'No results found on JustTCG for "{card_name}".',
            "searched": {"name": card_name, "number": set_number, "game": game},
        })

    candidates = []
    for card in products:
        variants = card.get("variants") or []
        nm_normal = next(
            (v for v in variants
             if v.get("condition") in ("Near Mint", "NM") and v.get("printing") == "Normal"),
            variants[0] if variants else {},
        )
        card_id = card.get("id", "")
        candidates.append({
            "card_id":    card_id,
            "name":       card.get("name", ""),
            "set_name":   card.get("set_name") or card.get("set") or "",
            "set_number": card.get("number") or "",
            "game":       card.get("game") or "",
            "rarity":     card.get("rarity") or "",
            "nm_price":   nm_normal.get("price"),
            "url":        card.get("url") or
                          (f"https://justtcg.com/cards/{card_id}" if card_id else ""),
        })

    return jsonify({
        "status":     "ok",
        "candidates": candidates,
        "searched":   {"name": card_name, "number": set_number, "game": game},
    })


@app.route("/justtcg_fetch/<int:record_id>", methods=["POST"])
def justtcg_fetch(record_id):
    """
    Saves JustTCG pricing data for a specific card onto this record.

    If the POST body contains a `card_id`, that exact card is looked up and
    saved — this is the path taken after the user has chosen from the picker.

    If no `card_id` is provided the first API result is used (legacy / bulk
    fetch behaviour).

    On success the pricing snapshot is stored under extracted_data['tcgplayer'].
    """
    record = ScanRecord.query.get_or_404(record_id)
    ext    = record.extracted_data or {}

    body      = request.get_json() or {}
    chosen_id = body.get("card_id", "").strip()

    card_name, set_number, game = _resolve_card_name_and_game(ext)

    if not card_name and not chosen_id:
        return jsonify({
            "status":  "error",
            "message": "No card name found on this record (product_name / name / card_name).",
        }), 400

    # ── If a specific card_id was supplied, fetch that card directly ──────────
    if chosen_id:
        # Build a direct lookup URL using cardId so we get full variant data
        params = {"cardId": chosen_id, "include_price_history": "false"}
        url = JUSTTCG_SEARCH_URL + "?" + urllib.parse.urlencode(params)
        headers = {
            "Accept":     "application/json",
            "User-Agent": "CardCollectorInventoryManager/1.0",
            "x-api-key":  get_api_key("JUSTTCG_API_KEY"),
        }
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=JUSTTCG_TIMEOUT) as resp:
                api_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return jsonify({
                "status":  "error",
                "message": f"JustTCG API returned HTTP {exc.code}: {exc.reason}",
            }), 502
        except urllib.error.URLError as exc:
            return jsonify({
                "status":  "error",
                "message": f"Could not reach JustTCG: {exc.reason}",
            }), 502
        except (json.JSONDecodeError, ValueError) as exc:
            return jsonify({
                "status":  "error",
                "message": f"Unexpected response from JustTCG: {exc}",
            }), 502

        products = api_data.get("data") or []
        if not products:
            return jsonify({
                "status":  "error",
                "message": f"JustTCG returned no data for card ID '{chosen_id}'.",
            }), 502
        hit = products[0]

    else:
        # ── No card_id: fall back to name search and take the first result ────
        try:
            api_data = _justtcg_search(card_name, number=set_number, game=game)
        except urllib.error.HTTPError as exc:
            return jsonify({
                "status":  "error",
                "message": f"JustTCG API returned HTTP {exc.code}: {exc.reason}",
            }), 502
        except urllib.error.URLError as exc:
            return jsonify({
                "status":  "error",
                "message": f"Could not reach JustTCG: {exc.reason}",
            }), 502
        except (json.JSONDecodeError, ValueError) as exc:
            return jsonify({
                "status":  "error",
                "message": f"Unexpected response from JustTCG: {exc}",
            }), 502

        products = api_data.get("data") or []
        if not products:
            return jsonify({
                "status":   "not_found",
                "message":  f'No results found on JustTCG for "{card_name}".',
                "searched": {"name": card_name, "number": set_number, "game": game},
            })

        hit = products[0]
        if set_number:
            for card in products:
                if str(card.get("number", "")).strip().lower() == set_number.lower():
                    hit = card
                    break

    entry = _build_entry_from_hit(hit, card_name, set_number)

    updated = dict(record.extracted_data or {})
    updated["tcgplayer"] = entry
    record.extracted_data = updated
    db.session.commit()

    return jsonify({
        "status":  "success",
        "message": f'Price data saved for "{entry["product_name"]}".',
        "entry":   entry,
    })


@app.route("/tcg_save_url/<int:record_id>", methods=["POST"])
def tcg_save_url(record_id):
    """
    Legacy compatibility shim — kept so the copy-from feature (which may copy
    a pricing URL from an older record) still works without errors.
    Saves a plain URL entry under extracted_data['tcgplayer'].
    """
    record = ScanRecord.query.get_or_404(record_id)
    data   = request.get_json() or {}
    url    = data.get("url", "").strip()

    if not url:
        return jsonify({"status": "error", "message": "No URL provided"}), 400

    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url

    entry = {
        "url":      url,
        "full_url": url,
        "saved_at": datetime.utcnow().isoformat(),
        "source":   "manual",
    }

    updated = dict(record.extracted_data or {})
    updated["tcgplayer"] = entry
    record.extracted_data = updated
    db.session.commit()

    return jsonify({"status": "success", "message": "URL saved", "url": url})


@app.route("/tcg_clear_url/<int:record_id>", methods=["POST"])
def tcg_clear_url(record_id):
    """Remove the saved JustTCG / pricing data from this record."""
    record  = ScanRecord.query.get_or_404(record_id)
    updated = dict(record.extracted_data or {})
    updated.pop("tcgplayer", None)
    record.extracted_data = updated
    db.session.commit()
    return jsonify({"status": "success", "message": "Pricing data removed"})


# ====================== OCR IDENTIFICATION ROUTES ======================
@app.route("/ocr_identify/<int:record_id>", methods=["GET"])
def ocr_identify(record_id):
    """
    OCR the front image of this record (top band -> name, bottom band -> N/M
    collector number) and rank existing records by how well they match. Saves
    nothing — the UI uses this to present a picker.

    Query params:
      catalog=1   -> match only against imported catalog (reference) rows.

    Response shape:
      { status, ocr: { name_guess, number_guess, set_code_guess,
                       conf_top, conf_bottom, raw_top, raw_bottom },
        candidates: [ { record_id, name, serial, set, game, thumbnail,
                        score, serial_match, name_similarity } ] }
    """
    record = ScanRecord.query.get_or_404(record_id)

    if card_ocr is None:
        return jsonify({
            "status":  "error",
            "message": "OCR is unavailable: install it with `pip install pytesseract` "
                       "and the tesseract-ocr binary on the host.",
        }), 503

    abs_path = _abs_record_image_path(record.image_path)
    if not abs_path or not os.path.exists(abs_path):
        return jsonify({
            "status":  "error",
            "message": "This record has no readable front image to OCR.",
        }), 400

    try:
        _g = (record.extracted_data or {}).get("game", "")
        ocr = card_ocr.ocr_card_front(abs_path, game=_g,
                                      type_refs=_load_prepared_type_refs(_g))
    except Exception as exc:  # never let OCR crash the request
        return jsonify({"status": "error", "message": f"OCR failed: {exc}"}), 500

    if not ocr.get("ocr_available"):
        return jsonify({
            "status":  "error",
            "message": "The tesseract-ocr binary was not found on the host. "
                       "Install it (e.g. `apt install tesseract-ocr`) and retry.",
        }), 503

    catalog_only = request.args.get("catalog", "").strip().lower() in ("1", "true", "yes")
    candidates = _build_ocr_candidates(exclude_record_id=record.id, catalog_only=catalog_only)
    matches = card_ocr.match_ocr_to_records(ocr, candidates)
    for m in matches:
        m["source"] = "record"

    # Also match against the downloaded tcgcsv catalog for this record's game,
    # if that game has been synced. Reference matches carry rich fields the UI
    # can auto-fill (set, rarity, TCGplayer URL, ...).
    ext = record.extracted_data or {}
    category_id, ref_game = _resolve_category_for_game(ext.get("game", ""))
    ref_matches = _reference_candidates_for_ocr(category_id, ocr) if category_id else []

    # Combine, sorted best-first, so number-matched cards float to the top
    # regardless of which source they came from.
    combined = sorted(ref_matches + matches, key=lambda c: c.get("score", 0), reverse=True)

    return jsonify({
        "status": "ok",
        "ocr": {
            "name_guess":     ocr.get("name_guess", ""),
            "number_guess":   ocr.get("number_guess", ""),
            "set_code_guess": ocr.get("set_code_guess", ""),
            "type_guess":     ocr.get("type_guess", ""),
            "type_confidence": ocr.get("type_confidence", 0.0),
            "conf_top":       ocr.get("conf_top", -1.0),
            "conf_bottom":    ocr.get("conf_bottom", -1.0),
            "raw_top":        ocr.get("raw_top", ""),
            "raw_bottom":     ocr.get("raw_bottom", ""),
        },
        "reference": {
            "game":         ref_game or ext.get("game", ""),
            "synced":       bool(category_id),
            "match_count":  len(ref_matches),
        },
        # Whether this record already has an identity (name or set number). The
        # UI uses this to decide between auto-applying a confident match on a
        # fresh record vs. opening the picker for manual correction.
        "already_populated": bool(_get_name(ext) or _get_serial(ext)),
        "candidates": combined,
    })


@app.route("/ocr_apply/<int:record_id>", methods=["POST"])
def ocr_apply(record_id):
    """
    Commit an OCR result onto this record. Body is JSON, one of:

      { "reference_product_id": <id> } -> fill fields from a tcgcsv reference
                                          card (name/number/set/rarity/game +
                                          TCGplayer URL under 'tcgplayer').
      { "source_record_id": <id> }     -> copy identity fields from a matched
                                          existing record.
      { "name": "...", "number": "..." }  -> write the raw OCR reading.

    Merges into extracted_data, then re-runs the Product match. Mirrors
    /update_scan semantics.
    """
    record = ScanRecord.query.get_or_404(record_id)
    body   = request.get_json() or {}
    updates = {}

    ref_pid = body.get("reference_product_id")
    source_id = body.get("source_record_id")

    if ref_pid:
        try:
            ref = ReferenceCard.query.filter_by(product_id=int(ref_pid)).first()
        except (TypeError, ValueError):
            ref = None
        if ref is None:
            return jsonify({"status": "error", "message": "reference_product_id not found."}), 404
        for key, attr in _REFERENCE_APPLY_MAP.items():
            val = getattr(ref, attr, None)
            if str(val or "").strip():
                updates[key] = val
        # Preserve the TCGplayer link + market price the way JustTCG data is stored,
        # so the price panel and export keep working.
        if ref.url:
            updates["tcgplayer"] = {
                "url":          ref.url,
                "full_url":     ref.url,
                "source":       "tcgcsv",
                "saved_at":     datetime.utcnow().isoformat(),
                "product_id":   str(ref.product_id),
                "product_name": ref.name or "",
                "set_name":     ref.set_name or "",
                "set_number":   ref.number or "",
                "prices":       {"market": ref.market_price} if ref.market_price is not None else {},
            }
        if getattr(ref, "market_price", None) is not None and \
           not str((record.extracted_data or {}).get("current_value") or "").strip():
            updates["current_value"] = ref.market_price
    elif source_id:
        try:
            source = ScanRecord.query.get(int(source_id))
        except (TypeError, ValueError):
            source = None
        if source is None:
            return jsonify({"status": "error", "message": "source_record_id not found."}), 404
        sdata = source.extracted_data or {}
        for k in _OCR_COPY_KEYS:
            val = sdata.get(k)
            if str(val or "").strip():
                updates[k] = val
    else:
        name   = str(body.get("name", "")).strip()
        number = str(body.get("number", "")).strip()
        if name:
            updates["name"] = name
        if number:
            # pad to the CSV convention (N to M's digit width) so it lines up
            # with the reference catalog's format, e.g. 24/112 -> 024/112.
            updates["set_number"] = _canonical_collector_number(number)

    if not updates:
        return jsonify({
            "status":  "error",
            "message": "Nothing to apply — provide source_record_id, or a name/number.",
        }), 400

    merged = {**(record.extracted_data or {}), **updates}
    record.extracted_data = merged

    matched = match_product_from_extracted(merged)
    if matched:
        record.matched_product_id = matched.id
    db.session.commit()

    return jsonify({
        "status":         "success",
        "message":        "Identification applied.",
        "applied":        updates,
        "extracted_data": merged,
    })


# ====================== TCGCSV REFERENCE-CATALOG ROUTES ======================
# Small in-memory cache for the categories list so the picker doesn't re-hit
# tcgcsv every time it's opened (their data only changes once a day).
_REF_CATEGORIES_CACHE = {"at": None, "data": None}


def _reference_recount(category_id):
    """Recompute cached counts for a category's ReferenceSync row."""
    rs = ReferenceSync.query.filter_by(category_id=category_id).first()
    if rs is None:
        return None
    rs.product_count = ReferenceCard.query.filter_by(category_id=category_id).count()
    rs.group_count = (ReferenceCard.query
                      .filter_by(category_id=category_id)
                      .with_entities(ReferenceCard.group_id).distinct().count())
    return rs


@app.route("/reference/status")
def reference_status():
    """Which games have been downloaded, with counts and last-sync time."""
    syncs = ReferenceSync.query.order_by(ReferenceSync.game).all()
    return jsonify({
        "status": "ok",
        "available": ref_sync is not None,
        "games": [{
            "category_id":   s.category_id,
            "game":          s.game,
            "product_count": s.product_count,
            "group_count":   s.group_count,
            "last_synced":   s.last_synced.isoformat() if s.last_synced else None,
            "remote_updated": s.remote_updated,
            "status":        s.status,
        } for s in syncs],
    })


@app.route("/reference/categories")
def reference_categories():
    """List the games (categories) available on tcgcsv, for the sync picker."""
    if ref_sync is None:
        return jsonify({"status": "error", "message": "Reference sync source unavailable."}), 503

    # Serve from cache if fetched within the last 30 minutes.
    now = datetime.utcnow()
    cached = _REF_CATEGORIES_CACHE
    if cached["data"] and cached["at"] and (now - cached["at"]).total_seconds() < 1800:
        cats = cached["data"]
    else:
        try:
            raw = ref_sync.get_categories()
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Could not reach reference source: {exc}"}), 502
        cats = sorted(
            [{"category_id": c.get("categoryId"),
              "name": c.get("displayName") or c.get("name") or "",
              "popularity": c.get("popularity", 0)} for c in raw],
            key=lambda c: c.get("popularity", 0), reverse=True,
        )
        _REF_CATEGORIES_CACHE["data"] = cats
        _REF_CATEGORIES_CACHE["at"] = now

    return jsonify({"status": "ok", "categories": cats,
                    "last_updated": ref_sync.get_last_updated()})


@app.route("/reference/groups/<int:category_id>")
def reference_groups(category_id):
    """List a category's groups (sets) — the work items the client loops over."""
    if ref_sync is None:
        return jsonify({"status": "error", "message": "Reference sync source unavailable."}), 503
    try:
        groups = ref_sync.get_groups(category_id)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Could not reach reference source: {exc}"}), 502
    return jsonify({
        "status": "ok",
        "groups": [{"group_id": g.get("groupId"), "name": g.get("name") or ""} for g in groups],
    })


@app.route("/reference/sync_group", methods=["POST"])
def reference_sync_group():
    """
    Download ONE group's cards from tcgcsv and upsert them into ReferenceCard.
    The client loops this over every group (mirroring the grade/price batch
    pattern) so progress + Stop work naturally and rate-limiting is inherent.

    Body: { category_id, category_name, group_id, group_name }
    """
    if ref_sync is None:
        return jsonify({"status": "error", "message": "Reference sync source unavailable."}), 503

    body = request.get_json() or {}
    try:
        category_id = int(body.get("category_id"))
        group_id    = int(body.get("group_id"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "category_id and group_id are required."}), 400
    category_name = str(body.get("category_name") or "").strip()
    group_name    = str(body.get("group_name") or "").strip()

    try:
        cards = ref_sync.fetch_group_cards(category_id, category_name, group_id, group_name)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Reference fetch failed: {exc}"}), 502

    for rec in cards:
        _reference_upsert(rec)

    # Ensure a ReferenceSync row exists and refresh its counts.
    rs = ReferenceSync.query.filter_by(category_id=category_id).first()
    if rs is None:
        rs = ReferenceSync(category_id=category_id)
        db.session.add(rs)
    rs.game = category_name or rs.game
    rs.status = "ok"
    rs.remote_updated = ref_sync.get_last_updated() or rs.remote_updated
    db.session.flush()
    _reference_recount(category_id)
    db.session.commit()

    return jsonify({
        "status": "ok",
        "added": len(cards),
        "group_name": group_name,
        "product_count": rs.product_count,
    })


@app.route("/reference/clear", methods=["POST"])
def reference_clear():
    """Delete all cached reference cards for a game (category_id in body)."""
    body = request.get_json() or {}
    try:
        category_id = int(body.get("category_id"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "category_id is required."}), 400

    deleted = ReferenceCard.query.filter_by(category_id=category_id).delete()
    ReferenceSync.query.filter_by(category_id=category_id).delete()
    db.session.commit()
    return jsonify({"status": "success", "deleted": deleted})


# ====================== TYPE-ICON LIBRARY ROUTES ======================
# A per-game library of labelled "type" icons (Pokemon energy, Yu-Gi-Oh
# attribute, ...). Populate it by uploading an icon image or capturing one from a
# scanned card; type detection then matches a card's icon against this library.
def _known_games_for_types():
    """Distinct games seen in inventory + defined templates, for the picker."""
    games = set()
    for rec in ScanRecord.query.all():
        g = str((rec.extracted_data or {}).get("game") or "").strip()
        if g and not _is_catalog_only(rec.extracted_data or {}):
            games.add(g)
    for t in get_template_names():
        games.add(t.replace("_", " "))
    return sorted(games, key=str.lower)


def _save_type_reference_image(icon_bgr, game_key, type_name, region):
    """Persist an icon crop under type_refs/<game>/<region>/<type>/ and return
    its upload-relative path."""
    ensure_dirs()
    safe_game = secure_filename(game_key) or "game"
    safe_type = secure_filename(type_name) or "type"
    safe_region = secure_filename(region) or "top_right"
    folder = os.path.join(app.config["TYPE_REF_FOLDER"], safe_game, safe_region, safe_type)
    os.makedirs(folder, exist_ok=True)
    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    abs_path = os.path.join(folder, fname)
    cv2.imwrite(abs_path, icon_bgr)
    return normalize_to_upload_relative(
        os.path.join("type_refs", safe_game, safe_region, safe_type, fname))


@app.route("/types")
def type_references_page():
    """Management page for the type-icon library."""
    rows = TypeReference.query.order_by(TypeReference.game, TypeReference.type_name,
                                        TypeReference.id).all()
    library = {}
    for r in rows:
        library.setdefault(r.game, {}).setdefault(r.type_name, []).append({
            "id": r.id,
            "url": build_uploaded_file_url(r.image_path),
            "source": r.source,
            "region": r.region or "top_right",
        })
    return render_template("type_references.html",
                           library=library,
                           games=_known_games_for_types())


@app.route("/types/add", methods=["POST"])
def type_reference_add():
    """
    Add a reference image. Two ways:
      • upload an image        (form 'image')
      • capture from a card    (form 'record_id' — crops the marker from its front)
    Common form fields: game (required), type_name (required),
                        region ('top_left' | 'top_right' | 'header', default top_right).
    Corner regions capture a small icon; the 'header' region captures the whole
    top band (for card kinds whose full header design differs).
    """
    ensure_dirs()
    game = (request.form.get("game") or "").strip()
    type_name = (request.form.get("type_name") or "").strip()
    region = card_ocr.normalize_type_region(request.form.get("region")) if card_ocr \
        else (request.form.get("region") or "top_right")
    if not game or not type_name:
        return jsonify({"status": "error", "message": "Game and type name are required."}), 400

    mode = card_ocr.region_mode(region) if card_ocr else "icon"
    game_key = _type_game_key(game)
    icon = None
    source = "upload"

    up = request.files.get("image")
    record_id = (request.form.get("record_id") or "").strip()

    if up and up.filename:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "message": "Could not read the uploaded image."}), 400
        if mode == "band":
            # A full-card upload -> crop the header band; a header strip -> as-is.
            h, w = img.shape[:2]
            if h > w and card_ocr:
                crop = card_ocr._extract_type_region(img, region)
                icon = crop if crop is not None else img
            else:
                icon = img
        else:
            # A whole card -> crop the corner icon; a tight icon -> as-is.
            icon = card_ocr._extract_type_region(img, region) if card_ocr else None
            if icon is None:
                icon = img
    elif record_id:
        try:
            record = ScanRecord.query.get(int(record_id))
        except (TypeError, ValueError):
            record = None
        if record is None:
            return jsonify({"status": "error", "message": f"No record #{record_id}."}), 404
        abs_path = _abs_record_image_path(record.image_path)
        if not abs_path or not os.path.exists(abs_path) or card_ocr is None:
            return jsonify({"status": "error", "message": "That card has no readable front image."}), 400
        card = cv2.imread(abs_path)
        if card is not None:
            card, _ = card_ocr.normalize_card_image(card)
        icon = card_ocr._extract_type_region(card, region) if card is not None else None
        if icon is None:
            where = "header" if mode == "band" else region.replace("_", "-")
            return jsonify({"status": "error", "message":
                            f"Couldn't read the {where} of that card. Try a different "
                            "region, or upload an image instead."}), 422
        source = "capture"
    else:
        return jsonify({"status": "error", "message": "Provide an icon image or a card to capture from."}), 400

    rel = _save_type_reference_image(icon, game_key, type_name, region)
    row = TypeReference(game=game_key, type_name=type_name, region=region,
                        image_path=rel, source=source)
    db.session.add(row)
    db.session.commit()
    _bump_type_refs()
    return jsonify({
        "status": "success",
        "reference": {"id": row.id, "game": game_key, "type_name": type_name,
                      "region": region, "url": build_uploaded_file_url(rel),
                      "source": source},
    })


@app.route("/types/delete/<int:ref_id>", methods=["POST"])
def type_reference_delete(ref_id):
    row = TypeReference.query.get_or_404(ref_id)
    remove_file_if_exists(row.image_path)
    db.session.delete(row)
    db.session.commit()
    _bump_type_refs()
    return jsonify({"status": "success", "deleted": ref_id})


# ====================== DUPLICATE IMAGE MANAGER ROUTES ======================
@app.route("/duplicates")
def duplicates():
    """
    Group records that share all three of:
      - name         (case-insensitive, first non-empty of _NAME_KEYS)
      - serial       (case-insensitive, first non-empty of _SERIAL_KEYS)
      - edition    (normalised string)

    All three must match. Records missing name or serial are excluded.
    Groups where all members already share the same image_path are also
    excluded — they are already resolved and won't reappear after a reload.
    """
    records = ScanRecord.query.order_by(ScanRecord.scan_date.desc()).all()

    groups_map = {}
    for record in records:
        data     = record.extracted_data or {}
        if _is_catalog_only(data):
            continue
        name     = _get_name(data)
        serial   = _get_serial(data)
        if not name or not serial:
            continue
        edition    = _get_edition(data)
        groups_map.setdefault((name, serial, edition), []).append(record)

    groups        = []
    total_records = 0

    for (name, serial, edition), recs in groups_map.items():
        if len(recs) < 2:
            continue
        # Skip groups where all *effective display* images are already
        # identical (already resolved). We check display_image_path first
        # since that's what the Inventory page actually shows for the
        # group's representative; falling back to image_path covers
        # groups that were never run through the duplicate resolver.
        if len(set((r.display_image_path or r.image_path) for r in recs)) <= 1:
            continue

        display_name = (
            recs[0].extracted_data.get("product_name")
            or recs[0].extracted_data.get("name")
            or recs[0].extracted_data.get("card_name")
            or name
        )
        groups.append({
            "name":    display_name,
            "serial":  serial,
            "edition": edition,
            "records": recs,
        })
        total_records += len(recs)

    groups.sort(key=lambda g: str(g["name"]).lower())

    return render_template(
        "duplicates.html",
        groups=groups,
        total_records=total_records,
    )


@app.route("/duplicates/resolve", methods=["POST"])
def duplicates_resolve():
    """
    Resolve one duplicate group.

    Payload: { "canonical_id": int, "record_ids": [int, ...] }

    This ONLY sets display_image_path on every record in the group to the
    canonical record's image_path. That field is purely a display override
    used when a record is chosen as a duplicate group's representative on
    the Inventory page (see build_group_info / inventory.html) — it does
    NOT touch image_path, so each record's own inventory_detail page keeps
    showing the photo it was actually scanned with, and no image files are
    deleted from disk.

    After resolution all records in the group share the same effective
    display image (display_image_path), so the group is filtered out on
    the next /duplicates load.
    """
    data         = request.get_json() or {}
    canonical_id = data.get("canonical_id")
    record_ids   = data.get("record_ids", [])

    if not canonical_id or not isinstance(record_ids, list) or len(record_ids) < 2:
        return jsonify({
            "status":  "error",
            "message": "canonical_id and at least 2 record_ids are required",
        }), 400

    canonical = ScanRecord.query.get(canonical_id)
    if not canonical:
        return jsonify({"status": "error", "message": f"Canonical record #{canonical_id} not found"}), 404

    canonical_path = canonical.image_path
    records        = ScanRecord.query.filter(ScanRecord.id.in_(record_ids)).all()

    if not records:
        return jsonify({"status": "error", "message": "No matching records found"}), 404

    updated = 0
    for record in records:
        if record.display_image_path != canonical_path:
            record.display_image_path = canonical_path
            updated += 1

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Database error: {exc}"}), 500

    return jsonify({
        "status":  "success",
        "message": (
            f"Image from Record #{canonical_id} will now represent {updated} record(s) "
            f"on the Inventory page. Each record's own detail page is unchanged."
        ),
    })


# ====================== IMPORT SPLIT / ALIGN ROUTES ======================
@app.route("/pdf_extract_pages", methods=["POST"])
def pdf_extract_pages():
    """
    Rasterizes every page of an uploaded PDF into a PNG image so the existing
    single-image 9-pocket splitter can process it. Pages come back in order;
    the front-end pairs them up two at a time (page 1 = front, page 2 = back,
    page 3 = front, page 4 = back, ...) and feeds them into /run_import_split
    one after another.
    """
    ensure_dirs()

    if fitz is None:
        return jsonify({
            "status": "error",
            "message": "PDF support isn't installed on the server. Run: pip install PyMuPDF"
        }), 500

    file = request.files.get("pdf")
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No PDF provided"}), 400

    _, ext = os.path.splitext(file.filename)
    if ext.lower() != ".pdf":
        return jsonify({"status": "error", "message": "File must be a .pdf"}), 400

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pdf_path = os.path.join(app.config["TEMP_PDF_FOLDER"], f"upload_{batch_id}.pdf")
    file.save(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        try:
            if doc.page_count < 1:
                return jsonify({"status": "error", "message": "PDF has no pages"}), 400

            # Render each page at its embedded scan's NATIVE resolution (clamped
            # between PDF_RASTER_ZOOM and PDF_RASTER_ZOOM_MAX) so scanned cards
            # keep full scanner detail instead of being re-sampled to a fixed DPI.
            # One page is rendered and saved at a time, so peak memory is a single
            # page's pixmap.
            pages = []
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=_pdf_render_matrix(page))
                page_filename = f"pdfpage_{batch_id}_{page_index + 1:03d}.png"
                page_path = os.path.join(app.config["TEMP_PDF_FOLDER"], page_filename)
                pix.save(page_path)
                del pix, page

                pages.append({
                    "index":    page_index + 1,
                    "filename": page_filename,
                    "url":      url_for("temp_pdf_file", filename=page_filename),
                })
        finally:
            doc.close()
    except Exception as e:
        return jsonify({"status": "error", "message": f"Could not read PDF: {e}"}), 500
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass

    return jsonify({
        "status":  "success",
        "message": f"Extracted {len(pages)} page(s) from PDF.",
        "pages":   pages,
    })


@app.route("/run_import_split", methods=["POST"])
def run_import_split():
    ensure_dirs()

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No image provided"}), 400

    game  = request.form.get("game",  "").strip()
    album = request.form.get("album", "").strip()
    page  = request.form.get("page",  "").strip()
    side  = request.form.get("side",  "front").strip().lower()
    edge_type = normalize_card_edge_type(request.form.get("card_edge_type"))

    if side not in ("front", "back"):
        side = "front"

    if not game or not album or not page:
        return jsonify({"status": "error", "message": "Game, album, and page are required"}), 400

    try:
        v_cut1 = float(request.form.get("vcut1", 33))
        v_cut2 = float(request.form.get("vcut2", 66))
        h_cut1 = float(request.form.get("hcut1", 33))
        h_cut2 = float(request.form.get("hcut2", 66))
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid numeric form values"}), 400

    if not (v_cut1 < v_cut2 and h_cut1 < h_cut2):
        return jsonify({"status": "error", "message": "Cut 1 must be less than Cut 2"}), 400

    # Auto-import mode: run the whole pipeline automatically (cut -> align/crop
    # -> identify -> file to inventory) with no per-tile review. It uses the
    # SAME align/crop as manual mode; the only difference here is that on a hard
    # alignment failure we still file the raw cut, so all 9 cards reach inventory
    # unattended rather than waiting for manual adjustment.
    auto_import = request.form.get("auto_import", "").strip().lower() in ("1", "true", "yes", "on")

    safe_game  = secure_filename(game)  or "game"
    safe_album = secure_filename(album) or "album"
    safe_page  = secure_filename(page)  or "page"

    base_name      = f"{safe_game}_{safe_album}_{safe_page}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    import_filename = f"{base_name}_page.png"
    import_path     = os.path.join(app.config["TEMP_IMPORT_FOLDER"], import_filename)
    file.save(import_path)

    try:
        pil_image = Image.open(import_path).convert("RGB")
        pieces    = split_image_3x3(pil_image, v_cut1, v_cut2, h_cut1, h_cut2)

        results = []
        for photographed_idx, piece in enumerate(pieces, start=1):
            slot_num      = resolve_slot_number(photographed_idx, side)
            slot_filename = f"{safe_game}-{safe_album}-{safe_page}-{slot_num}-{side}.png"
            split_path    = os.path.join(app.config["TEMP_SPLIT_FOLDER"], slot_filename)
            piece.save(split_path)

            try:
                split_cv = cv2.imread(split_path)
                if split_cv is None:
                    raise ValueError("Could not read split image")

                # Detect the card in this tile and crop tightly to it, using the
                # corner strategy for the selected edge type. If no card outline
                # is found, fall back to a slight deskew so the tile is still
                # usable (and can be Manual-Adjusted afterwards).
                cropped, ok = detect_and_crop_card(split_cv, edge_type)
                processed = cropped if ok else straighten_split_image(split_cv)

                card_path = os.path.join(app.config["TEMP_CARD_FOLDER"], slot_filename)
                cv2.imwrite(card_path, processed)

                results.append({
                    "slot": slot_num, "filename": slot_filename,
                    "status": "processed",
                    "url": url_for("temp_card_file", filename=slot_filename),
                })
            except Exception:
                # Alignment couldn't run on this tile. In auto-import we still
                # file the raw cut so the card reaches inventory unattended; in
                # manual mode we surface it as a fallback for review.
                if auto_import:
                    try:
                        card_path = os.path.join(app.config["TEMP_CARD_FOLDER"], slot_filename)
                        piece.save(card_path)
                        results.append({
                            "slot": slot_num, "filename": slot_filename,
                            "status": "processed",
                            "url": url_for("temp_card_file", filename=slot_filename),
                        })
                        continue
                    except Exception:
                        pass
                results.append({
                    "slot": slot_num, "filename": slot_filename,
                    "status": "fallback",
                    "url": url_for("temp_split_file", filename=slot_filename),
                })

        processed_count = sum(1 for r in results if r["status"] == "processed")
        fallback_count  = sum(1 for r in results if r["status"] == "fallback")

        if auto_import:
            message = f"Auto-import — {processed_count} card(s) aligned and cut, filing to inventory."
        else:
            message = f"Import completed. {processed_count} auto-processed, {fallback_count} fallback."

        return jsonify({
            "status":  "success",
            "message": message,
            "files":   results,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/manual_process_card", methods=["POST"])
def manual_process_card():
    data   = request.get_json() or {}
    filename = data.get("filename")
    points   = data.get("points", [])

    if not filename:
        return jsonify({"status": "error", "message": "Missing filename"}), 400

    if not isinstance(points, list) or len(points) != 4:
        return jsonify({"status": "error", "message": "Exactly 4 points are required"}), 400

    split_path = os.path.join(app.config["TEMP_SPLIT_FOLDER"], filename)
    if not os.path.exists(split_path):
        return jsonify({"status": "error", "message": f"Fallback image not found: {filename}"}), 404

    image = cv2.imread(split_path)
    if image is None:
        return jsonify({"status": "error", "message": "Could not read fallback image"}), 400

    try:
        pts = np.array([[float(p["x"]), float(p["y"])] for p in points], dtype="float32")
    except Exception:
        return jsonify({"status": "error", "message": "Invalid point format"}), 400

    try:
        warped   = four_point_transform(image, pts)
        warped   = sharpen_image(warped)
        out_path = os.path.join(app.config["TEMP_CARD_FOLDER"], filename)
        cv2.imwrite(out_path, warped)

        return jsonify({
            "status":   "success",
            "message":  "Manual card alignment completed",
            "url":      url_for("temp_card_file", filename=filename),
            "filename": filename,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ====================== BATCH IMPORT ROUTE (SSE streaming) ======================
@app.route("/import_finalize_batch", methods=["POST"])
def import_finalize_batch():
    """
    Streams Server-Sent Events so the browser can show real per-card progress.

    Takes the 9 already-cropped/aligned card image filenames plus the chosen
    Game (template) name, and creates one inventory record per card with all
    of that Game's fields present but blank — ready to be filled in by hand.
    There is no OCR step anymore; this just files the photos away.

    Event types
    -----------
    progress  — one card finished; payload has slot index + result data
    done      — all cards finished; payload has summary + full results list
    error     — fatal setup error before processing began
    """
    from flask import stream_with_context, Response

    data          = request.get_json() or {}
    template_name = data.get("template_name", "product_label")
    filenames     = data.get("filenames", [])
    collection    = _clean_collection(data.get("collection"))

    def sse(event, payload):
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    def generate():
        if not filenames or len(filenames) != 9:
            yield sse("error", {"message": "Exactly 9 aligned card filenames are required"})
            return

        try:
            template = load_template(template_name)
        except Exception as e:
            yield sse("error", {"message": f"Could not load game '{template_name}': {e}"})
            return

        # Refuse if the inventory is already at the cap.
        if _inventory_remaining() <= 0:
            yield sse("error", {"cap_reached": True, "message": UPGRADE_MESSAGE})
            return

        # Blank starting values for every field this Game defines.
        blank_fields = {field_key: "" for field_key in (template.get("fields", {}) or {}).keys()}
        results = []

        for idx, filename in enumerate(filenames, start=1):
            temp_image_path = os.path.join(app.config["TEMP_CARD_FOLDER"], filename)

            if not os.path.exists(temp_image_path):
                result = {"filename": filename, "status": "error", "message": "Aligned card image not found"}
                results.append(result)
                yield sse("progress", {"slot": idx, "total": len(filenames), "result": result})
                continue

            filename_fields = parse_card_filename(filename)
            extracted = {**blank_fields, **filename_fields}
            if extracted.get("game"):
                # Capitalize the stored game (e.g. 'pokemon' -> 'Pokemon'). Done
                # here, before the pocket lookup below, so a normalized front and
                # its back still resolve to the same key and merge (not duplicate).
                extracted["game"] = _normalize_game_name(extracted["game"])
            if collection:
                extracted["collection"] = collection
            side = filename_fields.get("side", "front")

            game  = extracted.get("game",  "")
            album = extracted.get("album", "")
            page  = extracted.get("page",  "")
            slot  = extracted.get("slot",  "")

            try:
                final_relative_image_path = move_temp_card_to_inventory(filename)
                existing = find_existing_record_for_key(game, album, page, slot)

                if existing:
                    # Same physical pocket already has a record (from the other
                    # side, or a re-import) — attach this image to it instead of
                    # creating a duplicate row. Any fields the person has
                    # already filled in by hand are left alone.
                    old_image = existing.image_path_back if side == "back" else existing.image_path
                    if side == "back":
                        existing.image_path_back = final_relative_image_path
                    else:
                        existing.image_path = final_relative_image_path

                    merged = dict(existing.extracted_data or {})
                    for key, val in extracted.items():
                        if not merged.get(key) and val:
                            merged[key] = val
                    existing.extracted_data = merged

                    db.session.commit()

                    if old_image and old_image != "__blank__":
                        remove_file_if_exists(old_image)

                    record          = existing
                    matched_product = existing.matched_product
                else:
                    create_kwargs = dict(
                        template_name=template_name,
                        extracted=extracted,
                    )
                    if side == "back":
                        # No front photo yet — hold its place with the blank
                        # sentinel so the back image still has a record to
                        # live on; the front slot fills in once it's imported.
                        create_kwargs["image_path"]      = "__blank__"
                        create_kwargs["image_path_back"] = final_relative_image_path
                    else:
                        create_kwargs["image_path"] = final_relative_image_path

                    matched_product, record = create_scan_record(**create_kwargs)

                # ── Auto-identify at the end of import (FRONTS ONLY) ──
                # Run the same OCR identification the detail page uses on a
                # 100% match, then apply + save; otherwise leave the entry blank
                # for manual entry later. Only front pages (odd pages) carry the
                # name/serial — even pages are card BACKS and are never OCR
                # name/serial-checked. Also skips if already identified/filled.
                ident = {"identified": False, "reason": "skipped"}
                try:
                    rext = record.extracted_data or {}
                    has_front = record.image_path and record.image_path != "__blank__"
                    if side == "front" and has_front and not (_get_name(rext) or _get_serial(rext)):
                        ident = auto_identify_record(record)
                        if ident.get("identified") or ident.get("type_applied"):
                            db.session.commit()
                            matched_product = record.matched_product or matched_product
                    elif side == "back":
                        ident = {"identified": False, "reason": "back_page_skipped"}
                except Exception:
                    db.session.rollback()
                    ident = {"identified": False, "reason": "error"}

                result = {
                    "filename":        filename,
                    "status":          "success",
                    "record_id":       record.id,
                    "side":            side,
                    "image_url":       build_uploaded_file_url(record.image_path),
                    "image_url_back":  build_uploaded_file_url(record.image_path_back) if record.image_path_back else None,
                    "extracted":       extracted,
                    "matched_product": matched_product.product_name if matched_product else "No match",
                    "identified":      bool(ident.get("identified")),
                    "identified_name": (ident.get("applied") or {}).get("name", "") if ident.get("identified") else "",
                    "card_type":       (ident.get("type_applied") or {}).get("value", ""),
                }
            except InventoryCapError:
                # Hit the cap partway through — stop, keeping everything filed so
                # far, and tell the browser to show the upgrade path.
                yield sse("done", {
                    "status":      "cap_reached",
                    "cap_reached": True,
                    "message":     (f"Import stopped at the {INVENTORY_MAX_RECORDS:,}-entry cap. "
                                    + UPGRADE_MESSAGE),
                    "results":     results,
                })
                return
            except Exception as e:
                result = {"filename": filename, "status": "error", "message": str(e)}

            results.append(result)
            yield sse("progress", {"slot": idx, "total": len(filenames), "result": result})

        success_count = sum(1 for r in results if r["status"] == "success")
        yield sse("done", {
            "status":  "success",
            "message": f"Import completed for {success_count} of {len(filenames)} cards",
            "results": results,
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


# ====================== JUSTTCG MANUAL SEARCH ======================
@app.route("/justtcg_search_manual", methods=["GET"])
def justtcg_search_manual():
    """
    Perform a JustTCG search using a card name and optional game supplied
    directly as query parameters — no inventory record required.

    Query params:
        name  (required)  — card name to search
        game  (optional)  — human-readable game name, mapped to API slug

    Response shape mirrors /justtcg_search/<record_id>:
        { status, candidates: [...], searched: {name, game} }
    """
    card_name = request.args.get("name", "").strip()
    raw_game  = request.args.get("game", "").strip()
    game      = _JUSTTCG_GAME_MAP.get(raw_game.lower(), "")

    if not card_name:
        return jsonify({"status": "error", "message": "Card name is required."}), 400

    try:
        api_data = _justtcg_search(card_name, game=game)
    except urllib.error.HTTPError as exc:
        return jsonify({
            "status":  "error",
            "message": f"JustTCG API returned HTTP {exc.code}: {exc.reason}",
        }), 502
    except urllib.error.URLError as exc:
        return jsonify({
            "status":  "error",
            "message": f"Could not reach JustTCG: {exc.reason}",
        }), 502
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({
            "status":  "error",
            "message": f"Unexpected response from JustTCG: {exc}",
        }), 502

    products = api_data.get("data") or []

    if not products:
        return jsonify({
            "status":   "not_found",
            "message":  f'No results found on JustTCG for "{card_name}".',
            "searched": {"name": card_name, "game": raw_game},
        })

    candidates = []
    for card in products:
        variants  = card.get("variants") or []
        nm_normal = next(
            (v for v in variants
             if v.get("condition") in ("Near Mint", "NM") and v.get("printing") == "Normal"),
            variants[0] if variants else {},
        )
        card_id = card.get("id", "")
        candidates.append({
            "card_id":    card_id,
            "name":       card.get("name", ""),
            "set_name":   card.get("set_name") or card.get("set") or "",
            "set_number": card.get("number") or "",
            "game":       card.get("game") or "",
            "rarity":     card.get("rarity") or "",
            "nm_price":   nm_normal.get("price"),
            "url":        card.get("url") or
                          (f"https://justtcg.com/cards/{card_id}" if card_id else ""),
        })

    return jsonify({
        "status":     "ok",
        "candidates": candidates,
        "searched":   {"name": card_name, "game": raw_game},
    })


# ====================== IMAGE SEARCH ======================
def _orb_descriptors(img_bgr):
    """Return ORB keypoint descriptors for an image (BGR numpy array), or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb  = cv2.ORB_create(nfeatures=500)
    _, desc = orb.detectAndCompute(gray, None)
    return desc


def _orb_descriptors_cached(record):
    """
    ORB descriptors for a record's stored front image, cached to disk so the
    photo search doesn't re-decode every image and recompute features on every
    query. The cache is a small per-record .npy under ORB_CACHE_FOLDER and is
    transparently rebuilt whenever the source image is newer than the cache
    (or the cache is missing). Returns a uint8 (N, 32) array, or None.
    """
    rel = normalize_to_upload_relative(record.image_path)
    if not rel or rel == "__blank__" or rel.startswith(("http://", "https://")):
        return None
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], rel)
    if not os.path.exists(img_path):
        return None

    cache_path = os.path.join(app.config["ORB_CACHE_FOLDER"], f"{record.id}.npy")
    try:
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(img_path):
            desc = np.load(cache_path, allow_pickle=False)
            return desc if getattr(desc, "size", 0) else None
    except Exception:
        pass  # unreadable/stale cache — fall through and recompute

    img = cv2.imread(img_path)
    if img is None:
        return None
    desc = _orb_descriptors(img)
    try:
        os.makedirs(app.config["ORB_CACHE_FOLDER"], exist_ok=True)
        # Store an empty array for "no features" so we still hit the cache next time.
        np.save(cache_path, desc if desc is not None else np.empty((0, 32), dtype=np.uint8))
    except Exception:
        pass
    return desc


# ---------------------------------------------------------------------------- #
# Approximate nearest-neighbour shortlist for photo search (optional).
#
# Brute-force ORB matching is O(n) per search. For large collections we first use
# a cheap per-image "global descriptor" (a normalized 32x32 thumbnail vector) in
# an hnswlib index to shortlist the most visually similar candidates, then run
# the SAME ORB matcher on just that shortlist to produce the final ranking — so
# ranking quality is unchanged, only the candidate set shrinks. Everything here
# is optional and degrades safely: no hnswlib, no/stale index, or a small
# collection all fall back to full brute force. Recent records are always folded
# in so cards added since the last index rebuild are never missed.
# ---------------------------------------------------------------------------- #
ANN_DIM = 32 * 32
ANN_MIN_RECORDS = 3000      # below this, brute force is fast enough — skip ANN
ANN_SHORTLIST   = 400       # candidates pulled from the index per search
ANN_RECENT_TOPUP = 500      # newest records always considered (may post-date index)


def _try_import_hnswlib():
    try:
        import hnswlib
        return hnswlib
    except Exception:
        return None


def _global_descriptor(img_bgr, size=32):
    """Cheap, L2-normalized thumbnail feature vector for coarse similarity."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    small -= float(small.mean())
    norm = float(np.linalg.norm(small))
    if norm > 1e-6:
        small /= norm
    return small


def _global_descriptor_cached(record):
    """Per-record global descriptor, cached next to the ORB cache."""
    rel = normalize_to_upload_relative(record.image_path)
    if not rel or rel == "__blank__" or rel.startswith(("http://", "https://")):
        return None
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], rel)
    if not os.path.exists(img_path):
        return None
    cache_path = os.path.join(app.config["ORB_CACHE_FOLDER"], f"{record.id}.gvec.npy")
    try:
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(img_path):
            v = np.load(cache_path, allow_pickle=False)
            return v if getattr(v, "size", 0) == ANN_DIM else None
    except Exception:
        pass
    img = cv2.imread(img_path)
    if img is None:
        return None
    v = _global_descriptor(img)
    try:
        os.makedirs(app.config["ORB_CACHE_FOLDER"], exist_ok=True)
        np.save(cache_path, v)
    except Exception:
        pass
    return v


def _ann_paths():
    folder = app.config["ORB_CACHE_FOLDER"]
    return os.path.join(folder, "ann_index.bin"), os.path.join(folder, "ann_ids.npy")


def _active_inventory_query():
    """Owned, active records only (exclude archived + catalog)."""
    return ScanRecord.query.filter(
        db.func.coalesce(ScanRecord.is_archived, False) == False,   # noqa: E712
        db.func.coalesce(ScanRecord.is_catalog, False) == False,    # noqa: E712
    )


def rebuild_ann_index():
    """(Re)build the hnswlib global-descriptor index over active records.
    Returns a status dict. Safe no-op (ok=False) if hnswlib isn't installed."""
    hnswlib = _try_import_hnswlib()
    if hnswlib is None:
        return {"ok": False, "message": "hnswlib is not installed (pip install hnswlib)."}

    ensure_dirs()
    ids, vecs = [], []
    for r in _active_inventory_query().all():
        v = _global_descriptor_cached(r)
        if v is not None:
            ids.append(r.id)
            vecs.append(v)
    if not ids:
        return {"ok": False, "message": "No images available to index."}

    mat = np.vstack(vecs).astype(np.float32)
    index = hnswlib.Index(space="cosine", dim=ANN_DIM)
    index.init_index(max_elements=len(ids), ef_construction=200, M=16)
    index.add_items(mat, np.arange(len(ids)))
    index.set_ef(max(64, ANN_SHORTLIST))

    index_path, ids_path = _ann_paths()
    index.save_index(index_path)
    np.save(ids_path, np.array(ids, dtype=np.int64))
    return {"ok": True, "count": len(ids), "message": f"Indexed {len(ids)} images."}


def _ann_candidate_ids(query_vec, k):
    """Return up to k candidate record ids from the ANN index, or None if the
    index is unavailable/unusable (caller then falls back to brute force)."""
    hnswlib = _try_import_hnswlib()
    if hnswlib is None or query_vec is None:
        return None
    index_path, ids_path = _ann_paths()
    if not (os.path.exists(index_path) and os.path.exists(ids_path)):
        return None
    try:
        ids = np.load(ids_path, allow_pickle=False)
        n = int(ids.shape[0])
        if n == 0:
            return None
        index = hnswlib.Index(space="cosine", dim=ANN_DIM)
        index.load_index(index_path, max_elements=n)
        index.set_ef(max(64, min(k * 2, n)))
        labels, _dist = index.knn_query(np.asarray(query_vec, dtype=np.float32), k=min(k, n))
        return [int(ids[lbl]) for lbl in labels[0]]
    except Exception:
        return None


@app.route("/search_by_image/rebuild_index", methods=["POST"])
def search_by_image_rebuild_index():
    result = rebuild_ann_index()
    return jsonify({"status": "success" if result.get("ok") else "error", **result})


def _match_score(desc_query, desc_ref):
    """BFMatcher Hamming ratio-test score: 0.0–1.0 (higher = better match)."""
    if desc_query is None or desc_ref is None:
        return 0.0
    if len(desc_query) < 2 or len(desc_ref) < 2:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(desc_query, desc_ref, k=2)
    except cv2.error:
        return 0.0
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good) / max(len(matches), 1)


@app.route("/search_by_image_page")
def search_by_image_page():
    return render_template("search_by_image.html")


@app.route("/settings")
def settings_page():
    """Settings landing page. The gear icon in the top nav points here; the
    former top-level tabs (Templates, Type Icons, Shops, Search by Image,
    Duplicates) are reached from the sidebar on these pages."""
    return render_template("settings.html")


# ============================================================================ #
# General settings
# ============================================================================ #
def _general_status():
    swap = _system_swap_bytes()
    return {
        "unlimited_native_import": _native_import_unlimited(),
        "capped_dpi": int(PDF_CAPPED_DPI),
        "swap_bytes": swap,
        "swap_gb": round(swap / 1000 ** 3, 1),
        "required_swap_gb": int(REQUIRED_SWAP_BYTES / 1000 ** 3),
        "swap_ok": swap >= REQUIRED_SWAP_BYTES,
        "env_forced": os.environ.get("PDF_UNLIMITED_NATIVE") is not None,
    }


@app.route("/settings/general")
def general_page():
    return render_template("general.html", general=_general_status())


@app.route("/settings/general/native_import", methods=["POST"])
def general_native_import():
    body = request.get_json(silent=True) or request.form
    enabled = str(body.get("enabled", "")).lower() in ("1", "true", "yes", "on")
    try:
        set_native_import_unlimited(enabled)
    except ValueError as exc:
        # Precondition failed (not enough swap) — report it, leave setting off.
        return jsonify({"status": "error", "message": str(exc),
                        "general": _general_status()}), 400
    return jsonify({
        "status": "success",
        "message": ("Unlimited native-resolution import enabled."
                    if enabled else
                    f"Native import capped at {int(PDF_CAPPED_DPI)} DPI."),
        "general": _general_status(),
    })


# ============================================================================ #
# API keys (stored in the DB, editable at runtime)
# ============================================================================ #
def _mask_secret(val):
    if not val:
        return ""
    val = str(val)
    if len(val) <= 8:
        return "•" * len(val)
    return f"{val[:4]}{'•' * 6}{val[-4:]}"


def _api_keys_status():
    out = []
    for spec in KNOWN_API_KEYS:
        val = get_api_key(spec["key"])
        out.append({
            "key": spec["key"],
            "label": spec["label"],
            "description": spec["description"],
            "docs": spec.get("docs", ""),
            "is_set": bool(val),
            "masked": _mask_secret(val),
        })
    return out


@app.route("/settings/api")
def api_keys_page():
    return render_template("api.html", api_keys=_api_keys_status())


@app.route("/settings/api/save", methods=["POST"])
def api_keys_save():
    """Update known API keys. Only fields the user actually typed into are
    changed; a per-key clear flag removes one. Values are never echoed back."""
    payload = request.get_json(silent=True) or request.form
    changed, cleared = [], []
    for spec in KNOWN_API_KEYS:
        name = spec["key"]
        if str(payload.get(f"clear_{name}", "")).lower() in ("1", "true", "on", "yes"):
            delete_setting(name)
            cleared.append(name)
            continue
        new_val = payload.get(name)
        if new_val is None:
            continue
        new_val = str(new_val).strip()
        if new_val:                       # blank input = leave unchanged
            set_setting(name, new_val)
            changed.append(name)
    msg_parts = []
    if changed:
        msg_parts.append(f"Updated {len(changed)} key(s)")
    if cleared:
        msg_parts.append(f"cleared {len(cleared)}")
    message = (", ".join(msg_parts) + ".") if msg_parts else "No changes."
    return jsonify({"status": "success", "message": message,
                    "api_keys": _api_keys_status()})


# ============================================================================ #
# Archiving — move cold records out of the hot working set
# ============================================================================ #
def _set_archived(ids, value):
    """Set archived=value on the given record ids. Writing through
    extracted_data keeps it the source of truth; the mapper event resyncs the
    is_archived column so hot queries (Inventory, photo search) skip these rows."""
    ids = [int(i) for i in ids if str(i).strip().isdigit()]
    if not ids:
        return 0
    changed = 0
    for r in ScanRecord.query.filter(ScanRecord.id.in_(ids)).all():
        data = dict(r.extracted_data or {})
        if bool(_bool_from(data, "archived")) == bool(value):
            continue
        data["archived"] = bool(value)
        r.extracted_data = data   # reassign -> row dirty -> before_update resyncs column
        changed += 1
    if changed:
        db.session.commit()
    return changed


def _archive_ids_from_request():
    body = request.get_json(silent=True) or {}
    ids = body.get("record_ids") or body.get("ids") or []
    if not ids:
        ids = request.form.getlist("record_ids") or request.form.getlist("ids")
    return ids


@app.route("/inventory/archive", methods=["POST"])
def inventory_archive():
    n = _set_archived(_archive_ids_from_request(), True)
    return jsonify({"status": "success", "archived": n,
                    "message": f"Archived {n} record(s)."})


@app.route("/inventory/unarchive", methods=["POST"])
def inventory_unarchive():
    n = _set_archived(_archive_ids_from_request(), False)
    return jsonify({"status": "success", "unarchived": n,
                    "message": f"Restored {n} record(s) from the archive."})


# ---------------------------------------------------------------------------- #
# Embedded target-side artifacts shipped inside every migration bundle so the
# desktop (PostgreSQL) build has a self-contained on-ramp.
# ---------------------------------------------------------------------------- #
POSTGRES_SCHEMA_SQL = r"""-- PostgreSQL schema for the desktop Card Collector build.
-- Range-partitioned scan_records + JSONB + object-storage image keys. Designed
-- for tens/hundreds of millions of rows. Run once before importing.

CREATE TABLE IF NOT EXISTS products (
    id            BIGINT PRIMARY KEY,
    brand         TEXT NOT NULL,
    product_name  TEXT NOT NULL,
    sku           TEXT UNIQUE,
    price         DOUBLE PRECISION,
    stock         INTEGER DEFAULT 0
);

-- Partitioned by scan_date. The partition key must be part of the PK.
CREATE TABLE IF NOT EXISTS scan_records (
    id                    BIGINT NOT NULL,
    image_path            TEXT,
    image_path_back       TEXT,
    display_image_path    TEXT,
    scan_date             TIMESTAMPTZ NOT NULL DEFAULT now(),
    template_used         TEXT,
    extracted_data        JSONB,
    matched_product_id    BIGINT,
    game_key              TEXT,
    album_key             TEXT,
    name_key              TEXT,
    card_type_key         TEXT,
    dup_hash              TEXT,
    is_finalized          BOOLEAN DEFAULT FALSE,
    is_catalog            BOOLEAN DEFAULT FALSE,
    is_archived           BOOLEAN DEFAULT FALSE,
    image_object_key      TEXT,
    image_object_key_back TEXT,
    PRIMARY KEY (id, scan_date)
) PARTITION BY RANGE (scan_date);

-- Catch-all partition; the importer also creates per-year partitions.
CREATE TABLE IF NOT EXISTS scan_records_default PARTITION OF scan_records DEFAULT;

-- Indexes (created on the parent -> propagate to partitions).
CREATE INDEX IF NOT EXISTS idx_scan_extracted_gin ON scan_records USING GIN (extracted_data jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_scan_hot   ON scan_records (game_key, is_catalog, is_archived, scan_date DESC);
CREATE INDEX IF NOT EXISTS idx_scan_dup   ON scan_records (dup_hash);
CREATE INDEX IF NOT EXISTS idx_scan_album ON scan_records (album_key);
CREATE INDEX IF NOT EXISTS idx_scan_name  ON scan_records (name_key);

CREATE TABLE IF NOT EXISTS listings (
    id           BIGINT PRIMARY KEY,
    record_id    BIGINT,
    marketplace  TEXT NOT NULL,
    sku          TEXT,
    external_id  TEXT,
    external_url TEXT,
    title        TEXT,
    price        DOUBLE PRECISION,
    currency     TEXT DEFAULT 'USD',
    quantity     INTEGER DEFAULT 0,
    status       TEXT DEFAULT 'not_listed',
    last_error   TEXT,
    last_synced  TIMESTAMPTZ,
    extra        JSONB
);
CREATE INDEX IF NOT EXISTS idx_listing_record ON listings (record_id);
CREATE INDEX IF NOT EXISTS idx_listing_mkt_status ON listings (marketplace, status);

CREATE TABLE IF NOT EXISTS sale_events (
    id            BIGINT PRIMARY KEY,
    source        TEXT DEFAULT 'tcgplayer',
    order_id      TEXT,
    item_title    TEXT,
    qty           INTEGER DEFAULT 1,
    price         DOUBLE PRECISION,
    record_id     BIGINT,
    status        TEXT DEFAULT 'unmatched',
    detail        TEXT,
    email_subject TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS type_references (
    id         BIGINT PRIMARY KEY,
    game       TEXT NOT NULL,
    type_name  TEXT NOT NULL,
    region     TEXT DEFAULT 'top_right',
    image_path TEXT NOT NULL,
    source     TEXT DEFAULT 'upload',
    note       TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shop_connections (
    id            BIGINT PRIMARY KEY,
    marketplace   TEXT UNIQUE NOT NULL,
    enabled       BOOLEAN DEFAULT FALSE,
    config        JSONB,
    status        TEXT DEFAULT 'disconnected',
    status_detail TEXT,
    connected_at  TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS email_monitors (
    id             BIGINT PRIMARY KEY,
    enabled        BOOLEAN DEFAULT FALSE,
    host TEXT, port INTEGER DEFAULT 993, use_ssl BOOLEAN DEFAULT TRUE,
    username TEXT, password TEXT, folder TEXT DEFAULT 'INBOX',
    sender_filter TEXT, subject_filter TEXT, source TEXT DEFAULT 'tcgplayer',
    mark_seen BOOLEAN DEFAULT TRUE, poll_interval INTEGER DEFAULT 0,
    last_uid BIGINT DEFAULT 0, last_checked TIMESTAMPTZ,
    status TEXT DEFAULT 'disconnected', status_detail TEXT, updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app_settings (
    id BIGINT PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    value TEXT,
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reference_syncs (
    id             BIGINT PRIMARY KEY,
    category_id    BIGINT UNIQUE NOT NULL,
    game           TEXT, product_count INTEGER DEFAULT 0, group_count INTEGER DEFAULT 0,
    remote_updated TEXT, last_synced TIMESTAMPTZ DEFAULT now(),
    status TEXT DEFAULT 'idle', status_detail TEXT
);

CREATE TABLE IF NOT EXISTS reference_cards (
    id BIGINT PRIMARY KEY,
    category_id BIGINT NOT NULL, group_id BIGINT NOT NULL, product_id BIGINT UNIQUE NOT NULL,
    game TEXT, set_name TEXT, name TEXT, clean_name TEXT, number TEXT, rarity TEXT,
    image_url TEXT, url TEXT, market_price DOUBLE PRECISION, extended JSONB, updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_refcard_game_number ON reference_cards (game, number);
"""


POSTGRES_IMPORTER_PY = r'''#!/usr/bin/env python3
"""
import_to_postgres.py — load a Card Collector migration bundle into PostgreSQL.

Usage:
    pip install psycopg2-binary
    python import_to_postgres.py --dsn "postgresql://user:pass@host/db" --bundle .

Optional image sync to object storage (S3/MinIO), rewriting object keys:
    pip install boto3
    python import_to_postgres.py --dsn ... --bundle . \
        --images-dir /path/to/uploads \
        --s3-endpoint http://localhost:9000 --s3-bucket cards \
        --s3-key AKIA... --s3-secret ...

The bundle is the extracted migration folder (contains manifest.json,
schema_postgres.sql, *.jsonl). Rows preserve their original ids.
"""
import argparse, json, os, sys

LOAD_ORDER = [
    "products", "scan_records", "listings", "sale_events", "type_references",
    "shop_connections", "email_monitors", "app_settings", "reference_syncs", "reference_cards",
]
JSONB_COLUMNS = {
    "scan_records": {"extracted_data"}, "listings": {"extra"},
    "shop_connections": {"config"}, "reference_cards": {"extended"},
}


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_year_partitions(cur, bundle):
    """Create per-year partitions of scan_records for the years present."""
    years = set()
    p = os.path.join(bundle, "scan_records.jsonl")
    if not os.path.exists(p):
        return
    for row in iter_jsonl(p):
        sd = (row.get("scan_date") or "")[:4]
        if sd.isdigit():
            years.add(int(sd))
    for y in sorted(years):
        name = f"scan_records_{y}"
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF scan_records "
            f"FOR VALUES FROM ('{y}-01-01') TO ('{y + 1}-01-01')"
        )


def load_table(cur, bundle, table):
    import psycopg2.extras as extras
    path = os.path.join(bundle, table + ".jsonl")
    if not os.path.exists(path):
        print(f"  - {table}: no file, skipped")
        return 0
    rows = list(iter_jsonl(path))
    if not rows:
        print(f"  - {table}: empty")
        return 0
    cols = list(rows[0].keys())
    jsonb = JSONB_COLUMNS.get(table, set())

    def coerce(row):
        vals = []
        for c in cols:
            v = row.get(c)
            if c in jsonb and v is not None and not isinstance(v, str):
                v = json.dumps(v)
            vals.append(v)
        return vals

    collist = ", ".join('"%s"' % c for c in cols)
    template = "(" + ", ".join("%s" for _ in cols) + ")"
    sql = f'INSERT INTO {table} ({collist}) VALUES %s ON CONFLICT DO NOTHING'
    extras.execute_values(cur, sql, [coerce(r) for r in rows], template=template, page_size=1000)
    print(f"  - {table}: {len(rows)} rows")
    return len(rows)


def sync_images(bundle, images_dir, s3):
    manifest = os.path.join(bundle, "images_manifest.jsonl")
    if not (s3 and os.path.exists(manifest)):
        return
    bucket = s3["bucket"]
    import boto3
    client = boto3.client("s3", endpoint_url=s3.get("endpoint"),
                          aws_access_key_id=s3.get("key"), aws_secret_access_key=s3.get("secret"))
    n = 0
    for row in iter_jsonl(manifest):
        for side in ("front", "back"):
            rel, key = row.get(side + "_path"), row.get(side + "_key")
            if rel and key:
                src = os.path.join(images_dir, rel)
                if os.path.exists(src):
                    client.upload_file(src, bucket, key)
                    n += 1
    print(f"  - images uploaded: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--bundle", default=".")
    ap.add_argument("--images-dir")
    ap.add_argument("--s3-endpoint"); ap.add_argument("--s3-bucket")
    ap.add_argument("--s3-key"); ap.add_argument("--s3-secret")
    args = ap.parse_args()

    import psycopg2
    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False
    cur = conn.cursor()

    print("Applying schema...")
    with open(os.path.join(args.bundle, "schema_postgres.sql"), "r", encoding="utf-8") as fh:
        cur.execute(fh.read())

    print("Creating year partitions...")
    ensure_year_partitions(cur, args.bundle)

    print("Loading tables...")
    for table in LOAD_ORDER:
        load_table(cur, args.bundle, table)

    conn.commit()
    print("Analyzing...")
    cur.execute("ANALYZE")
    conn.commit()

    if args.images_dir and args.s3_bucket:
        print("Syncing images to object storage...")
        sync_images(args.bundle, args.images_dir, {
            "endpoint": args.s3_endpoint, "bucket": args.s3_bucket,
            "key": args.s3_key, "secret": args.s3_secret,
        })

    cur.close(); conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
'''


MIGRATION_README_MD = """# Migrating to the desktop (PostgreSQL) build

This bundle is a complete, portable export of your SQLite inventory, ready to
load into the desktop build that uses PostgreSQL + table partitioning + object
storage for images.

## Contents
- `manifest.json` — versions, row counts, capacity, notes.
- `*.jsonl` — one file per table, one JSON object per line (streamed, so large
  tables export without high memory use).
- `images_manifest.jsonl` — maps each record's image file path to a suggested
  object-storage key. Image **bytes are not included** (they can be terabytes);
  sync them from your image folder during import.
- `schema_postgres.sql` — the partitioned PostgreSQL schema.
- `import_to_postgres.py` — loader script.

## Steps
1. Provision PostgreSQL (14+ recommended) on the dedicated machine and create a
   database.
2. (Optional) Stand up object storage (S3 or MinIO) and a bucket for images.
3. Extract this bundle, then run:

       pip install psycopg2-binary
       python import_to_postgres.py --dsn "postgresql://user:pass@host/db" --bundle .

   To also upload images to object storage:

       pip install boto3
       python import_to_postgres.py --dsn "postgresql://user:pass@host/db" --bundle . \\
           --images-dir /path/to/your/uploads \\
           --s3-endpoint http://localhost:9000 --s3-bucket cards \\
           --s3-key KEY --s3-secret SECRET

4. Point the desktop application at the same DSN and bucket.

## Notes
- Record ids are preserved, and the loader uses `ON CONFLICT DO NOTHING`, so it
  is safe to re-run.
- `reference_cards` is rebuildable from tcgcsv.com and is skipped unless you
  chose to include it.
- **This bundle can contain marketplace and mailbox credentials**
  (`shop_connections`, `email_monitors`). Store and transfer it securely.
"""


# ============================================================================ #
# Upgrade / migration workflow to the desktop (PostgreSQL) build
# ============================================================================ #
# The SQLite build is capped (see INVENTORY_MAX_RECORDS). When a collection
# outgrows it, this exports a self-contained migration bundle — every table as
# streamed JSONL (memory-safe), an image manifest (keys, not bytes), plus the
# PostgreSQL schema, an importer script, and a README — for the desktop build
# that uses PostgreSQL + partitioning + object storage.
MIGRATION_BUNDLE_VERSION = "1.0"

# Models included in the export: (filename_stem, ORM class, rebuildable?)
def _migration_tables():
    return [
        ("products",         Product,        False),
        ("scan_records",     ScanRecord,     False),
        ("listings",         Listing,        False),
        ("sale_events",      SaleEvent,      False),
        ("type_references",  TypeReference,  False),
        ("shop_connections", ShopConnection, False),
        ("email_monitors",   EmailMonitor,   False),
        ("app_settings",     AppSetting,     False),
        ("reference_syncs",  ReferenceSync,  False),
        # Large and rebuildable from tcgcsv.com — skipped by default.
        ("reference_cards",  ReferenceCard,  True),
    ]


def _row_to_dict(row):
    """Serialize an ORM row to a JSON-safe dict using its table columns."""
    out = {}
    for col in row.__table__.columns:
        val = getattr(row, col.name)
        if isinstance(val, datetime):
            val = val.isoformat()
        out[col.name] = val
    return out


def _capacity_status():
    mode = _system_mode()
    count = _inventory_count(refresh=True)
    cap = _effective_cap()
    if cap is None:  # Dedicated Server — uncapped
        return {
            "mode": mode, "uncapped": True, "count": count, "limit": None,
            "remaining": None, "percent": 0.0, "at_cap": False, "near_cap": False,
            "is_pi": _is_raspberry_pi(),
        }
    remaining = max(0, cap - count)
    pct = (count / cap * 100.0) if cap else 0.0
    return {
        "mode": mode, "uncapped": False, "count": count, "limit": cap,
        "remaining": remaining, "percent": round(min(pct, 100.0), 2),
        "at_cap": remaining <= 0, "near_cap": pct >= 90.0,
        "is_pi": _is_raspberry_pi(),
    }


def _stream_table_jsonl(model, fh, batch=2000):
    """Write every row of `model` to an open text file as JSON lines, using
    keyset iteration so memory stays flat for very large tables."""
    n = 0
    last_id = 0
    has_int_pk = hasattr(model, "id")
    if has_int_pk:
        while True:
            rows = (model.query.filter(model.id > last_id)
                    .order_by(model.id).limit(batch).all())
            if not rows:
                break
            for r in rows:
                fh.write(json.dumps(_row_to_dict(r), default=str) + "\n")
            n += len(rows)
            last_id = rows[-1].id
            db.session.expunge_all()   # release hydrated objects
    else:
        for r in model.query.all():
            fh.write(json.dumps(_row_to_dict(r), default=str) + "\n")
            n += 1
    return n


def _object_key_for(record_id, relpath, side):
    ext = os.path.splitext(relpath or "")[1] or ".png"
    return f"cards/{record_id}/{side}{ext}"


def _write_images_manifest(fh):
    """One JSON line per record that has local image files, mapping the current
    upload-relative paths to suggested object-storage keys."""
    n = 0
    last_id = 0
    while True:
        rows = (ScanRecord.query.filter(ScanRecord.id > last_id)
                .order_by(ScanRecord.id).limit(2000).all())
        if not rows:
            break
        for r in rows:
            entry = {"record_id": r.id}
            fp = normalize_to_upload_relative(r.image_path)
            bp = normalize_to_upload_relative(r.image_path_back) if r.image_path_back else ""
            wrote = False
            if fp and fp != "__blank__" and not fp.startswith(("http://", "https://")):
                entry["front_path"] = fp
                entry["front_key"] = _object_key_for(r.id, fp, "front")
                wrote = True
            if bp and bp != "__blank__" and not bp.startswith(("http://", "https://")):
                entry["back_path"] = bp
                entry["back_key"] = _object_key_for(r.id, bp, "back")
                wrote = True
            if wrote:
                fh.write(json.dumps(entry, default=str) + "\n")
                n += 1
        last_id = rows[-1].id
        db.session.expunge_all()
    return n


def _build_migration_bundle(include_reference=False, include_images_manifest=True):
    """Assemble the migration bundle as a .tar.gz on disk and return its path."""
    import tarfile
    ensure_dirs()
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    work = tempfile.mkdtemp(prefix="migration_", dir=app.config["TEMP_PDF_FOLDER"])
    bundle_dir = os.path.join(work, f"ccim_migration_{stamp}")
    os.makedirs(bundle_dir, exist_ok=True)

    counts = {}
    for stem, model, rebuildable in _migration_tables():
        if rebuildable and not include_reference:
            counts[stem] = "skipped (rebuildable from tcgcsv.com)"
            continue
        with open(os.path.join(bundle_dir, f"{stem}.jsonl"), "w", encoding="utf-8") as fh:
            counts[stem] = _stream_table_jsonl(model, fh)

    image_count = 0
    if include_images_manifest:
        with open(os.path.join(bundle_dir, "images_manifest.jsonl"), "w", encoding="utf-8") as fh:
            image_count = _write_images_manifest(fh)

    # Static target-side artifacts.
    with open(os.path.join(bundle_dir, "schema_postgres.sql"), "w", encoding="utf-8") as fh:
        fh.write(POSTGRES_SCHEMA_SQL)
    with open(os.path.join(bundle_dir, "import_to_postgres.py"), "w", encoding="utf-8") as fh:
        fh.write(POSTGRES_IMPORTER_PY)
    with open(os.path.join(bundle_dir, "README_MIGRATION.md"), "w", encoding="utf-8") as fh:
        fh.write(MIGRATION_README_MD)

    roots = app.config.get("STORAGE_ROOTS", {})
    manifest = {
        "bundle_version": MIGRATION_BUNDLE_VERSION,
        "created_utc": datetime.utcnow().isoformat(),
        "source": "Card Collector Inventory Manager (SQLite build)",
        "source_cap": INVENTORY_MAX_RECORDS,
        "capacity": _capacity_status(),
        "row_counts": counts,
        "image_manifest_rows": image_count,
        "images_source_dir": roots.get("uploads", ""),
        "reference_cards_included": bool(include_reference),
        "notes": [
            "Load order: products, scan_records, listings, sale_events, type_references, "
            "shop_connections, email_monitors, reference_syncs, reference_cards.",
            "extracted_data is JSON per line; load into a jsonb column.",
            "Image bytes are NOT in this bundle. images_manifest.jsonl maps each record's "
            "current file path to a suggested object-storage key; run the importer's image "
            "sync against images_source_dir.",
            "shop_connections and email_monitors may contain credentials — treat this bundle as a secret.",
        ],
    }
    with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    out_dir = os.path.join(app.config["UPLOAD_FOLDER"], "migration_exports")
    os.makedirs(out_dir, exist_ok=True)
    tar_path = os.path.join(out_dir, f"ccim_migration_{stamp}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=os.path.basename(bundle_dir))
    shutil.rmtree(work, ignore_errors=True)
    return tar_path, manifest


@app.route("/settings/upgrade")
def upgrade_page():
    return render_template("upgrade.html", capacity=_capacity_status())


@app.route("/settings/upgrade/status")
def upgrade_status():
    return jsonify({"status": "success", "capacity": _capacity_status()})


@app.route("/settings/upgrade/export", methods=["POST"])
def upgrade_export():
    include_reference = str(request.form.get("include_reference", "")).lower() in ("1", "true", "yes", "on")
    try:
        tar_path, manifest = _build_migration_bundle(include_reference=include_reference)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Export failed: {exc}"}), 500
    size = os.path.getsize(tar_path)
    return jsonify({
        "status": "success",
        "message": "Migration bundle created.",
        "download_url": url_for("uploaded_file", filename="migration_exports/" + os.path.basename(tar_path)),
        "filename": os.path.basename(tar_path),
        "size_bytes": size,
        "size_human": _human_size(size),
        "manifest": manifest,
    })


# ============================================================================ #
# Reset system — wipe database + storage, return to first-run setup
# ============================================================================ #
RESET_CONFIRM_PHRASE = "RESET"


def _wipe_storage_contents():
    """Delete all managed data files (images, caches, temp, exports) but keep
    the directories. The database itself is emptied separately."""
    roots = app.config.get("STORAGE_ROOTS", {})
    targets = []
    up = roots.get("uploads", "")
    if up:
        targets += [os.path.join(up, s) for s in STORAGE_UPLOAD_SUBDIRS]
        targets += [os.path.join(up, "migration_exports"), os.path.join(up, "game_icons"),
                    os.path.join(up, "album_covers")]
    tmp = roots.get("temp", "")
    if tmp:
        targets += [os.path.join(tmp, s) for s in STORAGE_TEMP_SUBDIRS]
    for path in targets:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass
    ensure_dirs()


@app.route("/settings/reset")
def reset_page():
    return render_template("reset.html", capacity=_capacity_status(),
                           confirm_phrase=RESET_CONFIRM_PHRASE)


@app.route("/settings/reset/confirm", methods=["POST"])
def reset_confirm():
    phrase = (request.form.get("confirm") or (request.get_json(silent=True) or {}).get("confirm") or "").strip()
    if phrase != RESET_CONFIRM_PHRASE:
        return jsonify({"status": "error",
                        "message": f'Type "{RESET_CONFIRM_PHRASE}" to confirm the wipe.'}), 400
    try:
        # Empty every table, then recreate the fresh schema (all columns present).
        db.session.remove()
        db.drop_all()
        db.create_all()
        _inv_count_cache["n"] = 0

        # Delete stored images, caches, temp files, and exports.
        _wipe_storage_contents()

        # Return to first-run: next request will hit the setup gate.
        SYSTEM["mode"] = None
        save_system_config()
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Reset failed: {exc}"}), 500

    return jsonify({"status": "success",
                    "message": "System wiped. Choose an implementation to start over.",
                    "redirect": url_for("setup_page")})


# ============================================================================ #
# Storage settings — relocate sizable files/folders at runtime
# ============================================================================ #
def _dir_size_bytes(path):
    """Total size of everything under `path` (0 if it doesn't exist)."""
    total = 0
    if not path or not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _human_size(n):
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < step:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} PB"


def _slot_owned_paths(slot, root):
    """Concrete paths a slot occupies under `root` (for sizing + migration)."""
    if slot == "uploads":
        return [os.path.join(root, s) for s in STORAGE_UPLOAD_SUBDIRS]
    if slot == "temp":
        return [os.path.join(root, s) for s in STORAGE_TEMP_SUBDIRS]
    if slot == "roi":
        return [root]                       # the ROI root IS the owned folder
    if slot == "db":
        return [root + sfx for sfx in ("", "-wal", "-shm")]
    return []


def _free_bytes(path):
    """Free space on the filesystem that would host `path`."""
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or BASE_DIR).free
    except OSError:
        return 0


STORAGE_SLOT_META = {
    "uploads": ("Image & upload storage",
                "Inventory card images, type-icon library, and import staging."),
    "temp":    ("Temporary working directory",
                "Scratch space for PDF rasterization, splitting, and imports. Safe to place on fast/scratch storage."),
    "roi":     ("ROI template folder",
                "Small per-game region-of-interest template files."),
    "db":      ("Database file (inventory.db)",
                "The SQLite database holding all records, listings, and settings."),
}


def _storage_status():
    """Build the per-slot status list the Storage page renders."""
    roots = app.config.get("STORAGE_ROOTS", {})
    slots = []
    for key in STORAGE_SLOTS:
        root = roots.get(key, "")
        size = sum(_dir_size_bytes(p) for p in _slot_owned_paths(key, root))
        title, desc = STORAGE_SLOT_META[key]
        slots.append({
            "key": key,
            "title": title,
            "description": desc,
            "path": root,
            "is_file": key == "db",
            "size_bytes": size,
            "size_human": _human_size(size),
            "free_human": _human_size(_free_bytes(root)),
            "shared_with_images": key == "temp" and os.path.abspath(root) == os.path.abspath(roots.get("uploads", "")),
        })
    return slots


@app.route("/settings/storage")
def storage_page():
    ensure_dirs()
    return render_template("storage.html", slots=_storage_status(),
                           config_path=STORAGE_CONFIG_PATH)


@app.route("/settings/storage/status")
def storage_status():
    return jsonify({"status": "success", "slots": _storage_status()})


def _migrate_tree(src, dst):
    """Copy directory `src` into `dst` (merging), then remove `src`. No-op if
    src is missing. Raises on copy failure (caller reports it)."""
    if not os.path.exists(src):
        return
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    os.makedirs(os.path.dirname(dst.rstrip(os.sep)) or dst, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    shutil.rmtree(src, ignore_errors=True)


def _move_folder_slot(slot, new_path):
    """Relocate a folder slot (uploads/temp/roi): copy its owned subfolders to
    the new root, delete the originals, then repoint the live + persisted
    config. Applies immediately — every path is read from app.config."""
    roots = app.config.get("STORAGE_ROOTS", {})
    old_root = roots.get(slot, "")
    new_root = _resolve_storage_path(new_path)

    if os.path.abspath(old_root) == os.path.abspath(new_root):
        return {"status": "success", "message": "Location unchanged.", "needs_restart": False}

    os.makedirs(new_root, exist_ok=True)
    if not os.access(new_root, os.W_OK):
        return {"status": "error", "message": f"New location isn't writable: {new_root}"}

    moved = 0
    if slot == "roi":
        # The root itself is the owned folder.
        if os.path.exists(old_root):
            _migrate_tree(old_root, new_root)
            moved += 1
    else:
        subdirs = STORAGE_UPLOAD_SUBDIRS if slot == "uploads" else STORAGE_TEMP_SUBDIRS
        for name in subdirs:
            src = os.path.join(old_root, name)
            if os.path.exists(src):
                _migrate_tree(src, os.path.join(new_root, name))
                moved += 1

    STORAGE[slot] = new_path
    save_storage_config(STORAGE)
    apply_storage_config(STORAGE)
    ensure_dirs()
    return {"status": "success",
            "message": f"Moved {moved} folder(s) to {new_root}. New location is live.",
            "needs_restart": False}


def _move_db_slot(new_path):
    """Relocate the SQLite database. Because the SQLAlchemy engine already holds
    the current file open, the copy is made now and the switch completes on the
    next app start (the old file is removed then, once we're safely running on
    the new one). Reads keep working in the meantime."""
    roots = app.config.get("STORAGE_ROOTS", {})
    old_path = os.path.abspath(roots.get("db", ""))

    new_path_resolved = _resolve_storage_path(new_path)
    # Allow pointing at a directory: keep the inventory.db filename inside it.
    if os.path.isdir(new_path_resolved) or new_path.endswith(("/", os.sep)):
        new_path_resolved = os.path.join(new_path_resolved, "inventory.db")
        new_path = new_path_resolved
    new_abs = os.path.abspath(new_path_resolved)

    if new_abs == old_path:
        return {"status": "success", "message": "Location unchanged.", "needs_restart": False}

    new_dir = os.path.dirname(new_abs) or BASE_DIR
    os.makedirs(new_dir, exist_ok=True)
    if not os.access(new_dir, os.W_OK):
        return {"status": "error", "message": f"New location isn't writable: {new_dir}"}
    if os.path.exists(new_abs):
        return {"status": "error", "message": f"A file already exists there: {new_abs}"}

    # Release file locks so the copy is clean. Checkpoint first so the WAL is
    # folded into the main .db file and the copy is self-consistent.
    try:
        with app.app_context():
            with db.engine.begin() as conn:
                conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass
    try:
        db.session.remove()
    except Exception:
        pass
    try:
        with app.app_context():
            db.engine.dispose()
    except Exception:
        pass

    # Copy the DB and any SQLite sidecar files.
    try:
        for sfx in ("", "-wal", "-shm"):
            s = old_path + sfx
            if os.path.exists(s):
                shutil.copy2(s, new_abs + sfx)
    except OSError as exc:
        return {"status": "error", "message": f"Copy failed: {exc}"}

    # Persist the new location and queue the old files for deletion on next
    # start (safe once we're provably running on the new copy).
    STORAGE["db"] = new_path
    pend = list(STORAGE.get("pending_deletions") or [])
    for sfx in ("", "-wal", "-shm"):
        s = old_path + sfx
        if os.path.exists(s):
            pend.append(s)
    STORAGE["pending_deletions"] = pend
    save_storage_config(STORAGE)
    apply_storage_config(STORAGE)  # updates the URI for the next start

    return {"status": "success", "needs_restart": True,
            "message": (f"Database copied to {new_abs}. Restart the app to switch to it — "
                        "until you restart, changes still write to the old file, which is "
                        "removed automatically on the next start.")}


@app.route("/settings/storage/update", methods=["POST"])
def storage_update():
    slot = (request.form.get("slot") or "").strip()
    new_path = (request.form.get("path") or "").strip()
    if slot not in STORAGE_SLOTS:
        return jsonify({"status": "error", "message": "Unknown storage slot."}), 400
    if not new_path:
        return jsonify({"status": "error", "message": "A new path is required."}), 400

    try:
        if slot == "db":
            result = _move_db_slot(new_path)
        else:
            result = _move_folder_slot(slot, new_path)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Move failed: {exc}"}), 500

    code = 200 if result.get("status") == "success" else 400
    if result.get("status") == "success":
        result["slots"] = _storage_status()
    return jsonify(result), code


def process_pending_deletions():
    """Remove files queued for deletion by a previous DB move, now that we're
    running on the new location. Called once at startup."""
    pend = list(STORAGE.get("pending_deletions") or [])
    if not pend:
        return
    current_db = os.path.abspath(_sqlite_uri_to_path(app.config["SQLALCHEMY_DATABASE_URI"]))
    remaining = []
    for p in pend:
        # Never delete the file we're currently using.
        if os.path.abspath(p) == current_db:
            continue
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            remaining.append(p)  # try again next start
    STORAGE["pending_deletions"] = remaining
    try:
        save_storage_config(STORAGE)
    except OSError:
        pass


@app.route("/search_by_image", methods=["POST"])
def search_by_image():
    """
    Accept a photo of a card, run ORB feature matching against every inventory
    image that has a file on disk, and return the top-N closest records.
    """
    TOP_N = 10

    file = request.files.get("image")
    if not file:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    np_buf    = np.frombuffer(file.read(), np.uint8)
    query_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if query_img is None:
        return jsonify({"status": "error", "message": "Could not decode image"}), 400

    # Try to auto-align the card the same way the import pipeline does.
    # Falls back to the raw image if alignment fails (e.g. card already cropped).
    try:
        query_img = process_card_image(query_img)
    except Exception:
        pass

    query_desc = _orb_descriptors(query_img)

    # Owned, active inventory only — skip archived (cold) and catalog rows.
    active_q = _active_inventory_query()

    # For large collections, shortlist visually-similar candidates via the ANN
    # index instead of scanning everything. Falls back to the full set when the
    # index is unavailable or the collection is small. Newly-added records (which
    # may post-date the last index build) are always folded in so nothing is
    # missed, and the final ranking still comes from the ORB matcher below.
    records = None
    try:
        total_active = active_q.count()
        if total_active >= ANN_MIN_RECORDS:
            cand_ids = _ann_candidate_ids(_global_descriptor(query_img), ANN_SHORTLIST)
            if cand_ids:
                id_set = set(cand_ids)
                recent = (active_q.order_by(ScanRecord.scan_date.desc())
                          .limit(ANN_RECENT_TOPUP).with_entities(ScanRecord.id).all())
                id_set.update(rid for (rid,) in recent)
                records = active_q.filter(ScanRecord.id.in_(id_set)).all()
    except Exception:
        records = None  # any trouble -> brute force

    if records is None:
        records = active_q.all()
    scored  = []

    for record in records:
        if not record.image_path or record.image_path == "__blank__":
            continue
        # Cached descriptors: computed once per image and reused across searches,
        # so this loop no longer re-reads and re-featurizes every card each time.
        ref_desc = _orb_descriptors_cached(record)
        if ref_desc is None:
            continue
        score = _match_score(query_desc, ref_desc)
        if score > 0:
            scored.append((score, record))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, record in scored[:TOP_N]:
        data = record.extracted_data or {}
        name = (
            data.get("product_name")
            or data.get("name")
            or data.get("card_name")
            or data.get("title")
            or f"Record #{record.id}"
        )
        results.append({
            "record_id":  record.id,
            "name":       name,
            "score":      round(score, 4),
            "game":       data.get("game", ""),
            "album":      data.get("album", ""),
            "page":       data.get("page", ""),
            "slot":       data.get("slot", ""),
            "image_url":  build_uploaded_file_url(record.image_path),
            "detail_url": url_for("inventory_detail", record_id=record.id),
        })

    return jsonify({"status": "success", "results": results})


# ====================== XIMILAR CARD GRADING (fast "condition" endpoint) ======================
#
# Ximilar's Card Grading service is asynchronous-only: you POST a job to
#   https://api.ximilar.com/account/v2/request/
# then poll
#   https://api.ximilar.com/account/v2/request/<id>
# until status == "DONE" and read response.records.
#
# Our card images live on local disk (served from /uploads/...), which Ximilar's
# workers can't reach over the network, so local files are sent as `_base64`.
# Images that were imported as external URLs are passed straight through as `_url`.
import base64
import time

XIMILAR_API_TOKEN       = None  # read at call time via get_api_key("XIMILAR_API_TOKEN")
XIMILAR_REQUEST_URL     = "https://api.ximilar.com/account/v2/request/"
XIMILAR_CONNECT_TIMEOUT = 40      # seconds per individual HTTP call
XIMILAR_POLL_INTERVAL   = 2.0     # seconds between status polls
XIMILAR_POLL_MAX_WAIT   = 120     # seconds to wait for a single job to finish

# Naming schemes accepted by the /condition endpoint's `mode` field.
XIMILAR_CONDITION_MODES = {"ebay", "tcgplayer", "cardmarket", "ximilar"}


def _image_source_for_grading(path_value, max_side=1600, jpeg_quality=90):
    """
    Turn a stored image_path into a Ximilar record source dict.

    Returns:
      {"_url": "https://..."}  for images stored as external URLs (Ximilar fetches them)
      {"_base64": "..."}       for local files (downscaled + JPEG-encoded from disk)
      None                     when there is no usable image on this side

    Local files are downscaled to `max_side` px on the long edge and re-encoded as
    JPEG before base64 so the request body stays small. With native-resolution
    imports a full-size page can be tens of MB; sending that raw overruns Ximilar's
    request-size limit / upload timeout and surfaces as a 502. The condition model
    downsamples internally, so ~1600 px is ample and keeps the upload fast.
    """
    relative = normalize_to_upload_relative(path_value)
    if not relative or relative == "__blank__":
        return None

    if relative.startswith("http://") or relative.startswith("https://"):
        return {"_url": relative}

    abs_path = os.path.join(app.config["UPLOAD_FOLDER"], relative)
    if not os.path.exists(abs_path):
        return None

    # Preferred path: decode, downscale, JPEG-encode (small payload).
    try:
        img = cv2.imread(abs_path)
        if img is not None:
            h, w = img.shape[:2]
            scale = max_side / float(max(h, w)) if max(h, w) > max_side else 1.0
            if scale < 1.0:
                img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if ok:
                return {"_base64": base64.b64encode(buf.tobytes()).decode("ascii")}
    except Exception:
        pass

    # Fallback: send the raw file bytes (e.g. an exotic format cv2 can't decode).
    try:
        with open(abs_path, "rb") as fh:
            return {"_base64": base64.b64encode(fh.read()).decode("ascii")}
    except OSError:
        return None


def _ximilar_http(method, url, payload=None):
    """Small JSON HTTP helper for the Ximilar async request API."""
    data = None
    headers = {
        "Authorization": f"Token {get_api_key('XIMILAR_API_TOKEN')}",
        "Accept":        "application/json",
        "User-Agent":    "CardCollectorInventoryManager/1.0",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=XIMILAR_CONNECT_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ximilar_condition_records(records, mode="ebay"):
    """
    Submit a batch of image records to the Card Grading *condition* endpoint and
    block until the async job finishes.

    `records` — list of dicts, each already carrying a source (`_url` / `_base64`)
    plus an `_id` marker used to map results back. Max 10 per Ximilar request.

    Returns response.records (order + `_id` preserved). Raises RuntimeError with a
    human-readable message on any failure.
    """
    if not get_api_key("XIMILAR_API_TOKEN"):
        raise RuntimeError(
            "XIMILAR_API_TOKEN is not set. Add it in Settings \u2192 API Keys "
            "to enable card condition grading."
        )
    if not records:
        raise RuntimeError("No usable card images to grade.")

    submit_body = {
        "type":     "card-grader",
        "endpoint": "condition",
        "mode":     mode if mode in XIMILAR_CONDITION_MODES else "ebay",
        "records":  records,
    }

    # 1) Submit the job
    try:
        submitted = _ximilar_http("POST", XIMILAR_REQUEST_URL, submit_body)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:300]
        except Exception:
            pass
        raise RuntimeError(f"Ximilar submit failed (HTTP {exc.code} {exc.reason}). {detail}".strip())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ximilar: {exc.reason}")
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Unexpected response from Ximilar on submit: {exc}")

    job_id = submitted.get("id")
    if not job_id:
        raise RuntimeError("Ximilar did not return a job id for the request.")

    # 2) Poll the details endpoint until DONE / timeout
    poll_url = XIMILAR_REQUEST_URL.rstrip("/") + "/" + job_id
    deadline = time.time() + XIMILAR_POLL_MAX_WAIT
    job = {}
    while True:
        try:
            job = _ximilar_http("GET", poll_url)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ximilar poll failed (HTTP {exc.code} {exc.reason}).")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ximilar while polling: {exc.reason}")
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Unexpected response from Ximilar while polling: {exc}")

        status = (job.get("status") or "").upper()
        if status == "DONE":
            break
        if status in ("FAILED", "ERROR"):
            raise RuntimeError(f"Ximilar reported job status {status}.")
        if time.time() >= deadline:
            raise RuntimeError("Ximilar grading timed out. Please try again.")
        time.sleep(XIMILAR_POLL_INTERVAL)

    return (job.get("response") or {}).get("records") or []


def _parse_condition_record(rec):
    """
    Pull the useful bits out of one condition response record.

    Returns a compact dict, or {"error": "..."} if that image failed.
    """
    if not isinstance(rec, dict):
        return {"error": "Malformed response record."}

    rstatus = rec.get("_status") or {}
    code = rstatus.get("code")
    if code is not None and code != 200:
        return {"error": rstatus.get("text") or f"Image failed (code {code})."}

    objects = rec.get("_objects") or []
    if not objects:
        return {"error": "No card detected in this image."}

    obj = objects[0]

    # Best category name (e.g. "Card/Sport Card" or "Card/Trading Card Game")
    category = ""
    cats = obj.get("Category") or []
    if cats:
        category = cats[0].get("name", "")

    cond_list = obj.get("Condition") or []
    if not cond_list:
        return {"error": "No condition returned for this image.", "category": category}

    cond = cond_list[0]
    return {
        "label":           cond.get("label", ""),
        "value":           cond.get("value"),
        "mode":            cond.get("mode", ""),
        "scale":           cond.get("scale") or [],
        "scale_value":     cond.get("scale_value"),
        "max_scale_value": cond.get("max_scale_value"),
        "category":        category,
    }


def _grade_condition_single(record, mode="ebay"):
    """
    Grade one ScanRecord's front (and back, if present) via the condition
    endpoint, persist the result onto record.extracted_data['grading'], and
    return the grading dict. Raises RuntimeError on failure.
    """
    records = []
    front_src = _image_source_for_grading(record.image_path)
    back_src  = _image_source_for_grading(record.image_path_back)

    if front_src:
        front_src["_id"] = "front"
        records.append(front_src)
    if back_src:
        back_src["_id"] = "back"
        records.append(back_src)

    if not records:
        raise RuntimeError("This record has no front or back image to grade.")

    resp_records = _ximilar_condition_records(records, mode=mode)

    # Map results back by the `_id` marker we submitted (fall back to order).
    by_id = {}
    for i, rec in enumerate(resp_records):
        marker = rec.get("_id")
        if marker not in ("front", "back"):
            marker = "front" if i == 0 else "back"
        by_id[marker] = _parse_condition_record(rec)

    grading = {
        "mode":      mode if mode in XIMILAR_CONDITION_MODES else "ebay",
        "graded_at": datetime.utcnow().isoformat() + "Z",
        "front":     by_id.get("front"),
        "back":      by_id.get("back"),
    }

    updated = dict(record.extracted_data or {})
    updated["grading"] = grading
    record.extracted_data = updated
    db.session.commit()

    return grading


@app.route("/grade_condition/<int:record_id>", methods=["POST"])
def grade_condition(record_id):
    """
    Send this record's front & back images to the Ximilar Card Grading
    *condition* endpoint and save the returned condition label(s).

    Optional JSON body: { "mode": "ebay" | "tcgplayer" | "cardmarket" | "ximilar" }
    """
    record = ScanRecord.query.get_or_404(record_id)
    body = request.get_json(silent=True) or {}
    mode = (body.get("mode") or "ebay").strip().lower()
    if mode not in XIMILAR_CONDITION_MODES:
        mode = "ebay"

    try:
        grading = _grade_condition_single(record, mode=mode)
    except RuntimeError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 502

    # Build a short human summary for toasts / inline badges.
    parts = []
    if grading.get("front") and grading["front"].get("label"):
        parts.append(f"Front: {grading['front']['label']}")
    if grading.get("back") and grading["back"].get("label"):
        parts.append(f"Back: {grading['back']['label']}")
    summary = " · ".join(parts) if parts else "Graded"

    return jsonify({
        "status":    "success",
        "message":   f"Condition checked ({summary}).",
        "record_id": record_id,
        "grading":   grading,
    })


# ====================== SHOPS (marketplace sync) ======================
import shop_providers
import email_monitor
from shop_providers import MARKETPLACES, SECRET_FIELDS, get_provider

SHOP_SKU_PREFIX = "CCIM-"


def _shop_persist():
    db.session.commit()


def _get_connection(marketplace, create=False):
    conn = ShopConnection.query.filter_by(marketplace=marketplace).first()
    if conn is None and create:
        conn = ShopConnection(marketplace=marketplace, enabled=False, config={},
                              status="disconnected")
        db.session.add(conn)
        db.session.commit()
    return conn


def _record_display_name(record):
    data = record.extracted_data or {}
    return (data.get("product_name") or data.get("name") or data.get("card_name")
            or data.get("title") or f"Record #{record.id}")


def _record_market_price(record):
    data = record.extracted_data or {}
    try:
        mp = (data.get("tcgplayer") or {}).get("prices", {}).get("market")
        if mp:
            return round(float(mp), 2)
    except (TypeError, ValueError):
        pass
    for k in ("price", "market_price", "sell_price"):
        v = data.get(k)
        try:
            if v not in (None, ""):
                return round(float(v), 2)
        except (TypeError, ValueError):
            continue
    return 0.0


def _record_quantity(record):
    data = record.extracted_data or {}
    for k in ("qty", "quantity", "stock", "count"):
        v = data.get(k)
        try:
            if v not in (None, ""):
                return max(int(float(v)), 0)
        except (TypeError, ValueError):
            continue
    return 1


def _record_image_sources(record):
    """Return (public_image_urls, base64_image_strings) for a record's front/back."""
    urls, b64s = [], []
    data = record.extracted_data or {}

    # Any URL-typed fields the record already carries (e.g. CSV "Photo URL").
    for key in ("photo_url", "image_url", "photo url", "picture_url", "img_url"):
        val = str(data.get(key, "")).strip()
        if val.startswith("http"):
            urls.append(val)

    for path in (record.image_path, record.image_path_back):
        rel = normalize_to_upload_relative(path)
        if not rel or rel == "__blank__":
            continue
        if rel.startswith("http://") or rel.startswith("https://"):
            urls.append(rel)
            continue
        abs_path = os.path.join(app.config["UPLOAD_FOLDER"], rel)
        if os.path.exists(abs_path):
            try:
                with open(abs_path, "rb") as fh:
                    b64s.append(base64.b64encode(fh.read()).decode("ascii"))
            except OSError:
                pass
    # De-dupe URLs while preserving order.
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq, b64s


def _condition_to_ebay(record):
    """Map the saved Ximilar condition (if any) to an eBay condition enum."""
    data = record.extracted_data or {}
    label = ((data.get("grading") or {}).get("front") or {}).get("label", "")
    m = {
        "gem mint": "LIKE_NEW", "mint": "LIKE_NEW", "near mint": "USED_LIKE_NEW",
        "excellent": "USED_EXCELLENT", "very good": "USED_VERY_GOOD",
        "lightly played": "USED_VERY_GOOD", "good": "USED_GOOD",
        "moderately played": "USED_GOOD", "heavily played": "USED_ACCEPTABLE",
        "played": "USED_ACCEPTABLE", "poor": "USED_ACCEPTABLE", "damaged": "USED_ACCEPTABLE",
    }
    return m.get(str(label).lower(), "USED_VERY_GOOD")


def _build_payload(record, marketplace, price=None, quantity=None):
    data = record.extracted_data or {}
    name = _record_display_name(record)
    game = data.get("game", "")
    serial = _get_serial(data)
    title = name
    if game:
        title = f"{name} - {game}"
    if serial:
        title = f"{title} #{serial}"

    urls, b64s = _record_image_sources(record)

    desc_bits = []
    for k in ("game", "set", "edition", "rarity", "holographic"):
        v = data.get(k)
        if v:
            desc_bits.append(f"{k.title()}: {v}")
    grade = ((data.get("grading") or {}).get("front") or {}).get("label")
    if grade:
        desc_bits.append(f"Condition (AI): {grade}")
    description = "<br>".join(desc_bits) or name

    # Map the AI condition label into each marketplace's condition vocabulary.
    grade_label = str(grade or "").lower()
    manapool_cond_map = {
        "gem mint": "NM", "mint": "NM", "near mint": "NM", "excellent": "LP",
        "very good": "LP", "lightly played": "LP", "good": "MP",
        "moderately played": "MP", "played": "HP", "heavily played": "HP",
        "poor": "DMG", "damaged": "DMG",
    }
    foil = str(data.get("holographic", "")).strip().lower() in (
        "true", "1", "yes", "foil", "holo", "holographic")

    return {
        "sku": f"{SHOP_SKU_PREFIX}{record.id}",
        "title": title,
        "description": description,
        "price": price if price is not None else _record_market_price(record),
        "currency": "USD",
        "quantity": quantity if quantity is not None else _record_quantity(record),
        "brand": game or "Trading Card",
        "category": data.get("set") or "Trading Card",
        "tags": ", ".join([t for t in (game, data.get("set"), data.get("rarity")) if t]),
        "image_urls": urls,
        "image_b64": b64s,
        "foil": foil,
        "language": data.get("language") or "English",
        # eBay
        "ebay_condition": _condition_to_ebay(record),
        # TCGplayer
        "tcgplayer_sku_id": data.get("tcgplayer_sku_id") or (data.get("tcgplayer") or {}).get("sku_id"),
        # Mana Pool (MTG-only)
        "manapool_product_id": data.get("manapool_product_id") or (data.get("manapool") or {}).get("product_id"),
        "manapool_condition": manapool_cond_map.get(grade_label, "NM"),
        # Cardmarket
        "cardmarket_product_id": (data.get("cardmarket_product_id")
                                  or (data.get("cardmarket") or {}).get("product_id")
                                  or (data.get("cardmarket") or {}).get("idProduct")),
        "cardmarket_condition": shop_providers.CardmarketProvider.CONDITION_MAP.get(grade_label, "NM"),
        # CardTrader
        "cardtrader_blueprint_id": (data.get("cardtrader_blueprint_id")
                                    or (data.get("cardtrader") or {}).get("blueprint_id")),
        "cardtrader_condition": shop_providers.CardTraderProvider.CONDITION_MAP.get(grade_label, "Near Mint"),
    }


def _listing_for(record_id, marketplace):
    return Listing.query.filter_by(record_id=record_id, marketplace=marketplace).first()


def _connection_public_view(conn, marketplace):
    """Config safe to send to the browser: secrets replaced with a set/unset flag."""
    meta = MARKETPLACES[marketplace]
    cfg = (conn.config if conn else {}) or {}
    public = {}
    secrets_set = {}
    for field in meta["fields"]:
        k = field["key"]
        if field.get("secret") or k in SECRET_FIELDS.get(marketplace, set()):
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
        "ebay_authorized": bool((cfg.get("refresh_token") or cfg.get("access_token"))) if marketplace == "ebay" else None,
    }


def _sellable_records_query():
    """Owned inventory only — exclude catalog-only reference rows and blank slots."""
    records = ScanRecord.query.order_by(ScanRecord.scan_date.desc(), ScanRecord.id.desc()).all()
    out = []
    for r in records:
        data = r.extracted_data or {}
        if _is_catalog_only(data):
            continue
        if str(data.get("empty", "")).strip().lower() == "true":
            continue
        out.append(r)
    return out


@app.route("/shops")
def shops_page():
    connections = {}
    for mk in MARKETPLACES:
        conn = _get_connection(mk)
        connections[mk] = _connection_public_view(conn, mk)

    # Summary counts
    records = _sellable_records_query()
    total = len(records)
    active_counts = {mk: 0 for mk in MARKETPLACES}
    for lst in Listing.query.filter(Listing.status.in_(["active", "draft"])).all():
        if lst.marketplace in active_counts:
            active_counts[lst.marketplace] += 1

    return render_template(
        "shops.html",
        marketplaces=MARKETPLACES,
        connections=connections,
        total_records=total,
        active_counts=active_counts,
        email_monitor=_email_public_view(_get_email_monitor(create=True)),
    )


@app.route("/shops/save/<marketplace>", methods=["POST"])
def shops_save(marketplace):
    if marketplace not in MARKETPLACES:
        return jsonify({"status": "error", "message": "Unknown marketplace"}), 404
    conn = _get_connection(marketplace, create=True)
    cfg = dict(conn.config or {})

    for field in MARKETPLACES[marketplace]["fields"]:
        k = field["key"]
        submitted = request.form.get(k, None)
        if submitted is None:
            continue
        submitted = submitted.strip()
        is_secret = field.get("secret") or k in SECRET_FIELDS.get(marketplace, set())
        # For secret fields, an empty submission means "keep what's stored".
        if is_secret and submitted == "":
            continue
        cfg[k] = submitted

    conn.config = cfg
    conn.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"status": "success", "message": f"{MARKETPLACES[marketplace]['label']} settings saved."})


@app.route("/shops/test/<marketplace>", methods=["POST"])
def shops_test(marketplace):
    if marketplace not in MARKETPLACES:
        return jsonify({"status": "error", "message": "Unknown marketplace"}), 404
    conn = _get_connection(marketplace, create=True)
    provider = get_provider(marketplace, conn, persist=_shop_persist)
    result = provider.test_connection()

    conn.status = "connected" if result.get("ok") else "error"
    conn.status_detail = result.get("message", "")
    if result.get("ok"):
        conn.enabled = True
        conn.connected_at = conn.connected_at or datetime.utcnow()
    conn.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        "status": "success" if result.get("ok") else "error",
        "message": result.get("message", ""),
        "connected": bool(result.get("ok")),
    })


@app.route("/shops/disconnect/<marketplace>", methods=["POST"])
def shops_disconnect(marketplace):
    if marketplace not in MARKETPLACES:
        return jsonify({"status": "error", "message": "Unknown marketplace"}), 404
    conn = _get_connection(marketplace)
    if conn:
        cfg = dict(conn.config or {})
        # Drop cached tokens but keep the app credentials the user typed in.
        for tok in ("access_token", "refresh_token", "access_expires_at",
                    "refresh_expires_at", "bearer_token", "bearer_expires_at"):
            cfg.pop(tok, None)
        conn.config = cfg
        conn.enabled = False
        conn.status = "disconnected"
        conn.status_detail = "Disconnected."
        conn.connected_at = None
        db.session.commit()
    return jsonify({"status": "success", "message": "Disconnected."})


@app.route("/shops/ebay/connect")
def shops_ebay_connect():
    conn = _get_connection("ebay", create=True)
    provider = get_provider("ebay", conn, persist=_shop_persist)
    if provider._need("client_id", "ru_name"):
        return redirect(url_for("shops_page") + "?ebay_error=Set+App+ID+and+RuName+first")
    return redirect(provider.authorize_url(state="ebay"))


@app.route("/shops/ebay/callback")
def shops_ebay_callback():
    code = request.args.get("code", "")
    err = request.args.get("error_description") or request.args.get("error")
    if err:
        return redirect(url_for("shops_page") + "?ebay_error=" + urllib.parse.quote(err))
    if not code:
        return redirect(url_for("shops_page") + "?ebay_error=No+authorization+code+returned")

    conn = _get_connection("ebay", create=True)
    provider = get_provider("ebay", conn, persist=_shop_persist)
    result = provider.exchange_code(code)
    if result.get("ok"):
        conn.status = "connected"
        conn.enabled = True
        conn.connected_at = datetime.utcnow()
        conn.status_detail = "Account connected."
        db.session.commit()
        return redirect(url_for("shops_page") + "?ebay_ok=1")
    return redirect(url_for("shops_page") + "?ebay_error=" + urllib.parse.quote(result.get("message", "Failed")))


@app.route("/shops/listings")
def shops_listings():
    """Paginated inventory with per-marketplace listing status (JSON for the table)."""
    page = max(int(request.args.get("page", 1) or 1), 1)
    per_page = min(max(int(request.args.get("per_page", 25) or 25), 1), 100)
    only_marketplace = request.args.get("marketplace", "")
    unlisted_only = request.args.get("unlisted", "") == "1"

    records = _sellable_records_query()

    # Pre-index listings by (record_id, marketplace)
    all_listings = Listing.query.all()
    idx = {}
    for lst in all_listings:
        idx[(lst.record_id, lst.marketplace)] = lst

    def record_row(r):
        data = r.extracted_data or {}
        row_listings = {}
        for mk in MARKETPLACES:
            lst = idx.get((r.id, mk))
            row_listings[mk] = {
                "status": lst.status if lst else "not_listed",
                "listing_id": lst.id if lst else None,
                "url": lst.external_url if lst else "",
                "price": lst.price if lst else None,
            } if True else None
        return {
            "record_id": r.id,
            "name": _record_display_name(r),
            "game": data.get("game", ""),
            "set": data.get("set", ""),
            "price": _record_market_price(r),
            "quantity": _record_quantity(r),
            "image_url": build_uploaded_file_url(r.image_path),
            "detail_url": url_for("inventory_detail", record_id=r.id),
            "listings": row_listings,
        }

    rows = [record_row(r) for r in records]

    if unlisted_only and only_marketplace in MARKETPLACES:
        rows = [row for row in rows if row["listings"][only_marketplace]["status"] == "not_listed"]

    total = len(rows)
    start = (page - 1) * per_page
    page_rows = rows[start:start + per_page]

    return jsonify({
        "status": "success",
        "rows": page_rows,
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": max((total + per_page - 1) // per_page, 1),
    })


@app.route("/shops/push", methods=["POST"])
def shops_push():
    body = request.get_json(silent=True) or {}
    marketplace = body.get("marketplace", "")
    record_ids = body.get("record_ids", []) or []
    price_override = body.get("price", None)
    qty_override = body.get("quantity", None)

    if marketplace not in MARKETPLACES:
        return jsonify({"status": "error", "message": "Unknown marketplace"}), 400
    conn = _get_connection(marketplace)
    if not conn or not conn.enabled:
        return jsonify({"status": "error", "message": f"{MARKETPLACES[marketplace]['label']} isn't connected."}), 400

    provider = get_provider(marketplace, conn, persist=_shop_persist)
    results = []

    for rid in record_ids:
        record = ScanRecord.query.get(rid)
        if not record:
            results.append({"record_id": rid, "ok": False, "message": "Record not found."})
            continue

        payload = _build_payload(record, marketplace,
                                 price=price_override, quantity=qty_override)

        listing = _listing_for(rid, marketplace)
        if listing and listing.external_id:
            payload["external_id"] = listing.external_id
            payload["extra"] = listing.extra or {}

        result = provider.push(payload)

        if listing is None:
            listing = Listing(record_id=rid, marketplace=marketplace)
            db.session.add(listing)
        listing.sku = payload["sku"]
        listing.title = payload["title"]
        listing.price = payload["price"]
        listing.quantity = payload["quantity"]
        listing.last_synced = datetime.utcnow()
        if result.get("ok"):
            listing.external_id = result.get("external_id") or listing.external_id
            listing.external_url = result.get("external_url") or listing.external_url
            listing.status = result.get("status", "active")
            listing.last_error = None
            if result.get("extra"):
                merged = dict(listing.extra or {}); merged.update(result["extra"])
                listing.extra = merged
        else:
            listing.status = result.get("status", "error")
            listing.last_error = result.get("message", "")
            if result.get("external_id"):
                listing.external_id = result["external_id"]
            if result.get("extra"):
                merged = dict(listing.extra or {}); merged.update(result["extra"])
                listing.extra = merged
        db.session.commit()

        results.append({
            "record_id": rid,
            "ok": bool(result.get("ok")),
            "status": listing.status,
            "url": listing.external_url,
            "message": result.get("message", ""),
        })

    ok_count = sum(1 for r in results if r["ok"])
    return jsonify({
        "status": "success",
        "ok_count": ok_count,
        "fail_count": len(results) - ok_count,
        "results": results,
    })


@app.route("/shops/unlist", methods=["POST"])
def shops_unlist():
    body = request.get_json(silent=True) or {}
    listing_id = body.get("listing_id")
    listing = Listing.query.get(listing_id) if listing_id else None
    if not listing:
        return jsonify({"status": "error", "message": "Listing not found."}), 404
    conn = _get_connection(listing.marketplace)
    if not conn:
        return jsonify({"status": "error", "message": "Marketplace not connected."}), 400
    provider = get_provider(listing.marketplace, conn, persist=_shop_persist)
    result = provider.end_listing(listing)
    if result.get("ok"):
        listing.status = result.get("status", "ended")
        listing.last_error = None
        listing.last_synced = datetime.utcnow()
    else:
        listing.last_error = result.get("message", "")
    db.session.commit()
    return jsonify({
        "status": "success" if result.get("ok") else "error",
        "message": result.get("message", ""),
    })


@app.route("/shops/pull/<marketplace>", methods=["POST"])
def shops_pull(marketplace):
    if marketplace not in MARKETPLACES:
        return jsonify({"status": "error", "message": "Unknown marketplace"}), 404
    conn = _get_connection(marketplace)
    if not conn or not conn.enabled:
        return jsonify({"status": "error", "message": f"{MARKETPLACES[marketplace]['label']} isn't connected."}), 400

    provider = get_provider(marketplace, conn, persist=_shop_persist)
    result = provider.pull()
    if not result.get("ok"):
        return jsonify({"status": "error", "message": result.get("message", "Pull failed.")}), 502

    matched = 0
    for item in result.get("items", []):
        sku = str(item.get("sku", ""))
        if not sku.startswith(SHOP_SKU_PREFIX):
            continue
        try:
            rid = int(sku[len(SHOP_SKU_PREFIX):])
        except ValueError:
            continue
        record = ScanRecord.query.get(rid)
        if not record:
            continue
        listing = _listing_for(rid, marketplace)
        if listing is None:
            listing = Listing(record_id=rid, marketplace=marketplace)
            db.session.add(listing)
        listing.sku = sku
        listing.external_id = item.get("external_id") or listing.external_id
        listing.external_url = item.get("external_url") or listing.external_url
        listing.title = item.get("title") or listing.title
        listing.price = item.get("price", listing.price)
        listing.quantity = item.get("quantity", listing.quantity)
        listing.status = item.get("status") or "active"
        listing.last_synced = datetime.utcnow()
        if item.get("extra"):
            merged = dict(listing.extra or {}); merged.update(item["extra"])
            listing.extra = merged
        matched += 1
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": f"{result.get('message','')} Matched {matched} to inventory by SKU.",
        "fetched": len(result.get("items", [])),
        "matched": matched,
    })


# ---------------------------------------------------------------------------
# TCGplayer: no-API listing via a Seller Portal import CSV
#
# TCGplayer's developer API is closed, but Level 4 sellers can bulk-list by
# importing a CSV in the Seller Portal (Pricing tab -> Import to Staged ->
# Move to Live). This app generates that import file from selected inventory
# records. Crucially, each row needs TCGplayer's own product/SKU id ("TCGplayer
# Id"), so as a side effect of generating the file we record that id back onto
# the record (tcgplayer_sku_id). That becomes the durable join key: when a sale
# later comes back via the Orders CSV / email, it carries the same TCGplayer Id,
# giving exact matching without ever needing to stamp our CCIM SKU onto TCGplayer.
# ---------------------------------------------------------------------------
import csv as _csv
import io as _io

# Column order mirrors the TCGplayer Pricing-tab export so the file imports cleanly.
TCGPLAYER_CSV_COLUMNS = [
    "TCGplayer Id", "Product Line", "Set Name", "Product Name", "Title", "Number",
    "Rarity", "Condition", "TCG Market Price", "TCG Direct Low",
    "TCG Low Price With Shipping", "TCG Low Price", "Total Quantity",
    "Add to Quantity", "TCG Marketplace Price", "Photo URL",
]


def _record_tcgplayer_id(record):
    """Best-effort TCGplayer product/SKU id from whatever the record already stores."""
    d = record.extracted_data or {}
    tcg = d.get("tcgplayer") if isinstance(d.get("tcgplayer"), dict) else {}
    candidates = (
        d.get("tcgplayer_sku_id"), tcg.get("sku_id"), tcg.get("skuId"),
        d.get("tcgplayer_id"), tcg.get("product_id"), tcg.get("productId"),
        tcg.get("id"), d.get("tcg_id"),
    )
    for v in candidates:
        if v not in (None, "", 0):
            return str(v).strip()
    return ""


def _condition_to_tcgplayer(record):
    """Map the saved AI condition to TCGplayer's condition vocabulary (+ Foil suffix)."""
    d = record.extracted_data or {}
    label = ((d.get("grading") or {}).get("front") or {}).get("label", "")
    m = {
        "gem mint": "Near Mint", "mint": "Near Mint", "near mint": "Near Mint",
        "excellent": "Lightly Played", "very good": "Lightly Played",
        "lightly played": "Lightly Played", "good": "Moderately Played",
        "moderately played": "Moderately Played", "played": "Moderately Played",
        "heavily played": "Heavily Played", "poor": "Damaged", "damaged": "Damaged",
    }
    cond = m.get(str(label).lower(), "Near Mint")
    foil = str(d.get("holographic", "")).strip().lower() in (
        "true", "1", "yes", "foil", "holo", "holographic")
    return f"{cond} Foil" if foil else cond


def _records_from_request(body):
    """Records for a TCGplayer CSV request: explicit ids, else all sellable inventory."""
    ids = body.get("record_ids")
    if ids:
        out = []
        for i in ids:
            try:
                r = ScanRecord.query.get(int(i))
            except (TypeError, ValueError):
                r = None
            if r:
                out.append(r)
        return out
    return _sellable_records_query()


def _tcgplayer_rows(records, fallback_price=None):
    """
    Build CSV-ready rows for records that can be listed, plus a `skipped` list
    explaining any exclusions (missing TCGplayer Id or missing price).
    """
    try:
        fb = float(fallback_price) if fallback_price not in (None, "") else 0.0
    except (TypeError, ValueError):
        fb = 0.0

    rows, skipped = [], []
    for r in records:
        d = r.extracted_data or {}
        name = _record_display_name(r)
        tcg_id = _record_tcgplayer_id(r)
        if not tcg_id:
            skipped.append({"record_id": r.id, "name": name,
                            "reason": "No TCGplayer Id on record"})
            continue

        market = _record_market_price(r)
        price = market
        if not price or price <= 0:
            if fb > 0:
                price = round(fb, 2)
            else:
                skipped.append({"record_id": r.id, "name": name,
                                "reason": "No price (add a market price or set a fallback)"})
                continue

        qty = _record_quantity(r)
        urls, _ = _record_image_sources(r)
        rows.append({
            "record_id": r.id,
            "tcg_id": tcg_id,
            "name": name,
            "price": round(float(price), 2),
            "qty": qty,
            "csv": {
                "TCGplayer Id": tcg_id,
                "Product Line": d.get("game", ""),
                "Set Name": d.get("set", ""),
                "Product Name": name,
                "Title": "",  # buyer-facing; only used for custom photo listings
                "Number": _get_serial(d),
                "Rarity": d.get("rarity", ""),
                "Condition": _condition_to_tcgplayer(r),
                "TCG Market Price": f"{market:.2f}" if market else "",
                "TCG Direct Low": "",
                "TCG Low Price With Shipping": "",
                "TCG Low Price": "",
                "Total Quantity": "",
                "Add to Quantity": qty,             # additive: adds this stock on import
                "TCG Marketplace Price": f"{round(float(price), 2):.2f}",  # required
                "Photo URL": urls[0] if urls else "",
            },
        })
    return rows, skipped


@app.route("/shops/tcgplayer/preview", methods=["POST"])
def shops_tcgplayer_preview():
    """Report how many records can be listed and which would be skipped (and why)."""
    body = request.get_json(silent=True) or {}
    records = _records_from_request(body)
    rows, skipped = _tcgplayer_rows(records, fallback_price=body.get("fallback_price"))
    return jsonify({
        "status": "success",
        "total": len(records),
        "eligible": len(rows),
        "skipped": skipped,
        "sample": [{"record_id": x["record_id"], "name": x["name"], "tcg_id": x["tcg_id"],
                    "price": x["price"], "qty": x["qty"]} for x in rows[:25]],
    })


@app.route("/shops/tcgplayer/export_csv", methods=["POST"])
def shops_tcgplayer_export_csv():
    """
    Return a TCGplayer Seller-Portal import CSV for the requested records, and as
    a side effect record each row's TCGplayer Id onto its record and upsert a
    draft TCGplayer Listing so the item shows as pending-import in the table.
    """
    body = request.get_json(silent=True) or {}
    records = _records_from_request(body)
    rows, skipped = _tcgplayer_rows(records, fallback_price=body.get("fallback_price"))

    buf = _io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=TCGPLAYER_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row["csv"])
    csv_text = buf.getvalue()

    # Side effects: backfill the TCGplayer Id + mark a draft listing.
    for row in rows:
        rec = ScanRecord.query.get(row["record_id"])
        if not rec:
            continue
        data = dict(rec.extracted_data or {})
        if str(data.get("tcgplayer_sku_id", "")) != row["tcg_id"]:
            data["tcgplayer_sku_id"] = row["tcg_id"]
            rec.extracted_data = data

        listing = _listing_for(rec.id, "tcgplayer")
        if listing is None:
            listing = Listing(record_id=rec.id, marketplace="tcgplayer")
            db.session.add(listing)
        listing.sku = f"{SHOP_SKU_PREFIX}{rec.id}"
        listing.external_id = row["tcg_id"]
        listing.title = row["name"]
        listing.price = row["price"]
        listing.quantity = row["qty"]
        listing.status = "draft"          # generated into an import file, awaiting upload
        listing.last_error = None
        listing.last_synced = datetime.utcnow()
    db.session.commit()

    fname = f"tcgplayer_import_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    resp = app.response_class(csv_text, mimetype="text/csv")
    resp.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
    resp.headers["X-CCIM-Included"] = str(len(rows))
    resp.headers["X-CCIM-Skipped"] = str(len(skipped))
    resp.headers["Access-Control-Expose-Headers"] = "X-CCIM-Included, X-CCIM-Skipped"
    return resp


# ---------------------------------------------------------------------------
# TCGplayer: bootstrap importer — backfill TCGplayer Ids from a Live export
#
# The user exports their existing TCGplayer Live inventory to CSV (Pricing tab ->
# Export from Live) and uploads it here. We match each export row back to an
# inventory ScanRecord by name/set/number/condition/foil and write the row's
# TCGplayer Id onto the record (tcgplayer_sku_id). After that, sales match
# exactly and the listing-CSV generator can include those items.
#
# Matching is tiered by confidence:
#   exact  = name + set + number + condition + foil
#   strong = name + set + number + foil
#   loose  = name + set + foil
# A record is only auto-assigned when its best available tier resolves to a
# single TCGplayer Id; multiple candidates are reported as "ambiguous" instead.
# ---------------------------------------------------------------------------

# AI/grade label -> normalized TCGplayer base condition
_TCG_COND_FROM_RECORD = {
    "gem mint": "near mint", "mint": "near mint", "near mint": "near mint",
    "excellent": "lightly played", "very good": "lightly played",
    "lightly played": "lightly played", "good": "moderately played",
    "moderately played": "moderately played", "played": "moderately played",
    "heavily played": "heavily played", "poor": "damaged", "damaged": "damaged",
}


def _norm_txt(s):
    return _re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _norm_num(s):
    return _re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _split_condition(cond_str):
    """'Near Mint Foil' -> ('near mint', True)."""
    c = str(cond_str or "").lower()
    foil = "foil" in c
    c = c.replace("foil", "")
    return _norm_txt(c), foil


def _record_condition_norm(record, assume_nm):
    d = record.extracted_data or {}
    label = d.get("condition") or ((d.get("grading") or {}).get("front") or {}).get("label")
    if label:
        return _TCG_COND_FROM_RECORD.get(str(label).lower(), _norm_txt(label))
    return "near mint" if assume_nm else None


def _record_foil(record):
    d = record.extracted_data or {}
    return str(d.get("holographic", "")).strip().lower() in (
        "true", "1", "yes", "foil", "holo", "holographic")


def _pick_col(fieldnames, *aliases):
    norm = {}
    for f in (fieldnames or []):
        norm[_re.sub(r"[^a-z0-9]+", "", (f or "").lower())] = f
    for a in aliases:
        k = _re.sub(r"[^a-z0-9]+", "", a.lower())
        if k in norm:
            return norm[k]
    return None


def _parse_tcg_export(file_storage):
    """Parse an uploaded TCGplayer export into normalized rows. Returns (rows, cols) or (None, None)."""
    raw = file_storage.read()
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")
    else:
        text = str(raw)

    reader = _csv.DictReader(_io.StringIO(text))
    fn = reader.fieldnames or []
    cols = {
        "id":     _pick_col(fn, "TCGplayer Id", "TCGplayerID", "Id"),
        "name":   _pick_col(fn, "Product Name", "Name"),
        "set":    _pick_col(fn, "Set Name", "Set"),
        "number": _pick_col(fn, "Number"),
        "cond":   _pick_col(fn, "Condition"),
        "price":  _pick_col(fn, "TCG Marketplace Price", "TCG Market Price", "Marketplace Price", "Price"),
        "qty":    _pick_col(fn, "Total Quantity", "Add to Quantity", "Quantity"),
    }
    if not cols["id"] or not cols["name"]:
        return None, None

    rows = []
    for r in reader:
        tcg_id = str(r.get(cols["id"], "")).strip()
        name = r.get(cols["name"], "")
        if not tcg_id or not str(name).strip():
            continue
        base_cond, foil = _split_condition(r.get(cols["cond"], "")) if cols["cond"] else ("", False)
        try:
            price = float(str(r.get(cols["price"], "") or 0).replace("$", "").replace(",", "")) if cols["price"] else 0.0
        except ValueError:
            price = 0.0
        try:
            qty = int(float(r.get(cols["qty"], "") or 0)) if cols["qty"] else 0
        except ValueError:
            qty = 0
        rows.append({
            "tcg_id": tcg_id,
            "name": _norm_txt(name),
            "set": _norm_txt(r.get(cols["set"], "")) if cols["set"] else "",
            "number": _norm_num(r.get(cols["number"], "")) if cols["number"] else "",
            "cond": base_cond,
            "foil": foil,
            "price": round(price, 2),
            "qty": qty,
        })
    return rows, cols


def _index_tcg_rows(rows):
    idx = {"full": {}, "strong": {}, "loose": {}}
    for row in rows:
        n, s, num, c, fo = row["name"], row["set"], row["number"], row["cond"], row["foil"]
        if num and c:
            idx["full"].setdefault((n, s, num, c, fo), []).append(row)
        if num:
            idx["strong"].setdefault((n, s, num, fo), []).append(row)
        idx["loose"].setdefault((n, s, fo), []).append(row)
    return idx


def _match_record_to_tcg(record, idx, assume_nm, allowed_tiers):
    d = record.extracted_data or {}
    n = _norm_txt(_record_display_name(record))
    s = _norm_txt(d.get("set", ""))
    num = _norm_num(_get_serial(d))
    fo = _record_foil(record)
    c = _record_condition_norm(record, assume_nm)

    tiers = []
    if "full" in allowed_tiers and num and c:
        tiers.append(("exact", idx["full"].get((n, s, num, c, fo))))
    if "strong" in allowed_tiers and num:
        tiers.append(("strong", idx["strong"].get((n, s, num, fo))))
    if "loose" in allowed_tiers:
        tiers.append(("loose", idx["loose"].get((n, s, fo))))

    for conf, cands in tiers:
        if not cands:
            continue
        ids = sorted({row["tcg_id"] for row in cands})
        if len(ids) == 1:
            return {"tcg_id": ids[0], "confidence": conf, "row": cands[0]}
        return {"ambiguous": True, "confidence": conf, "candidates": ids[:6]}
    return None


@app.route("/shops/tcgplayer/import_ids", methods=["POST"])
def shops_tcgplayer_import_ids():
    """
    Backfill TCGplayer Ids onto inventory records from an uploaded Live-inventory
    export. mode=preview reports matches without writing; mode=apply writes the
    ids (and optionally marks the items as active TCGplayer listings).
    """
    f = request.files.get("file")
    if not f:
        return jsonify({"status": "error", "message": "No CSV file uploaded."}), 400

    mode        = request.form.get("mode", "preview")
    min_conf    = request.form.get("min_confidence", "strong")
    only_missing = request.form.get("only_missing", "1") == "1"
    assume_nm   = request.form.get("assume_nm", "1") == "1"
    create_listings = request.form.get("create_listings", "1") == "1"

    rows, _cols = _parse_tcg_export(f)
    if rows is None:
        return jsonify({"status": "error",
                        "message": "This doesn't look like a TCGplayer export "
                                   "(couldn't find 'TCGplayer Id' / 'Product Name' columns)."}), 400
    if not rows:
        return jsonify({"status": "error", "message": "No usable rows found in the file."}), 400

    allowed = {"exact": ["full"], "strong": ["full", "strong"],
               "loose": ["full", "strong", "loose"]}.get(min_conf, ["full", "strong"])
    idx = _index_tcg_rows(rows)

    records = _sellable_records_query()
    matched, ambiguous, unmatched = [], [], []
    used_ids = set()
    skipped_existing = applied = listings_created = 0

    for r in records:
        if only_missing and _record_tcgplayer_id(r):
            skipped_existing += 1
            continue
        m = _match_record_to_tcg(r, idx, assume_nm, allowed)
        name = _record_display_name(r)
        if not m:
            unmatched.append({"record_id": r.id, "name": name})
            continue
        if m.get("ambiguous"):
            ambiguous.append({"record_id": r.id, "name": name,
                              "confidence": m["confidence"], "candidates": m["candidates"]})
            continue

        entry = {"record_id": r.id, "name": name, "tcg_id": m["tcg_id"],
                 "confidence": m["confidence"], "price": m["row"]["price"], "qty": m["row"]["qty"]}
        matched.append(entry)
        used_ids.add(m["tcg_id"])

        if mode == "apply":
            data = dict(r.extracted_data or {})
            data["tcgplayer_sku_id"] = m["tcg_id"]
            r.extracted_data = data
            applied += 1
            if create_listings:
                listing = _listing_for(r.id, "tcgplayer")
                if listing is None:
                    listing = Listing(record_id=r.id, marketplace="tcgplayer")
                    db.session.add(listing)
                    listings_created += 1
                listing.sku = f"{SHOP_SKU_PREFIX}{r.id}"
                listing.external_id = m["tcg_id"]
                listing.title = name
                listing.price = m["row"]["price"] or listing.price
                listing.quantity = m["row"]["qty"] or _record_quantity(r)
                listing.status = "active"      # these are live TCGplayer listings
                listing.last_error = None
                listing.last_synced = datetime.utcnow()

    if mode == "apply":
        db.session.commit()

    by_conf = {"exact": 0, "strong": 0, "loose": 0}
    for e in matched:
        by_conf[e["confidence"]] = by_conf.get(e["confidence"], 0) + 1
    csv_unused = sum(1 for row in rows if row["tcg_id"] not in used_ids)

    return jsonify({
        "status": "success",
        "mode": mode,
        "csv_rows": len(rows),
        "records_considered": len(records) - skipped_existing,
        "skipped_existing": skipped_existing,
        "matched_count": len(matched),
        "by_confidence": by_conf,
        "ambiguous_count": len(ambiguous),
        "unmatched_count": len(unmatched),
        "csv_unused_count": csv_unused,
        "applied": applied,
        "listings_created": listings_created,
        "matched_sample": matched[:25],
        "ambiguous_sample": ambiguous[:25],
        "unmatched_sample": unmatched[:25],
    })


# ---------------------------------------------------------------------------
# Sale processing + IMAP email monitor
#
# TCGplayer's API is closed, so a sale becomes known to this app via its
# notification email. A watched IMAP mailbox is polled (manually or on an
# interval); matching messages are parsed (email_monitor.parse_tcgplayer_sale),
# each sold line is matched to an inventory record, the record's quantity is
# decremented, and — when it hits zero — the item is delisted on every other
# connected marketplace via the providers' end_listing(). SaleEvent gives an
# audit trail and idempotency so the same order is never processed twice.
# ---------------------------------------------------------------------------

def _mark_record_sold(record, sold_qty, source, order_id, sold_price=None):
    """Decrement a record's quantity for a sale and fan out delists to other shops."""
    data = dict(record.extracted_data or {})
    remaining = max(_record_quantity(record) - int(sold_qty or 1), 0)
    data["quantity"] = remaining
    log = list(data.get("sales_log", []))
    log.append({"source": source, "order_id": order_id, "qty": int(sold_qty or 1),
                "price": sold_price, "at": datetime.utcnow().isoformat()})
    data["sales_log"] = log
    if remaining == 0:
        data["sold"] = True
        data["sold_at"] = datetime.utcnow().isoformat()
    record.extracted_data = data

    results = []
    for listing in list(record.listings):
        # The channel the sale came from: just reflect the new quantity/status.
        if listing.marketplace == source:
            listing.quantity = remaining
            listing.status = "ended" if remaining == 0 else listing.status
            listing.last_synced = datetime.utcnow()
            continue

        conn = _get_connection(listing.marketplace)
        if not conn or not conn.enabled or listing.status not in ("active", "draft"):
            continue
        provider = get_provider(listing.marketplace, conn, persist=_shop_persist)
        if provider is None:
            continue

        if remaining == 0:
            r = provider.end_listing(listing)
            listing.status = "ended" if r.get("ok") else "error"
        else:
            payload = _build_payload(record, listing.marketplace, quantity=remaining)
            payload["external_id"] = listing.external_id
            payload["extra"] = listing.extra or {}
            r = provider.push(payload)
        listing.last_error = None if r.get("ok") else r.get("message")
        listing.last_synced = datetime.utcnow()
        results.append({"marketplace": listing.marketplace, "ok": bool(r.get("ok")),
                        "message": r.get("message", "")})

    db.session.commit()
    return {"record_id": record.id, "remaining": remaining, "delisted": results}


def _match_email_item_to_record(item, source="tcgplayer"):
    """Resolve a parsed sale line to a single ScanRecord, else (None, reason)."""
    records = _sellable_records_query()

    tid = str(item.get("tcgplayer_id") or "").strip()
    if tid:
        hits = [r for r in records if _record_tcgplayer_id(r) == tid]
        if len(hits) == 1:
            return hits[0], "id"
        if len(hits) > 1:
            # Narrow multiple copies of the same SKU to one (any is fine).
            return hits[0], "id"

    name = _norm_txt(item.get("name", ""))
    if not name:
        return None, "no-name"

    cands = [r for r in records if _norm_txt(_record_display_name(r)) == name]
    if not cands:
        return None, "no-match"

    st = _norm_txt(item.get("set", ""))
    if st:
        refined = [r for r in cands if _norm_txt((r.extracted_data or {}).get("set", "")) == st]
        if refined:
            cands = refined

    fo = bool(item.get("foil"))
    refined = [r for r in cands if _record_foil(r) == fo]
    if refined:
        cands = refined

    # Prefer records actually listed on the source marketplace.
    listed = [r for r in cands
              if (_listing_for(r.id, source) and _listing_for(r.id, source).status in ("active", "draft"))]
    if listed:
        cands = listed

    if len(cands) == 1:
        return cands[0], "name"
    return None, "ambiguous"


def _process_sale_email(parsed, source="tcgplayer"):
    """Turn one parsed email into SaleEvents + delist fan-out. Idempotent per order line."""
    order_id = parsed.get("order_id") or ""
    subject = parsed.get("subject", "")
    out = {"order_id": order_id, "subject": subject,
           "processed": 0, "unmatched": 0, "duplicate": 0, "items": []}

    items = parsed.get("items", [])
    if not items:
        title = "(unparsed) " + (subject[:120] or "no items found")
        if not SaleEvent.query.filter_by(source=source, order_id=order_id, item_title=title).first():
            db.session.add(SaleEvent(source=source, order_id=order_id, item_title=title,
                                     status="unparsed", email_subject=subject,
                                     detail=(parsed.get("excerpt", "") or "")[:500]))
            db.session.commit()
        out["items"].append({"title": title, "status": "unparsed"})
        return out

    for item in items:
        title = (item.get("name") or "").strip() or "(unknown)"
        if item.get("set"):
            title += f" ({item['set']})"

        if SaleEvent.query.filter_by(source=source, order_id=order_id, item_title=title).first():
            out["duplicate"] += 1
            out["items"].append({"title": title, "status": "duplicate"})
            continue

        rec, conf = _match_email_item_to_record(item, source)
        ev = SaleEvent(source=source, order_id=order_id, item_title=title,
                       qty=int(item.get("qty", 1)), price=item.get("price"),
                       record_id=rec.id if rec else None, email_subject=subject,
                       status="matched" if rec else "unmatched",
                       detail=None if rec else f"reason: {conf}")
        db.session.add(ev)
        db.session.flush()

        if rec:
            fan = _mark_record_sold(rec, item.get("qty", 1), source, order_id, item.get("price"))
            ev.status = "processed"
            ev.detail = json.dumps(fan.get("delisted", []))[:500]
            out["processed"] += 1
            out["items"].append({"title": title, "status": "processed", "record_id": rec.id,
                                 "remaining": fan["remaining"], "delisted": fan["delisted"]})
        else:
            out["unmatched"] += 1
            out["items"].append({"title": title, "status": "unmatched", "reason": conf})

    db.session.commit()
    return out


# -- Email monitor config + routes -----------------------------------------
def _get_email_monitor(create=False):
    m = EmailMonitor.query.first()
    if m is None and create:
        m = EmailMonitor()
        db.session.add(m)
        db.session.commit()
    return m


def _email_cfg(m):
    return {
        "host": m.host, "port": m.port, "use_ssl": m.use_ssl,
        "username": m.username, "password": m.password, "folder": m.folder,
        "sender_filter": m.sender_filter, "subject_filter": m.subject_filter,
        "mark_seen": m.mark_seen,
    }


def _email_public_view(m):
    return {
        "enabled": bool(m.enabled), "host": m.host or "", "port": m.port or 993,
        "use_ssl": bool(m.use_ssl), "username": m.username or "",
        "password_set": bool(m.password), "folder": m.folder or "INBOX",
        "sender_filter": m.sender_filter or "", "subject_filter": m.subject_filter or "",
        "source": m.source or "tcgplayer", "mark_seen": bool(m.mark_seen),
        "poll_interval": m.poll_interval or 0,
        "status": m.status or "disconnected", "status_detail": m.status_detail or "",
        "last_checked": m.last_checked.strftime("%Y-%m-%d %H:%M") if m.last_checked else "",
    }


@app.route("/shops/email/save", methods=["POST"])
def shops_email_save():
    m = _get_email_monitor(create=True)
    f = request.form
    m.host = f.get("host", m.host or "").strip()
    m.username = f.get("username", m.username or "").strip()
    pw = f.get("password", None)
    if pw:  # blank keeps the stored password
        m.password = pw
    try:
        m.port = int(f.get("port") or m.port or 993)
    except ValueError:
        m.port = 993
    m.use_ssl = f.get("use_ssl", "1") == "1"
    m.folder = f.get("folder", m.folder or "INBOX").strip() or "INBOX"
    m.sender_filter = f.get("sender_filter", m.sender_filter or "").strip()
    m.subject_filter = f.get("subject_filter", m.subject_filter or "").strip()
    m.source = f.get("source", m.source or "tcgplayer").strip() or "tcgplayer"
    m.mark_seen = f.get("mark_seen", "1") == "1"
    try:
        m.poll_interval = max(int(f.get("poll_interval") or 0), 0)
    except ValueError:
        m.poll_interval = 0
    db.session.commit()
    return jsonify({"status": "success", "message": "Email monitor settings saved."})


@app.route("/shops/email/test", methods=["POST"])
def shops_email_test():
    m = _get_email_monitor(create=True)
    result = email_monitor.test_imap(_email_cfg(m))
    m.status = "connected" if result.get("ok") else "error"
    m.status_detail = result.get("message", "")
    if result.get("ok"):
        m.enabled = True
    db.session.commit()
    return jsonify({"status": "success" if result.get("ok") else "error",
                    "message": result.get("message", ""), "connected": bool(result.get("ok"))})


@app.route("/shops/email/check", methods=["POST"])
def shops_email_check():
    m = _get_email_monitor(create=True)
    if not (m.host and m.username and m.password):
        return jsonify({"status": "error", "message": "Configure and save the mailbox first."}), 400

    res = email_monitor.fetch_sale_emails(_email_cfg(m), since_uid=m.last_uid or 0)
    if not res.get("ok"):
        m.status = "error"
        m.status_detail = res.get("message", "")
        db.session.commit()
        return jsonify({"status": "error", "message": res.get("message", "Fetch failed.")}), 502

    summary = {"emails": len(res["emails"]), "processed": 0, "unmatched": 0,
               "duplicate": 0, "unparsed": 0, "details": []}
    for pe in res["emails"]:
        out = _process_sale_email(pe, source=m.source or "tcgplayer")
        summary["processed"] += out["processed"]
        summary["unmatched"] += out["unmatched"]
        summary["duplicate"] += out["duplicate"]
        summary["unparsed"] += sum(1 for i in out["items"] if i["status"] == "unparsed")
        summary["details"].append(out)

    m.last_uid = res.get("max_uid", m.last_uid)
    m.last_checked = datetime.utcnow()
    m.status = "connected"
    m.status_detail = (f"Checked {summary['emails']} email(s): "
                       f"{summary['processed']} sold, {summary['unmatched']} unmatched.")
    db.session.commit()

    return jsonify({"status": "success",
                    "message": m.status_detail,
                    "summary": summary})


@app.route("/shops/email/disconnect", methods=["POST"])
def shops_email_disconnect():
    m = _get_email_monitor()
    if m:
        m.password = None
        m.enabled = False
        m.status = "disconnected"
        m.status_detail = "Disconnected."
        db.session.commit()
    return jsonify({"status": "success", "message": "Email monitor disconnected."})


# Optional background poller (opt-in via EMAIL_MONITOR_BACKGROUND=1). The manual
# "Check now" button is the primary, always-available path.
_email_poller_started = False


def start_email_poller():
    global _email_poller_started
    if _email_poller_started:
        return
    _email_poller_started = True
    import threading
    import time as _time

    def _loop():
        while True:
            interval = 60
            try:
                with app.app_context():
                    m = _get_email_monitor()
                    if m and m.enabled and (m.poll_interval or 0) > 0 and m.host and m.password:
                        res = email_monitor.fetch_sale_emails(_email_cfg(m), since_uid=m.last_uid or 0)
                        if res.get("ok"):
                            for pe in res["emails"]:
                                _process_sale_email(pe, source=m.source or "tcgplayer")
                            m.last_uid = res.get("max_uid", m.last_uid)
                            m.last_checked = datetime.utcnow()
                            m.status = "connected"
                            db.session.commit()
                        interval = max(int(m.poll_interval or 60), 15)
            except Exception:
                interval = 60
            _time.sleep(interval)

    threading.Thread(target=_loop, daemon=True).start()


# ====================== START ======================
def migrate_add_image_path_back_column():
    """
    db.create_all() only creates missing tables — it never alters existing
    ones. Since image_path_back is new, existing installs with an already-
    created scan_records table need the column added by hand. This runs a
    plain ALTER TABLE the first time (SQLite supports adding a nullable
    column live) and is a no-op on every run after that.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "scan_records" not in inspector.get_table_names():
        return  # fresh DB — db.create_all() already created it with the column

    columns = {col["name"] for col in inspector.get_columns("scan_records")}
    if "image_path_back" not in columns:
        with db.engine.begin() as conn:
            conn.execute(text("ALTER TABLE scan_records ADD COLUMN image_path_back VARCHAR(255)"))


def migrate_add_display_image_path_column():
    """
    Same rationale as migrate_add_image_path_back_column(), for the new
    display_image_path column used by the duplicate-image resolver to
    override only the Inventory page's stacked thumbnail without touching
    each record's own image_path.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "scan_records" not in inspector.get_table_names():
        return  # fresh DB — db.create_all() already created it with the column

    columns = {col["name"] for col in inspector.get_columns("scan_records")}
    if "display_image_path" not in columns:
        with db.engine.begin() as conn:
            conn.execute(text("ALTER TABLE scan_records ADD COLUMN display_image_path VARCHAR(255)"))


def migrate_add_type_reference_region_column():
    """
    Add the type_references.region column ('top_left' | 'top_right') to installs
    that created the table before it existed. No-op afterwards.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "type_references" not in inspector.get_table_names():
        return  # fresh DB — db.create_all() already created it with the column

    columns = {col["name"] for col in inspector.get_columns("type_references")}
    if "region" not in columns:
        with db.engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE type_references ADD COLUMN region VARCHAR(20) DEFAULT 'top_right'"))


def migrate_add_performance_indexes():
    """
    Create indexes that keep reads fast as the collection grows into the tens of
    thousands. This is the scalable alternative to splitting the DB per game:
      • scan_date / template_used / matched_product_id — ordering & direct filters
      • expression indexes on json_extract(extracted_data,'$.game'|'$.album') so
        the Inventory game/album filters (now pushed into SQL) are index-served
      • a composite reference_cards(game, number) for card identification lookups
    All are IF NOT EXISTS, so this is a cheap no-op on every start after the first.
    A final PRAGMA optimize lets SQLite refresh stats only when worthwhile.
    """
    from sqlalchemy import text

    statements = [
        "CREATE INDEX IF NOT EXISTS idx_scan_scan_date ON scan_records(scan_date)",
        "CREATE INDEX IF NOT EXISTS idx_scan_template ON scan_records(template_used)",
        "CREATE INDEX IF NOT EXISTS idx_scan_matched_product ON scan_records(matched_product_id)",
        "CREATE INDEX IF NOT EXISTS idx_scan_game ON scan_records(json_extract(extracted_data, '$.game'))",
        "CREATE INDEX IF NOT EXISTS idx_scan_album ON scan_records(json_extract(extracted_data, '$.album'))",
        "CREATE INDEX IF NOT EXISTS idx_refcard_game_number ON reference_cards(game, number)",
        "CREATE INDEX IF NOT EXISTS idx_listing_marketplace_status ON listings(marketplace, status)",
    ]
    try:
        with db.engine.begin() as conn:
            for sql in statements:
                conn.exec_driver_sql(sql)
    except Exception:
        pass  # never block startup on index creation


def optimize_database():
    """Ask SQLite to update planner statistics where beneficial (cheap)."""
    try:
        with db.engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA optimize")
    except Exception:
        pass


def _backfill_scan_columns(batch=1000):
    """One-time population of the denormalized columns for pre-existing rows,
    using keyset iteration so it stays O(n) even on very large tables."""
    last_id = 0
    while True:
        rows = (ScanRecord.query
                .filter(ScanRecord.id > last_id)
                .order_by(ScanRecord.id)
                .limit(batch).all())
        if not rows:
            break
        for r in rows:
            _derive_scan_columns(r)   # marks the row dirty -> written on commit
        db.session.commit()
        last_id = rows[-1].id


def migrate_add_scan_scaling_columns():
    """
    Add the denormalized scaling columns (game_key, album_key, name_key,
    card_type_key, dup_hash, is_finalized, is_catalog, is_archived) plus their
    indexes to installs whose scan_records table predates them, then backfill
    existing rows once. No-op on every start afterwards. Fresh databases get
    the columns straight from db.create_all(), so nothing is backfilled there.
    """
    from sqlalchemy import inspect

    inspector = inspect(db.engine)
    if "scan_records" not in inspector.get_table_names():
        return  # fresh DB — create_all() already made the columns

    existing = {c["name"] for c in inspector.get_columns("scan_records")}
    new_cols = {
        "game_key":      "VARCHAR(120)",
        "album_key":     "VARCHAR(200)",
        "name_key":      "VARCHAR(300)",
        "card_type_key": "VARCHAR(80)",
        "dup_hash":      "VARCHAR(64)",
        "is_finalized":  "BOOLEAN",
        "is_catalog":    "BOOLEAN",
        "is_archived":   "BOOLEAN",
    }
    added = []
    with db.engine.begin() as conn:
        for name, decl in new_cols.items():
            if name not in existing:
                conn.exec_driver_sql(f"ALTER TABLE scan_records ADD COLUMN {name} {decl}")
                added.append(name)

    # Indexes (names match SQLAlchemy's so a fresh DB's create_all doesn't dupe).
    index_sql = [
        "CREATE INDEX IF NOT EXISTS ix_scan_records_game_key ON scan_records(game_key)",
        "CREATE INDEX IF NOT EXISTS ix_scan_records_album_key ON scan_records(album_key)",
        "CREATE INDEX IF NOT EXISTS ix_scan_records_name_key ON scan_records(name_key)",
        "CREATE INDEX IF NOT EXISTS ix_scan_records_card_type_key ON scan_records(card_type_key)",
        "CREATE INDEX IF NOT EXISTS ix_scan_records_dup_hash ON scan_records(dup_hash)",
        "CREATE INDEX IF NOT EXISTS ix_scan_records_is_finalized ON scan_records(is_finalized)",
        "CREATE INDEX IF NOT EXISTS ix_scan_records_is_catalog ON scan_records(is_catalog)",
        "CREATE INDEX IF NOT EXISTS ix_scan_records_is_archived ON scan_records(is_archived)",
        "CREATE INDEX IF NOT EXISTS idx_scan_hot ON scan_records(game_key, is_catalog, is_archived, scan_date)",
        "CREATE INDEX IF NOT EXISTS idx_scan_album_hot ON scan_records(album_key, is_catalog, is_archived)",
    ]
    with db.engine.begin() as conn:
        for sql in index_sql:
            conn.exec_driver_sql(sql)

    if added:
        _backfill_scan_columns()


if __name__ == "__main__":
    # Complete any deferred cleanup from a previous DB relocation now that we're
    # (re)starting on the current configured database.
    process_pending_deletions()

    with app.app_context():
        db.create_all()
        migrate_add_image_path_back_column()
        migrate_add_display_image_path_column()
        migrate_add_type_reference_region_column()
        migrate_add_scan_scaling_columns()
        migrate_add_performance_indexes()
        optimize_database()
        load_settings()   # load API keys/settings; one-time seed from .env

        # First-run vs existing install: if the mode was never chosen but the
        # database already has records, adopt Sorting Machine (backward-compatible,
        # capped) so upgrades aren't forced through setup. A genuinely fresh,
        # empty install stays unconfigured and lands on the setup gate.
        if _system_mode() is None:
            try:
                if ScanRecord.query.count() > 0:
                    set_system_mode("sorting_machine")
            except Exception:
                pass

    ensure_dirs()

    # Optional background sale-email polling (off by default; "Check now" always works).
    if os.environ.get("EMAIL_MONITOR_BACKGROUND") == "1":
        start_email_poller()

    _roots = app.config.get("STORAGE_ROOTS", {})
    print("Card Collector Inventory Manager")
    print(" • Storage (editable in Settings → Storage):")
    print(f"     images/uploads : {_roots.get('uploads')}")
    print(f"     temp           : {_roots.get('temp')}")
    print(f"     roi templates  : {_roots.get('roi')}")
    print(f"     database       : {_roots.get('db')}")
    _mode = _system_mode() or "(unset — first-run setup pending)"
    print(f" • Implementation : {_mode}"
          + ("  [Raspberry Pi detected]" if _is_raspberry_pi() else ""))
    print(" • /             — Create/edit Game templates (name + blank field list)")
    print(" • /inventory    — Inventory list with server-side pagination and filters")
    print(" • /inventory/<id> — Record detail with edit, TCGPlayer link, copy-from, edition")
    print(" • /inventory/filter_options — Dropdown options API")
    print(" • /duplicates   — Duplicate image manager (name + serial + edition)")
    print(" • /migrate_clean_legacy_fields — POST: scrub first_edition, limited_edition, boolean holographic, empty")
    print(" • /albums       — Album grid view")
    print(" • /search_by_image_page     — Photo search UI")
    print(" • /search_by_image          — POST: ORB feature-match a card photo against inventory")
    print(" • /justtcg_search_manual    — GET: search JustTCG by name/game without a record")
    print(" • /import       — 3x3 split, alignment, manual corner override, blank batch import")
    print(" • /records_summary — Copy-from dropdown API")
    print(" • /template_save   — Save a Game definition (name + fields)")
    print(" • /justtcg_fetch/<id>              — Fetch live price from JustTCG API (POST, manual trigger)")
    print(" • /tcg_save_url/<id>, /tcg_clear_url/<id> — Legacy URL save / pricing data clear")
    print(" • Visit: http://127.0.0.1:5000")

    app.run(host="0.0.0.0", port=5005, debug=True)
