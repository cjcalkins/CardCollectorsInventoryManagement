from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, redirect, Response, session, g
import os
import posixpath
import re as _re
# Decoded-image size ceiling, in megapixels. This is the core defense against
# image "decompression bombs" — a tiny file that declares enormous dimensions and
# exhausts RAM when decoded. Both OpenCV (imread/imdecode) and Pillow honor it, so
# every image decode in the app is bounded. It's generous by default so real
# high-DPI scans still load (a 2400-DPI letter page is ~0.54 G px); raise
# MAX_IMAGE_MEGAPIXELS if your scans are bigger and you have the RAM, or lower it
# to tighten. PDFs are bounded separately by the render-DPI cap, not this value.
try:
    _MAX_IMAGE_MP = float(os.environ.get("MAX_IMAGE_MEGAPIXELS", "2000") or "2000")
except ValueError:
    _MAX_IMAGE_MP = 2000.0
_MAX_IMAGE_PIXELS = max(1, int(_MAX_IMAGE_MP * 1_000_000))
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(_MAX_IMAGE_PIXELS))
import cv2
import json
import shutil
import tempfile
import threading as _threading
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
# Same ceiling for Pillow (its default is ~89 MP, far below legitimate scans).
Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
import io as _io

# ── Lenient image decoding (OpenCV → Pillow fallback) ──
# OpenCV 4.13's PNG reader rejects images whose ancillary chunks (iTXt / zTXt /
# iCCP / eXIf metadata — as embedded by phone cameras, Photoshop/XMP, ICC colour
# profiles, and many AI tools) exceed libpng's ~8 MB per-chunk cap: cv2.imread /
# cv2.imdecode then return None and log
# "grfmt_png.cpp read_chunk chunk data is too large". There is no runtime knob for
# that limit, so we fall back to Pillow, which decodes such files fine. All image
# reads in the app go through these two wrappers. The decompression-bomb guards
# still apply: Pillow honours Image.MAX_IMAGE_PIXELS (set above), and OPENCV's
# pixel ceiling is unchanged. `_cv2_imread`/`_cv2_imdecode` are captured before the
# wrappers exist so the fallback can call the originals without recursion.
_cv2_imread = cv2.imread
_cv2_imdecode = cv2.imdecode


def _pil_to_bgr(im):
    """PIL image -> 3-channel BGR ndarray (matches cv2 IMREAD_COLOR)."""
    return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def _imread(path, flags=cv2.IMREAD_COLOR):
    """Drop-in cv2.imread with a Pillow fallback on decode failure. Returns an
    ndarray, or None (like cv2) if the file can't be read by either backend."""
    img = _cv2_imread(path, flags)
    if img is not None:
        return img
    try:
        with Image.open(path) as im:
            return _pil_to_bgr(im)
    except Exception:
        return None


def _imdecode(buf, flags=cv2.IMREAD_COLOR):
    """Drop-in cv2.imdecode with a Pillow fallback on decode failure. `buf` is a
    uint8 ndarray of encoded bytes (as produced by np.frombuffer)."""
    img = _cv2_imdecode(buf, flags)
    if img is not None:
        return img
    try:
        with Image.open(_io.BytesIO(bytes(buf))) as im:
            return _pil_to_bgr(im)
    except Exception:
        return None
from werkzeug.utils import secure_filename
from models import (db, Product, ScanRecord, ShopConnection, Listing, EmailMonitor,
                    SaleEvent, ReferenceCard, ReferenceSync, TypeReference, AppSetting,
                    CollectionPrice)
from dotenv import load_dotenv
load_dotenv()

# PyMuPDF is used to rasterize uploaded PDF pages (see /pdf_open + /pdf_render_page).
# Import is optional at module load time so the rest of the app keeps working
# even on installs that haven't run `pip install PyMuPDF` yet — the PDF
# import route will just report a clear error instead of crashing the app.
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# card_ocr provides front-image OCR (top/bottom band -> name + N/M collector
# number) and a matcher against existing records. It uses RapidOCR (PP-OCRv5
# mobile via ONNX Runtime). Imported optionally so the app still boots on installs
# that haven't run `pip install rapidocr onnxruntime` yet; the /ocr_identify route
# reports a clear error instead of crashing.
try:
    import card_ocr
except Exception:
    card_ocr = None

# Reference-catalog provider: downloads a game's card catalog into ReferenceCard
# rows so OCR results can be matched to a real card and auto-fill entry data.
#
# tcgcsv.com mirrors EVERY TCGplayer game (Pokemon, Magic, Yu-Gi-Oh, Lorcana,
# One Piece, Digimon, Flesh and Blood, ...), so it is the default source and the
# game picker lists everything it offers. pokemontcg.io is Pokemon-only but has
# richer vintage WOTC set data (Base, Jungle, Fossil, Gym, Neo, ...); set
# REFERENCE_PROVIDER=pokemontcg to prefer it if you only collect Pokemon and want
# that extra fidelity. Both adapters expose the same interface (get_categories /
# get_groups / fetch_group_cards / get_last_updated / normalize_product), so
# either can back ref_sync. Imported optionally so the app still boots without
# network/module access; the /reference routes then report a clear error.
_REFERENCE_PROVIDER = (os.environ.get("REFERENCE_PROVIDER") or "tcgcsv").strip().lower()
_ref_provider_order = (["pokemontcg_sync", "tcgcsv_sync"]
                       if _REFERENCE_PROVIDER == "pokemontcg"
                       else ["tcgcsv_sync", "pokemontcg_sync"])
ref_sync = None
for _ref_mod in _ref_provider_order:
    try:
        ref_sync = __import__(_ref_mod)
        break
    except Exception:
        ref_sync = None

app = Flask(__name__, template_folder="templates")

# ====================== CONFIG ======================
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ── Security hardening ──
def _persisted_secret_key():
    """Read (or create once) the session-signing key kept beside the application.

    This exists because the key has to be the same in every process. The __main__
    block below keeps a copy in app_settings, which works when the app is launched
    with `python app.py` — but a WSGI server imports this module and never runs that
    block, so under gunicorn every worker used to mint its own os.urandom() key. A
    session or CSRF token issued by one worker is then rejected by the next, which
    presents as random logouts and "your session's security token expired" on a form
    that was filled in seconds ago. It also meant every restart logged everyone out.

    A file rather than the database because this runs at import, before db.create_all()
    and load_settings() have had a chance to run under any launch method. Reaching for
    the DB here would be an ordering hazard for the sake of a value that does not need
    to be in the DB.

    os.path.dirname(__file__) rather than BASE_DIR: that constant is defined further
    down this file and does not exist yet. instance/ is Flask's conventional spot for
    per-deployment state and is already gitignored (twice: /instance/ and *.key).
    0600 because anyone who reads this file can mint a session cookie for any user."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance",
                        "session_secret.key")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            existing = fh.read().strip()
        if existing:
            return existing
    except OSError:
        pass
    key = os.urandom(32).hex()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Create with 0600 already set rather than chmod-ing after: writing first and
        # tightening second leaves a window where the key is world-readable.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key)
    except OSError:
        # Read-only deployment: fall back to a per-process key. Single-worker still
        # works; multi-worker degrades to the old behaviour rather than failing to boot.
        pass
    return key


# Env wins, so a container or systemd unit can inject the key without any file.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or _persisted_secret_key()

# Cap request bodies to blunt memory/disk exhaustion from oversized uploads.
# High-DPI PDF/image imports are expected, so the default is generous and tunable
# via MAX_UPLOAD_MB (set MAX_UPLOAD_MB=0 to disable the cap entirely).
try:
    _max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "1024"))
except (TypeError, ValueError):
    _max_upload_mb = 1024
if _max_upload_mb > 0:
    app.config["MAX_CONTENT_LENGTH"] = _max_upload_mb * 1024 * 1024

# Session cookie hardening. The session is load-bearing, not hypothetical: it carries
# the signed-in user id, the CSRF token and the eBay OAuth nonce.
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Secure must be decided HERE, at import, because that is the only code that runs under
# every launch method. The __main__ block below also sets it, but a WSGI server
# (gunicorn/uWSGI/mod_wsgi) imports this module and never executes __main__ — so the
# flag used to be left at Flask's default of False for exactly the deployments most
# likely to be on a real network.
#
# Defaults OFF so the supported plain-HTTP modes keep working out of the box: marking
# the cookie Secure on an http:// origin means the browser never sends it back, which
# presents as "login silently does nothing". Behind a TLS-terminating proxy the app
# sees plain http and cannot infer this, so it has to be told.
app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get("SESSION_COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes", "on"))


@app.after_request
def _security_headers(resp):
    """Conservative, app-safe response headers. (No strict CSP on app pages — the UI
    relies on inline scripts/styles — but these block MIME sniffing, clickjacking,
    and referrer leakage.) User-uploaded files are additionally served sandboxed so
    a booby-trapped SVG/HTML upload can't execute script in the app's origin."""
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    p = request.path or ""
    if p.startswith(("/uploads/", "/temp_cards/", "/temp_split/", "/temp_pdf/")):
        # Treat served user content as inert: no scripts, sandboxed, own origin.
        resp.headers["Content-Security-Policy"] = (
            "sandbox; default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'")
    return resp


def _within_dir(base, target):
    """True if `target` resolves to a location inside `base` (blocks ../ traversal
    and absolute-path escapes when a user-influenced name is joined onto a dir)."""
    try:
        base_r = os.path.realpath(base)
        target_r = os.path.realpath(target)
        return target_r == base_r or target_r.startswith(base_r + os.sep)
    except Exception:
        return False


# ── Decompression-bomb guard for uploaded images ──
# A legitimate high-res scan is a LARGE FILE that decodes to a large image (a
# modest ratio); a decompression bomb is a TINY FILE that claims a huge image (an
# extreme ratio). We read the header dimensions cheaply — without decoding pixels
# — and reject only the over-ceiling or extreme-ratio cases, so genuine scans of
# any realistic size pass. The finite MAX_IMAGE_MEGAPIXELS ceiling on OpenCV/Pillow
# is the hard backstop; this is the early, clearly-explained check.
class ImageRejected(Exception):
    """Raised when an uploaded image is over the pixel ceiling or looks like a bomb."""


try:
    _IMG_MAX_DECODE_RATIO = float(os.environ.get("MAX_IMAGE_DECODE_RATIO", "300") or "300")
except ValueError:
    _IMG_MAX_DECODE_RATIO = 300.0
_IMG_BOMB_FLOOR_PX = 40 * 1_000_000     # don't ratio-check images under ~40 MP


def _peek_image_size(src):
    """(width, height) from an image header without decoding pixels, or None if it
    isn't a readable image. Propagates Pillow's DecompressionBombError."""
    try:
        with Image.open(src) as im:
            return int(im.width), int(im.height)
    except Image.DecompressionBombError:
        raise
    except Exception:
        return None


_IMAGE_UPLOAD_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _validated_image_ext(file_storage, default=".jpg"):
    """Extension to save a user-uploaded image under, or None if the upload must
    be rejected. The client-supplied extension is only trusted when it's on the
    image allowlist AND the file's header actually parses as an image — uploads
    are served same-origin from /uploads, so an .html/.svg file saved with its
    original extension would execute as script in the viewer's session. Leaves
    the stream rewound so the caller can still save it."""
    ext = os.path.splitext(secure_filename(file_storage.filename or ""))[1].lower()
    if not ext:
        ext = default
    if ext not in _IMAGE_UPLOAD_EXTS:
        return None
    stream = getattr(file_storage, "stream", None)
    try:
        if stream is not None:
            stream.seek(0)
        real = _peek_image_size(stream if stream is not None else file_storage)
    except Exception:
        real = None
    finally:
        try:
            if stream is not None:
                stream.seek(0)
        except Exception:
            pass
    if real is None:
        return None
    return ext


def _guard_image_upload(file_storage):
    """Validate a Werkzeug FileStorage image against decompression-bomb heuristics
    before it is saved or decoded. Raises ImageRejected on a likely bomb / oversized
    image. Non-images are ignored (downstream handles them). Leaves the upload
    stream rewound so the caller can still save it."""
    if file_storage is None:
        return
    stream = getattr(file_storage, "stream", None)
    if stream is None:
        return
    try:
        start = stream.tell()
    except Exception:
        start = 0
    try:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
    except Exception:
        size = 0
    finally:
        try:
            stream.seek(0)
        except Exception:
            pass

    try:
        dims = _peek_image_size(stream)
    except Image.DecompressionBombError:
        _rewind(stream, start)
        raise ImageRejected("Image exceeds the decoded-size limit (possible decompression bomb).")
    _rewind(stream, start)

    if not dims:
        return   # not a decodable image header; let the route decide what to do
    w, h = dims
    px = w * h
    if px > _MAX_IMAGE_PIXELS:
        raise ImageRejected(
            f"Image is {px / 1_000_000:.0f} MP, over the {_MAX_IMAGE_MP:.0f} MP limit. "
            f"If it's a genuine scan, raise MAX_IMAGE_MEGAPIXELS.")
    decoded = px * 4   # worst-case bytes at RGBA
    if size > 0 and px > _IMG_BOMB_FLOOR_PX and decoded > size * _IMG_MAX_DECODE_RATIO:
        raise ImageRejected(
            "This image looks like a decompression bomb — its decoded size is far "
            "larger than the file. If it's a real scan, re-save it as PNG or JPEG and retry.")


def _rewind(stream, pos):
    try:
        stream.seek(pos)
    except Exception:
        try:
            stream.seek(0)
        except Exception:
            pass


def _reject_if_bomb(*file_storages):
    """Guard one or more uploaded images; return a Flask (json, 413) error response
    if any is a decompression bomb / oversized, else None. Usage in a route:
        bad = _reject_if_bomb(file);  if bad: return bad"""
    try:
        for fs in file_storages:
            _guard_image_upload(fs)
        return None
    except ImageRejected as exc:
        return jsonify({"status": "error", "message": str(exc)}), 413


def _same_origin_next(nxt, fallback):
    """Return `nxt` if it is a site-relative path the browser cannot resolve off this
    origin; otherwise `fallback`. For post-login "?next=" style redirect targets.

    Two things make this harder than "starts with / and not //":

    1. Browsers DELETE ASCII tab, LF and CR from a URL before parsing it (WHATWG URL:
       "Remove all ASCII tab or newline from input"). So the string this function
       inspects is not the string the browser resolves. "/<tab>/evil.com" reads as a
       path here and lands on https://evil.com/ there. They are stripped first, and
       the STRIPPED value is what gets returned -- validating one string and handing
       back another is how this check would go quietly out of sync with itself.
    2. A backslash starts an authority just like a slash, because the parser treats
       \\ as / for special schemes. So "/\\evil.com" and "/\\/evil.com" are
       protocol-relative too, not paths.

    Verified against a real WHATWG parser rather than reasoned about: of the payload
    shapes tried, this admits none that resolve to another origin, and still admits
    "/a/b?x=1#f", "/ /x", "/%2f/x" and "/%5cx", which are ordinary same-origin paths."""
    s = str(nxt or "").translate({0x09: None, 0x0A: None, 0x0D: None})
    if not s.startswith("/") or s[1:2] in ("/", "\\"):
        return fallback
    return s


def _external_http_url(u):
    """Return `u` if it is an absolute http(s) URL, else "". For values that reach an
    href/src, where the danger is the SCHEME rather than the characters.

    Escaping does not help at this sink and it is worth being explicit about why:
    Jinja autoescaping (and the templates' esc()) stop a value from breaking OUT of the
    attribute, which is a different attack. They leave "javascript:alert(1)" exactly as
    it is, and it is still a working href. Only an allowlist of schemes closes it.

    Trim first, then lower, because leading whitespace and mixed case are both accepted
    by browsers in a URL scheme. Relative URLs are deliberately NOT allowed here: every
    caller is rendering a link OUT to a marketplace, so "" (render no link) is the
    correct answer for anything else."""
    s = str(u or "").strip()
    return s if s.lower().startswith(("http://", "https://")) else ""


def _csv_safe(v):
    """Neutralize spreadsheet formula injection: a cell that begins with = + - @ or
    a control char is prefixed with an apostrophe so Excel/Sheets/Numbers treat it
    as text rather than executing it. Use on every user-influenced CSV cell."""
    if v is None:
        return ""
    s = str(v)
    if s[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + s
    return s



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
# External card-identification providers used when local OCR/catalog lookup fails.
VALID_IDENTIFY_PROVIDERS = ("none", "ximilar", "cardsight")
SYSTEM = {"mode": None, "unlimited_native_import": False,
          "ximilar_fallback_enabled": True, "identify_provider": "ximilar"}

# Minimum swap required before the operator may enable unlimited-native import
# (rendering full-resolution scans can need multiple GB per page).
REQUIRED_SWAP_BYTES = 8 * 1000 ** 3   # ~8 GB


def load_system_config():
    try:
        with open(SYSTEM_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        mode = data.get("mode")
        prov = data.get("identify_provider")
        return {
            "mode": mode if mode in VALID_MODES else None,
            "unlimited_native_import": bool(data.get("unlimited_native_import", False)),
            "ximilar_fallback_enabled": bool(data.get("ximilar_fallback_enabled", True)),
            "identify_provider": prov if prov in VALID_IDENTIFY_PROVIDERS else None,
        }
    except (OSError, ValueError):
        return {"mode": None, "unlimited_native_import": False,
                "ximilar_fallback_enabled": True, "identify_provider": None}


def save_system_config():
    tmp = SYSTEM_CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({
            "mode": SYSTEM["mode"],
            "unlimited_native_import": bool(SYSTEM.get("unlimited_native_import", False)),
            "ximilar_fallback_enabled": bool(SYSTEM.get("ximilar_fallback_enabled", True)),
            "identify_provider": _identify_provider(),
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


def _ximilar_fallback_on():
    """Whether the Ximilar card-ID fallback toggle is ON — independent of whether
    an API key is configured (so we can surface a 'key missing' error when it's on
    but unusable). Env XIMILAR_IDENTIFY_FALLBACK overrides the stored setting."""
    env = os.environ.get("XIMILAR_IDENTIFY_FALLBACK")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(SYSTEM.get("ximilar_fallback_enabled", True))


def set_ximilar_fallback(enabled):
    SYSTEM["ximilar_fallback_enabled"] = bool(enabled)
    save_system_config()


def _identify_provider():
    """Which external identification service to use when the local OCR/catalog
    lookup fails: 'ximilar', 'cardsight', or 'none'. This is the single control
    for both the import auto-identify and the manual inventory-detail identify.
    Env IDENTIFY_PROVIDER overrides the stored setting. Falls back to the legacy
    ximilar on/off toggle when no explicit provider has been chosen yet."""
    env = os.environ.get("IDENTIFY_PROVIDER")
    if env is not None:
        env = env.strip().lower()
        return env if env in VALID_IDENTIFY_PROVIDERS else "none"
    val = SYSTEM.get("identify_provider")
    if val in VALID_IDENTIFY_PROVIDERS:
        return val
    # Legacy config (before the provider selector existed): derive from the old
    # Ximilar on/off toggle so existing installs keep their behaviour.
    return "ximilar" if SYSTEM.get("ximilar_fallback_enabled", True) else "none"


def set_identify_provider(provider):
    provider = str(provider or "").strip().lower()
    if provider not in VALID_IDENTIFY_PROVIDERS:
        raise ValueError("invalid identification provider")
    SYSTEM["identify_provider"] = provider
    # Keep the legacy Ximilar toggle in sync so any old code/UI that still reads
    # it stays consistent (on unless the provider is 'none').
    SYSTEM["ximilar_fallback_enabled"] = (provider != "none")
    save_system_config()


def _identify_provider_label(provider=None):
    return {"ximilar": "Ximilar", "cardsight": "CardSight", "none": "None"}.get(
        provider or _identify_provider(), "None")


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
        "key": "CARDSIGHT_API_KEY",
        "label": "CardSight AI API Key",
        "description": "Image-based card identification with a free tier (750 calls/month, no credit card). "
                       "Used when 'CardSight' is the selected identification service.",
        "docs": "https://cardsight.ai/documentation",
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


def _record_storage_type(record):
    """A record's storage container kind: 'box' or 'album' (default)."""
    val = str((record.extracted_data or {}).get("storage_type", "") or "").strip().lower()
    return "box" if val == "box" else "album"


def build_storage_index():
    """
    Group owned records into their storage containers (Albums and Boxes).

    A container is identified by its name (the extracted_data 'album' field, kept
    for data continuity). Its kind comes from the records' 'storage_type': an
    Album has pages + 9 slots, a Box is a flat 1..N run with no page numbers.
    When records disagree, the majority kind wins (they share a kind at import).
    """
    # Only rows that can land in a container load: non-catalog (is_catalog
    # mirrors _is_catalog_only) with a non-empty album (album_key is NULL
    # exactly when the album strips to empty). The loop's checks still run.
    from sqlalchemy import func as _f
    records = (ScanRecord.query
               .filter(_f.coalesce(ScanRecord.is_catalog, False) == False,  # noqa: E712
                       ScanRecord.album_key.isnot(None))
               .order_by(ScanRecord.scan_date.desc()).all())
    storage_map = {}

    for record in records:
        data = record.extracted_data or {}
        if _is_catalog_only(data):
            continue

        name = get_record_value(record, "album")
        if not name:
            continue

        game_name = get_record_value(record, "game")

        info = storage_map.setdefault(
            name,
            {
                "name": name,
                "count": 0,
                "latest_scan": record.scan_date,
                "records": [],
                "games": set(),
                "box_votes": 0,
                "album_votes": 0,
            },
        )
        info["count"] += 1
        info["records"].append(record)
        if _record_storage_type(record) == "box":
            info["box_votes"] += 1
        else:
            info["album_votes"] += 1

        if game_name:
            info["games"].add(game_name)

        if record.scan_date and (info["latest_scan"] is None or record.scan_date > info["latest_scan"]):
            info["latest_scan"] = record.scan_date

    containers = []
    for info in storage_map.values():
        stype = "box" if info["box_votes"] > info["album_votes"] else "album"
        containers.append(
            {
                "name": info["name"],
                "count": info["count"],
                "latest_scan": info["latest_scan"],
                "records": info["records"],
                "games": sorted(info["games"]),
                "storage_type": stype,
                "is_box": stype == "box",
            }
        )

    return sorted(
        containers,
        key=lambda item: item["latest_scan"] or datetime.min,
        reverse=True,
    )


# Backwards-compatible alias: older callers referenced build_album_index.
build_album_index = build_storage_index


# ====================== GAME TEMPLATES ======================
# A "template" file (templates/roi/<name>.json) now represents a Game
# definition: just a name plus a flat list of fields that belong to every
# entry for that game. There are no OCR zones / ROI coordinates anymore —
# imported cards are simply created with these fields blank, ready to be
# filled in by hand from the Inventory / Inventory Detail pages.
def load_template(template_name="product_label"):
    # Prefer fields derived from this game's downloaded tcgcsv catalog columns, so
    # each game's entry fields mirror the real per-game columns from the source.
    # Falls back to the on-disk ROI template when the game has no catalog yet.
    try:
        derived = _reference_template_for_name(template_name)
        if derived is not None:
            return derived
    except Exception:
        pass

    folder = app.config["ROI_TEMPLATE_FOLDER"]
    template_path = os.path.join(folder, f"{template_name}.json")
    # Contain within the templates folder: a crafted game/template name must never
    # be able to read arbitrary files via ../ or an absolute path.
    if not _within_dir(folder, template_path) or not os.path.exists(template_path):
        template_path = os.path.join(folder, "product_label.json")

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
    names = {
        f.replace(".json", "")
        for f in os.listdir(app.config["ROI_TEMPLATE_FOLDER"])
        if f.endswith(".json")
    }
    # Every downloaded tcgcsv game is a usable "game" too — its entry fields are
    # derived from the catalog columns even without an on-disk template file.
    try:
        for rs in ReferenceSync.query.all():
            slug = _slugify_template_name(rs.game)
            if slug:
                names.add(slug)
    except Exception:
        pass
    if not names:
        names = {"product_label"}
    return sorted(names)


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

    # Narrow to the album's rows with the indexed keys first — a normalized
    # match is a strict superset of the raw comparisons below, so this can
    # never exclude a row the old full-table scan would have returned — then
    # confirm with exactly the same raw comparisons as before.
    candidates = (
        ScanRecord.query
        .filter(ScanRecord.game_key == str(game).strip().lower(),
                ScanRecord.album_key == str(album).strip().lower())
        .order_by(ScanRecord.id)
        .all()
    )
    for record in candidates:
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

    extracted = dict(extracted)
    # Every new entry starts "Held" (in your possession). Catalog/reference rows
    # from CSV import are not owned inventory, so they don't get the flag.
    if "held" not in extracted and not _is_catalog_only(extracted):
        extracted["held"] = True

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


def _reference_upsert(rec, cache=None):
    """Insert or update a ReferenceCard from a normalized tcgcsv product dict
    (see ref_sync.normalize_product). Keyed on the upstream productId.

    cache: optional {product_id: ReferenceCard} the caller preloaded for its
    whole payload. When given it replaces the per-row lookup entirely — one
    SELECT per sync instead of one per card — and rows created here are added
    to it so a duplicate productId later in the same payload updates the row
    instead of inserting twice (the same net behavior autoflush gave the
    per-row lookup)."""
    pid = rec["product_id"]
    if cache is not None:
        existing = cache.get(pid)
    else:
        existing = ReferenceCard.query.filter_by(product_id=pid).first()
    if existing is None:
        existing = ReferenceCard(product_id=pid)
        db.session.add(existing)
        if cache is not None:
            cache[pid] = existing
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


# ============================================================================ #
# Per-game entry fields derived from the tcgcsv catalog columns
# ----------------------------------------------------------------------------
# A game's entry fields come from the columns tcgcsv actually provides for that
# game rather than a hand-authored ROI template. That means the core per-card
# columns (name, set, collector number, rarity) plus every distinct extendedData
# key the catalog carries — which differ per game (Pokemon: HP/Stage/Energy Type;
# Magic: Mana Cost/Power/Toughness; Yu-Gi-Oh: Attribute/Level; ...). A column with
# a small, categorical value set becomes a dropdown (options taken from the
# catalog); everything else is free text. Results are cached per category and
# auto-invalidated when the catalog's card count changes.
_REF_FIELDS_CACHE = {}            # category_id -> (product_count, fields_dict)
_REF_FIELD_DROPDOWN_MAX = 40      # distinct values at/under which a column is a dropdown
_REF_FIELD_SAMPLE = 5000          # cards scanned per game to discover columns/values


def _infer_ref_field(values):
    """Pick {field_type[, dropdown_options]} for a column from its observed values.
    Short, low-cardinality, non-numeric value sets become dropdowns; the rest is text."""
    distinct = sorted({str(v).strip() for v in values if str(v).strip()})
    n = len(distinct)
    if (2 <= n <= _REF_FIELD_DROPDOWN_MAX
            and all(len(v) <= 40 for v in distinct)
            and not all(_re.fullmatch(r"-?\d+(?:\.\d+)?", v) for v in distinct)):
        return {"field_type": "dropdown", "dropdown_options": distinct}
    return {"field_type": "text"}


def _reference_fields_config(category_id):
    """Derive a downloaded game's entry-field definitions (template "fields" shape)
    from its tcgcsv catalog columns. Returns {} if the game isn't downloaded."""
    rs = ReferenceSync.query.filter_by(category_id=category_id).first()
    if rs is None:
        return {}
    count = rs.product_count or 0
    cached = _REF_FIELDS_CACHE.get(category_id)
    if cached and cached[0] == count:
        return cached[1]

    cards = (ReferenceCard.query
             .filter_by(category_id=category_id)
             .order_by(ReferenceCard.id)
             .limit(_REF_FIELD_SAMPLE).all())

    core_keys = {"name", "set", "number", "rarity"}
    set_vals, rarity_vals = set(), set()
    ext_values = {}       # field_key -> set of distinct values (bounded)
    ext_order = []        # first-seen field_key order
    ext_capped = set()    # keys whose value set overflowed -> forced to text

    for c in cards:
        if c.set_name:
            set_vals.add(str(c.set_name).strip())
        if c.rarity:
            rarity_vals.add(str(c.rarity).strip())
        for raw_key, raw_val in (c.extended or {}).items():
            fk = _slugify_template_name(raw_key)
            if not fk or fk in core_keys:
                continue
            if fk not in ext_values and fk not in ext_capped:
                ext_values[fk] = set()
                ext_order.append(fk)
            if fk in ext_capped:
                continue
            sv = str(raw_val or "").strip()
            if sv:
                ext_values[fk].add(sv)
                if len(ext_values[fk]) > _REF_FIELD_DROPDOWN_MAX:
                    ext_capped.add(fk)   # too many distinct values to be a dropdown

    fields = {
        "name":   {"field_type": "text"},
        "set":    _infer_ref_field(set_vals),
        "number": {"field_type": "text"},
        "rarity": _infer_ref_field(rarity_vals),
    }
    for fk in ext_order:
        if fk in fields:
            continue
        fields[fk] = ({"field_type": "text"} if fk in ext_capped
                      else _infer_ref_field(ext_values.get(fk, ())))

    _REF_FIELDS_CACHE[category_id] = (count, fields)
    return fields


def _reference_template_for_name(template_name):
    """If `template_name` maps to a downloaded tcgcsv game, return a template dict
    whose fields are derived from that game's catalog columns. A matching on-disk
    file (if any) supplies optional per-field overrides (type/hidden) and a legacy
    csv_column_mapping. Returns None when the game has no downloaded catalog."""
    cat_id, cat_game = _resolve_category_for_game(template_name)
    if cat_id is None:
        return None
    derived = _reference_fields_config(cat_id)
    if not derived:
        return None

    fields = {k: dict(v) for k, v in derived.items()}

    # Fold in optional overrides from an on-disk template, so the live field-type
    # / hide edits and any legacy CSV column mapping still apply to catalog games.
    csv_map = None
    try:
        slug = _slugify_template_name(template_name)
        fpath = os.path.join(app.config["ROI_TEMPLATE_FOLDER"], f"{slug}.json")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as fh:
                on_disk = json.load(fh) or {}
            for fk, cfg in (on_disk.get("fields") or {}).items():
                if fk not in fields or not isinstance(cfg, dict):
                    continue
                ftype = cfg.get("field_type")
                if ftype in ("text", "dropdown", "boolean"):
                    fields[fk]["field_type"] = ftype
                    if ftype == "dropdown" and cfg.get("dropdown_options"):
                        fields[fk]["dropdown_options"] = list(cfg["dropdown_options"])
                    elif ftype != "dropdown":
                        fields[fk].pop("dropdown_options", None)
                if cfg.get("hidden"):
                    fields[fk]["hidden"] = True
            csv_map = on_disk.get("csv_column_mapping") or None
    except Exception:
        pass

    out = {"name": _slugify_template_name(template_name) or template_name,
           "fields": fields, "source": "reference",
           "category_id": cat_id, "category_game": cat_game}
    if csv_map:
        out["csv_column_mapping"] = csv_map
    return out


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


# ---- FTS5 search helpers ---------------------------------------------------
# The FTS tables (scan_search / ref_search, trigram tokenizer) are created by
# migrate_add_search_fts(). These helpers return the SAME logical filter either
# way — a LIKE '%q%' match, ASCII-case-insensitive — but route it through the
# FTS index when the table exists and the query has >= 3 characters (a trigram
# needs three); otherwise the original scan runs unchanged.
_FTS_READY = {}   # {table_name: bool}, probed once per process

# Lightweight handles for the FTS virtual tables (not in db.metadata, so
# create_all/drop_all never touch them). Using Core constructs here means
# .like() gets a fresh anonymous bindparam per call — two conditions in one
# statement can never collide the way fixed-name text() params silently do.
from sqlalchemy import table as _sa_lite_table, column as _sa_lite_column
_scan_search_t = _sa_lite_table("scan_search", _sa_lite_column("rowid"),
                                _sa_lite_column("extracted_data"))
_ref_search_t = _sa_lite_table("ref_search", _sa_lite_column("rowid"),
                               _sa_lite_column("name"))


def _fts_ready(table):
    """Is the FTS table actually usable on this connection?

    A real probe query on a dedicated connection, not a sqlite_master lookup:
    a DB file can carry the FTS tables while the runtime's SQLite lacks the
    fts5 module or trigram tokenizer (e.g. the same collection opened on an
    older Pi build), and on non-SQLite backends the tables never exist at
    all. Any failure caches False — searches then use the portable LIKE scan
    for the rest of the process, which returns identical rows, just slower.
    migrate_add_search_fts() clears the cache after (re)creating the tables.
    """
    if table not in _FTS_READY:
        try:
            if db.engine.dialect.name != "sqlite":
                _FTS_READY[table] = False
            else:
                with db.engine.connect() as conn:
                    conn.exec_driver_sql(f"SELECT rowid FROM {table} LIMIT 0")
                _FTS_READY[table] = True
        except Exception:
            _FTS_READY[table] = False
    return _FTS_READY[table]


def _scan_search_condition(search):
    """'This text appears anywhere in the record JSON', as a filter clause."""
    pattern = f"%{search}%"
    if len(search) >= 3 and _fts_ready("scan_search"):
        return ScanRecord.id.in_(
            db.select(_scan_search_t.c.rowid)
              .where(_scan_search_t.c.extracted_data.like(pattern)))
    return ScanRecord.extracted_data.cast(db.Text).ilike(pattern)


def _ref_name_condition(fragment):
    """'This text appears in the reference card name', as a filter clause."""
    pattern = f"%{fragment}%"
    if len(fragment) >= 3 and _fts_ready("ref_search"):
        return ReferenceCard.id.in_(
            db.select(_ref_search_t.c.rowid)
              .where(_ref_search_t.c.name.like(pattern)))
    return ReferenceCard.name.ilike(pattern)


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
        narrowed = base.filter(_ref_name_condition(first)).limit(500).all()
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


def _database_match_candidates(category_id, name, number, limit=25):
    """
    Match a typed Name + collector Number against a game's downloaded catalog
    (no OCR/image). Returns (candidates, exact_ids):

      candidates -> best-first list of dicts (product_id, name, number, set,
                    rarity, market_price, image_url, url, score, exact)
      exact_ids  -> set of product_ids that match exactly. "Exact" means: both
                    the normalized name AND the collector number match when both
                    were supplied; otherwise the single supplied field matches.

    Number matching is padding-insensitive (24/112 == 024/112). The catalog is
    pre-narrowed by number, then by the name's first token, so we never fuzzy
    score an entire game.
    """
    import difflib

    def _nname(s):
        return _re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()

    qname = _nname(name)
    qnum  = str(number or "").strip()
    qvariants = _collector_number_variants(qnum) or ({qnum} if qnum else set())
    qcanon = _canonical_collector_number(qnum) if qnum else ""

    base = ReferenceCard.query.filter(ReferenceCard.category_id == category_id)

    pool = []
    if qnum:
        pool = base.filter(ReferenceCard.number.in_(list(qvariants) or [qnum])).limit(400).all()
    if qname and not pool:
        first = (qname.split() or [qname])[0]
        pool = base.filter(_ref_name_condition(first)).limit(600).all()
    elif qname and qnum:
        # Add name-token matches too, so a padding/number mismatch still surfaces
        # the right card in the picker rather than hiding it.
        first = (qname.split() or [qname])[0]
        seen = {c.product_id for c in pool}
        pool += [c for c in base.filter(_ref_name_condition(first)).limit(400).all()
                 if c.product_id not in seen]
    if not pool:
        pool = base.limit(800).all()

    def _num_ok(cardnum):
        cn = str(cardnum or "").strip()
        if not (qnum and cn):
            return False
        return cn in qvariants or (bool(qcanon) and _canonical_collector_number(cn) == qcanon)

    out, exact_ids = [], set()
    for rc in pool:
        cn_norm  = _nname(rc.name)
        name_sim = difflib.SequenceMatcher(None, qname, cn_norm).ratio() if qname else 0.0
        name_ok  = bool(qname) and cn_norm == qname
        num_ok   = _num_ok(rc.number)

        if qname and qnum:
            exact = name_ok and num_ok
            score = (0.5 if num_ok else 0.0) + 0.5 * name_sim
        elif qnum:
            exact = num_ok
            score = 1.0 if num_ok else 0.0
        else:  # name only
            exact = name_ok
            score = name_sim

        if score <= 0 and not exact:
            continue
        if exact:
            exact_ids.add(rc.product_id)
        out.append({
            "product_id":   rc.product_id,
            "name":         rc.name or "",
            "number":       rc.number or "",
            "set":          rc.set_name or "",
            "rarity":       rc.rarity or "",
            "market_price": rc.market_price,
            "image_url":    rc.image_url or "",
            "url":          rc.url or "",
            "score":        round(min(1.0, score), 4),
            "exact":        bool(exact),
        })

    out.sort(key=lambda c: (c["exact"], c["score"]), reverse=True)
    return out[:limit], exact_ids


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


def _raw_field(data: dict, keys) -> str:
    """First non-empty value among `keys`, preserving original case/format.
    Unlike _get_name/_get_serial (which lowercase for matching), this returns the
    value as the user typed it — used when we need to display or re-match it."""
    for k in keys:
        v = str((data or {}).get(k, "")).strip()
        if v:
            return v
    return ""


# ====================== AUTO-IDENTIFY (used at end of import) ======================
# Minimum match score to auto-apply the top OCR identification on import. Scores
# come from match_ocr_to_records (capped at 1.0): an exact collector-number (N/M)
# match contributes +0.5 and the card-name similarity contributes up to +0.65, so
# 0.60 means "a confident combined name + number match" — e.g. an exact number
# plus even a partial name read, or a near-exact name on its own. Below this the
# entry is left blank for the person to check by hand.
#
# This is no longer a fixed number: it's a user setting (Settings → "Auto-accept
# confidence" slider) persisted in the AppSetting key/value store, so it can be
# tuned without editing code or restarting. Raise it for fewer wrong auto-fills
# and more cards left blank; lower it for more hands-off imports at the cost of
# occasional misidentifications. The AUTO_IDENTIFY_MIN_SCORE env var still works
# as an override for headless deploys (get_setting prefers the stored value, then
# the environment, then the default below).
AUTO_IDENTIFY_MIN_SCORE_KEY     = "AUTO_IDENTIFY_MIN_SCORE"
AUTO_IDENTIFY_MIN_SCORE_DEFAULT = 0.60
# Slider bounds. The floor keeps the setting inside the range where a score is
# still meaningful: match_ocr_to_records only *returns* candidates at >= 0.45, so
# anything below that would accept whatever the matcher happened to surface.
AUTO_IDENTIFY_MIN_SCORE_FLOOR   = 0.45
AUTO_IDENTIFY_MIN_SCORE_CEIL    = 1.00


def _coerce_min_score(value, default=AUTO_IDENTIFY_MIN_SCORE_DEFAULT):
    """Parse a threshold from a setting value / form field / JSON body into a
    0..1 float, clamped to the slider bounds.

    Accepts either a fraction ("0.75") or a percentage ("75", "75%") so the UI
    can post whichever is convenient and old env values keep working. Anything
    unparseable falls back to `default`."""
    raw = str(value if value is not None else "").strip().rstrip("%").strip()
    if not raw:
        return default
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return default
    if num > 1.0:                     # given as a percentage (e.g. 75 -> 0.75)
        num = num / 100.0
    return max(AUTO_IDENTIFY_MIN_SCORE_FLOOR, min(AUTO_IDENTIFY_MIN_SCORE_CEIL, num))


def auto_identify_min_score():
    """The current auto-accept threshold (0..1). Read fresh on every call so a
    change made in Settings takes effect on the very next card — no restart."""
    return _coerce_min_score(get_setting(AUTO_IDENTIFY_MIN_SCORE_KEY, ""),
                             AUTO_IDENTIFY_MIN_SCORE_DEFAULT)


def set_auto_identify_min_score(value):
    """Persist the auto-accept threshold. Returns the clamped value actually
    stored, so the caller can echo back what the slider should snap to."""
    score = _coerce_min_score(value, AUTO_IDENTIFY_MIN_SCORE_DEFAULT)
    set_setting(AUTO_IDENTIFY_MIN_SCORE_KEY, f"{score:.2f}")
    return score


def auto_identify_min_percent():
    """The threshold as a whole-number percentage, for display."""
    return int(round(auto_identify_min_score() * 100))


# Ties are decided at FULL precision: the decimal places matter. Scores arrive
# rounded to three decimals from match_ocr_to_records, so 0.884 and 0.881 are two
# different scores and the higher one wins outright — only genuinely identical
# scores are a tie. The epsilon absorbs float representation error, nothing more.
_TIE_EPSILON = 1e-9


def rank_reference_matches(ref_matches, min_score=None):
    """
    Decide whether a set of REFERENCE-CATALOG matches identifies a card.

    This is the single ranking rule, shared by the import auto-identify and the
    card-detail page so the two can never disagree. Only reference (catalog)
    matches are considered — matches against the user's own existing records are
    never used to auto-fill, because the collector-number bonus lets an unrelated
    card that merely shares a number (e.g. "25/102") outscore the real match.

    Rules, in order:
      1. The highest-scoring match at or above `min_score` wins. Scores are
         compared at full precision, so the higher of two close scores (0.884 vs
         0.881) is the winner rather than a tie.
      2. If two or more matches share the identical top score, nothing is
         applied — the card is ambiguous and goes to manual review.
      3. If nothing reaches `min_score`, nothing is applied.

    Returns a decision dict:
        { decision: "apply" | "ambiguous" | "below_threshold" | "no_candidates",
          winner:   the winning candidate, or None,
          tied:     the tied candidates when decision == "ambiguous",
          top_score, runner_up_score, min_score }
    """
    min_score = auto_identify_min_score() if min_score is None else _coerce_min_score(min_score)
    out = {"decision": "no_candidates", "winner": None, "tied": [],
           "top_score": None, "runner_up_score": None, "min_score": min_score}

    ranked = sorted((c for c in (ref_matches or []) if c is not None),
                    key=lambda c: float(c.get("score", 0) or 0), reverse=True)
    if not ranked:
        return out

    top_score = float(ranked[0].get("score", 0) or 0)
    out["top_score"] = top_score
    if len(ranked) > 1:
        out["runner_up_score"] = float(ranked[1].get("score", 0) or 0)

    # Rule 3 — nothing is confident enough. (The epsilon keeps a score that is
    # exactly the threshold from being rejected by float representation.)
    if top_score < float(min_score) - _TIE_EPSILON:
        out["decision"] = "below_threshold"
        return out

    # Rule 2 — only an identical score ties. A card scoring even 0.001 higher
    # than the next is a clear winner under rule 1.
    tied = [c for c in ranked
            if abs(float(c.get("score", 0) or 0) - top_score) <= _TIE_EPSILON]
    if len(tied) > 1:
        out["decision"] = "ambiguous"
        out["tied"] = tied
        return out

    # Rule 1 — a single clear winner.
    out["decision"] = "apply"
    out["winner"] = ranked[0]
    return out

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
        img = _imread(abs_path)
        if img is None:
            continue
        raw.append({"type_name": r.type_name, "region": r.region, "image": img})
    prepared = card_ocr.prepare_type_references(raw)
    _TYPE_REF_CACHE[key] = (_TYPE_REF_VERSION, prepared)
    return prepared


def _reference_extended_fill(record, ref):
    """
    Non-destructive fill of a game's tcgcsv extendedData columns (HP, Stage,
    Energy Type, attacks, Mana Cost, Power/Toughness, ...) from a matched
    reference card. Returns {field_key: value} for the game's fields the record
    hasn't filled yet — it never overwrites a value already present and never
    blanks anything.

    This lets an imported / OCR-identified card carry the same rich column data
    that a Database Match applies, without clobbering hand-entered values. (The
    identity fields name/number/set/rarity/game are handled separately by
    _REFERENCE_APPLY_MAP; here we only add the extra per-game columns.)
    """
    ext = {}
    for raw_k, raw_v in (getattr(ref, "extended", None) or {}).items():
        fk = _slugify_template_name(raw_k)
        sv = "" if raw_v is None else str(raw_v).strip()
        if fk and sv:
            ext.setdefault(fk, sv)              # first non-empty value wins
    if not ext:
        return {}

    # Restrict to the game's known field set so we never inject stray columns.
    try:
        _g = (record.extracted_data or {}).get("game", "") or record.template_used or "product_label"
        tpl_fields = set((load_template(_g).get("fields") or {}).keys())
    except Exception:
        tpl_fields = set()

    data = record.extracted_data or {}
    out = {}
    for fk, val in ext.items():
        if tpl_fields and fk not in tpl_fields:
            continue
        if fk in _OVERWRITE_PROTECT:
            continue
        if str(data.get(fk) or "").strip():     # keep anything already entered
            continue
        out[fk] = val
    return out


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
        # Fill the game's remaining tcgcsv columns (HP/Stage/attacks/...) from the
        # catalog card — non-destructive, so nothing already entered is touched.
        for _fk, _val in _reference_extended_fill(record, ref).items():
            updates.setdefault(_fk, _val)
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


def auto_identify_record(record, min_score=None):
    """
    Identify a record from its FRONT image and fill the entry in, or leave it
    blank for manual review.

    Obtain
      1. OCR the front image (name + collector number).
      2. Score that read against the REFERENCE DATA for the game the scan was
         filed under.

    Rank
      1. The reference match with the highest percentage, provided it is at or
         above the configured minimum, is applied automatically.
      2. If two or more reference matches tie at that top percentage, neither is
         applied — the entry is left blank for manual review.
      3. If no reference match reaches the minimum, the entry is left blank for
         manual review.

    Matches against the user's OWN existing records are deliberately not used to
    auto-fill: the matcher's collector-number bonus lets an unrelated card that
    merely shares a number outscore the real one. They remain available in the
    manual picker on the card-detail page, where a person is choosing.

    `min_score` defaults to the "Auto-accept confidence" setting (Settings page
    slider, 60% out of the box), resolved on every call so a change to the slider
    affects the very next card scanned.

    Never raises and never commits: any OCR/matching problem simply yields
    identified=False, and the caller decides when to persist.

    Separately from the identity, it also fills the Game's "type" field (e.g.
    Pokemon energy type) when the card provides one — from the matched reference
    card if identified, otherwise from a confident VISUAL reading of the type
    icon. This runs even when there's no identity match, so cards can still be
    sorted by type.

    Returns: { identified, reason, score, name, applied: {..}, min_score,
               runner_up_score, candidates_considered, tied: [..],
               type_guess, type_confidence, type_applied: {field, value}|None }

    `reason` is one of: applied | applied_<provider> | ambiguous_match |
    below_threshold | no_candidates | no_reference_data | apply_failed |
    ocr_unavailable | ocr_error | no_front_image | <provider>_error.
    """
    # Resolve the user-configured auto-accept threshold now (callers may pass an
    # explicit override). Reported back in `min_score` so the UI / import summary
    # can say what bar a card had to clear.
    min_score = auto_identify_min_score() if min_score is None else _coerce_min_score(min_score)

    out = {"identified": False, "reason": "", "score": None, "name": "",
           "applied": {}, "type_guess": "", "type_confidence": 0.0, "type_applied": None,
           "source": "", "error": "", "min_score": min_score,
           "runner_up_score": None, "candidates_considered": 0, "tied": []}

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

    # ── Obtain step 2: score the OCR read against this game's REFERENCE DATA ──
    category_id, _ = _resolve_category_for_game(game)
    ref_matches = _reference_candidates_for_ocr(category_id, ocr) if category_id else []
    out["candidates_considered"] = len(ref_matches)

    # ── Rank step: one clear winner over the bar, or leave it blank ──
    decision = rank_reference_matches(ref_matches, min_score)
    out["runner_up_score"] = decision["runner_up_score"]
    if decision["top_score"] is not None:
        out["score"] = decision["top_score"]

    winner = decision["winner"]
    if winner is not None:
        out["name"] = winner.get("name", "")
    elif ref_matches:
        # Nothing was applied, but report the closest read so the import summary
        # can show how near the card came. Callers gate the displayed name on
        # `identified`, so this is diagnostic only and never shown as the answer.
        best = max(ref_matches, key=lambda c: float(c.get("score", 0) or 0))
        out["name"] = best.get("name", "")

    if decision["decision"] == "apply":
        applied = _apply_ocr_candidate(record, winner)
        if applied:
            out["identified"] = True
            out["reason"] = "applied"
            out["applied"] = applied
            out["source"] = "reference"
        else:
            # The catalog row behind the winning match has gone (catalog re-synced
            # or pruned since scoring). Nothing to apply — treat it as unidentified
            # so the fallback below still gets its turn.
            out["reason"] = "apply_failed"
    elif decision["decision"] == "ambiguous":
        # Rule 2 — two or more reference cards are equally good reads. Picking one
        # would be a coin flip, so the entry stays blank and both are reported.
        out["reason"] = "ambiguous_match"
        out["tied"] = [{"name": c.get("name", ""), "serial": c.get("serial", ""),
                        "set": c.get("set", ""), "score": c.get("score"),
                        "product_id": c.get("product_id")}
                       for c in decision["tied"]]
    elif decision["decision"] == "below_threshold":
        out["reason"] = "below_threshold"          # Rule 3
    else:
        # No scored candidates at all: either the game's catalog was never synced
        # or the OCR read matched nothing in it.
        out["reason"] = "no_reference_data" if not category_id else "no_candidates"

    # ── External identification fallback (opt-in) ──
    # Runs only when the reference data did NOT produce a confident single match,
    # using the provider selected in Settings → General. With the provider set to
    # 'none' — the local-database-only configuration — this is skipped entirely
    # and the entry is simply left blank, exactly as the rank rules specify.
    _provider = _identify_provider()
    if not out["identified"] and _provider != "none":
        xi, xi_err = _external_identify_card_ex(record.image_path)
        if xi:
            applied = _apply_external_identification(record, xi, category_id)
            if applied:
                out["identified"] = True
                out["reason"] = f"applied_{_provider}"
                out["applied"] = applied
                out["source"] = _provider
                out["name"] = applied.get("name") or xi.get("name") or out["name"]
        elif xi_err:
            out["error"] = xi_err
            # Keep the local reason (why the catalog didn't decide) and record the
            # provider failure alongside it, rather than masking one with the other.
            if not out["reason"]:
                out["reason"] = f"{_provider}_error"

    # ── Type field — independent of the identity match ──
    # Prefer the catalog value (authoritative) when a reference card was applied;
    # otherwise use a confident visual guess. _apply_ocr_candidate may already
    # have filled it from the catalog, in which case _fill_type_field is a no-op.
    # This runs even for ambiguous/blank entries so cards remain sortable by type.
    type_value = ""
    if out["identified"] and winner is not None and out["source"] == "reference":
        type_value = _reference_type_value(winner.get("product_id"))
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
# ====================== AUTHENTICATION & ROLES ======================
# Session-based login with per-role, per-tab/tool "view" / "edit" permissions.
# Enforced centrally in a before_request gate: GET/HEAD needs "view", any mutating
# method needs "edit". Admin roles bypass all checks. A fresh install (no accounts)
# is routed to a one-time setup page to create the first administrator, and the
# DISABLE_AUTH=1 env var is a kill-switch so you can never be permanently locked out.
from werkzeug.security import generate_password_hash, check_password_hash

# The tabs/tools that can be permissioned. (key, human label)
PROTECTED_RESOURCES = [
    ("templates",    "Game Templates (home)"),
    ("inventory",    "Inventory"),
    ("albums",       "Albums"),
    ("import",       "Import / scanning"),
    ("duplicates",   "Duplicates"),
    ("analytics",    "Analytics"),
    ("reports",      "Financial reports"),
    ("quick_scan",   "Quick Scan (camera)"),
    ("image_search", "Search by Image"),
    ("reference",    "Reference Data"),
    ("shops",        "Shops / listings"),
    ("pricing",      "Pricing lookups"),
    ("identify",     "Card identification"),
    ("api_keys",     "API Keys"),
    ("storage",      "Storage locations"),
    ("network",      "Network name"),
    ("settings",     "General settings"),
    ("upgrade",      "Upgrade / export"),
]
_RESOURCE_KEYS = {k for k, _ in PROTECTED_RESOURCES}
_PERM_RANK = {"none": 0, "view": 1, "edit": 2}


class Role(db.Model):
    __tablename__ = "auth_roles"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    permissions = db.Column(db.JSON, default=dict)   # {resource_key: none|view|edit}
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class User(db.Model):
    __tablename__ = "auth_users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role_id = db.Column(db.Integer, db.ForeignKey("auth_roles.id"))
    role = db.relationship("Role")
    active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        try:
            return check_password_hash(self.password_hash, pw)
        except Exception:
            return False


class SecurityEvent(db.Model):
    """Append-only record of security-relevant actions.

    Names are captured AS THEY WERE, not looked up later through a foreign key: the
    point of this table is to survive the account being renamed or deleted, which is
    exactly what someone covering their tracks would do.
    """
    __tablename__ = "auth_security_events"
    id = db.Column(db.Integer, primary_key=True)
    at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    action = db.Column(db.String(40), nullable=False)
    actor_id = db.Column(db.Integer)          # null for a failed sign-in or anonymous
    actor_name = db.Column(db.String(64))
    target_id = db.Column(db.Integer)
    target_name = db.Column(db.String(64))
    source_ip = db.Column(db.String(45))      # fits IPv6
    detail = db.Column(db.String(80))         # vocabulary only -- never caller text


# The complete vocabulary. `action` and `detail` are checked against these sets, so
# nothing a caller controls can reach either column. The only free-ish values in the
# table are the captured names, and usernames are NOT charset-validated anywhere in
# this app -- which is why the viewer escapes every field it renders.
_EVENT_ACTIONS = frozenset({
    "sign_in", "sign_in_failed", "sign_in_throttled", "sign_out", "first_admin_created",
    "user_created", "user_updated", "user_deleted", "password_changed",
    "role_saved", "role_deleted", "storage_root_changed", "migration_bundle_exported",
})
_EVENT_DETAILS = frozenset({
    "", "by_admin", "self_service", "role_changed", "activated", "deactivated",
    "password_reset", "role_and_password", "bad_password", "unknown_user", "inactive_user",
})
EVENT_LOG_KEEP = 5000          # rows retained; the oldest are pruned periodically
_EVENT_PRUNE_EVERY = 64        # prune once per this many writes (amortized)
_EVENT_TABLE_READY = {"ok": False}


def _event_table_ready():
    """Create the table on first use.

    db.create_all() and every migration in this app run only from the __main__
    block, so a WSGI deployment would otherwise come up without this table and every
    write below would raise. checkfirst=True makes it a no-op once it exists.
    """
    if _EVENT_TABLE_READY["ok"]:
        return True
    try:
        SecurityEvent.__table__.create(db.engine, checkfirst=True)
        _EVENT_TABLE_READY["ok"] = True
    except Exception:
        return False
    return True


def _clip(v, n=64):
    return (str(v)[:n] if v is not None else None)


def log_security_event(action, actor=None, target=None, detail="",
                       actor_name=None, target_name=None):
    """Best-effort append to the security log. NEVER raises.

    An audit write that can 500 a sign-in is worse than no audit write: it converts a
    logging fault into an outage and hands an attacker a way to deny the action it was
    meant to record. Everything here is swallowed, and the caller is not told.
    """
    try:
        if action not in _EVENT_ACTIONS or detail not in _EVENT_DETAILS:
            return                      # programming error, not a caller's input
        if not _event_table_ready():
            return
        ip = None
        try:
            ip = _clip(request.remote_addr, 45)
        except Exception:
            pass
        row = SecurityEvent(
            action=action, detail=detail, source_ip=ip,
            actor_id=(getattr(actor, "id", None) if actor is not None else None),
            actor_name=_clip(actor_name if actor_name is not None
                             else getattr(actor, "username", None)),
            target_id=(getattr(target, "id", None) if target is not None else None),
            target_name=_clip(target_name if target_name is not None
                              else getattr(target, "username", None)),
        )
        db.session.add(row)
        db.session.commit()
        # Amortized prune: every write used to COUNT the whole table. The id
        # is monotonic (SQLite rowids never reuse a deleted max), so gating on
        # it prunes once per _EVENT_PRUNE_EVERY writes; the table stays within
        # EVENT_LOG_KEEP + _EVENT_PRUNE_EVERY rows instead of exactly
        # EVENT_LOG_KEEP, and the other writes cost nothing extra.
        if row.id is None or row.id % _EVENT_PRUNE_EVERY == 0:
            _prune_security_events()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass


def _prune_security_events():
    """Keep the newest EVENT_LOG_KEEP rows. Unbounded growth is its own availability
    problem on a device with a small disk, which is what this app often runs on.

    The cutoff query below is also the "is there anything to prune" check: it
    returns NULL until the table exceeds EVENT_LOG_KEEP rows, so the COUNT(*)
    that used to precede it was pure overhead."""
    try:
        cutoff = (db.session.query(SecurityEvent.id)
                  .order_by(SecurityEvent.id.desc())
                  .offset(EVENT_LOG_KEEP).limit(1).scalar())
        if cutoff:
            db.session.query(SecurityEvent).filter(SecurityEvent.id <= cutoff).delete(
                synchronize_session=False)
            db.session.commit()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass


def _auth_disabled():
    """Kill-switch: DISABLE_AUTH=1 turns authentication off entirely."""
    return os.environ.get("DISABLE_AUTH", "").strip().lower() in ("1", "true", "yes", "on")


def _users_exist():
    try:
        return db.session.query(User.id).first() is not None
    except Exception:
        return False


def _admin_count():
    try:
        return (db.session.query(User.id)
                .join(Role, User.role_id == Role.id)
                .filter(Role.is_admin.is_(True), User.active.is_(True)).count())
    except Exception:
        return 0


def _session_auth_hash(u):
    """A token tying a session to the password it was issued under.

    Derived from the stored password hash, so changing the password changes this
    and every session minted under the old one stops matching. It is keyed with
    the app secret and truncated because it is a comparison token that travels in
    the cookie, not a credential -- and it is NOT the stored hash itself.

    This is deliberately derived rather than stored in a new column: a column
    would need a migration, and migrations on this app only run from the __main__
    block, so a WSGI deployment would come up with the column missing and every
    login broken. Nothing to migrate cannot fail to migrate.
    """
    key = app.config.get("SECRET_KEY") or ""
    if isinstance(key, str):
        key = key.encode("utf-8")
    return _hmac.new(key, (u.password_hash or "").encode("utf-8"),
                     _hashlib.sha256).hexdigest()[:32]


def _current_user():
    uid = session.get("uid")
    if not uid:
        return None
    try:
        u = User.query.get(uid)
    except Exception:
        u = None
    if u is None or not u.active:
        return None
    # A session is only valid for the password it was issued under. Without this,
    # changing the password of an account you believe is compromised does nothing
    # to the attacker: their stolen cookie keeps working for the full signature
    # lifetime. Deactivating the user already took effect immediately (above);
    # changing the password did not.
    got = session.get("sauth")
    if not isinstance(got, str) or not _hmac.compare_digest(got, _session_auth_hash(u)):
        return None
    return u


def _role_allows(role, resource, need):
    if role is None:
        return False
    if role.is_admin:
        return True
    have = (role.permissions or {}).get(resource, "none")
    return _PERM_RANK.get(have, 0) >= _PERM_RANK.get(need, 1)


# Returned for paths whose GET responses are deliberately readable by every signed-in
# user, whatever their role: every card image is served through /uploads (uploaded_file
# says so), and the temp servers hand back in-flight scan images while an import or a
# quick scan is running. This is a distinct value rather than None so the gate can tell
# "deliberately public" apart from "nobody mapped this" — the two now fail differently,
# and a sentinel keeps that decision in the one function that parses the path instead of
# a second prefix test that could disagree with this one about what a path means.
PUBLIC_READ = "__public_read__"

# Reachable by every signed-in account whatever its role, because the route acts on
# nobody but the caller and takes no user id. Distinct from PUBLIC_READ, which is
# read-only by design: this one must permit a POST.
SELF_SERVICE = "__self_service__"


def _resource_for_path(path):
    """Map a request path to a protected resource key, '__admin__' for the user/role
    manager, PUBLIC_READ for the deliberately-open file servers, or None when nothing
    maps the path — which the gate treats as a refusal, not as permission."""
    p = (path or "/").rstrip("/")
    seg = [s for s in p.split("/") if s]
    # Matched as a WHOLE PATH, not by first segment. A segment rule would hand every
    # future /account/* route to every role by inheritance, which is the failure this
    # map's default-deny exists to prevent.
    if p == "/account/password":
        return SELF_SERVICE
    if not seg:
        return "templates"
    head = seg[0]
    if head == "settings":
        sub = seg[1] if len(seg) > 1 else ""
        if sub in ("users", "roles", "security"):
            return "__admin__"
        # "reset" is the factory wipe (drops the DB + storage) — admins only, never
        # the plain "settings" default it would otherwise inherit.
        return {"reference": "reference", "api": "api_keys", "storage": "storage",
                "network": "network", "identify": "identify", "upgrade": "upgrade",
                "reset": "__admin__", "general": "settings"}.get(sub, "settings")
    return {
        "inventory": "inventory",
        "albums": "albums", "album": "albums",
        "import": "import", "run_import_split": "import", "manual_process_card": "import",
        "import_single_card": "import", "import_finalize_batch": "import",
        "pdf_open": "import", "pdf_render_page": "import", "pdf_close": "import",
        # Inventory record actions (buttons on the inventory pages).
        "update_scan": "inventory", "update_scan_image": "inventory",
        "realign_record_image": "inventory", "delete_scan": "inventory",
        "delete_scans": "inventory", "add_custom_field": "inventory",
        "grade_condition": "inventory", "ocr_apply": "inventory",
        "wrong_match": "inventory",
        "duplicates": "duplicates",
        "analytics": "analytics", "analytics_page": "analytics",
        "reports": "reports",
        "quickscan": "quick_scan",
        "search_by_image": "image_search", "search_by_image_page": "image_search",
        "reference": "reference",
        "shops": "shops", "shop": "shops",
        # Shipping blueprint is dormant (unregistered) today; map it now so the gate
        # covers /shipping/* the moment INTEGRATION.md wires it up. Reuses shops.
        "shipping": "shops",
        "justtcg_fetch": "pricing", "justtcg_search_manual": "pricing",
        "tcg_save_url": "pricing", "tcg_clear_url": "pricing",
        "save_tcgplayer_link": "pricing", "collections": "pricing",
        # Type-icon library is scanner training data — reachable mid-import, so it
        # rides the resource an importing user has by definition.
        "types": "import",
        "cloud_identify": "identify", "identify": "identify", "identify_diagnose": "identify",
        "storage": "storage", "upgrade": "upgrade",
        # update_field_type/hidden rewrite every game template that has the field,
        # so they are template edits (Chris-approved), not record edits.
        "template": "templates", "template_save": "templates",
        "template_delete": "templates", "templates": "templates",
        "template_config": "templates",
        "update_field_type": "templates", "update_field_hidden": "templates",
        # First-run system-mode select (/setup, /setup/select) + the orphaned
        # legacy-field migration: admin-only. At true first-run the gate
        # short-circuits on _users_exist() before this runs, and the first
        # account is an admin by construction, so /setup stays reachable then.
        "setup": "__admin__", "migrate_clean_legacy_fields": "__admin__",
        # ── Reads that were previously unmapped, and so were reachable by any
        # signed-in account regardless of role. Each rides the resource that already
        # governs the data it hands back:
        # /sold is literally the same view function as /inventory (two @app.route
        # decorators on one def), and records_summary/game_fields enumerate record
        # fields, so all three are inventory reads.
        "sold": "inventory", "records_summary": "inventory", "game_fields": "inventory",
        # The bulk-price UI asks which records still need a price, then fetches each
        # through justtcg_fetch — which is already "pricing", so gating the first step
        # any lower would protect nothing.
        "justtcg_missing_ids": "pricing", "justtcg_search": "pricing",
        # Identify-and-propose reads ride the resource of the apply step they feed:
        # both hand their choice to /ocr_apply, which is "inventory", and both work
        # from data the caller can already see (local OCR, local tcgcsv catalog).
        # cloud_identify stays "identify" because it spends an external API key.
        "ocr_identify": "inventory", "database_match": "inventory",
        # Deliberately readable by any signed-in user — see PUBLIC_READ.
        "uploads": PUBLIC_READ, "temp_cards": PUBLIC_READ,
        "temp_split": PUBLIC_READ, "temp_pdf": PUBLIC_READ,
    }.get(head, None)


_AUTH_PUBLIC_PREFIXES = ("/auth/", "/static/")


def _wants_json():
    return (request.method not in ("GET", "HEAD")
            or request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or "application/json" in (request.headers.get("Accept") or ""))


def _forbidden(msg):
    if _wants_json():
        return jsonify({"status": "error", "forbidden": True, "message": msg}), 403
    body = (f"<div style='font-family:system-ui;max-width:560px;margin:80px auto;text-align:center'>"
            f"<h1 style='font-size:20px'>403 &mdash; Not allowed</h1>"
            f"<p style='color:#6b7280'>{msg}</p>"
            f"<p><a href='{url_for('index')}'>Home</a> &nbsp;·&nbsp; "
            f"<a href='{url_for('auth_logout')}'>Sign out</a></p></div>")
    return Response(body, status=403, mimetype="text/html")


def _require_admin(msg="Administrator access is required to test a saved connection."):
    """Return a 403 Response if the caller is not an administrator, else None.

    Gates actions whose damage is not captured by any single resource permission:
      - connecting to a user-supplied host while replaying a stored secret
        (mailbox/shop 'Test' + mailbox 'Check') — a non-admin with shops:edit could
        otherwise repoint the saved host and exfiltrate the stored credential;
      - handing the caller a migration bundle, which is a plaintext dump of every
        stored credential (see _build_migration_bundle).
    `msg` is the 403 text, so each call site can say which action it refused."""
    if _auth_disabled():
        return None
    u = getattr(g, "user", None) or _current_user()
    if u is not None and u.role and u.role.is_admin:
        return None
    return _forbidden(msg)


# Shared 403 text for the migration-bundle routes, so the export, the download and
# the legacy /uploads path all refuse with the same sentence.
_BUNDLE_ADMIN_MSG = ("Administrator access is required to export or download a migration "
                     "bundle: it contains stored credentials in plaintext.")


# GET routes that perform a privileged write and so must require edit despite
# their method — kept deliberately tiny (the eBay OAuth handshake stores a
# marketplace token). This is not a general per-route level table.
_EBAY_OAUTH_WRITE_PATHS = ("/shops/ebay/connect", "/shops/ebay/callback")


@app.before_request
def _auth_gate():
    if _auth_disabled():
        return
    path = request.path or "/"
    if path == "/favicon.ico" or any(path.startswith(p) for p in _AUTH_PUBLIC_PREFIXES):
        return
    # Fresh install: no accounts yet -> force first-administrator creation.
    if not _users_exist():
        if _wants_json():
            return jsonify({"status": "error", "needs_auth_setup": True,
                            "message": "Create the first administrator account."}), 409
        return redirect(url_for("auth_setup_page"))
    user = _current_user()
    if user is None:
        if _wants_json():
            return jsonify({"status": "error", "auth_required": True,
                            "message": "Sign in required."}), 401
        return redirect(url_for("auth_login_page", next=path))
    g.user = user
    if user.role and user.role.is_admin:
        return
    resource = _resource_for_path(path)
    if resource == PUBLIC_READ:
        # The file servers, open to any signed-in user by design. Reads only: none of
        # them defines a mutating method today, and if one is ever added it has to be
        # mapped to a real resource like anything else rather than inheriting this.
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return
        return _forbidden("This action isn't available to your role.")
    if resource == SELF_SERVICE:
        # Every signed-in account, including one whose role grants nothing anywhere:
        # a person must be able to change their own password without an administrator.
        # Admins already returned above. The route enforces the "own" part.
        return
    if resource is None:
        # Default-deny in BOTH directions. This used to refuse unmapped *mutating*
        # methods and wave unmapped reads through on nothing but a valid session,
        # which meant a page nobody remembered to map was readable by every account:
        # /sold, the second route on the /inventory view, served the full listing —
        # names, serials and prices — to a role with every resource set to "none".
        # Forgetting to map a route is now a 403 instead of a silent disclosure.
        # NOTE: this still keys on the METHOD for the message only. The real
        # guarantee is that an unmapped path is refused however it is requested.
        return _forbidden("This isn't available to your role.")
    if resource == "__admin__":
        return _forbidden("Administrator access is required for user and role management.")
    # Compare the same rstripped form _resource_for_path uses, so the two agree
    # on normalization (a trailing-slash variant can't slip past this set).
    force_edit = path.rstrip("/") in _EBAY_OAUTH_WRITE_PATHS
    need = "edit" if (force_edit or request.method not in ("GET", "HEAD", "OPTIONS")) else "view"
    if not _role_allows(user.role, resource, need):
        label = dict(PROTECTED_RESOURCES).get(resource, resource)
        return _forbidden(f"Your role doesn't have &ldquo;{need}&rdquo; access to {label}.")


@app.context_processor
def _inject_auth():
    u = getattr(g, "user", None) or _current_user()

    def _perm_check(resource, need):
        if _auth_disabled():
            return True
        if u is None:
            return False
        if u.role and u.role.is_admin:
            return True
        if resource not in _RESOURCE_KEYS:
            return True   # not a gated resource -> any signed-in user may see it
        return _role_allows(u.role, resource, need)

    return {
        "auth_user": u,
        "auth_username": (u.username if u else ""),
        "auth_is_admin": bool(u and u.role and u.role.is_admin),
        "auth_enabled": not _auth_disabled(),
        "can_view": (lambda r: _perm_check(r, "view")),
        "can_edit": (lambda r: _perm_check(r, "edit")),
        # Templates rendering a stored URL into an href must pass it through this.
        "external_http_url": _external_http_url,
    }


import hmac as _hmac
import secrets as _secrets

# ── CSRF protection (synchronizer token) ──
# A per-session token is embedded in every page and auto-attached to all
# same-origin state-changing requests by a small injected script that patches
# fetch/XMLHttpRequest/form submits — so existing pages need no changes. The
# server validates the token on every mutating request.
def _csrf_token():
    tok = session.get("_csrf")
    if not tok:
        tok = _secrets.token_hex(32)
        session["_csrf"] = tok
    return tok


def csrf_exempt(view):
    """Decorator to opt a view out of CSRF checks (e.g. a machine-to-machine
    endpoint). Not currently used, but available for future non-browser callers."""
    view._csrf_exempt = True
    return view


_CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


@app.before_request
def _csrf_protect():
    if request.method in _CSRF_SAFE_METHODS:
        return
    path = request.path or "/"
    if path.startswith("/static/"):
        return
    fn = app.view_functions.get(request.endpoint)
    if fn is not None and getattr(fn, "_csrf_exempt", False):
        return
    expected = session.get("_csrf")
    sent = (request.headers.get("X-CSRFToken")
            or request.headers.get("X-CSRF-Token")
            or request.form.get("csrf_token"))
    if not expected or not sent or not _hmac.compare_digest(str(sent), str(expected)):
        if _wants_json():
            return jsonify({"status": "error", "csrf": True,
                            "message": "Your session's security token is missing or expired. "
                                       "Refresh the page and try again."}), 400
        return Response("<div style='font-family:system-ui;max-width:520px;margin:80px auto;text-align:center'>"
                        "<h1 style='font-size:20px'>Security check failed</h1>"
                        "<p style='color:#6b7280'>Please refresh the page and try again.</p></div>",
                        status=400, mimetype="text/html")


# Client-side shim: sets the token and transparently attaches it to same-origin
# mutating fetch/XHR requests and form posts. Injected right after <body> so it's
# active before any page script runs.
_CSRF_SCRIPT = ("<script>(function(){var T=\"__CSRF__\";"
                "function so(u){try{return new URL(u,location.href).origin===location.origin;}catch(e){return true;}}"
                "if(window.fetch){var _f=window.fetch;window.fetch=function(input,init){init=init||{};"
                "var m=(init.method||(input&&typeof input!=='string'&&input.method)||'GET').toUpperCase();"
                "var u=(typeof input==='string')?input:(input&&input.url)||'';"
                "if(m!=='GET'&&m!=='HEAD'&&so(u)){var base=init.headers||(input&&typeof input!=='string'&&input.headers)||{};"
                "var h=new Headers(base);if(!h.has('X-CSRFToken'))h.set('X-CSRFToken',T);init.headers=h;}"
                "return _f.call(this,input,init);};}"
                "if(window.XMLHttpRequest){var o=XMLHttpRequest.prototype.open,s=XMLHttpRequest.prototype.send;"
                "XMLHttpRequest.prototype.open=function(m,u){this.__m=(m||'GET').toUpperCase();this.__u=u;return o.apply(this,arguments);};"
                "XMLHttpRequest.prototype.send=function(b){try{if(this.__m&&this.__m!=='GET'&&this.__m!=='HEAD'&&so(this.__u))this.setRequestHeader('X-CSRFToken',T);}catch(e){}return s.apply(this,arguments);};}"
                "document.addEventListener('submit',function(e){var f=e.target;if(!f||!f.tagName||f.tagName!=='FORM')return;"
                "if((f.method||'get').toUpperCase()!=='POST')return;if(f.querySelector('input[name=csrf_token]'))return;"
                "var i=document.createElement('input');i.type='hidden';i.name='csrf_token';i.value=T;f.appendChild(i);},true);"
                "})();</script>")

_FLOAT_STYLE = ('position:fixed;bottom:16px;z-index:2147483000;'
                'background:#111827;color:#fff;padding:8px 14px;border-radius:999px;'
                'font:600 13px system-ui,-apple-system,Segoe UI,Roboto,sans-serif;'
                'text-decoration:none;box-shadow:0 3px 12px rgba(0,0,0,.3);opacity:.92')

# The password page is the only way a non-administrator can reach it: /settings is
# role-gated, so a link there would be invisible to exactly the people who need it.
# This block is already injected on every signed-in page, so it is the one place that
# reaches every account.
_LOGOUT_FLOAT = ('<a href="/account/password" title="Change your password" '
                 'style="' + _FLOAT_STYLE + ';right:120px">&#128273; Password</a>'
                 '<a href="/auth/logout" title="Sign out (%s)" '
                 'style="' + _FLOAT_STYLE + ';right:16px">'
                 '&#10150; Sign out</a>')


@app.after_request
def _inject_client_helpers(resp):
    """Inject the CSRF shim (all HTML pages) and, as a safety net, a floating
    logout on signed-in pages whose nav doesn't already have one."""
    try:
        if getattr(resp, "direct_passthrough", False):
            return resp
        if "text/html" not in (resp.content_type or ""):
            return resp
        body = resp.get_data(as_text=True)
        if "</body>" not in body:
            return resp
        changed = False

        # CSRF shim — right after the opening <body> tag so it runs first.
        script = _CSRF_SCRIPT.replace("__CSRF__", _csrf_token())
        m = _re.search(r"<body[^>]*>", body, _re.IGNORECASE)
        if m:
            body = body[:m.end()] + script + body[m.end():]
        else:
            body = body.replace("</body>", script + "</body>", 1)
        changed = True

        # Floating logout fallback for signed-in pages missing a nav logout.
        if session.get("uid") and "/auth/logout" not in body:
            u = getattr(g, "user", None) or _current_user()
            if u is not None:
                uname = (u.username or "").replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
                body = body.replace("</body>", (_LOGOUT_FLOAT % uname) + "</body>", 1)

        if changed:
            resp.set_data(body)
    except Exception:
        pass
    return resp


@app.context_processor
def _inject_csrf_token():
    # Also expose the token to templates that want to add it to a form manually.
    return {"csrf_token": _csrf_token}


_SETUP_ALLOWED_PREFIXES = ("/setup", "/static", "/auth")


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


_AUTH_CSS = """
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#f5f7fb; color:#1f2937; margin:0; padding:24px; }
  .card { max-width:640px; margin:6vh auto 0; background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:26px 28px; box-shadow:0 10px 30px rgba(0,0,0,.06); }
  .wide { max-width:960px; }
  h1 { font-size:22px; margin:0 0 6px; } p.sub { color:#6b7280; margin:0 0 20px; }
  label.fld { display:block; font-weight:700; margin:14px 0 6px; }
  input[type=text], input[type=password], select { width:100%; box-sizing:border-box; border:1px solid #d1d5db; border-radius:10px; padding:10px 12px; font-size:15px; }
  button { background:#4f46e5; color:#fff; border:0; border-radius:10px; padding:10px 18px; font-weight:700; cursor:pointer; }
  button.sec { background:#eef2ff; color:#4338ca; } button.danger { background:#fee2e2; color:#991b1b; }
  button:disabled { opacity:.6; cursor:default; }
  .row { display:flex; gap:10px; align-items:center; margin-top:18px; flex-wrap:wrap; }
  a { color:#4f46e5; text-decoration:none; } a:hover { text-decoration:underline; }
  .msg { margin-top:16px; padding:12px 14px; border-radius:10px; display:none; } .msg.show { display:block; }
  .msg.ok { background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; } .msg.err { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  table { width:100%; border-collapse:collapse; margin-top:10px; } th,td { text-align:left; padding:8px 10px; border-bottom:1px solid #eef0f4; font-size:14px; vertical-align:middle; }
  th { color:#6b7280; font-weight:700; } .pill { font-size:12px; font-weight:700; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#4338ca; }
  .links { margin-top:22px; font-size:14px; color:#6b7280; } .links a { margin-right:14px; }
  .grid { width:100%; border-collapse:collapse; margin-top:8px; } .grid th,.grid td { padding:6px 8px; border-bottom:1px solid #eef0f4; font-size:14px; }
  .seg { display:inline-flex; border:1px solid #d1d5db; border-radius:8px; overflow:hidden; } .seg label { padding:4px 10px; cursor:pointer; font-size:13px; }
  .seg input { display:none; } .seg label.on { background:#4f46e5; color:#fff; }
"""

# Minimum length for any account password. Single source of truth for every
# enforcement point (first-run setup, add-user, password change) and the UI
# hint text below — the "__MINPW__" token in the auth HTML is substituted with
# this value. Existing passwords stay valid until changed; raising this does
# not force-expire anyone.
MIN_PASSWORD_LEN = 10

_AUTH_SETUP_HTML = ("""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Create administrator</title>
<style>""" + _AUTH_CSS + """</style></head><body>
<div class=card>
  <h1>Welcome &mdash; create your administrator</h1>
  <p class=sub>This is the first account. It gets full access and can create roles and other users.</p>
  <label class=fld for=u>Username</label><input id=u type=text autocomplete=username autofocus>
  <label class=fld for=p>Password</label><input id=p type=password autocomplete=new-password placeholder="at least __MINPW__ characters">
  <label class=fld for=p2>Confirm password</label><input id=p2 type=password autocomplete=new-password>
  <div class=row><button id=go>Create administrator</button></div>
  <div id=msg class=msg></div>
</div>
<script>
  var msg=document.getElementById('msg');
  function show(t,ok){msg.textContent=t;msg.className='msg show '+(ok?'ok':'err');}
  document.getElementById('go').addEventListener('click',async function(){
    var u=document.getElementById('u').value.trim(),p=document.getElementById('p').value,p2=document.getElementById('p2').value;
    if(u.length<3){show('Username needs 3+ characters.',false);return;}
    if(p.length<__MINPW__){show('Password needs __MINPW__+ characters.',false);return;}
    if(p!==p2){show('Passwords do not match.',false);return;}
    this.disabled=true;
    try{var r=await fetch('/auth/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p})});
      var d=await r.json(); if(d.status==='success'){location.href=d.redirect||'/';} else {show(d.message||'Failed.',false);this.disabled=false;}
    }catch(e){show('Error: '+e.message,false);this.disabled=false;}
  });
</script></body></html>""").replace("__MINPW__", str(MIN_PASSWORD_LEN))

_AUTH_LOGIN_HTML = ("""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Sign in</title>
<style>""" + _AUTH_CSS + """</style></head><body>
<div class=card>
  <h1>Sign in</h1>
  <p class=sub>Card Collector Inventory Manager</p>
  <label class=fld for=u>Username</label><input id=u type=text autocomplete=username autofocus>
  <label class=fld for=p>Password</label><input id=p type=password autocomplete=current-password>
  <div class=row><button id=go>Sign in</button></div>
  <div id=msg class=msg></div>
</div>
<script>
  var msg=document.getElementById('msg');
  function show(t,ok){msg.textContent=t;msg.className='msg show '+(ok?'ok':'err');}
  function nextParam(){var m=location.search.match(/[?&]next=([^&]+)/);return m?decodeURIComponent(m[1]):'';}
  async function submit(){
    var b=document.getElementById('go');b.disabled=true;
    try{var r=await fetch('/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({username:document.getElementById('u').value.trim(),password:document.getElementById('p').value,next:nextParam()})});
      var d=await r.json(); if(d.status==='success'){location.href=d.redirect||'/';} else {show(d.message||'Failed.',false);b.disabled=false;}
    }catch(e){show('Error: '+e.message,false);b.disabled=false;}
  }
  document.getElementById('go').addEventListener('click',submit);
  document.getElementById('p').addEventListener('keydown',function(e){if(e.key==='Enter')submit();});
</script></body></html>""")


@app.route("/auth/setup", methods=["GET"])
def auth_setup_page():
    if _users_exist():
        return redirect(url_for("auth_login_page"))
    return Response(_AUTH_SETUP_HTML, mimetype="text/html")


@app.route("/auth/setup", methods=["POST"])
def auth_setup_submit():
    if _users_exist():
        return jsonify({"status": "error", "message": "Already set up."}), 409
    body = request.get_json(silent=True) or request.form
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    if len(username) < 3 or len(password) < MIN_PASSWORD_LEN:
        return jsonify({"status": "error",
                        "message": f"Username (3+) and password ({MIN_PASSWORD_LEN}+) required."}), 400
    admin  = Role(name="Administrator", is_admin=True, permissions={})
    editor = Role(name="Editor", is_admin=False, permissions={k: "edit" for k in _RESOURCE_KEYS})
    viewer = Role(name="Viewer", is_admin=False, permissions={k: "view" for k in _RESOURCE_KEYS})
    db.session.add_all([admin, editor, viewer])
    db.session.flush()
    u = User(username=username, role_id=admin.id, active=True)
    u.set_password(password)
    db.session.add(u)
    db.session.commit()
    session["uid"] = u.id
    session["sauth"] = _session_auth_hash(u)
    log_security_event("first_admin_created", actor=u, target=u)
    return jsonify({"status": "success", "redirect": url_for("index")})


@app.route("/auth/login", methods=["GET"])
def auth_login_page():
    if not _users_exist():
        return redirect(url_for("auth_setup_page"))
    if _current_user() is not None:
        return redirect(url_for("index"))
    return Response(_AUTH_LOGIN_HTML, mimetype="text/html")


_LOGIN_FAILS = {}
_LOGIN_FAILS_LOCK = _threading.Lock()
_LOGIN_MAX_FAILS = 10          # failures per (IP + username) before a cooldown
_LOGIN_WINDOW = 600            # rolling window / cooldown, seconds


def _login_throttle_key(username):
    # Keyed by IP + username so an attacker hammering one account is throttled
    # without locking that user out from a different machine.
    return (request.remote_addr or "?") + "|" + (username or "").strip().lower()


def _login_is_locked(key):
    now = time.time()
    with _LOGIN_FAILS_LOCK:
        fails = [t for t in _LOGIN_FAILS.get(key, []) if now - t < _LOGIN_WINDOW]
        _LOGIN_FAILS[key] = fails
        return len(fails) >= _LOGIN_MAX_FAILS


def _login_record_fail(key):
    now = time.time()
    with _LOGIN_FAILS_LOCK:
        fails = [t for t in _LOGIN_FAILS.get(key, []) if now - t < _LOGIN_WINDOW]
        fails.append(now)
        _LOGIN_FAILS[key] = fails


def _login_clear(key):
    with _LOGIN_FAILS_LOCK:
        _LOGIN_FAILS.pop(key, None)


@app.route("/auth/login", methods=["POST"])
def auth_login_submit():
    body = request.get_json(silent=True) or request.form
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    key = _login_throttle_key(username)
    if _login_is_locked(key):
        log_security_event("sign_in_throttled", actor_name=username)
        return jsonify({"status": "error",
                        "message": "Too many failed attempts. Wait a few minutes and try again."}), 429
    u = User.query.filter(db.func.lower(User.username) == username.lower()).first()
    if u is None or not u.active or not u.check_password(password):
        _login_record_fail(key)
        # The reason is recorded for the log only. The RESPONSE stays the single
        # generic message -- distinguishing them to the caller is user enumeration.
        why = ("unknown_user" if u is None
               else "inactive_user" if not u.active else "bad_password")
        log_security_event("sign_in_failed", actor=u, actor_name=username, detail=why)
        return jsonify({"status": "error", "message": "Invalid username or password."}), 401
    _login_clear(key)
    session["uid"] = u.id
    session["sauth"] = _session_auth_hash(u)
    log_security_event("sign_in", actor=u)
    return jsonify({"status": "success",
                    "redirect": _same_origin_next(body.get("next"), url_for("index"))})


# ---- Self-service password change (any signed-in account, own password only) ----
_ACCOUNT_PW_HTML = ("""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Change password</title>
<style>""" + _AUTH_CSS + """</style></head><body>
<div class=card>
  <h1>Change your password</h1>
  <p class=sub>Signed in as <b>__USER__</b>. This changes your own password only &mdash;
     an administrator changes anyone else&rsquo;s from Manage users.</p>
  <label class=fld for=c>Current password</label>
  <input id=c type=password autocomplete=current-password autofocus>
  <label class=fld for=n>New password (__MINPW__+ characters)</label>
  <input id=n type=password autocomplete=new-password>
  <label class=fld for=n2>Confirm new password</label>
  <input id=n2 type=password autocomplete=new-password>
  <div class=row><button id=go>Change password</button></div>
  <div id=msg class=msg></div>
  <div class=links><a href="/">Back to the app</a><a href="/auth/logout">Sign out</a></div>
</div>
<script>
  var msg=document.getElementById('msg');
  function show(t,ok){msg.textContent=t;msg.className='msg show '+(ok?'ok':'err');}
  async function submit(){
    var b=document.getElementById('go');b.disabled=true;
    try{
      var r=await fetch('/account/password',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({current_password:document.getElementById('c').value,
                             new_password:document.getElementById('n').value,
                             confirm_password:document.getElementById('n2').value})});
      var d=await r.json();
      if(d.status==='success'){
        show(d.message||'Password changed.',true);
        document.getElementById('c').value='';document.getElementById('n').value='';document.getElementById('n2').value='';
      } else { show(d.message||'Could not change the password.',false); }
    }catch(e){ show('Error: '+e.message,false); }
    b.disabled=false;
  }
  document.getElementById('go').addEventListener('click',submit);
  document.getElementById('n2').addEventListener('keydown',function(e){if(e.key==='Enter')submit();});
</script></body></html>""")


@app.route("/account/password", methods=["GET"])
def account_password_page():
    u = _current_user()
    if u is None:
        return redirect(url_for("auth_login_page", next="/account/password"))
    name = (u.username or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _ACCOUNT_PW_HTML.replace("__USER__", name).replace("__MINPW__", str(MIN_PASSWORD_LEN))


@app.route("/account/password", methods=["POST"])
def account_password_change():
    """Change the CALLER'S password. Deliberately takes no user id.

    An id parameter here would be an authorization decision made from request data on
    a route every signed-in account can reach -- the shape that put ten paths at the
    mercy of the permission map before. The account changed is whichever one the
    session already proved, and there is no argument that can redirect it.
    """
    u = _current_user()
    if u is None:
        return jsonify({"status": "error", "auth_required": True,
                        "message": "Sign in required."}), 401
    body = request.get_json(silent=True) or request.form
    current = str(body.get("current_password", "") or "")
    new = str(body.get("new_password", "") or "")
    confirm = str(body.get("confirm_password", "") or "")

    # Re-authenticate. A hijacked session should not be able to lock the real owner
    # out, so the current password is required -- and guessing it is throttled on the
    # same counter as the login form.
    key = _login_throttle_key(u.username)
    if _login_is_locked(key):
        return jsonify({"status": "error",
                        "message": "Too many failed attempts. Wait a few minutes and try again."}), 429
    if not u.check_password(current):
        _login_record_fail(key)
        return jsonify({"status": "error", "message": "Your current password is incorrect."}), 403
    _login_clear(key)

    if len(new) < MIN_PASSWORD_LEN:
        return jsonify({"status": "error",
                        "message": f"The new password must be at least {MIN_PASSWORD_LEN} characters."}), 400
    if new != confirm:
        return jsonify({"status": "error", "message": "The new passwords do not match."}), 400
    if new == current:
        return jsonify({"status": "error",
                        "message": "The new password must be different from your current one."}), 400

    u.set_password(new)
    db.session.commit()
    # Every OTHER session for this account is now invalid, which is the whole point of
    # changing a password you believe someone else knows. Re-stamp this one so the
    # person doing it is not signed out by their own action.
    session["sauth"] = _session_auth_hash(u)
    log_security_event("password_changed", actor=u, target=u, detail="self_service")
    return jsonify({"status": "success",
                    "message": "Password changed. Any other sessions signed in as you have been signed out."})


@app.route("/auth/logout")
def auth_logout():
    log_security_event("sign_out", actor=_current_user())
    session.pop("uid", None)
    session.pop("sauth", None)
    return redirect(url_for("auth_login_page"))


# ---- User management (admin only; gated by _resource_for_path -> '__admin__') ----
_AUTH_USERS_HTML = ("""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Users</title>
<style>""" + _AUTH_CSS + """</style></head><body>
<div class="card wide">
  <h1>Users</h1>
  <p class=sub>Create accounts and assign each a role. Roles decide what each account can view or edit.</p>
  <table id=tbl><thead><tr><th>Username</th><th>Role</th><th>Active</th><th></th></tr></thead><tbody></tbody></table>

  <h1 style="font-size:17px;margin-top:26px">Add a user</h1>
  <div class=row style="align-items:flex-end">
    <div><label class=fld for=nu>Username</label><input id=nu type=text style="width:200px"></div>
    <div><label class=fld for=np>Password</label><input id=np type=password style="width:200px" placeholder="__MINPW__+ chars"></div>
    <div><label class=fld for=nr>Role</label><select id=nr style="width:200px"></select></div>
    <button id=addBtn>Add user</button>
  </div>
  <div id=msg class=msg></div>
  <div class=links><a href="/settings/roles">Manage roles</a><a href="/settings/security">Security log</a><a href="/account/password">My password</a><a href="/settings">All settings</a><a href="/auth/logout">Sign out</a></div>
</div>
<script>
  var msg=document.getElementById('msg');
  function show(t,ok){msg.textContent=t;msg.className='msg show '+(ok?'ok':'err');}
  function opt(sel,val,txt,cur){var o=document.createElement('option');o.value=val;o.textContent=txt;if(String(val)===String(cur))o.selected=true;sel.appendChild(o);}
  var DATA={users:[],roles:[],me:null};
  async function load(){
    var d=await (await fetch('/settings/users/list',{headers:{'X-Requested-With':'XMLHttpRequest'}})).json();
    DATA=d; var tb=document.querySelector('#tbl tbody'); tb.innerHTML='';
    d.users.forEach(function(u){
      var tr=document.createElement('tr');
      var td1=document.createElement('td'); td1.textContent=u.username; if(u.id===d.me){var b=document.createElement('span');b.className='pill';b.textContent='you';b.style.marginLeft='8px';td1.appendChild(b);} tr.appendChild(td1);
      var td2=document.createElement('td'); var sel=document.createElement('select'); d.roles.forEach(function(r){opt(sel,r.id,r.name+(r.is_admin?' (admin)':''),u.role_id);});
      sel.addEventListener('change',function(){upd(u.id,{role_id:parseInt(sel.value,10)});}); td2.appendChild(sel); tr.appendChild(td2);
      var td3=document.createElement('td'); var cb=document.createElement('input'); cb.type='checkbox'; cb.checked=u.active;
      cb.addEventListener('change',function(){upd(u.id,{active:cb.checked});}); td3.appendChild(cb); tr.appendChild(td3);
      var td4=document.createElement('td');
      var rp=document.createElement('button'); rp.className='sec'; rp.textContent='Reset password'; rp.style.marginRight='6px';
      rp.addEventListener('click',function(){var p=prompt('New password for '+u.username+' (__MINPW__+ chars):');if(p){if(p.length<__MINPW__){show('Password too short.',false);return;}upd(u.id,{password:p});}});
      var del=document.createElement('button'); del.className='danger'; del.textContent='Delete';
      del.addEventListener('click',function(){if(confirm('Delete user '+u.username+'?'))act('/settings/users/delete',{id:u.id});});
      td4.appendChild(rp); td4.appendChild(del); tr.appendChild(td4); tb.appendChild(tr);
    });
    var nr=document.getElementById('nr'); nr.innerHTML=''; d.roles.forEach(function(r){opt(nr,r.id,r.name+(r.is_admin?' (admin)':''),null);});
  }
  async function act(url,payload){
    try{var r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'XMLHttpRequest'},body:JSON.stringify(payload)});
      var d=await r.json(); if(d.status==='success'){show('Saved.',true);load();} else show(d.message||'Failed.',false);
    }catch(e){show('Error: '+e.message,false);}
  }
  function upd(id,patch){patch.id=id;act('/settings/users/update',patch);}
  document.getElementById('addBtn').addEventListener('click',function(){
    var u=document.getElementById('nu').value.trim(),p=document.getElementById('np').value,r=document.getElementById('nr').value;
    if(u.length<3||p.length<__MINPW__){show('Username 3+ and password __MINPW__+ required.',false);return;}
    act('/settings/users/create',{username:u,password:p,role_id:parseInt(r,10)});
    document.getElementById('nu').value='';document.getElementById('np').value='';
  });
  load();
</script></body></html>""").replace("__MINPW__", str(MIN_PASSWORD_LEN))


@app.route("/settings/users")
def users_page():
    return Response(_AUTH_USERS_HTML, mimetype="text/html")


@app.route("/settings/users/list")
def users_list():
    users = User.query.order_by(User.username).all()
    roles = Role.query.order_by(Role.is_admin.desc(), Role.name).all()
    return jsonify({
        "status": "ok",
        "me": session.get("uid"),
        "users": [{"id": u.id, "username": u.username, "role_id": u.role_id,
                   "role_name": (u.role.name if u.role else "\u2014"),
                   "is_admin": bool(u.role and u.role.is_admin), "active": u.active} for u in users],
        "roles": [{"id": r.id, "name": r.name, "is_admin": r.is_admin} for r in roles],
    })


@app.route("/settings/users/create", methods=["POST"])
def users_create():
    body = request.get_json(silent=True) or request.form
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    if len(username) < 3 or len(password) < MIN_PASSWORD_LEN:
        return jsonify({"status": "error",
                        "message": f"Username (3+) and password ({MIN_PASSWORD_LEN}+) required."}), 400
    if User.query.filter(db.func.lower(User.username) == username.lower()).first():
        return jsonify({"status": "error", "message": "That username already exists."}), 409
    rid = body.get("role_id")
    role = Role.query.get(int(rid)) if rid else None
    u = User(username=username, role_id=(role.id if role else None), active=True)
    u.set_password(password)
    db.session.add(u)
    db.session.commit()
    log_security_event("user_created", actor=_current_user(), target=u)
    return jsonify({"status": "success"})


@app.route("/settings/users/update", methods=["POST"])
def users_update():
    body = request.get_json(silent=True) or request.form
    u = User.query.get(int(body.get("id", 0) or 0))
    if u is None:
        return jsonify({"status": "error", "message": "No such user."}), 404
    # Guard against removing the last active administrator.
    currently_admin = bool(u.role and u.role.is_admin and u.active)
    if currently_admin:
        new_role = Role.query.get(int(body["role_id"])) if body.get("role_id") else u.role
        will_admin = bool(new_role and new_role.is_admin)
        will_active = bool(body["active"]) if "active" in body else u.active
        if (not (will_admin and will_active)) and _admin_count() <= 1:
            return jsonify({"status": "error",
                            "message": "This is the only administrator — keep at least one."}), 400
    changed_role = bool(body.get("role_id")) and int(body["role_id"]) != u.role_id
    changed_active = "active" in body and bool(body["active"]) != bool(u.active)
    if body.get("role_id"):
        u.role_id = int(body["role_id"])
    if "active" in body:
        u.active = bool(body["active"])
    pw = str(body.get("password", "") or "")
    if pw:
        if len(pw) < MIN_PASSWORD_LEN:
            return jsonify({"status": "error",
                            "message": f"Password must be {MIN_PASSWORD_LEN}+ characters."}), 400
        u.set_password(pw)
    db.session.commit()
    if pw and u.id == session.get("uid"):
        # Every other session for this user is now invalid, which is the point.
        # Re-stamp the one making the request so an admin changing their own
        # password is not logged out by their own action.
        session["sauth"] = _session_auth_hash(u)
    actor = _current_user()
    if pw:
        log_security_event("password_changed", actor=actor, target=u,
                           detail=("role_and_password" if changed_role else "by_admin"))
    if changed_role or changed_active:
        log_security_event("user_updated", actor=actor, target=u,
                           detail=("role_changed" if changed_role
                                   else "activated" if bool(body.get("active")) else "deactivated"))
    return jsonify({"status": "success"})


@app.route("/settings/users/delete", methods=["POST"])
def users_delete():
    body = request.get_json(silent=True) or request.form
    u = User.query.get(int(body.get("id", 0) or 0))
    if u is None:
        return jsonify({"status": "error", "message": "No such user."}), 404
    if u.role and u.role.is_admin and u.active and _admin_count() <= 1:
        return jsonify({"status": "error", "message": "Can't delete the only administrator."}), 400
    gone_id, gone_name = u.id, u.username
    db.session.delete(u)
    db.session.commit()
    log_security_event("user_deleted", actor=_current_user(),
                       target_name=gone_name)
    return jsonify({"status": "success"})


# ---- Role management (admin only) ----
_AUTH_ROLES_HTML = ("""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Roles</title>
<style>""" + _AUTH_CSS + """</style></head><body>
<div class="card wide">
  <h1>Roles</h1>
  <p class=sub>A role sets, for each tab and tool, whether members can <b>view</b>, <b>edit</b>, or have no access.
     &ldquo;Edit&rdquo; implies view. Administrator roles have full access and manage users and roles.</p>
  <div id=roles></div>

  <h1 style="font-size:17px;margin-top:26px">Create a role</h1>
  <div class=row style="align-items:flex-end">
    <div><label class=fld for=rn>Role name</label><input id=rn type=text style="width:240px" placeholder="e.g. Front desk"></div>
    <label style="font-weight:600"><input type=checkbox id=radmin> Administrator (full access)</label>
    <button id=newBtn>Create role</button>
  </div>
  <div id=msg class=msg></div>
  <div class=links><a href="/settings/users">Manage users</a><a href="/settings">All settings</a><a href="/auth/logout">Sign out</a></div>
</div>
<script>
  var msg=document.getElementById('msg'),RES=[],ROLES=[];
  function show(t,ok){msg.textContent=t;msg.className='msg show '+(ok?'ok':'err');}
  function seg(role,key){
    var cur=(role.permissions||{})[key]||'none';
    var wrap=document.createElement('div');wrap.className='seg';
    ['none','view','edit'].forEach(function(lvl){
      var id='r'+role.id+'_'+key+'_'+lvl;
      var lab=document.createElement('label');lab.textContent=lvl;lab.className=(cur===lvl?'on':'');
      var inp=document.createElement('input');inp.type='radio';inp.name='r'+role.id+'_'+key;inp.value=lvl;if(cur===lvl)inp.checked=true;
      lab.addEventListener('click',function(){wrap.querySelectorAll('label').forEach(function(l){l.className='';});lab.className='on';inp.checked=true;});
      lab.prepend(inp);wrap.appendChild(lab);
    });
    return wrap;
  }
  function renderRole(role){
    var box=document.createElement('div');box.style.cssText='border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin-bottom:14px';
    var h=document.createElement('div');h.style.cssText='display:flex;align-items:center;gap:10px;flex-wrap:wrap';
    var nm=document.createElement('input');nm.type='text';nm.value=role.name;nm.style.cssText='font-weight:700;font-size:15px;width:240px';
    h.appendChild(nm);
    var adm=document.createElement('label');adm.style.fontWeight='600';var ac=document.createElement('input');ac.type='checkbox';ac.checked=role.is_admin;adm.appendChild(ac);adm.appendChild(document.createTextNode(' Administrator'));
    h.appendChild(adm);
    var inuse=document.createElement('span');inuse.className='pill';inuse.textContent=role.in_use+' user'+(role.in_use===1?'':'s');h.appendChild(inuse);
    box.appendChild(h);
    var gridWrap=document.createElement('div');gridWrap.style.marginTop='10px';
    var tbl=document.createElement('table');tbl.className='grid';
    RES.forEach(function(rc){
      var tr=document.createElement('tr');var td1=document.createElement('td');td1.textContent=rc[1];td1.style.width='55%';tr.appendChild(td1);
      var td2=document.createElement('td');td2.appendChild(seg(role,rc[0]));tr.appendChild(td2);tbl.appendChild(tr);
    });
    gridWrap.appendChild(tbl);
    function setGridDisabled(dis){gridWrap.style.opacity=dis?'.45':'1';gridWrap.style.pointerEvents=dis?'none':'auto';}
    setGridDisabled(role.is_admin);ac.addEventListener('change',function(){setGridDisabled(ac.checked);});
    box.appendChild(gridWrap);
    var row=document.createElement('div');row.className='row';
    var save=document.createElement('button');save.textContent='Save';
    save.addEventListener('click',function(){
      var perms={};RES.forEach(function(rc){var sel=box.querySelector('input[name="r'+role.id+'_'+rc[0]+'"]:checked');perms[rc[0]]=sel?sel.value:'none';});
      act('/settings/roles/save',{id:role.id,name:nm.value.trim(),is_admin:ac.checked,permissions:perms});
    });
    row.appendChild(save);
    if(role.in_use===0){var del=document.createElement('button');del.className='danger';del.textContent='Delete';del.addEventListener('click',function(){if(confirm('Delete role '+role.name+'?'))act('/settings/roles/delete',{id:role.id});});row.appendChild(del);}
    box.appendChild(row);return box;
  }
  async function load(){
    var d=await (await fetch('/settings/roles/list',{headers:{'X-Requested-With':'XMLHttpRequest'}})).json();
    RES=d.resources;ROLES=d.roles;var c=document.getElementById('roles');c.innerHTML='';d.roles.forEach(function(r){c.appendChild(renderRole(r));});
  }
  async function act(url,payload){
    try{var r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'XMLHttpRequest'},body:JSON.stringify(payload)});
      var d=await r.json();if(d.status==='success'){show('Saved.',true);load();}else show(d.message||'Failed.',false);
    }catch(e){show('Error: '+e.message,false);}
  }
  document.getElementById('newBtn').addEventListener('click',function(){
    var n=document.getElementById('rn').value.trim();if(!n){show('Enter a role name.',false);return;}
    act('/settings/roles/save',{name:n,is_admin:document.getElementById('radmin').checked,permissions:{}});
    document.getElementById('rn').value='';document.getElementById('radmin').checked=false;
  });
  load();
</script></body></html>""")


# ---- Security event log (admin only; gated by _resource_for_path -> '__admin__') ----
_AUTH_EVENTS_HTML = ("""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Security log</title>
<style>""" + _AUTH_CSS + """
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #e5e7eb;vertical-align:top}
 th{color:#6b7280;font-weight:600;white-space:nowrap}
 td.mono{font-family:ui-monospace,Menlo,Consolas,monospace;white-space:nowrap;color:#6b7280}
 .act{font-weight:600}
</style></head><body>
<div class="card wide">
  <h1>Security log</h1>
  <p class=sub>Sign-ins, account and role changes, password resets, storage moves and
     migration-bundle exports. Newest first; the oldest entries are pruned once the log
     passes __KEEP__ rows.</p>
  <div id=rows></div>
  <div id=msg class=msg></div>
  <div class=links><a href="/settings/users">Manage users</a><a href="/settings/roles">Manage roles</a><a href="/settings">All settings</a></div>
</div>
<script>
  var msg=document.getElementById('msg');
  function show(t,ok){msg.textContent=t;msg.className='msg show '+(ok?'ok':'err');}
  function cell(row,text,cls){var td=document.createElement('td');if(cls)td.className=cls;
    td.textContent=(text===null||text===undefined||text==='')?'\u2014':String(text);row.appendChild(td);}
  async function load(){
    try{
      var r=await fetch('/settings/security/list');var d=await r.json();
      if(d.status!=='ok'){show(d.message||'Could not load the log.',false);return;}
      var wrap=document.getElementById('rows');wrap.textContent='';
      if(!d.events.length){show('No events recorded yet.',true);return;}
      var t=document.createElement('table');
      var hr=document.createElement('tr');
      ['When (UTC)','Action','Detail','Actor','Target','From'].forEach(function(h){
        var th=document.createElement('th');th.textContent=h;hr.appendChild(th);});
      t.appendChild(hr);
      d.events.forEach(function(e){
        var tr=document.createElement('tr');
        cell(tr,e.at,'mono');cell(tr,e.action,'act');cell(tr,e.detail);
        cell(tr,e.actor_name);cell(tr,e.target_name);cell(tr,e.source_ip,'mono');
        t.appendChild(tr);});
      wrap.appendChild(t);
    }catch(e){show('Error: '+e.message,false);}
  }
  load();
</script></body></html>""").replace("__KEEP__", str(EVENT_LOG_KEEP))


@app.route("/settings/security")
def security_log_page():
    return Response(_AUTH_EVENTS_HTML, mimetype="text/html")


@app.route("/settings/security/list")
def security_log_list():
    """Rows as JSON. Every value is rendered by the page with textContent, never as
    markup: usernames are not charset-validated anywhere in this app, so a user can be
    created called <img src=x onerror=...> and this viewer is where it would land."""
    if not _event_table_ready():
        return jsonify({"status": "ok", "events": []})
    try:
        rows = (SecurityEvent.query.order_by(SecurityEvent.id.desc()).limit(500).all())
    except Exception:
        return jsonify({"status": "ok", "events": []})
    return jsonify({"status": "ok", "events": [{
        "at": (e.at.strftime("%Y-%m-%d %H:%M:%S") if e.at else ""),
        "action": e.action, "detail": e.detail or "",
        "actor_name": e.actor_name, "target_name": e.target_name,
        "source_ip": e.source_ip,
    } for e in rows]})


@app.route("/settings/roles")
def roles_page():
    return Response(_AUTH_ROLES_HTML, mimetype="text/html")


@app.route("/settings/roles/list")
def roles_list():
    roles = Role.query.order_by(Role.is_admin.desc(), Role.name).all()
    return jsonify({
        "status": "ok",
        "resources": PROTECTED_RESOURCES,
        "roles": [{"id": r.id, "name": r.name, "is_admin": r.is_admin,
                   "permissions": (r.permissions or {}),
                   "in_use": User.query.filter_by(role_id=r.id).count()} for r in roles],
    })


@app.route("/settings/roles/save", methods=["POST"])
def roles_save():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    if not name:
        return jsonify({"status": "error", "message": "Role name is required."}), 400
    is_admin = bool(body.get("is_admin"))
    perms_in = body.get("permissions") or {}
    perms = {k: (perms_in.get(k) if perms_in.get(k) in ("none", "view", "edit") else "none")
             for k in _RESOURCE_KEYS}
    rid = body.get("id")
    if rid:
        r = Role.query.get(int(rid))
        if r is None:
            return jsonify({"status": "error", "message": "No such role."}), 404
        # Don't let the last administrator lose admin.
        if r.is_admin and not is_admin and _admin_count() <= 1:
            return jsonify({"status": "error",
                            "message": "Can't remove admin from the only administrator."}), 400
        dup = Role.query.filter(db.func.lower(Role.name) == name.lower(), Role.id != r.id).first()
        if dup:
            return jsonify({"status": "error", "message": "A role with that name already exists."}), 409
        r.name, r.is_admin, r.permissions = name, is_admin, perms
    else:
        if Role.query.filter(db.func.lower(Role.name) == name.lower()).first():
            return jsonify({"status": "error", "message": "A role with that name already exists."}), 409
        r = Role(name=name, is_admin=is_admin, permissions=perms)
        db.session.add(r)
    db.session.commit()
    log_security_event("role_saved", actor=_current_user(),
                       target_name=r.name)
    return jsonify({"status": "success", "id": r.id})


@app.route("/settings/roles/delete", methods=["POST"])
def roles_delete():
    body = request.get_json(silent=True) or {}
    r = Role.query.get(int(body.get("id", 0) or 0))
    if r is None:
        return jsonify({"status": "error", "message": "No such role."}), 404
    if User.query.filter_by(role_id=r.id).count() > 0:
        return jsonify({"status": "error", "message": "Reassign users off this role before deleting it."}), 400
    if r.is_admin and _admin_count() <= 1:
        return jsonify({"status": "error", "message": "Can't delete the only administrator role."}), 400
    gone_role = r.name
    db.session.delete(r)
    db.session.commit()
    log_security_event("role_deleted", actor=_current_user(), target_name=gone_role)
    return jsonify({"status": "success"})


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
    # System flags/values, never shown as ad-hoc text columns.
    "held", "storage_type", "box_number",
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


def _held_from(data):
    """'Held' defaults to True: an entry is held unless explicitly marked sold
    (held == False). Missing/None/anything-but-false reads as held."""
    v = (data or {}).get("held", True)
    if v is None:
        return True
    return not (v is False or str(v).strip().lower() == "false")


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
    record.serial_key    = (_get_serial(data) or None)
    record.card_type_key = (_derive_card_type(data) or None)
    record.dup_hash      = _compute_dup_hash(data)
    record.is_finalized  = _bool_from(data, "finalized")
    record.is_catalog    = _is_catalog_only(data)
    record.is_archived   = _bool_from(data, "archived")
    record.is_held       = _held_from(data)


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
    """Render the game selection landing page for /inventory.

    Tile counts come from one SQL aggregation over the raw (trimmed) game
    value — no rows are hydrated for counting. NULL and empty games collapse
    into the "(Unknown Game)" tile inside the GROUP BY, same as the Python
    fold this replaces. Field discovery then samples up to 200 rows per tile
    through the indexed game_key; for mixed-case variants of one game the
    sample now spans the normalized game rather than each casing separately,
    consistent with how every filter treats casing since the normalization
    work.
    """
    from sqlalchemy import func as _f
    not_catalog = _f.coalesce(ScanRecord.is_catalog, False) == False  # noqa: E712
    game_expr = _f.coalesce(
        _f.nullif(_f.trim(_f.json_extract(ScanRecord.extracted_data, "$.game")), ""),
        "(Unknown Game)")
    album_expr = _f.nullif(_f.trim(_f.json_extract(ScanRecord.extracted_data, "$.album")), "")

    agg = (db.session.query(game_expr,
                            _f.count(ScanRecord.id),
                            _f.count(_f.distinct(album_expr)))
           .filter(not_catalog).group_by(game_expr).all())
    catalog_count = (db.session.query(_f.count(ScanRecord.id))
                     .filter(_f.coalesce(ScanRecord.is_catalog, False) == True)  # noqa: E712
                     .scalar()) or 0

    games = []
    for raw_name, count, album_count in agg:
        game_name = str(raw_name)
        gkey = game_name.strip().lower() or None
        key_cond = (ScanRecord.game_key.is_(None)
                    if game_name == "(Unknown Game)" else ScanRecord.game_key == gkey)
        sample = (ScanRecord.query.with_entities(ScanRecord.extracted_data)
                  .filter(not_catalog, key_cond).limit(200).all())
        sample_records_fake = [type("R", (), {"extracted_data": d})() for (d,) in sample]
        fields = discover_entry_fields(sample_records_fake)
        games.append({
            "name":        game_name,
            "count":       count,
            "album_count": album_count,
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


def _inventory_base_conditions(f_game, f_album, f_template, view_catalog, held_state=None):
    """WHERE conditions (on denormalized columns) shared by the fast-path
    representative query, count, member lookup, and field sampling.

    held_state: None = don't filter by Held (e.g. catalog view);
                True = only held (normal Inventory); False = only sold (Sold page).
    NULL is_held is treated as held (True) for rows created before the column."""
    from sqlalchemy import func as _f
    conds = [
        _f.coalesce(ScanRecord.is_catalog, False) == bool(view_catalog),
    ]
    if not view_catalog:
        conds.append(_f.coalesce(ScanRecord.is_archived, False) == False)  # noqa: E712
    if held_state is not None:
        conds.append(_f.coalesce(ScanRecord.is_held, True) == bool(held_state))
    if f_template:
        conds.append(ScanRecord.template_used == f_template)
    if f_game:
        conds.append(ScanRecord.game_key == f_game.strip().lower())
    if f_album:
        conds.append(ScanRecord.album_key == f_album.strip().lower())
    return conds


def _render_inventory_fast(f_game, f_album, f_template, view_catalog,
                           page, per_page, sort_col, sort_dir, held_state=None):
    """
    Fast Inventory path: de-duplicate and paginate entirely in SQL using the
    dup_hash column and window functions, loading only the page's ~50 rows
    instead of the whole filtered set. Used for the default (recency) view;
    the caller falls back to the Python path for free-text search and arbitrary
    field sorts. Raises on any DB error so the caller can fall back safely.
    """
    from sqlalchemy import select, func, cast, String, and_

    conds = _inventory_base_conditions(f_game, f_album, f_template, view_catalog, held_state)

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
        sold_view=(held_state is False),
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
    # The header is user-influenced, not just the body: `ordered` above is built from
    # extracted_data.keys(), and /update_scan lets an inventory:edit user name a field
    # whatever they like. The fixed strings around it are unaffected by _csv_safe.
    w.writerow([_csv_safe(h) for h in header])
    for label, i in flat:
        r = recs.get(i)
        if not r:
            w.writerow([_csv_safe(label), i] + [""] * (len(header) - 2))
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
        w.writerow([_csv_safe(c) for c in row])
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
    """Mark the picked records as sold (Held -> False) rather than deleting them.
    The cards have been physically pulled to build the pack/set, so they leave the
    Inventory list and move to the Sold view, but the entries (and their images,
    prices, sale history) are kept. Idempotent: records already sold are left as-is."""
    body = request.get_json(silent=True) or {}
    groups = body.get("groups") or []
    ids = [i for _l, i in _builder_flatten(groups)]
    if not ids:
        return jsonify({"status": "error", "message": "No cards to pull."}), 400

    recs = ScanRecord.query.filter(ScanRecord.id.in_(ids)).all()
    if not recs:
        return jsonify({"status": "success", "moved": 0, "message": "Nothing to move."})

    # Flip Held -> False (source of truth in extracted_data; the mapper event
    # resyncs is_held so these drop off Inventory and appear under Sold).
    _set_held([r.id for r in recs], False)
    moved = len(recs)

    return jsonify({"status": "success", "moved": moved,
                    "message": f"Pulled {moved} card(s) — marked Sold and moved to the Sold view."})


# ============================ ANALYTICS ============================
# A cross-inventory analytics page: query the Held ("in stock") and/or Sold
# entries, group by any field, aggregate counts + money fields, and view the
# result as a dashboard (charts) or export it as a spreadsheet (CSV). Aggregation
# runs in Python over the hot-column-filtered set (same approach as the inventory
# Python path); the SQL pre-filter keeps only relevant rows in play.

_ANALYTICS_MONEY_FIELDS = [
    ("intake_price",  "Intake $"),
    ("current_value", "Current $"),
    ("sold_price",    "Sold $"),
]

# Always-offered group-by dimensions. "__status__" and "template" are synthetic
# (derived), the rest are extracted_data keys. Dynamic entry fields discovered
# from a live sample are appended to these.
_ANALYTICS_BASE_DIMENSIONS = [
    ("game", "Game"), ("album", "Storage"), ("__status__", "Held / Sold"),
    ("storage_type", "Storage type"), ("rarity", "Rarity"),
    ("edition", "Edition"), ("holographic", "Holo"), ("condition", "Condition"),
]

_ANALYTICS_METRIC_LABELS = {
    "count":              "Count",
    "sum_intake_price":   "Sum Intake $",
    "avg_intake_price":   "Avg Intake $",
    "sum_current_value":  "Sum Current $",
    "avg_current_value":  "Avg Current $",
    "sum_sold_price":     "Sum Sold $",
    "avg_sold_price":     "Avg Sold $",
    "sum_profit":         "Sum Profit $",
    "avg_profit":         "Avg Profit $",
}
_ANALYTICS_ARR_BY_FIELD = {
    "intake_price": "intake", "current_value": "current",
    "sold_price": "sold", "profit": "profit",
}


def _analytics_money(v):
    """Parse a money-ish value ('$1,234.50', '12', '') into a float or None."""
    if v is None:
        return None
    s = str(v).strip().replace("$", "").replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _analytics_group_value(rec, field):
    data = rec.extracted_data or {}
    if not field:
        return "All"
    if field == "__status__":
        return "Held" if _held_from(data) else "Sold"
    if field == "template":
        return rec.template_used or "—"
    v = data.get(field, "")
    v = str(v).strip() if v is not None else ""
    return v or "—"


def _analytics_filtered_records(source, f_game, f_template, search):
    """Records matching the source (held/sold/all) + optional game/template/search.
    Catalog and archived rows are always excluded."""
    from sqlalchemy import func as _f, and_
    conds = [
        _f.coalesce(ScanRecord.is_catalog, False) == False,   # noqa: E712
        _f.coalesce(ScanRecord.is_archived, False) == False,  # noqa: E712
    ]
    if source == "held":
        conds.append(_f.coalesce(ScanRecord.is_held, True) == True)   # noqa: E712
    elif source == "sold":
        conds.append(_f.coalesce(ScanRecord.is_held, True) == False)  # noqa: E712
    # source == "all" -> both held and sold
    if f_game:
        conds.append(ScanRecord.game_key == f_game.strip().lower())
    if f_template:
        conds.append(ScanRecord.template_used == f_template)
    q = ScanRecord.query.filter(and_(*conds))
    if search:
        q = q.filter(_scan_search_condition(search))
    return q.all()


def _analytics_available_dimensions(sample):
    dims, have = [], set()
    for k, lbl in _ANALYTICS_BASE_DIMENSIONS:
        dims.append({"key": k, "label": lbl}); have.add(k)
    dims.append({"key": "template", "label": "Template"}); have.add("template")
    for f in discover_entry_fields(sample):
        if f not in have:
            dims.append({"key": f, "label": f.replace("_", " ").title()})
            have.add(f)
    return dims


def _analytics_metric_value(bucket, key):
    if key == "count":
        return bucket["count"]
    op, _, rest = key.partition("_")           # "sum" / "avg", "intake_price"...
    arr = bucket.get(_ANALYTICS_ARR_BY_FIELD.get(rest, ""), [])
    if op == "sum":
        return round(sum(arr), 2)
    if op == "avg":
        return round(sum(arr) / len(arr), 2) if arr else 0
    return 0


def _analytics_run(source, f_game, f_template, search, group_by, metrics):
    """Core aggregation shared by the query API and CSV export. Returns a dict
    with metric defs, per-group rows, grand totals, and record/group counts."""
    valid = [m for m in (metrics or []) if m in _ANALYTICS_METRIC_LABELS]
    if "count" not in valid:
        valid = ["count"] + valid
    records = _analytics_filtered_records(source, f_game, f_template, search)

    buckets, order = {}, []
    for rec in records:
        g = _analytics_group_value(rec, group_by)
        b = buckets.get(g)
        if b is None:
            b = buckets[g] = {"count": 0, "intake": [], "current": [], "sold": [], "profit": []}
            order.append(g)
        b["count"] += 1
        data = rec.extracted_data or {}
        ip = _analytics_money(data.get("intake_price"))
        cv = _analytics_money(data.get("current_value"))
        sp = _analytics_money(data.get("sold_price"))
        if ip is not None: b["intake"].append(ip)
        if cv is not None: b["current"].append(cv)
        if sp is not None: b["sold"].append(sp)
        if ip is not None and sp is not None: b["profit"].append(sp - ip)

    rows = []
    for g in order:
        b = buckets[g]
        rows.append({"group": g, "values": {m: _analytics_metric_value(b, m) for m in valid}})

    # Sort by the first money metric if present, else by count — descending.
    primary = next((m for m in valid if m != "count"), "count")
    rows.sort(key=lambda r: r["values"].get(primary, 0), reverse=True)

    # Grand totals across every matching record (one overall bucket).
    total_b = {"count": 0, "intake": [], "current": [], "sold": [], "profit": []}
    for b in buckets.values():
        total_b["count"] += b["count"]
        for k in ("intake", "current", "sold", "profit"):
            total_b[k].extend(b[k])
    totals = {m: _analytics_metric_value(total_b, m) for m in valid}

    return {
        "metrics": [{"key": m, "label": _ANALYTICS_METRIC_LABELS[m]} for m in valid],
        "rows": rows,
        "totals": totals,
        "record_count": len(records),
        "group_count": len(rows),
        "group_by": group_by,
    }


def _analytics_params(src):
    """Pull analytics params from a JSON body or form/query dict `src`."""
    source = (src.get("source") or "all").strip().lower()
    if source not in ("held", "sold", "all"):
        source = "all"
    group_by = (src.get("group_by") or "").strip()
    metrics = src.get("metrics") or ["count"]
    if isinstance(metrics, str):
        metrics = [m for m in metrics.split(",") if m]
    return {
        "source": source,
        "f_game": (src.get("game") or "").strip(),
        "f_template": (src.get("template") or "").strip(),
        "search": (src.get("search") or "").strip(),
        "group_by": group_by,
        "metrics": metrics,
    }


def _collection_lot_prices():
    """{name_key: bought_for} for collections that have a 'Bought For' lot price."""
    return {cp.name_key: cp.bought_for
            for cp in CollectionPrice.query.filter(CollectionPrice.bought_for.isnot(None)).all()}


def _collection_counts(records):
    """{name_key: number of cards} across the given records (the cards a lot covers)."""
    counts = {}
    for r in records:
        c = str((r.extracted_data or {}).get("collection") or "").strip().lower()
        if c:
            counts[c] = counts.get(c, 0) + 1
    return counts


def _record_effective_cost(rec, lot_prices, lot_counts):
    """Cost attributed to one card: an equal share of its collection's lot price
    when that collection has a 'Bought For' set (so the cards don't each need an
    intake_price), otherwise the card's own intake_price."""
    d = rec.extracted_data or {}
    ck = str(d.get("collection") or "").strip().lower()
    if ck and ck in lot_prices:
        n = lot_counts.get(ck, 0)
        return (lot_prices[ck] / n) if n else 0.0
    return _analytics_money(d.get("intake_price")) or 0.0


@app.route("/analytics")
def analytics_page():
    ensure_dirs()
    from sqlalchemy import func as _f
    games = _builder_games()
    sample = (ScanRecord.query
              .filter(_f.coalesce(ScanRecord.is_catalog, False) == False,    # noqa: E712
                      _f.coalesce(ScanRecord.is_archived, False) == False)   # noqa: E712
              .limit(500).all())
    dimensions = _analytics_available_dimensions(sample)
    return render_template(
        "analytics.html",
        games=games,
        dimensions=dimensions,
        money_fields=_ANALYTICS_MONEY_FIELDS,
        metric_labels=_ANALYTICS_METRIC_LABELS,
        templates=get_template_names(),
    )


@app.route("/analytics/query", methods=["POST"])
def analytics_query():
    p = _analytics_params(request.get_json(silent=True) or {})
    result = _analytics_run(**p)
    result["status"] = "success"
    return jsonify(result)


@app.route("/analytics/export", methods=["POST"])
def analytics_export():
    # Accept JSON or form so the button can post either way.
    src = request.get_json(silent=True) or request.form.to_dict() or {}
    p = _analytics_params(src)
    result = _analytics_run(**p)

    import csv as _csv
    from io import StringIO
    buf = StringIO()
    w = _csv.writer(buf)
    group_label = next((d["label"] for d in _analytics_available_dimensions([])
                        if d["key"] == p["group_by"]), p["group_by"] or "All")
    # group_label falls back to p["group_by"], which _analytics_params only .strip()s --
    # unlike "source" it is not checked against a whitelist, so it is request data.
    w.writerow([_csv_safe(group_label)] + [_csv_safe(m["label"]) for m in result["metrics"]])
    for row in result["rows"]:
        w.writerow([_csv_safe(row["group"])] + [_csv_safe(row["values"][m["key"]]) for m in result["metrics"]])
    w.writerow([])
    w.writerow(["TOTAL"] + [result["totals"][m["key"]] for m in result["metrics"]])

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"analytics_{p['source']}_{(p['group_by'] or 'all')}_{stamp}.csv"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.route("/analytics/overview")
def analytics_overview():
    """Landing chart data: per game, the combined recorded cost basis
    (sum of intake_price) vs the current value (sum of current_value). The
    client derives "expected proceeds at current market rate" from the current
    value and an adjustable marketplace-fee %, so the fee slider updates live.
    source=held (default) covers owned cards; source=all includes sold too."""
    source = (request.args.get("source") or "held").strip().lower()
    if source not in ("held", "all"):
        source = "held"
    records = _analytics_filtered_records(source, "", "", "")

    # Lot prices spread across a collection's cards; denominator is the full
    # collection (held+sold) so an owned subset gets its pro-rata share.
    lot_prices = _collection_lot_prices()
    lot_counts = _collection_counts(_analytics_filtered_records("all", "", "", "")) if lot_prices else {}

    by = {}
    for r in records:
        d = r.extracted_data or {}
        g = (str(d.get("game") or "").strip() or "—")
        b = by.get(g)
        if b is None:
            b = by[g] = {"game": g, "count": 0, "cost": 0.0, "value": 0.0}
        b["count"] += 1
        b["cost"] += _record_effective_cost(r, lot_prices, lot_counts)
        cv = _analytics_money(d.get("current_value"))
        if cv is not None: b["value"] += cv

    rows = sorted(by.values(), key=lambda x: x["value"], reverse=True)
    for b in rows:
        b["cost"] = round(b["cost"], 2)
        b["value"] = round(b["value"], 2)
    totals = {
        "count": sum(b["count"] for b in rows),
        "cost": round(sum(b["cost"] for b in rows), 2),
        "value": round(sum(b["value"] for b in rows), 2),
    }
    return jsonify({"status": "success", "source": source, "rows": rows, "totals": totals})


@app.route("/analytics/collections")
def analytics_collections():
    """Distinct collection names (with counts) for the collection dropdown."""
    records = _analytics_filtered_records("all", "", "", "")
    by = {}
    for r in records:
        c = str((r.extracted_data or {}).get("collection") or "").strip()
        if c:
            by[c] = by.get(c, 0) + 1
    rows = sorted(({"name": k, "count": v} for k, v in by.items()),
                  key=lambda x: x["name"].lower())
    return jsonify({"status": "success", "collections": rows})


@app.route("/analytics/collection")
def analytics_collection():
    """Per-game breakdown for one collection: what was paid (intake), the current
    value of still-held cards (the client turns this into expected proceeds using
    the market fee), and what was realized where a Sold Price is filled in. Held
    cards feed 'value' and sold cards feed 'sold', so the two never double-count
    the same card."""
    name = (request.args.get("name") or "").strip()
    if not name:
        return jsonify({"status": "error", "message": "No collection specified."}), 400
    target = name.lower()

    records = _analytics_filtered_records("all", "", "", "")
    lot_prices = _collection_lot_prices()
    lot_counts = _collection_counts(records)
    bought_for = lot_prices.get(target)

    by = {}
    for r in records:
        d = r.extracted_data or {}
        if str(d.get("collection") or "").strip().lower() != target:
            continue
        g = str(d.get("game") or "").strip() or "—"
        b = by.get(g)
        if b is None:
            b = by[g] = {"game": g, "count": 0, "cost": 0.0, "value": 0.0, "sold": 0.0}
        b["count"] += 1
        b["cost"] += _record_effective_cost(r, lot_prices, lot_counts)
        cv = _analytics_money(d.get("current_value"))
        sp = _analytics_money(d.get("sold_price"))
        if cv is not None and _held_from(d):     # expected = still-owned cards only
            b["value"] += cv
        if sp is not None:                        # realized = cards with a Sold Price
            b["sold"] += sp

    rows = sorted(by.values(), key=lambda x: (x["cost"] + x["value"] + x["sold"]), reverse=True)
    for b in rows:
        b["cost"] = round(b["cost"], 2)
        b["value"] = round(b["value"], 2)
        b["sold"] = round(b["sold"], 2)
    total_cost = bought_for if bought_for is not None else round(sum(b["cost"] for b in rows), 2)
    totals = {
        "count": sum(b["count"] for b in rows),
        "cost": round(total_cost, 2),
        "value": round(sum(b["value"] for b in rows), 2),
        "sold": round(sum(b["sold"] for b in rows), 2),
    }
    return jsonify({"status": "success", "name": name, "rows": rows, "totals": totals,
                    "bought_for": bought_for,
                    "cost_source": "lot" if bought_for is not None else "entries"})


# ====================== PERIOD REPORTS ======================
# Weekly / monthly / quarterly / annual PDF (and on-screen) reports covering
# acquisitions, sales by shop, derived strategies, and the financial outcome for
# the period. Acquisitions key off ScanRecord.scan_date; sales come from each
# record's extracted_data (integration sales_log entries carry a shop + price +
# date, manual sales carry sold_price + sold_at).

def _parse_dt(v):
    """Parse an ISO-ish datetime (or datetime) into a naive datetime, or None."""
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    s = str(v).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
    return None


def _report_period_range(period_type, ref=None):
    """Return (start, end, label, normalized_type) for the period containing ref."""
    ref = ref or datetime.now()
    day = datetime(ref.year, ref.month, ref.day)
    pt = (period_type or "monthly").lower()
    if pt == "weekly":
        start = day - timedelta(days=day.weekday())          # Monday
        end = start + timedelta(days=7)
        label = f"Week of {start:%b %d} – {end - timedelta(days=1):%b %d, %Y}"
    elif pt == "quarterly":
        q = (day.month - 1) // 3
        start = datetime(day.year, q * 3 + 1, 1)
        em = q * 3 + 4
        end = datetime(day.year + (1 if em > 12 else 0), ((em - 1) % 12) + 1, 1)
        label = f"Q{q + 1} {day.year}"
    elif pt == "annual":
        start = datetime(day.year, 1, 1)
        end = datetime(day.year + 1, 1, 1)
        label = f"{day.year}"
    else:
        pt = "monthly"
        start = datetime(day.year, day.month, 1)
        end = datetime(day.year + (1 if day.month == 12 else 0), (day.month % 12) + 1, 1)
        label = f"{start:%B %Y}"
    return start, end, label, pt


_SOURCE_LABELS = {"ebay": "eBay", "tcgplayer": "TCGplayer", "tcg": "TCGplayer",
                  "manual": "Manual / Direct", "": "Manual / Direct", "other": "Other"}


def _norm_source(s):
    k = (s or "").strip().lower()
    return _SOURCE_LABELS.get(k, k.title() if k else "Manual / Direct")


def _report_record_title(rec):
    d = rec.extracted_data or {}
    name = (d.get("name") or d.get("card_name") or d.get("title") or "").strip()
    num = (d.get("number") or d.get("card_number") or "").strip()
    setn = (d.get("set") or d.get("set_name") or "").strip()
    bits = [name or f"Record #{rec.id}"]
    if num:
        bits.append(f"#{num}")
    if setn:
        bits.append(f"({setn})")
    return " ".join(bits)


def _record_sale_in_period(rec, start, end):
    """If this record recorded a sale within [start, end), return
    {source, revenue, date, qty}; else None."""
    data = rec.extracted_data or {}
    entries = []
    for e in (data.get("sales_log") or []):
        d = _parse_dt(e.get("at"))
        if d:
            entries.append((d, e.get("source") or "other", _analytics_money(e.get("price"))))
    if entries:
        in_p = [(d, s, p) for (d, s, p) in entries if start <= d < end]
        if not in_p:
            return None
        revenue = sum(p for (_, _, p) in in_p if p is not None)
        if not revenue:
            sp = _analytics_money(data.get("sold_price"))
            revenue = sp if sp is not None else 0.0
        last = max(in_p, key=lambda t: t[0])
        return {"source": last[1], "revenue": revenue or 0.0, "date": last[0], "qty": len(in_p)}
    if not _held_from(data):
        sp = _analytics_money(data.get("sold_price"))
        at = _parse_dt(data.get("sold_at"))
        if sp is not None and at and start <= at < end:
            return {"source": "manual", "revenue": sp, "date": at, "qty": 1}
    return None


def _report_build(start, end, label, period_type):
    """Aggregate everything for the period into a plain dict (money as floats)."""
    from sqlalchemy import func as _f, and_
    money = _analytics_money
    records = ScanRecord.query.filter(and_(
        _f.coalesce(ScanRecord.is_catalog, False) == False,   # noqa: E712
        _f.coalesce(ScanRecord.is_archived, False) == False,  # noqa: E712
    )).all()

    # ── Acquisitions in period, grouped by collection ──
    acq, acq_tot = {}, {"count": 0, "intake": 0.0, "market": 0.0, "sold_count": 0, "sold_value": 0.0}
    for rec in records:
        d = rec.scan_date
        if not (d and start <= d < end):
            continue
        data = rec.extracted_data or {}
        coll = (data.get("collection") or "").strip() or "Uncategorized"
        ip = money(data.get("intake_price")) or 0.0
        cv = money(data.get("current_value")) or 0.0
        sp = money(data.get("sold_price"))
        sold = not _held_from(data)
        b = acq.setdefault(coll, {"collection": coll, "count": 0, "intake": 0.0, "market": 0.0,
                                  "sold_count": 0, "sold_value": 0.0, "games": set()})
        b["count"] += 1
        b["intake"] += ip
        b["market"] += cv
        g = (data.get("game") or "").strip()
        if g:
            b["games"].add(g)
        acq_tot["count"] += 1
        acq_tot["intake"] += ip
        acq_tot["market"] += cv
        if sold and sp is not None:
            b["sold_count"] += 1
            b["sold_value"] += sp
            acq_tot["sold_count"] += 1
            acq_tot["sold_value"] += sp
    acq_rows = sorted(acq.values(), key=lambda r: r["intake"], reverse=True)
    for r in acq_rows:
        r["games"] = ", ".join(sorted(r["games"])) if r["games"] else "—"

    # ── Sales in period, grouped by shop/source ──
    sales, sales_tot = {}, {"count": 0, "revenue": 0.0, "cost": 0.0, "profit": 0.0}
    sale_items, undated = [], 0
    for rec in records:
        s = _record_sale_in_period(rec, start, end)
        if s is None:
            data = rec.extracted_data or {}
            if (not _held_from(data)) and not (data.get("sales_log")) \
               and money(data.get("sold_price")) is not None and _parse_dt(data.get("sold_at")) is None:
                undated += 1
            continue
        data = rec.extracted_data or {}
        intake = money(data.get("intake_price")) or 0.0
        src = _norm_source(s["source"])
        b = sales.setdefault(src, {"source": src, "count": 0, "revenue": 0.0, "cost": 0.0, "profit": 0.0})
        b["count"] += s["qty"]
        b["revenue"] += s["revenue"]
        b["cost"] += intake
        b["profit"] += (s["revenue"] - intake)
        sales_tot["count"] += s["qty"]
        sales_tot["revenue"] += s["revenue"]
        sales_tot["cost"] += intake
        sales_tot["profit"] += (s["revenue"] - intake)
        sale_items.append({"title": _report_record_title(rec), "source": src, "date": s["date"],
                           "revenue": s["revenue"], "intake": intake, "profit": s["revenue"] - intake,
                           "collection": (data.get("collection") or "—")})
    sales_rows = sorted(sales.values(), key=lambda r: r["revenue"], reverse=True)
    sale_items.sort(key=lambda i: i["date"], reverse=True)

    # ── Strategies (explicit tag if present, else derived from holding behavior) ──
    strat = {}
    for rec in records:
        d = rec.scan_date
        if not (d and start <= d < end):
            continue
        data = rec.extracted_data or {}
        tag = (data.get("strategy") or "").strip()
        if not tag:
            if not _held_from(data):
                sale_dt = _parse_dt(data.get("sold_at"))
                if not sale_dt:
                    sl = [_parse_dt(e.get("at")) for e in (data.get("sales_log") or [])]
                    sl = [x for x in sl if x]
                    sale_dt = max(sl) if sl else None
                held_days = (sale_dt - d).days if sale_dt else None
                tag = "Flip (sold ≤30 days)" if (held_days is not None and held_days <= 30) else "Resold (held >30 days)"
            else:
                tag = "Buy & Hold"
        ip = money(data.get("intake_price")) or 0.0
        cv = money(data.get("current_value")) or 0.0
        sp = money(data.get("sold_price"))
        b = strat.setdefault(tag, {"strategy": tag, "count": 0, "cost": 0.0, "market": 0.0, "realized": 0.0})
        b["count"] += 1
        b["cost"] += ip
        b["market"] += cv
        if (not _held_from(data)) and sp is not None:
            b["realized"] += sp - ip
    strat_rows = sorted(strat.values(), key=lambda r: r["count"], reverse=True)

    # ── Unrealized gain on held items acquired this period ──
    unrealized = 0.0
    for rec in records:
        d = rec.scan_date
        if d and start <= d < end and _held_from(rec.extracted_data or {}):
            data = rec.extracted_data or {}
            unrealized += (money(data.get("current_value")) or 0.0) - (money(data.get("intake_price")) or 0.0)

    financial = {
        "acq_cost": acq_tot["intake"],
        "acq_market": acq_tot["market"],
        "sales_revenue": sales_tot["revenue"],
        "sales_cost": sales_tot["cost"],
        "realized_profit": sales_tot["profit"],
        "unrealized_gain": unrealized,
        "net_cash_flow": sales_tot["revenue"] - acq_tot["intake"],
    }

    return {
        "label": label, "period_type": period_type, "start": start, "end": end,
        "generated": datetime.now(),
        "acquisitions": {"rows": acq_rows, "totals": acq_tot},
        "sales": {"rows": sales_rows, "totals": sales_tot, "items": sale_items, "undated": undated},
        "strategies": strat_rows,
        "financial": financial,
    }


def _money_str(v):
    v = v or 0.0
    return ("-$%s" % format(abs(v), ",.2f")) if v < 0 else ("$%s" % format(v, ",.2f"))


def _report_from_args():
    period = (request.args.get("period") or "monthly").lower()
    ref = _parse_dt((request.args.get("date") or "").strip()) or datetime.now()
    start, end, label, pt = _report_period_range(period, ref)
    return _report_build(start, end, label, pt)


# ── On-screen HTML report ──
def _report_to_html(rep):
    from markupsafe import escape as e
    fin = rep["financial"]
    pt = rep["period_type"].title()

    def cell(v, cls=""):
        return f'<td class="{cls}">{e(v)}</td>'

    def money_td(v):
        cls = "pos" if (v or 0) > 0 else ("neg" if (v or 0) < 0 else "")
        return f'<td class="num {cls}">{e(_money_str(v))}</td>'

    # Financial summary
    fin_rows = "".join(
        f"<tr><th>{e(lbl)}</th>{money_td(val)}</tr>" for lbl, val in [
            ("Acquisition cost (spent)", fin["acq_cost"]),
            ("Acquisition market value", fin["acq_market"]),
            ("Sales revenue (income)", fin["sales_revenue"]),
            ("Cost basis of items sold", fin["sales_cost"]),
            ("Realized profit on sales", fin["realized_profit"]),
            ("Net cash flow (revenue − spend)", fin["net_cash_flow"]),
            ("Unrealized gain on held (acquired this period)", fin["unrealized_gain"]),
        ])

    st = rep["sales"]["totals"]
    sales_rows = "".join(
        f"<tr>{cell(r['source'])}<td class='num'>{r['count']}</td>{money_td(r['revenue'])}"
        f"{money_td(r['cost'])}{money_td(r['profit'])}</tr>" for r in rep["sales"]["rows"]) \
        or "<tr><td colspan=5 class='muted'>No sales recorded in this period.</td></tr>"
    sales_total = (f"<tr class='tot'><th>Total</th><td class='num'>{st['count']}</td>"
                   f"{money_td(st['revenue'])}{money_td(st['cost'])}{money_td(st['profit'])}</tr>")

    at = rep["acquisitions"]["totals"]
    acq_rows = "".join(
        f"<tr>{cell(r['collection'])}<td class='num'>{r['count']}</td>{money_td(r['intake'])}"
        f"{money_td(r['market'])}<td class='num'>{r['sold_count']}</td>{money_td(r['sold_value'])}"
        f"{cell(r['games'])}</tr>" for r in rep["acquisitions"]["rows"]) \
        or "<tr><td colspan=7 class='muted'>No acquisitions recorded in this period.</td></tr>"
    acq_total = (f"<tr class='tot'><th>Total</th><td class='num'>{at['count']}</td>"
                 f"{money_td(at['intake'])}{money_td(at['market'])}<td class='num'>{at['sold_count']}</td>"
                 f"{money_td(at['sold_value'])}<td></td></tr>")

    strat_rows = "".join(
        f"<tr>{cell(r['strategy'])}<td class='num'>{r['count']}</td>{money_td(r['cost'])}"
        f"{money_td(r['market'])}{money_td(r['realized'])}</tr>" for r in rep["strategies"]) \
        or "<tr><td colspan=5 class='muted'>No activity in this period.</td></tr>"

    items = rep["sales"]["items"][:60]
    item_rows = "".join(
        f"<tr>{cell(i['title'])}{cell(i['source'])}<td>{i['date']:%Y-%m-%d}</td>"
        f"{money_td(i['revenue'])}{money_td(i['intake'])}{money_td(i['profit'])}</tr>" for i in items) \
        or "<tr><td colspan=6 class='muted'>No sales to itemize.</td></tr>"

    undated_note = ""
    if rep["sales"]["undated"]:
        undated_note = (f"<p class='muted'>Note: {rep['sales']['undated']} manually-sold item(s) had no "
                        f"recorded sale date and were excluded from period sales. Re-marking them sold now "
                        f"stamps a date.</p>")

    qs = f"period={rep['period_type']}&date={rep['start']:%Y-%m-%d}"
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>{e(pt)} report — {e(rep['label'])}</title>
<style>
 body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2937;background:#f5f7fb;margin:0;padding:28px;}}
 .wrap{{max-width:1000px;margin:0 auto;background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:28px 32px;box-shadow:0 10px 30px rgba(0,0,0,.05);}}
 h1{{font-size:22px;margin:0 0 2px;}} .sub{{color:#6b7280;margin:0 0 18px;}}
 h2{{font-size:15px;text-transform:uppercase;letter-spacing:.04em;color:#4338ca;margin:26px 0 8px;}}
 table{{width:100%;border-collapse:collapse;margin-top:4px;font-size:14px;}}
 th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid #eef0f4;}}
 td.num,th.num{{text-align:right;}} .num{{text-align:right;font-variant-numeric:tabular-nums;}}
 tr.tot th,tr.tot td{{border-top:2px solid #d1d5db;font-weight:700;background:#fafafe;}}
 .pos{{color:#047857;}} .neg{{color:#b91c1c;}} .muted{{color:#9ca3af;}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 4px;}}
 .kpi{{flex:1;min-width:150px;border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;}}
 .kpi .l{{font-size:12px;color:#6b7280;}} .kpi .v{{font-size:20px;font-weight:700;}}
 .toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;}}
 a.btn{{background:#4f46e5;color:#fff;text-decoration:none;padding:9px 15px;border-radius:10px;font-weight:600;font-size:14px;}}
 a.btn.sec{{background:#eef2ff;color:#4338ca;}}
 @media print{{.toolbar{{display:none;}} body{{background:#fff;padding:0;}} .wrap{{box-shadow:none;border:0;}}}}
</style></head><body><div class=wrap>
 <div class=toolbar>
   <a class="btn" href="/reports/pdf?{qs}">Download PDF</a>
   <a class="btn sec" href="/reports">Choose another period</a>
 </div>
 <h1>{e(pt)} Report — {e(rep['label'])}</h1>
 <p class=sub>{rep['start']:%b %d, %Y} to {rep['end'] - timedelta(days=1):%b %d, %Y} · generated {rep['generated']:%Y-%m-%d %H:%M}</p>

 <div class=cards>
   <div class=kpi><div class=l>Acquisition cost</div><div class=v>{e(_money_str(fin['acq_cost']))}</div></div>
   <div class=kpi><div class=l>Sales revenue</div><div class=v>{e(_money_str(fin['sales_revenue']))}</div></div>
   <div class=kpi><div class=l>Realized profit</div><div class="v {'pos' if fin['realized_profit']>=0 else 'neg'}">{e(_money_str(fin['realized_profit']))}</div></div>
   <div class=kpi><div class=l>Net cash flow</div><div class="v {'pos' if fin['net_cash_flow']>=0 else 'neg'}">{e(_money_str(fin['net_cash_flow']))}</div></div>
 </div>

 <h2>Financial outcome</h2>
 <table>{fin_rows}</table>

 <h2>Sales by shop</h2>
 <table><thead><tr><th>Source</th><th class=num>Sales</th><th class=num>Revenue</th><th class=num>Cost basis</th><th class=num>Profit</th></tr></thead>
 <tbody>{sales_rows}{sales_total}</tbody></table>
 {undated_note}

 <h2>Acquisitions by collection</h2>
 <table><thead><tr><th>Collection</th><th class=num>Items</th><th class=num>Purchase</th><th class=num>Market</th><th class=num>Sold</th><th class=num>Sold value</th><th>Games</th></tr></thead>
 <tbody>{acq_rows}{acq_total}</tbody></table>

 <h2>Strategies</h2>
 <table><thead><tr><th>Strategy</th><th class=num>Items</th><th class=num>Cost</th><th class=num>Market</th><th class=num>Realized</th></tr></thead>
 <tbody>{strat_rows}</tbody></table>

 <h2>Itemized sales</h2>
 <table><thead><tr><th>Item</th><th>Source</th><th>Date</th><th class=num>Revenue</th><th class=num>Cost</th><th class=num>Profit</th></tr></thead>
 <tbody>{item_rows}</tbody></table>
</div></body></html>"""


# ── PDF report (reportlab) ──
def _report_to_pdf(rep):
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    INDIGO = colors.HexColor("#4338ca")
    GREY = colors.HexColor("#6b7280")
    LINE = colors.HexColor("#e5e7eb")
    POS = colors.HexColor("#047857")
    NEG = colors.HexColor("#b91c1c")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18, spaceAfter=2, textColor=colors.HexColor("#111827"))
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, textColor=GREY, spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, textColor=INDIGO, spaceBefore=14, spaceAfter=4)
    note = ParagraphStyle("note", parent=styles["Normal"], fontSize=8, textColor=GREY, spaceBefore=4)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                            leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            title=f"{rep['period_type'].title()} report {rep['label']}")
    story = []
    story.append(Paragraph(f"{rep['period_type'].title()} Report &mdash; {rep['label']}", h1))
    story.append(Paragraph(
        f"{rep['start']:%b %d, %Y} to {rep['end'] - timedelta(days=1):%b %d, %Y} &middot; "
        f"generated {rep['generated']:%Y-%m-%d %H:%M}", sub))

    def money_cells(row, idxs):
        """Build a TableStyle color list for money columns (green/red)."""
        cmds = []
        for (ri, ci, val) in idxs:
            cmds.append(("TEXTCOLOR", (ci, ri), (ci, ri), POS if (val or 0) > 0 else (NEG if (val or 0) < 0 else colors.black)))
        return cmds

    def make_table(header, rows, aligns, col_widths=None, money_cols=(), totals_row=False):
        data = [header] + rows
        t = Table(data, colWidths=col_widths, repeatRows=1)
        style = [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]
        for ci, a in enumerate(aligns):
            style.append(("ALIGN", (ci, 0), (ci, -1), a))
        # money color per data cell
        for (ri, row) in enumerate(rows, start=1):
            for ci in money_cols:
                raw = row[ci]
                v = 0.0
                try:
                    v = float(str(raw).replace("$", "").replace(",", ""))
                except Exception:
                    v = 0.0
                if v > 0:
                    style.append(("TEXTCOLOR", (ci, ri), (ci, ri), POS))
                elif v < 0:
                    style.append(("TEXTCOLOR", (ci, ri), (ci, ri), NEG))
        if totals_row:
            r = len(data) - 1
            style += [("FONT", (0, r), (-1, r), "Helvetica-Bold", 9),
                      ("LINEABOVE", (0, r), (-1, r), 1, colors.HexColor("#9ca3af")),
                      ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fafafe"))]
        t.setStyle(TableStyle(style))
        return t

    fin = rep["financial"]
    story.append(Paragraph("Financial outcome", h2))
    fin_data = [
        ("Acquisition cost (spent)", _money_str(fin["acq_cost"])),
        ("Acquisition market value", _money_str(fin["acq_market"])),
        ("Sales revenue (income)", _money_str(fin["sales_revenue"])),
        ("Cost basis of items sold", _money_str(fin["sales_cost"])),
        ("Realized profit on sales", _money_str(fin["realized_profit"])),
        ("Net cash flow (revenue - spend)", _money_str(fin["net_cash_flow"])),
        ("Unrealized gain on held (acquired this period)", _money_str(fin["unrealized_gain"])),
    ]
    story.append(make_table(["Metric", "Amount"], [list(r) for r in fin_data],
                            ["LEFT", "RIGHT"], col_widths=[4.7 * inch, 2.1 * inch], money_cols=(1,)))

    st = rep["sales"]["totals"]
    story.append(Paragraph("Sales by shop", h2))
    srows = [[r["source"], str(r["count"]), _money_str(r["revenue"]), _money_str(r["cost"]), _money_str(r["profit"])]
             for r in rep["sales"]["rows"]]
    if not srows:
        srows = [["No sales in this period", "", "", "", ""]]
    srows.append(["Total", str(st["count"]), _money_str(st["revenue"]), _money_str(st["cost"]), _money_str(st["profit"])])
    story.append(make_table(["Source", "Sales", "Revenue", "Cost basis", "Profit"], srows,
                            ["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
                            col_widths=[2.4 * inch, 0.8 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch],
                            money_cols=(2, 3, 4), totals_row=True))
    if rep["sales"]["undated"]:
        story.append(Paragraph(
            f"Note: {rep['sales']['undated']} manually-sold item(s) had no recorded sale date and were "
            f"excluded from period sales.", note))

    at = rep["acquisitions"]["totals"]
    story.append(Paragraph("Acquisitions by collection", h2))
    arows = [[r["collection"], str(r["count"]), _money_str(r["intake"]), _money_str(r["market"]),
              str(r["sold_count"]), _money_str(r["sold_value"]), r["games"]]
             for r in rep["acquisitions"]["rows"]]
    if not arows:
        arows = [["No acquisitions in this period", "", "", "", "", "", ""]]
    arows.append(["Total", str(at["count"]), _money_str(at["intake"]), _money_str(at["market"]),
                  str(at["sold_count"]), _money_str(at["sold_value"]), ""])
    story.append(make_table(["Collection", "Items", "Purchase", "Market", "Sold", "Sold value", "Games"], arows,
                            ["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "LEFT"],
                            col_widths=[1.7 * inch, 0.6 * inch, 0.95 * inch, 0.95 * inch, 0.55 * inch, 0.95 * inch, 1.35 * inch],
                            money_cols=(2, 3, 5), totals_row=True))

    story.append(Paragraph("Strategies", h2))
    strows = [[r["strategy"], str(r["count"]), _money_str(r["cost"]), _money_str(r["market"]), _money_str(r["realized"])]
              for r in rep["strategies"]] or [["No activity in this period", "", "", "", ""]]
    story.append(make_table(["Strategy", "Items", "Cost", "Market", "Realized"], strows,
                            ["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
                            col_widths=[2.7 * inch, 0.8 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch],
                            money_cols=(2, 3, 4)))

    items = rep["sales"]["items"][:40]
    if items:
        story.append(Paragraph("Itemized sales", h2))
        irows = [[i["title"][:44], i["source"], f"{i['date']:%Y-%m-%d}",
                  _money_str(i["revenue"]), _money_str(i["intake"]), _money_str(i["profit"])] for i in items]
        story.append(make_table(["Item", "Source", "Date", "Revenue", "Cost", "Profit"], irows,
                                ["LEFT", "LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT"],
                                col_widths=[2.5 * inch, 1.1 * inch, 0.9 * inch, 0.95 * inch, 0.85 * inch, 0.9 * inch],
                                money_cols=(3, 4, 5)))

    doc.build(story)
    return buf.getvalue()


_REPORTS_LANDING_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Reports</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2937;background:#f5f7fb;margin:0;padding:28px;}
 .card{max-width:640px;margin:6vh auto 0;background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:26px 28px;box-shadow:0 10px 30px rgba(0,0,0,.06);}
 h1{font-size:22px;margin:0 0 6px;} p.sub{color:#6b7280;margin:0 0 20px;}
 label{display:block;font-weight:700;margin:14px 0 6px;}
 select,input{width:100%;box-sizing:border-box;border:1px solid #d1d5db;border-radius:10px;padding:10px 12px;font-size:15px;}
 .row{display:flex;gap:12px;} .row>div{flex:1;}
 .btns{display:flex;gap:10px;margin-top:20px;flex-wrap:wrap;}
 button{background:#4f46e5;color:#fff;border:0;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer;font-size:14px;}
 button.sec{background:#eef2ff;color:#4338ca;}
 .links{margin-top:20px;font-size:14px;color:#6b7280;} .links a{margin-right:14px;}
 .hint{color:#6b7280;font-size:13px;margin-top:14px;line-height:1.5;}
</style></head><body>
<div class=card>
 <h1>Financial reports</h1>
 <p class=sub>Generate a report of acquisitions, sales by shop, strategies, and the financial outcome for a period.</p>
 <div class=row>
   <div>
     <label for=period>Period</label>
     <select id=period>
       <option value=weekly>Weekly</option>
       <option value=monthly selected>Monthly</option>
       <option value=quarterly>Quarterly</option>
       <option value=annual>Annual</option>
     </select>
   </div>
   <div>
     <label for=date>Any date in the period</label>
     <input type=date id=date>
   </div>
 </div>
 <div class=btns>
   <button id=view>View report</button>
   <button id=pdf class=sec>Download PDF</button>
 </div>
 <p class=hint>The report covers the week, month, quarter, or year containing the date you pick (defaults to today).
    Acquisitions are grouped by collection with purchase and market value; sales are split by shop (eBay, TCGplayer, etc.).</p>
 <div class=links><a href="/">Home</a><a href="/analytics">Analytics</a></div>
</div>
<script>
 var d=new Date();document.getElementById('date').value=d.toISOString().slice(0,10);
 function qs(){return 'period='+encodeURIComponent(document.getElementById('period').value)+'&date='+encodeURIComponent(document.getElementById('date').value);}
 document.getElementById('view').addEventListener('click',function(){location.href='/reports/view?'+qs();});
 document.getElementById('pdf').addEventListener('click',function(){location.href='/reports/pdf?'+qs();});
</script></body></html>"""


@app.route("/reports")
def reports_page():
    return Response(_REPORTS_LANDING_HTML, mimetype="text/html")


@app.route("/reports/view")
def reports_view():
    return Response(_report_to_html(_report_from_args()), mimetype="text/html")


@app.route("/reports/pdf")
def reports_pdf():
    rep = _report_from_args()
    try:
        pdf = _report_to_pdf(rep)
    except ImportError:
        return jsonify({"status": "error",
                        "message": "PDF generation needs the 'reportlab' package. Run: pip install reportlab"}), 500
    fname = "report_%s_%s.pdf" % (rep["period_type"], rep["start"].strftime("%Y%m%d"))
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


def _held_inventory_match(game, name, number):
    """Count HELD (unsold, non-catalog, non-archived) inventory records matching an
    identified card by name (and collector number when both have one). Read-only —
    used purely for the 'already in inventory' indicator."""
    from sqlalchemy import func as _f, and_
    tn = _re.sub(r"[^a-z0-9]+", "", (name or "").lower())
    if not tn:
        return 0
    tnum_variants = _collector_number_variants(number) if number else set()
    q = ScanRecord.query.filter(and_(
        _f.coalesce(ScanRecord.is_catalog, False) == False,   # noqa: E712
        _f.coalesce(ScanRecord.is_archived, False) == False,  # noqa: E712
        # is_held mirrors _held_from by derivation; sold rows never load.
        _f.coalesce(ScanRecord.is_held, True) == True,        # noqa: E712
    ))
    gk = (game or "").strip().lower()
    if gk:
        q = q.filter(ScanRecord.game_key == gk)
    count = 0
    for r in q.all():
        data = r.extracted_data or {}
        if not _held_from(data):
            continue
        rn = _re.sub(r"[^a-z0-9]+", "", (_get_name(data) or "").lower())
        if not rn or not (tn == rn or tn in rn or rn in tn):
            continue
        if tnum_variants:
            rnum = (_get_serial(data) or "").strip()
            rvar = _collector_number_variants(rnum) if rnum else set()
            if rvar and not (tnum_variants & rvar):
                continue   # both have numbers but they disagree
        count += 1
    return count


_QUICK_SCAN_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Quick Scan</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2937;background:#f5f7fb;margin:0;padding:18px;}
 .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px;}
 .bar h1{font-size:20px;margin:0;margin-right:auto;}
 select{border:1px solid #d1d5db;padding:9px 10px;border-radius:9px;font-size:14px;}
 button{background:#4f46e5;color:#fff;border:0;padding:9px 15px;cursor:pointer;border-radius:9px;font-size:14px;font-weight:700;}
 button.sec{background:#eef2ff;color:#4338ca;}
 label.file{background:#eef2ff;color:#4338ca;padding:9px 15px;cursor:pointer;border-radius:9px;font-size:14px;font-weight:700;}
 .note{color:#6b7280;font-size:13px;margin:6px 0 12px;line-height:1.5;}
 .stage{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;}
 .camwrap{position:relative;width:360px;max-width:100%;background:#000;border-radius:12px;overflow:hidden;}
 video{width:100%;display:block;} .frame{position:absolute;inset:8% 14%;border:3px dashed rgba(255,255,255,.7);border-radius:10px;pointer-events:none;}
 .side{flex:1;min-width:300px;}
 #status{min-height:22px;font-size:14px;margin:8px 0;font-weight:600;}
 table{width:100%;border-collapse:collapse;font-size:13px;} th,td{padding:6px 8px;border-bottom:1px solid #eef0f4;text-align:left;}
 th{color:#6b7280;} .miss{color:#b91c1c;} .ok{color:#047857;}
 .x{color:#9ca3af;cursor:pointer;} .links{margin-top:14px;font-size:14px;color:#6b7280;} .links a{margin-right:14px;}
 .overlay{position:fixed;left:50%;top:34%;transform:translate(-50%,-50%);z-index:99999;padding:18px 34px;border-radius:16px;font-size:22px;font-weight:800;color:#fff;box-shadow:0 12px 40px rgba(0,0,0,.35);opacity:0;transition:opacity .25s ease;pointer-events:none;text-align:center;}
 .overlay.show{opacity:1;} .overlay.have{background:#059669;} .overlay.new{background:#2563eb;} .overlay.proc{background:#4f46e5;}
</style></head><body>
 <div id=ovl class=overlay></div>
 <div class=bar>
   <h1>Quick Scan</h1>
   <select id=game></select>
   <button id=startBtn class=sec>Start camera</button>
   <button id=scanBtn>Scan card</button>
   <label class=file>Photo<input type=file id=file accept="image/*" capture="environment" style="display:none"></label>
   <label style="font-weight:500;font-size:13px;"><input type=checkbox id=auto> Auto every 3s</label>
   <button id=csv class=sec>Export CSV</button>
 </div>
 <p class=note>Point the camera at a card and press Scan. Each read is matched against the selected game's catalog and listed below &mdash; <b>nothing is saved to inventory</b>. Export the CSV when done.</p>
 <div class=stage>
   <div class=camwrap><video id=video autoplay playsinline muted></video><div class=frame></div></div>
   <div class=side>
     <div id=status></div>
     <table id=tbl><thead><tr><th>Held</th><th>OCR name</th><th>OCR #</th><th>Matched</th><th>#</th><th>Set</th><th>Rarity</th><th>Price</th><th>Score</th><th></th></tr></thead><tbody></tbody></table>
   </div>
 </div>
 <div class=links><a href="/">Home</a><a href="/settings/reference">Reference Data</a></div>
<script>
 var NL=String.fromCharCode(10);
 var gameSel=document.getElementById('game'),video=document.getElementById('video'),statusEl=document.getElementById('status');
 var rows=[], stream=null, autoTimer=null;
 function setStatus(t,cls){statusEl.textContent=t;statusEl.className=cls||'';}
 var ovl=document.getElementById('ovl'), ovlTimer=null;
 function flashOverlay(inInv,count){
   ovl.textContent = inInv ? ('\u2713 Already in held inventory'+(count>1?(' ('+count+')'):'')) : '\u2717 Not in held inventory';
   ovl.className='overlay show '+(inInv?'have':'new');
   if(ovlTimer)clearTimeout(ovlTimer);
   ovlTimer=setTimeout(function(){ovl.className='overlay '+(inInv?'have':'new');},2000);
 }
 function showProcessing(){ if(ovlTimer){clearTimeout(ovlTimer);ovlTimer=null;} ovl.textContent='\u23f3 Identifying\u2026 finding a match'; ovl.className='overlay show proc'; }
 function hideOverlay(){ if(ovlTimer){clearTimeout(ovlTimer);ovlTimer=null;} ovl.className='overlay'; }
 function overlayNoCard(){ if(ovlTimer)clearTimeout(ovlTimer); ovl.textContent='No card detected'; ovl.className='overlay show proc'; ovlTimer=setTimeout(function(){ovl.className='overlay';},1500); }
 async function loadGames(){
   try{
     var d=await (await fetch('/reference/status',{headers:{'X-Requested-With':'XMLHttpRequest'}})).json();
     var games=(d.games||[]);
     if(!games.length){gameSel.innerHTML='<option value="">No catalogs — download one in Reference Data</option>';return;}
     gameSel.innerHTML=''; games.forEach(function(g){var o=document.createElement('option');o.value=g.game;o.textContent=g.game;gameSel.appendChild(o);});
   }catch(e){gameSel.innerHTML='<option value="">Could not load games</option>';}
 }
 async function startCamera(){
   try{
     stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});
     video.srcObject=stream; setStatus('Camera on. Point at a card and press Scan.','ok');
   }catch(e){ setStatus('Live camera unavailable (needs HTTPS or localhost). Use the Photo button instead.','miss'); }
 }
 function grabBlob(cb){
   if(!video.videoWidth){setStatus('Camera not started. Use Start camera or the Photo button.','miss');return;}
   var c=document.createElement('canvas');c.width=video.videoWidth;c.height=video.videoHeight;
   c.getContext('2d').drawImage(video,0,0);c.toBlob(function(b){cb(b);},'image/jpeg',0.92);
 }
 async function scanBlob(blob){
   var game=gameSel.value;
   if(!game){setStatus('Choose a game first.','miss');return;}
   setStatus('Scanning...','');
   showProcessing();
   var fd=new FormData();fd.append('image',blob,'frame.jpg');fd.append('game',game);
   try{
     var d=await (await fetch('/quickscan/identify',{method:'POST',body:fd})).json();
     if(d.status!=='ok'){hideOverlay();setStatus(d.message||'Scan failed.','miss');return;}
     if(!d.detected && !d.ocr_name && !d.ocr_number){overlayNoCard();setStatus('No card detected — reposition and try again.','miss');return;}
     addRow(d);
     flashOverlay(d.in_inventory, d.inventory_count||0);
     setStatus(d.matched?('Matched: '+d.name+(d.number?(' #'+d.number):'')):'Read but no catalog match.', d.matched?'ok':'miss');
   }catch(e){hideOverlay();setStatus('Error: '+e.message,'miss');}
 }
 function addRow(d){
   rows.push(d);
   var tb=document.querySelector('#tbl tbody'),tr=document.createElement('tr');
   function td(v,cls){var c=document.createElement('td');c.textContent=(v==null?'':v);if(cls)c.className=cls;return c;}
   var ind=document.createElement('td');var mark=document.createElement('span');
   mark.style.fontWeight='800';mark.style.fontSize='16px';
   if(d.in_inventory){mark.textContent='\u2713';mark.style.color='#059669';mark.title='In held inventory'+(d.inventory_count>1?(' ('+d.inventory_count+')'):'');}
   else{mark.textContent='\u2717';mark.style.color='#2563eb';mark.title='Not in held inventory';}
   ind.appendChild(mark);tr.appendChild(ind);
   tr.appendChild(td(d.ocr_name));tr.appendChild(td(d.ocr_number));
   tr.appendChild(td(d.matched?d.name:'no match',d.matched?'ok':'miss'));
   tr.appendChild(td(d.number));tr.appendChild(td(d.set));tr.appendChild(td(d.rarity));
   tr.appendChild(td(d.market_price==null?'':('$'+d.market_price)));tr.appendChild(td(d.score==null?'':(d.score+'%')));
   var x=td('remove','x');x.onclick=function(){var i=rows.indexOf(d);if(i>=0)rows.splice(i,1);tr.remove();};tr.appendChild(x);
   tb.appendChild(tr);
 }
 function exportCsv(){
   if(!rows.length){setStatus('Nothing to export yet.','miss');return;}
   var head=['In Held Inventory','OCR Name','OCR Number','Matched Name','Number','Set','Rarity','Market Price','Score','Game'];
   var lines=[head];
   rows.forEach(function(r){lines.push([(r.in_inventory?'\u2713 Yes':'\u2717 No'),r.ocr_name,r.ocr_number,(r.matched?r.name:''),r.number,r.set,r.rarity,r.market_price,r.score,r.game]);});
   var csv=lines.map(function(row){return row.map(function(cell){var s=(cell==null?'':String(cell));if(s&&'=+-@'.indexOf(s.charAt(0))>=0)s="'"+s;return '"'+s.replace(/"/g,'""')+'"';}).join(',');}).join(NL);
   var blob=new Blob([String.fromCharCode(0xFEFF)+csv],{type:'text/csv;charset=utf-8'});var a=document.createElement('a');a.href=URL.createObjectURL(blob);
   a.download='quick_scan_'+Date.now()+'.csv';a.click();URL.revokeObjectURL(a.href);
 }
 document.getElementById('startBtn').addEventListener('click',startCamera);
 document.getElementById('scanBtn').addEventListener('click',function(){grabBlob(function(b){if(b)scanBlob(b);});});
 document.getElementById('file').addEventListener('change',function(e){var f=e.target.files[0];if(f)scanBlob(f);e.target.value='';});
 document.getElementById('csv').addEventListener('click',exportCsv);
 document.getElementById('auto').addEventListener('change',function(){
   if(this.checked){autoTimer=setInterval(function(){grabBlob(function(b){if(b)scanBlob(b);});},3000);}
   else if(autoTimer){clearInterval(autoTimer);autoTimer=null;}
 });
 loadGames();
</script></body></html>"""


@app.route("/quickscan")
def quick_scan_page():
    return Response(_QUICK_SCAN_HTML, mimetype="text/html")


@app.route("/quickscan/identify", methods=["POST"])
def quick_scan_identify():
    """Detect + OCR + catalog-match one camera frame and return the matched data.
    Does NOT create any inventory record — purely a lookup for CSV export."""
    if card_ocr is None:
        return jsonify({"status": "error",
                        "message": "OCR isn't installed. Add it with: "
                                   "pip install rapidocr onnxruntime"}), 503
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No image provided."}), 400
    bad = _reject_if_bomb(file)
    if bad:
        return bad
    game = (request.form.get("game") or "").strip()
    if not game:
        return jsonify({"status": "error", "message": "Choose a game first."}), 400

    ensure_dirs()
    tmp = os.path.join(app.config["TEMP_CARD_FOLDER"],
                       "quickscan_" + datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + ".png")
    try:
        file.save(tmp)
        bgr = _imread(tmp)
        if bgr is None:
            return jsonify({"status": "error", "message": "Could not read the image."}), 400
        cropped, ok = detect_and_crop_card(bgr, CARD_EDGE_DEFAULT)
        if ok:
            cv2.imwrite(tmp, cropped)
        try:
            ocr = card_ocr.ocr_card_front(tmp, game=game, type_refs=_load_prepared_type_refs(game))
        except Exception:
            ocr = {"ocr_available": False}
        if not ocr.get("ocr_available"):
            return jsonify({"status": "error", "message": "OCR is unavailable on the server."}), 503

        name_guess = (ocr.get("name_guess") or "").strip()
        number_guess = (ocr.get("number_guess") or "").strip()
        category_id, _ = _resolve_category_for_game(game)
        cands = _reference_candidates_for_ocr(category_id, ocr) if category_id else []
        top = cands[0] if cands else None

        # Non-destructive "do I already own this?" check against HELD inventory.
        match_name = (top.get("name") if top else "") or name_guess
        match_number = (top.get("serial") if top else "") or number_guess
        held_count = _held_inventory_match(game, match_name, match_number)

        return jsonify({
            "status": "ok",
            "detected": bool(ok),
            "ocr_name": name_guess,
            "ocr_number": number_guess,
            "matched": bool(top),
            "score": (round(float(top.get("score")), 1) if top and top.get("score") is not None else None),
            "name": (top.get("name") if top else ""),
            "number": (top.get("serial") if top else ""),
            "set": (top.get("set") if top else ""),
            "rarity": (top.get("rarity") if top else ""),
            "market_price": (top.get("market_price") if top else None),
            "thumbnail": (top.get("thumbnail") if top else ""),
            "url": (top.get("url") if top else ""),
            "game": game,
            "in_inventory": held_count > 0,
            "inventory_count": held_count,
        })
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


@app.route("/collections/prices")
def collections_prices():
    """List collections (from scanned cards + any saved prices) with their card
    count and 'Bought For' lot price, for the Inventory pricing modal."""
    records = _analytics_filtered_records("all", "", "", "")
    counts = _collection_counts(records)
    saved = {cp.name_key: cp for cp in CollectionPrice.query.all()}

    names = {}   # name_key -> display name (prefer the casing seen on cards)
    for r in records:
        c = str((r.extracted_data or {}).get("collection") or "").strip()
        if c:
            names.setdefault(c.lower(), c)
    for k, cp in saved.items():
        names.setdefault(k, cp.name)

    out = [{
        "name": disp,
        "count": counts.get(k, 0),
        "bought_for": (saved[k].bought_for if k in saved else None),
    } for k, disp in names.items()]
    out.sort(key=lambda x: x["name"].lower())
    return jsonify({"status": "success", "collections": out})


@app.route("/collections/price", methods=["POST"])
def collections_price_save():
    """Set (or clear, with a blank value) a collection's 'Bought For' lot price."""
    body = request.get_json(silent=True) or request.form.to_dict() or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"status": "error", "message": "Collection name is required."}), 400

    raw = body.get("bought_for")
    bought_for = None
    if raw not in (None, "", "null"):
        bought_for = _analytics_money(raw)
        if bought_for is None:
            return jsonify({"status": "error", "message": "Enter a valid number, or leave blank to clear."}), 400

    key = name.lower()
    cp = CollectionPrice.query.filter_by(name_key=key).first()
    if cp is None:
        cp = CollectionPrice(name_key=key, name=name)
        db.session.add(cp)
    cp.name = name
    cp.bought_for = bought_for
    db.session.commit()

    msg = "Cleared lot price." if bought_for is None else f"Saved: {name} bought for ${bought_for:,.2f}."
    return jsonify({"status": "success", "name": name, "bought_for": bought_for, "message": msg})


@app.route("/inventory")
@app.route("/sold", endpoint="sold")
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

    # Sold view: the /sold page (or ?sold=1) lists entries whose Held flag is
    # False — the ones that have been sold. Catalog view is never "sold".
    view_sold = (request.endpoint == "sold"
                 or request.args.get("sold", "").strip() in ("1", "true", "yes"))
    # held_state: None in catalog view (don't filter); False for Sold; True otherwise.
    held_state = None if view_catalog else (False if view_sold else True)

    # If no filter is active, show the game selection landing page — but the Sold
    # page shows its full list directly (no per-game landing).
    if (not view_sold and not f_game and not f_album and not f_template
            and not search and not view_catalog):
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
                f_game, f_album, f_template, view_catalog, page, per_page, sort_col, sort_dir, held_state)
        except Exception:
            db.session.rollback()  # fall back to the proven Python grouping path

    # Base query. game/album are filtered on the denormalized key columns with
    # the same strip/lower normalization the fast path uses (app.py:_inventory_
    # base_conditions), so both paths return the same set for the same URL —
    # the raw json_extract comparison this replaced was case-sensitive, which
    # made a search or column sort silently drop mixed-case records. The ORM
    # SELECT lists every mapped column anyway, so this path never worked
    # without the denormalized columns either. template_used and the JSON text
    # search are handled in SQL as before.
    query = ScanRecord.query.order_by(ScanRecord.scan_date.desc())

    if f_template:
        query = query.filter(ScanRecord.template_used == f_template)
    if search:
        query = query.filter(_scan_search_condition(search))
    if f_game:
        query = query.filter(ScanRecord.game_key == f_game.strip().lower())
    if f_album:
        query = query.filter(ScanRecord.album_key == f_album.strip().lower())

    # Catalog / archived / held scoping in SQL on the derived columns — the
    # same coalesce() conditions the fast path uses, and equal by construction
    # to the _is_catalog_only/_bool_from/_held_from checks that used to run
    # here in Python: the mapper events derive those columns from exactly
    # those helpers. Only the rows the page can actually show are loaded.
    from sqlalchemy import func as _f
    query = query.filter(_f.coalesce(ScanRecord.is_catalog, False) == bool(view_catalog))
    if not view_catalog:
        # Archived rows (cold storage) are hidden from the normal Inventory list.
        query = query.filter(_f.coalesce(ScanRecord.is_archived, False) == False)  # noqa: E712
        # Held vs Sold split: Inventory shows held entries; the Sold page shows
        # the rest. (Catalog view leaves held_state None and skips this.)
        if held_state is not None:
            query = query.filter(_f.coalesce(ScanRecord.is_held, True) == bool(held_state))

    all_records = query.all()

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
        sold_view=view_sold,
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

    # Build query — same logic as /inventory, but no pagination. All filters
    # run in SQL on the derived key columns (same strip/lower normalization as
    # the page), so only the exported rows are ever loaded.
    from sqlalchemy import func as _f
    query = (ScanRecord.query.order_by(ScanRecord.scan_date.desc())
             .filter(_f.coalesce(ScanRecord.is_catalog, False) == bool(view_catalog)))
    if f_template:
        query = query.filter(ScanRecord.template_used == f_template)
    if search:
        query = query.filter(_scan_search_condition(search))
    if f_game:
        query = query.filter(ScanRecord.game_key == f_game.strip().lower())
    if f_album:
        query = query.filter(ScanRecord.album_key == f_album.strip().lower())

    records = query.all()

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

    # Header row. The labels are derived from the caller's `columns` parameter
    # (entry_<field> -> "<Field>"), and .title() does not disarm a leading "=".
    writer.writerow([_csv_safe(label) for label, _ in columns])

    # Data rows
    for record in records:
        writer.writerow([_csv_safe(extractor(record)) for _, extractor in columns])

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

    # Aggregate in SQL: DISTINCT over the trimmed raw JSON values keeps the
    # display casing the dropdowns show, while the catalog scope filters on the
    # indexed is_catalog mirror of _is_catalog_only — no row hydration, no
    # per-row JSON parsing in Python.
    catalog_cond = db.func.coalesce(ScanRecord.is_catalog, False) == bool(view_catalog)

    def _distinct_json(path):
        expr = db.func.trim(db.func.json_extract(ScanRecord.extracted_data, path))
        q = (db.session.query(expr)
             .filter(catalog_cond, expr.isnot(None), expr != "")
             .distinct())
        return {str(v) for (v,) in q}

    games = _distinct_json("$.game")
    albums = _distinct_json("$.album")
    templates = {
        str(t).strip()
        for (t,) in (db.session.query(ScanRecord.template_used)
                     .filter(catalog_cond,
                             ScanRecord.template_used.isnot(None),
                             ScanRecord.template_used != "")
                     .distinct())
    }

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

    # Every filter runs in SQL on the derived columns (same strip/lower
    # normalization as the page), so this selects bare ids — no JSON is
    # transferred or parsed at all.
    from sqlalchemy import func as _f
    query = (ScanRecord.query.with_entities(ScanRecord.id)
             .filter(_f.coalesce(ScanRecord.is_catalog, False) == bool(view_catalog)))
    if f_template:
        query = query.filter(ScanRecord.template_used == f_template)
    if search:
        query = query.filter(_scan_search_condition(search))
    if f_game:
        query = query.filter(ScanRecord.game_key == f_game.strip().lower())
    if f_album:
        query = query.filter(ScanRecord.album_key == f_album.strip().lower())

    ids = [row_id for (row_id,) in query.all()]
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

    # The indexed game_key column carries the same strip/lower normalization
    # this comparison applied in Python — only the game's own rows load.
    records = (ScanRecord.query
               .filter(ScanRecord.game_key == game.strip().lower())
               .all())

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

    # Neighbors scoped to the same game via the indexed game_key column
    # (normalized like every other game filter), ordered like the inventory
    # list: scan_date DESC, id DESC, wrapping at both ends. Row-value
    # comparisons walk the (scan_date, id) ordering directly instead of
    # loading the whole table to find one position in it.
    from sqlalchemy import tuple_ as _sa_tuple

    _gkey = str(current_game or "").strip().lower() or None
    game_cond = (ScanRecord.game_key.is_(None) if _gkey is None
                 else ScanRecord.game_key == _gkey)
    _pos = (record.scan_date, record.id)
    base = ScanRecord.query.filter(game_cond)
    # Previous in DESC order = the smallest row strictly above this one.
    prev_record = (base.filter(_sa_tuple(ScanRecord.scan_date, ScanRecord.id) > _pos)
                   .order_by(ScanRecord.scan_date.asc(), ScanRecord.id.asc())
                   .first())
    if prev_record is None:  # wrap to the last entry of the DESC list
        prev_record = base.order_by(ScanRecord.scan_date.asc(), ScanRecord.id.asc()).first()
    # Next in DESC order = the largest row strictly below this one.
    next_record = (base.filter(_sa_tuple(ScanRecord.scan_date, ScanRecord.id) < _pos)
                   .order_by(ScanRecord.scan_date.desc(), ScanRecord.id.desc())
                   .first())
    if next_record is None:  # wrap to the first entry of the DESC list
        next_record = base.order_by(ScanRecord.scan_date.desc(), ScanRecord.id.desc()).first()
    if prev_record is not None and prev_record.id == record.id:
        # Wrapped straight back to this record: it is its game's only entry.
        prev_record = None
        next_record = None

    # ── Find all duplicate records that belong to the same stack ──
    # A stack is defined by matching: name, serial, edition, holographic, finalized==True.
    # Only groups where finalized is True are stacked (same logic as build_group_info).
    data = record.extracted_data or {}
    finalized = data.get("finalized", False)
    is_final = finalized is True or str(finalized).strip().lower() == "true"

    stack_locations = []  # list of dicts: {record_id, album, page, slot}
    if is_final and record.dup_hash:
        # dup_hash IS this stack's identity — sha1(name|serial|edition|holo),
        # computed only for finalized records and kept current by the mapper
        # events. An indexed lookup on it returns exactly the rows the old
        # full-table scan matched field-by-field.
        members = (
            ScanRecord.query
            .filter(ScanRecord.dup_hash == record.dup_hash,
                    ScanRecord.id != record.id)
            .order_by(ScanRecord.scan_date.asc(), ScanRecord.id.asc())
            .all()
        )
        for r in members:
            rdata = r.extracted_data or {}
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


@app.route("/storage")
def storage_home():
    containers = build_storage_index()
    if containers:
        return redirect(url_for("storage_detail", name=containers[0]["name"]))
    return render_template("storage.html", containers=[])


@app.route("/storage/list")
def storage_list():
    containers = build_storage_index()
    for c in containers:
        c["image_url"] = find_saved_image("albums", c["name"])
    return render_template("storage.html", containers=containers)


@app.route("/storage/next_index")
def storage_next_index():
    """Next 'sheet' index for a Box (or Album) container: max existing page + 1.

    Boxes hide the page field, but the 9-pocket importer still keys each physical
    pocket by (game, container, page, slot) so a card's front and back merge. The
    client fetches this so successive box imports don't collide on page 1.
    """
    name = (request.args.get("name") or "").strip().lower()
    nxt = 1
    if name:
        rows = (ScanRecord.query
                .filter(ScanRecord.album_key == name)
                .with_entities(ScanRecord.extracted_data)
                .all())
        max_page = 0
        for (data,) in rows:
            try:
                p = int((data or {}).get("page") or 0)
            except (TypeError, ValueError):
                p = 0
            if p > max_page:
                max_page = p
        nxt = max_page + 1
    return jsonify({"next": nxt})


@app.route("/storage/upload_image", methods=["POST"])
def storage_upload_image():
    # Accept the current 'storage_name' field and the legacy 'album_name'.
    name = (request.form.get("storage_name") or request.form.get("album_name") or "").strip()
    file = request.files.get("image")

    if not name or not file or not file.filename:
        return jsonify({"status": "error", "message": "Storage name and image file are required"}), 400
    bad = _reject_if_bomb(file)
    if bad:
        return bad

    # Cover images continue to live under uploads/albums/ for continuity with
    # any images uploaded before the Storage rename.
    img_folder = os.path.join(app.config["UPLOAD_FOLDER"], "albums")
    os.makedirs(img_folder, exist_ok=True)

    ext = _validated_image_ext(file)
    if ext is None:
        return jsonify({"status": "error",
                        "message": "Only image files (PNG, JPG, GIF, WebP, BMP) are allowed."}), 415

    safe_name = secure_filename(name)
    filename = f"{safe_name}{ext}"
    save_path = os.path.join(img_folder, filename)
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
    bad = _reject_if_bomb(file)
    if bad:
        return bad

    game_img_folder = os.path.join(app.config["UPLOAD_FOLDER"], "game_icons")
    os.makedirs(game_img_folder, exist_ok=True)

    ext = _validated_image_ext(file)
    if ext is None:
        return jsonify({"status": "error",
                        "message": "Only image files (PNG, JPG, GIF, WebP, BMP) are allowed."}), 415

    safe_game = secure_filename(game_name)
    filename = f"{safe_game}{ext}"
    save_path = os.path.join(game_img_folder, filename)
    file.save(save_path)

    relative_path = f"game_icons/{filename}"
    image_url = url_for("uploaded_file", filename=relative_path)
    return jsonify({"status": "success", "url": image_url})


def _storage_item_name(record, fallback=""):
    return (get_record_value(record, "product_name")
            or get_record_value(record, "name")
            or fallback)


@app.route("/storage/<path:name>")
def storage_detail(name):
    containers = build_storage_index()
    selected = next(
        (c for c in containers if c["name"].lower() == name.strip().lower()),
        None,
    )

    if not selected:
        return redirect(url_for("storage_list"))

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

    # ── Box: one flat, continuously numbered run, no pages ──
    if selected["is_box"]:
        items = []
        for i, record in enumerate(records, start=1):
            items.append({
                "number": i,
                "record": record,
                "name": _storage_item_name(record, f"Card {i}"),
            })
        return render_template(
            "box_detail.html",
            storage_name=selected["name"],
            items=items,
            total=len(items),
        )

    # ── Album: 3×3 pages keyed by slot (unchanged behavior) ──
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
                "name": _storage_item_name(record, f"Slot {slot_num}"),
            }

    return render_template(
        "album_detail.html",
        album_name=selected["name"],
        currentpage=current_page,
        maxpage=len(page_groups),
        grid=grid,
    )


# ── Legacy endpoint aliases ──────────────────────────────────────────────────
# The Album section became Storage, but not every template references the new
# endpoints yet (several partials/pages aren't touched by this change). Keep the
# old endpoint *names* alive — pointing at the same views, preserving the old
# `album_name` URL argument — so any lingering url_for('albums_list') /
# url_for('album_detail', album_name=...) keeps building instead of raising
# BuildError. Safe to delete once every template uses the storage_* endpoints.
def _legacy_album_detail(album_name):
    return storage_detail(album_name)

app.add_url_rule("/albums",                   endpoint="albums",             view_func=storage_home)
app.add_url_rule("/albums/list",              endpoint="albums_list",        view_func=storage_list)
app.add_url_rule("/albums/upload_image",      endpoint="album_upload_image", view_func=storage_upload_image, methods=["POST"])
app.add_url_rule("/albums/<path:album_name>", endpoint="album_detail",       view_func=_legacy_album_detail)


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
try:
    MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "5000") or "5000")  # bound processing time
except ValueError:
    MAX_PDF_PAGES = 5000


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


def _next_box_number(name):
    """Next sequential card number for a Box container (max existing + 1).

    Boxes number their cards 1..N in scan order with no page structure. Because
    create_scan_record commits each record before the next, successive single /
    PDF-pair imports into the same box see prior numbers and keep incrementing.
    """
    key = str(name or "").strip().lower()
    if not key:
        return 1
    rows = (ScanRecord.query
            .filter(ScanRecord.album_key == key)
            .with_entities(ScanRecord.extracted_data)
            .all())
    max_n = 0
    for (data,) in rows:
        try:
            n = int((data or {}).get("box_number") or 0)
        except (TypeError, ValueError):
            n = 0
        if n > max_n:
            max_n = n
    return max_n + 1


def _create_single_card(front_path, back_path, game, album, template,
                        collection="", storage_type="album"):
    """
    Create a blank inventory record for `game` (+optional album/collection) with
    the given front (required) and back (optional) image paths, then OCR-identify
    the FRONT ONLY — the back is never name/serial-checked. A match at or above the
    auto-identify threshold (the Settings slider, 60% by default) is applied and
    saved; anything less leaves the entry blank for manual entry.

    `storage_type` is 'album' (pages + slots) or 'box' (a flat 1..N run); for a
    box the card is stamped with the next sequential box_number.

    Returns (record, ident_dict).
    """
    blank_fields = {k: "" for k in (template.get("fields", {}) or {}).keys()}
    extracted = {**blank_fields, "game": _normalize_game_name(game)}
    if album:
        extracted["album"] = album
        if str(storage_type).strip().lower() == "box":
            extracted["storage_type"] = "box"
            extracted["box_number"] = _next_box_number(album)
        else:
            extracted["storage_type"] = "album"
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


def _finalize_pdf_pair(front_path, back_path, front_page, back_page, game, album, template,
                       collection="", storage_type="album"):
    """Create one inventory record from an already-saved front (+ optional back)
    image pair pulled from a PDF, and return the card dict the UI expects."""
    record, ident = _create_single_card(front_path, back_path, game, album, template,
                                         collection, storage_type)
    return {
        "record_id":       record.id,
        "front_page":      front_page,
        "back_page":       back_page,
        "identified":      bool(ident.get("identified")),
        "identified_name": (ident.get("applied") or {}).get("name", "") if ident.get("identified") else "",
        "ident_error":     ident.get("error", ""),
        # Why the card was (or wasn't) filled in, and how close it came — without
        # these the UI can't tell "no match" from "tied" from "just below the bar".
        "ident_reason":    ident.get("reason", ""),
        "ident_score":     ident.get("score"),
        "ident_min_score": ident.get("min_score"),
        "ident_tied":      ident.get("tied", []),
        "card_type":       (ident.get("type_applied") or {}).get("value", ""),
        "image_url":       build_uploaded_file_url(record.image_path),
        "detail_url":      url_for("inventory_detail", record_id=record.id),
    }


def _import_single_card_pdf(pdf_bytes, game, album, edge_type, template, collection="", storage_type="album"):
    """
    Import a PDF (given as raw bytes) as front/back pairs. Suitable for smaller
    uploads; inputs over 500 MB are spilled to a temp file automatically (see
    _iter_pdf_bgr_pages). Prefer _import_single_card_pdf_path when the upload has
    already been streamed to disk, to avoid ever holding the file in RAM.
    """
    return _import_pdf_pages(_iter_pdf_bgr_pages(pdf_bytes), game, album, edge_type, template, collection, storage_type)


def _import_single_card_pdf_path(pdf_path, game, album, edge_type, template, collection="", storage_type="album"):
    """
    Import an already-on-disk PDF as front/back pairs, rasterizing one page at a
    time straight from the file. The document is never loaded into RAM, so this
    keeps the process's memory well under the 500 MB cap no matter how large the
    PDF is. The caller owns and cleans up `pdf_path`.
    """
    return _import_pdf_pages(_iter_pdf_bgr_pages_from_path(pdf_path), game, album, edge_type, template, collection, storage_type)


def _import_pdf_pages(page_iter, game, album, edge_type, template, collection="", storage_type="album"):
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
                page_no, game, album, template, collection, storage_type))
            pending_front = None

        # A trailing odd page: a front with no back.
        if pending_front is not None:
            cards.append(_finalize_pdf_pair(
                pending_front["path"], None, pending_front["page_no"],
                None, game, album, template, collection, storage_type))
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
    storage_type = (request.form.get("storage_type") or "album").strip().lower()
    collection = _clean_collection(request.form.get("collection"))
    edge_type = normalize_card_edge_type(request.form.get("card_edge_type"))
    if not game:
        return jsonify({"status": "error", "message": "Game is required"}), 400

    front = request.files.get("front_image")
    if not front or not front.filename:
        return jsonify({"status": "error", "message": "A front image or PDF is required"}), 400
    back = request.files.get("back_image")
    bad = _reject_if_bomb(front, back)
    if bad:
        return bad

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
            return _import_single_card_pdf_path(front_tmp, game, album, edge_type, template, collection, storage_type)

        # ---- Single image: front (required) + optional back ----
        # Card photos are small, so decoding one from disk is well within budget.
        aligned_flags = {}
        try:
            front_img = _imread(front_tmp, cv2.IMREAD_COLOR)
            if front_img is None:
                return jsonify({"status": "error", "message": "Could not read the front image"}), 400
            front_path, front_cropped = _save_single_card_image(front_img, "front", edge_type)
            aligned_flags["front"] = front_cropped
            del front_img

            back_path = None
            if back and back.filename:
                back_tmp = os.path.join(upload_dir, "back_upload")
                back.save(back_tmp)
                back_img = _imread(back_tmp, cv2.IMREAD_COLOR)
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

    record, ident = _create_single_card(front_path, back_path, game, album, template, collection, storage_type)

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
        via = " via Ximilar" if ident.get("source") == "ximilar" else ""
        message += (f" Identified as \u201c{ident_name}\u201d{via} and filled in automatically."
                    if ident_name else f" Identified{via} and filled in automatically.")
    else:
        message += " Left blank for manual entry (no confident match)."

    if ident.get("error"):
        message += f" \u26a0 {ident['error']}"

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
        "ident_error":     ident.get("error", ""),
        "card_type":       card_type,
        "image_url":       build_uploaded_file_url(record.image_path),
        "image_url_back":  build_uploaded_file_url(record.image_path_back) if record.image_path_back else None,
        "detail_url":      url_for("inventory_detail", record_id=record.id),
    })



# ====================== FILE ROUTES ======================
def _no_sniff(resp):
    """Stop browsers MIME-sniffing user-supplied files into something executable
    (e.g. treating an uploaded file as HTML) — defense in depth alongside the
    image-extension allowlist on the upload routes."""
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    # /uploads is intentionally open to any signed-in user because every card
    # image is served through it. Migration bundles, however, live under
    # UPLOAD_FOLDER/migration_exports and contain secrets (FLASK_SECRET_KEY, shop
    # tokens, mailbox password). Bundles are now served by the dedicated,
    # map-gated upgrade_download route; this block stays as defense in depth so
    # that bundles reachable via the old /uploads/... URL shape are still gated.
    # CRITICAL: normalize BEFORE the prefix test — send_from_directory normalizes
    # afterwards, so a raw check on "cards/../migration_exports/x" (which the file
    # server resolves back inside the root) would miss it. normpath collapses the
    # traversal first; .lower() closes the case-insensitive-FS variant.
    norm = posixpath.normpath(str(filename).replace("\\", "/").lstrip("/"))
    if norm.lower().startswith("migration_exports/") or norm == ".." or norm.startswith("../"):
        # Admin, NOT _role_allows(..., "upgrade", "view"). Two reasons: the bundle is a
        # plaintext credential dump, so it should never have hung off an ordinary
        # grantable resource; and dropping "upgrade" from PROTECTED_RESOURCES would not
        # have revoked it, because _role_allows reads the role's own stored permissions
        # dict — every role created before this change still carries "upgrade": "edit".
        denied = _require_admin(_BUNDLE_ADMIN_MSG)
        if denied:
            return denied
        # Prune stale bundles on download too (not just on export), excluding the
        # one being served, so a single old bundle can't linger indefinitely.
        _prune_migration_bundles(os.path.join(app.config["UPLOAD_FOLDER"], "migration_exports"),
                                 keep=os.path.basename(norm))
    return _no_sniff(send_from_directory(app.config["UPLOAD_FOLDER"], filename))


@app.route("/temp_split/<path:filename>")
def temp_split_file(filename):
    return _no_sniff(send_from_directory(app.config["TEMP_SPLIT_FOLDER"], filename))


@app.route("/temp_cards/<path:filename>")
def temp_card_file(filename):
    return _no_sniff(send_from_directory(app.config["TEMP_CARD_FOLDER"], filename))


@app.route("/temp_pdf/<path:filename>")
def temp_pdf_file(filename):
    return _no_sniff(send_from_directory(app.config["TEMP_PDF_FOLDER"], filename))


# ====================== INVENTORY UPDATE ROUTES ======================
@app.route("/update_scan/<int:record_id>", methods=["POST"])
def update_scan(record_id):
    record = ScanRecord.query.get_or_404(record_id)
    data = request.get_json() or {}
    new_data = data.get("extracted_data", {})
    # Strip legacy/internal keys that should never be stored as editable fields.
    # tcgplayer_link is here because it is not a field: it is a URL that gets rendered
    # into an href, and /save_tcgplayer_link already refuses anything that is not
    # http(s). This route merged the whole extracted_data dict, so it was a second,
    # unvalidated door onto the same key -- the scheme check was on the wrong one.
    for hidden in ("card_lookup", "__roi_fields_used", "tcgplayer_link"):
        new_data.pop(hidden, None)
    record.extracted_data = {**(record.extracted_data or {}), **new_data}
    db.session.commit()
    return jsonify({"status": "success", "message": "Entry updated"})


@app.route("/update_scan_image/<int:record_id>", methods=["POST"])
def update_scan_image(record_id):
    record = ScanRecord.query.get_or_404(record_id)
    file = request.files.get("image")
    bad = _reject_if_bomb(file)
    if bad:
        return bad
    side = request.form.get("side", "front").strip().lower()
    if side not in ("front", "back"):
        side = "front"

    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    ensure_dirs()

    ext = _validated_image_ext(file, default=".png")
    if ext is None:
        return jsonify({"status": "error",
                        "message": "Only image files (PNG, JPG, GIF, WebP, BMP) are allowed."}), 415

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


@app.route("/realign_record_image/<int:record_id>", methods=["POST"])
def realign_record_image(record_id):
    """
    Manually re-align one side of a saved record's image using four corner
    points, exactly like the import page's "Manual Corner Selection" — but
    operating on the record's own stored front/back photo instead of a temp
    split tile.

    Body (JSON): { "side": "front"|"back",
                   "points": [ {"x":.., "y":..} x4 ] }

    The points are in the pixel coordinates of the currently-stored image (the
    front-end canvas already converts click positions back to full-res image
    coordinates). We perspective-warp + sharpen that image, save the result as a
    new inventory_cards file, point the record at it, and delete the old file —
    mirroring update_scan_image so paths/cleanup stay consistent.
    """
    record = ScanRecord.query.get_or_404(record_id)

    data   = request.get_json(silent=True) or {}
    side   = str(data.get("side", "front")).strip().lower()
    points = data.get("points", [])
    if side not in ("front", "back"):
        side = "front"

    if not isinstance(points, list) or len(points) != 4:
        return jsonify({"status": "error", "message": "Exactly 4 points are required"}), 400

    current_path = record.image_path_back if side == "back" else record.image_path
    if not current_path or current_path == "__blank__":
        return jsonify({
            "status":  "error",
            "message": f"This record has no {side} image to re-align.",
        }), 400

    relative_current = normalize_to_upload_relative(current_path)
    if relative_current.startswith("http://") or relative_current.startswith("https://"):
        return jsonify({
            "status":  "error",
            "message": "This image is stored as an external URL and can't be re-aligned.",
        }), 400

    abs_path = _abs_record_image_path(current_path)
    if not abs_path or not os.path.exists(abs_path):
        return jsonify({"status": "error", "message": "The image file could not be found on disk."}), 404

    image = _imread(abs_path)
    if image is None:
        return jsonify({"status": "error", "message": "Could not read the image for re-alignment."}), 400

    try:
        pts = np.array([[float(p["x"]), float(p["y"])] for p in points], dtype="float32")
    except (KeyError, TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid point format"}), 400

    ensure_dirs()
    try:
        # Straighten/crop to the four chosen corners ONLY. We deliberately do NOT
        # run sharpen_image() here (unlike the import pipeline): this image is
        # already a finished inventory photo, and the sharpening kernel visibly
        # boosts edge contrast and shifts colour/tone — re-aligning would sharpen
        # a second time and change how the card looks. The perspective warp keeps
        # the original colours and pixel detail; only the geometry changes.
        warped = four_point_transform(image, pts)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Re-alignment failed: {e}"}), 500

    suffix        = "back" if side == "back" else "front"
    final_name    = f"record_{record_id}_{suffix}_realigned_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    relative_path = normalize_to_upload_relative(os.path.join("inventory_cards", final_name))
    absolute_path = os.path.join(app.config["UPLOAD_FOLDER"], relative_path)

    if not cv2.imwrite(absolute_path, warped):
        return jsonify({"status": "error", "message": "Could not save the re-aligned image."}), 500

    if side == "back":
        record.image_path_back = relative_path
    else:
        record.image_path = relative_path
    db.session.commit()

    # Remove the previous file now that the record points at the new one.
    if relative_current and relative_current != "__blank__" and relative_current != relative_path:
        remove_file_if_exists(current_path)

    return jsonify({
        "status":    "success",
        "message":   f"{suffix.capitalize()} image re-aligned.",
        "side":      side,
        "image_url": build_uploaded_file_url(record.image_path_back if side == "back" else record.image_path),
    })


@app.route("/save_tcgplayer_link/<int:record_id>", methods=["POST"])
def save_tcgplayer_link(record_id):
    """
    Save (or clear) a manual TCGplayer URL shown in the record's "TCGPlayer Link"
    panel. Stored on extracted_data['tcgplayer_link']; an empty url removes it.
    """
    record = ScanRecord.query.get_or_404(record_id)
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()

    if url and not (url.lower().startswith("http://") or url.lower().startswith("https://")):
        return jsonify({"status": "error",
                        "message": "URL must start with http:// or https://"}), 400

    ext = dict(record.extracted_data or {})
    if url:
        ext["tcgplayer_link"] = url
    else:
        ext.pop("tcgplayer_link", None)
    # Reassign so SQLAlchemy marks the JSON column dirty (before_update resyncs the
    # denormalized hot columns from it).
    record.extracted_data = ext
    db.session.commit()

    return jsonify({
        "status":  "success",
        "url":     url,
        "message": "TCGplayer link saved." if url else "TCGplayer link removed.",
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
        # Same normalized comparisons the Python loop here made, as indexed
        # SQL: game via game_key, the optional catalog scope via is_catalog.
        from sqlalchemy import func as _f
        q = (ScanRecord.query.with_entities(ScanRecord.id)
             .filter(ScanRecord.game_key == game.strip().lower()))
        if scope_passed:
            q = q.filter(_f.coalesce(ScanRecord.is_catalog, False) == bool(want_catalog))
        matching_ids = [row_id for (row_id,) in q.all()]

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
            "message": "OCR is unavailable: install it with "
                       "`pip install rapidocr onnxruntime` on the host.",
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
            "message": "The OCR engine could not be initialised on the host "
                       "(RapidOCR missing, or its PP-OCRv5 mobile models could not "
                       "be downloaded on first use). Install with "
                       "`pip install rapidocr onnxruntime` and ensure network access "
                       "for the one-time model download, then retry.",
        }), 503

    catalog_only = request.args.get("catalog", "").strip().lower() in ("1", "true", "yes")

    # Also match against the downloaded tcgcsv catalog for this record's game,
    # if that game has been synced. Reference matches carry rich fields the UI
    # can auto-fill (set, rarity, TCGplayer URL, ...).
    ext = record.extracted_data or {}
    category_id, ref_game = _resolve_category_for_game(ext.get("game", ""))

    # The old separate "Database Match" action is folded in here: when the image
    # yields no name and no number (glare, low resolution, a card the detector
    # couldn't square up), fall back to the identity the user already typed on the
    # entry and match THAT against the catalog. So one button covers both "read
    # the card" and "look up what I typed", instead of the person having to know
    # which of two tools to reach for.
    lookup = ocr
    used_typed_fields = False
    if not (ocr.get("name_guess") or "").strip() and not (ocr.get("number_guess") or "").strip():
        typed_name   = _raw_field(ext, _NAME_KEYS)
        typed_serial = _raw_field(ext, _SERIAL_KEYS)
        if typed_name or typed_serial:
            lookup = {**ocr, "name_guess": typed_name, "number_guess": typed_serial}
            used_typed_fields = True

    ref_matches = _reference_candidates_for_ocr(category_id, lookup) if category_id else []

    # Matches against the user's own existing entries. These are offered in the
    # picker for a person to choose, but never auto-applied — see the note on
    # rank_reference_matches.
    candidates = _build_ocr_candidates(exclude_record_id=record.id, catalog_only=catalog_only)
    matches = card_ocr.match_ocr_to_records(lookup, candidates)
    for m in matches:
        m["source"] = "record"

    # Combine, sorted best-first, so number-matched cards float to the top
    # regardless of which source they came from. This combined list is for the
    # PICKER — a person choosing — so it still includes the user's own records.
    combined = sorted(ref_matches + matches, key=lambda c: c.get("score", 0), reverse=True)

    # The automatic decision, by contrast, considers reference data only and
    # applies the tie / threshold rules. Same function the import path calls.
    _auto_decision = rank_reference_matches(ref_matches)

    # Local-only: OCR + catalog/record matching. External (cloud) identification
    # is a separate, explicit action now — see /cloud_identify. `ximilar_error`
    # stays in the response (always empty here) for backward compatibility with
    # any client that reads it.
    ximilar_error = ""

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
        # True when the image was unreadable and the entry's typed Name/Number
        # were matched against the catalog instead (the folded-in Database Match).
        "used_typed_fields": used_typed_fields,
        "ximilar_error": ximilar_error,
        # The auto-accept decision, computed by the SAME rule the import path uses
        # (rank_reference_matches over reference data only). The page should apply
        # `winner` when decision == "apply" and otherwise open the picker — it must
        # not re-derive this from candidates[0], because `candidates` also contains
        # matches against the user's own records, which never auto-fill.
        "auto_accept": {
            "decision":        _auto_decision["decision"],
            "winner":          _auto_decision["winner"],
            "tied":            _auto_decision["tied"],
            "top_score":       _auto_decision["top_score"],
            "runner_up_score": _auto_decision["runner_up_score"],
            "min_score":       _auto_decision["min_score"],
            "min_percent":     int(round(_auto_decision["min_score"] * 100)),
        },
        # Kept for older clients that read the bare threshold.
        "auto_accept_score":   _auto_decision["min_score"],
        "auto_accept_percent": int(round(_auto_decision["min_score"] * 100)),
        # Whether this record already has an identity (name or set number). The
        # UI uses this to decide between auto-applying a confident match on a
        # fresh record vs. opening the picker for manual correction.
        "already_populated": bool(_get_name(ext) or _get_serial(ext)),
        "candidates": combined,
    })


@app.route("/cloud_identify/<int:record_id>", methods=["GET"])
def cloud_identify(record_id):
    """
    Cloud-only identification: send this record's FRONT image straight to the
    configured identification service (Settings → General: CardSight or Ximilar)
    and return whatever it recognizes. It does NO local OCR and NO local database
    matching — it's the deliberate "just ask the cloud" action.

    The provider's read is enriched against the local catalog when possible (so
    it can be applied with set/rarity/price/product_id), otherwise a raw
    name/number/set/rarity candidate is returned. Always reports a clear message
    for every non-success outcome (no provider selected, no key, network error,
    out of credits, or "couldn't identify").
    """
    record = ScanRecord.query.get_or_404(record_id)
    ext = record.extracted_data or {}

    provider = _identify_provider()
    if provider == "none":
        return jsonify({
            "status": "ok",
            "provider": "none",
            "provider_label": "None",
            "candidates": [],
            "error": "No cloud identification service is selected. Choose CardSight or Ximilar "
                     "in Settings \u2192 General.",
            "already_populated": bool(_get_name(ext) or _get_serial(ext)),
        })

    if not record.image_path or record.image_path == "__blank__":
        return jsonify({
            "status": "ok",
            "provider": provider,
            "provider_label": _identify_provider_label(provider),
            "candidates": [],
            "error": "This record has no front image to send to the cloud service.",
            "already_populated": bool(_get_name(ext) or _get_serial(ext)),
        })

    # category_id (the game's catalog) is used only to enrich the cloud read with
    # local catalog data — it does NOT gate the call.
    category_id, _ref_game = _resolve_category_for_game(ext.get("game", ""))
    cands, err = _external_identify_candidates(record, category_id)

    resp = {
        "status": "ok",
        "provider": provider,
        "provider_label": _identify_provider_label(provider),
        "candidates": cands,
        "error": err or "",
        "already_populated": bool(_get_name(ext) or _get_serial(ext)),
    }
    # /cloud_identify/<id>?debug=1 attaches the raw provider exchange so you can see
    # exactly what came back (HTTP status + JSON) when a result is unexpected.
    if request.args.get("debug", "").strip().lower() in ("1", "true", "yes") and provider == "cardsight":
        resp["debug"] = _cardsight_debug(record.image_path)
    return jsonify(resp)


# Inventory / location / ownership metadata that a full "overwrite to match the
# reference" must NEVER clobber — only the card's identity + catalog fields change.
_OVERWRITE_PROTECT = frozenset({
    "album", "collection", "page", "slot", "storage_type", "box_number",
    "held", "finalized", "archived", "sold_price", "intake_price", "current_value",
    "edition", "holographic", "grading", "empty", "catalog_only", "tcgplayer",
})


@app.route("/ocr_apply/<int:record_id>", methods=["POST"])
def ocr_apply(record_id):
    """
    Commit an OCR / match result onto this record. Body is JSON, one of:

      { "reference_product_id": <id> } -> fill fields from a tcgcsv reference
                                          card (name/number/set/rarity/game +
                                          TCGplayer URL under 'tcgplayer').
                                          Add "overwrite": true to make EVERY
                                          card field match the reference (fills
                                          the game's tcgcsv columns and blanks
                                          any the reference lacks); inventory /
                                          location metadata is always preserved.
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

        overwrite = bool(body.get("overwrite"))

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

        if overwrite:
            # Make EVERY card field match the reference: fill the game's fields
            # (name / number / set / rarity + the tcgcsv extendedData columns)
            # from the reference, blanking any the reference doesn't provide.
            # Inventory & location metadata (_OVERWRITE_PROTECT) is left intact.
            ext = {}
            for raw_k, raw_v in (ref.extended or {}).items():
                fk = _slugify_template_name(raw_k)
                if fk:
                    ext[fk] = raw_v
            try:
                _g = (record.extracted_data or {}).get("game", "") or record.template_used or "product_label"
                tpl_fields = list((load_template(_g).get("fields") or {}).keys())
            except Exception:
                tpl_fields = []
            ref_by_key = {
                "name": ref.name, "number": ref.number, "set_number": ref.number,
                "set": ref.set_name, "rarity": ref.rarity, "game": ref.game,
            }
            for fk in (set(tpl_fields) | set(ext.keys()) | set(ref_by_key.keys())):
                if fk in _OVERWRITE_PROTECT:
                    continue
                val = ref_by_key[fk] if fk in ref_by_key else ext.get(fk, "")
                updates[fk] = "" if val is None else val
            # Track the reference's market price on a full overwrite.
            if getattr(ref, "market_price", None) is not None:
                updates["current_value"] = ref.market_price
        else:
            # Non-destructive fill of the game's remaining tcgcsv columns
            # (HP/Stage/Energy Type/attacks/...), so an OCR-picked reference card
            # carries the same rich data a Database Match would — without
            # overwriting anything already entered.
            for _fk, _val in _reference_extended_fill(record, ref).items():
                updates.setdefault(_fk, _val)
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
        set_v  = str(body.get("set", "")).strip()
        rarity = str(body.get("rarity", "")).strip()
        if name:
            updates["name"] = name
        if number:
            # pad to the CSV convention (N to M's digit width) so it lines up
            # with the reference catalog's format, e.g. 24/112 -> 024/112.
            updates["set_number"] = _canonical_collector_number(number)
        if set_v:
            updates["set"] = set_v
        if rarity:
            updates["rarity"] = rarity

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


@app.route("/database_match/<int:record_id>", methods=["GET"])
def database_match(record_id):
    """
    Identify a record against the downloaded tcgcsv catalog using the Name and
    collector Number already on the entry (no image/OCR). Behaviour:

      * exactly one exact match  -> returned under `auto_apply` so the client
                                    applies it immediately.
      * several matches          -> returned under `candidates` for the user to
                                    pick from in a chooser.

    Applying a chosen card is done through /ocr_apply with its
    reference_product_id, so the fill logic stays in one place.
    """
    record = ScanRecord.query.get_or_404(record_id)
    data   = record.extracted_data or {}

    name   = _raw_field(data, _NAME_KEYS)
    number = _raw_field(data, _SERIAL_KEYS)
    if not name and not number:
        return jsonify({"status": "error",
                        "message": "Add a Name or Number to this entry first."}), 400

    category_id, ref_game = _resolve_category_for_game(data.get("game", ""))
    if not category_id:
        game_label = str(data.get("game", "")).strip() or "this game"
        return jsonify({
            "status": "error", "not_synced": True,
            "message": f"No downloaded database for {game_label}. "
                       f"Download it in Settings → Reference Data.",
        }), 400

    candidates, exact_ids = _database_match_candidates(category_id, name, number)

    auto_apply = None
    if len(exact_ids) == 1:
        pid = next(iter(exact_ids))
        auto_apply = next((c for c in candidates if c["product_id"] == pid), None)

    return jsonify({
        "status":      "ok",
        "game":        ref_game or str(data.get("game", "")),
        "query":       {"name": name, "number": number},
        "auto_apply":  auto_apply,        # non-null => apply without prompting
        "exact_count": len(exact_ids),
        "candidates":  candidates,        # populate the chooser when not auto-applying
    })


@app.route("/wrong_match/<int:record_id>", methods=["POST"])
def wrong_match(record_id):
    """
    Mark the current identification as wrong: blank this entry's card / catalog
    fields (name, collector number, set, rarity, and the game's tcgcsv columns)
    and drop the matched TCGplayer link, so the fields are empty and the user can
    type the correct details and run a fresh match. Inventory / location /
    ownership / price metadata (album, slot, held, intake & current value,
    edition, holographic, grading, ...) is preserved, and so is the game — the
    re-match needs it to know which database to search.
    """
    record = ScanRecord.query.get_or_404(record_id)
    data = dict(record.extracted_data or {})

    protect = set(_OVERWRITE_PROTECT) | {"game"}

    cleared = []
    for key in list(data.keys()):
        if key in protect or key.startswith("__"):   # keep metadata + internal keys
            continue
        if str(data.get(key) or "").strip():
            cleared.append(key)
        data[key] = ""

    # The matched TCGplayer link/price is match-specific — remove it outright.
    if data.pop("tcgplayer", None) is not None:
        cleared.append("tcgplayer")

    record.extracted_data = data
    db.session.commit()

    return jsonify({
        "status":         "success",
        "message":        f"Cleared {len(cleared)} field(s) — enter the correct details and re-match.",
        "cleared":        sorted(set(cleared)),
        "extracted_data": data,
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
    # Which of this category's sets already have cached cards? The client uses
    # this to skip sets it has already downloaded (unless "replace all" is on).
    synced_ids = {
        gid for (gid,) in (ReferenceCard.query
                           .filter_by(category_id=category_id)
                           .with_entities(ReferenceCard.group_id).distinct().all())
    }
    return jsonify({
        "status": "ok",
        "groups": [{"group_id": g.get("groupId"), "name": g.get("name") or "",
                    "synced": g.get("groupId") in synced_ids} for g in groups],
    })


def _reference_sync_one_group(category_id, category_name, group_id, group_name, replace):
    """Sync ONE set into ReferenceCard and refresh the game's ReferenceSync row.
    Returns {"status": "ok"|"skipped", "added": N}. Shared by the per-set route
    and the background sync worker so both behave identically."""
    already = (ReferenceCard.query
               .filter_by(category_id=category_id, group_id=group_id)
               .first() is not None)
    if already and not replace:
        return {"status": "skipped", "added": 0}

    cards = ref_sync.fetch_group_cards(category_id, category_name, group_id, group_name)

    if replace:
        ReferenceCard.query.filter_by(category_id=category_id, group_id=group_id).delete()
        db.session.flush()
    # One SELECT for the whole set instead of one per card: preload every
    # existing row for this payload's productIds (chunked under SQLite's
    # bound-parameter limit) and hand the map to the upserts.
    pids = [rec["product_id"] for rec in cards]
    cache = {}
    for i in range(0, len(pids), 500):
        for rc in ReferenceCard.query.filter(
                ReferenceCard.product_id.in_(pids[i:i + 500])).all():
            cache[rc.product_id] = rc
    for rec in cards:
        _reference_upsert(rec, cache=cache)

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
    _REF_FIELDS_CACHE.pop(category_id, None)
    return {"status": "ok", "added": len(cards)}


# ── Background reference-sync jobs ──────────────────────────────────────────
# A game's catalog is pulled set-by-set on a daemon thread so the download keeps
# running while the user navigates elsewhere in the app. Progress lives in memory
# and is polled by the Reference Data page; the SQLite WAL + busy_timeout config
# lets the worker write while the rest of the app keeps reading.
import threading as _threading
_REF_JOBS = {}                       # category_id -> job dict
_REF_JOBS_LOCK = _threading.Lock()

_REF_JOB_PUBLIC = ("category_id", "game", "status", "total", "done",
                   "ok", "cards", "skipped", "err", "current", "error", "replace")


def _reference_job_snapshot(job):
    return {k: job.get(k) for k in _REF_JOB_PUBLIC}


def _run_reference_sync(flask_app, category_id, category_name, replace):
    """Worker: download every set of a game, updating the shared job state."""
    import time as _t
    with flask_app.app_context():
        job = _REF_JOBS.get(category_id)
        if job is None:
            return
        try:
            groups = ref_sync.get_groups(category_id)
            job["total"] = len(groups)
            job["current"] = "Loading set list…"

            # Make the game show up in the downloaded list right away.
            rs = ReferenceSync.query.filter_by(category_id=category_id).first()
            if rs is None:
                rs = ReferenceSync(category_id=category_id, game=category_name)
                db.session.add(rs)
                rs.status = "syncing"
                db.session.commit()

            for g in groups:
                if job["stop"]:
                    break
                gid = g.get("groupId")
                gname = g.get("name") or ""
                job["current"] = gname or (f"Set #{gid}")
                try:
                    res = _reference_sync_one_group(category_id, category_name, gid, gname, replace)
                    if res["status"] == "ok":
                        job["ok"] += 1
                        job["cards"] += res["added"]
                    else:
                        job["skipped"] += 1
                except Exception:
                    db.session.rollback()
                    job["err"] += 1
                job["done"] += 1
                _t.sleep(0.12)   # be a good neighbour between sets

            job["status"] = "stopped" if job["stop"] else "done"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
        finally:
            try:
                db.session.remove()
            except Exception:
                pass


@app.route("/reference/start_sync", methods=["POST"])
def reference_start_sync():
    """Kick off a background download of a game's catalog and return immediately."""
    if ref_sync is None:
        return jsonify({"status": "error", "message": "Reference sync source unavailable."}), 503
    body = request.get_json() or {}
    try:
        category_id = int(body.get("category_id"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "category_id is required."}), 400
    category_name = str(body.get("category_name") or "").strip()
    replace = bool(body.get("replace"))

    with _REF_JOBS_LOCK:
        existing = _REF_JOBS.get(category_id)
        if existing and existing["status"] == "running":
            return jsonify({"status": "already_running", **_reference_job_snapshot(existing)})
        job = {"category_id": category_id, "game": category_name, "status": "running",
               "total": 0, "done": 0, "ok": 0, "cards": 0, "skipped": 0, "err": 0,
               "current": "Preparing…", "stop": False, "error": None, "replace": replace}
        _REF_JOBS[category_id] = job

    _threading.Thread(target=_run_reference_sync,
                      args=(app, category_id, category_name, replace),
                      daemon=True).start()
    return jsonify({"status": "started", **_reference_job_snapshot(job)})


@app.route("/reference/sync_progress")
def reference_sync_progress():
    """Snapshot of all reference-sync jobs (running and finished), for polling."""
    with _REF_JOBS_LOCK:
        jobs = [_reference_job_snapshot(j) for j in _REF_JOBS.values()]
    running = [j for j in jobs if j["status"] == "running"]
    return jsonify({"status": "ok", "jobs": jobs, "running": len(running)})


@app.route("/reference/stop_sync", methods=["POST"])
def reference_stop_sync():
    """Ask a running background sync to stop after the current set."""
    body = request.get_json() or {}
    try:
        category_id = int(body.get("category_id"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "category_id is required."}), 400
    with _REF_JOBS_LOCK:
        job = _REF_JOBS.get(category_id)
        if job:
            job["stop"] = True
    return jsonify({"status": "ok"})


@app.route("/reference/sync_group", methods=["POST"])
def reference_sync_group():
    """
    Download ONE group's cards from tcgcsv and upsert them into ReferenceCard.
    The client loops this over every group (mirroring the grade/price batch
    pattern) so progress + Stop work naturally and rate-limiting is inherent.

    Body: { category_id, category_name, group_id, group_name, replace }

    When `replace` is falsy (the default) and this set already has cached cards,
    the set is skipped without contacting the source — so re-running a game only
    fills in the sets it is missing. When `replace` is truthy the set's existing
    cards are cleared first and re-downloaded, overwriting stale data.
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
    replace       = bool(body.get("replace"))

    try:
        res = _reference_sync_one_group(category_id, category_name, group_id, group_name, replace)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Reference fetch failed: {exc}"}), 502

    rs = ReferenceSync.query.filter_by(category_id=category_id).first()
    return jsonify({
        "status": res["status"],
        "added": res["added"],
        "group_name": group_name,
        "product_count": rs.product_count if rs else None,
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


# Header aliases -> core ReferenceCard fields. Any header not matched becomes an
# extendedData column, so every CSV header ends up as an entry field for the game.
_CSV_CORE_ALIASES = {
    "name":         {"name", "card name", "cardname", "product name", "productname", "card", "title"},
    "number":       {"number", "no", "no.", "num", "card number", "cardnumber", "collector number",
                     "collectornumber", "set number", "setnumber", "#"},
    "set_name":     {"set", "set name", "setname", "expansion", "series"},
    "rarity":       {"rarity", "rare"},
    "market_price": {"price", "market price", "marketprice", "market", "value", "tcg price", "tcgprice"},
    "url":          {"url", "link", "tcgplayer", "tcgplayer url", "tcgplayerurl", "product url"},
    "image_url":    {"image", "image url", "imageurl", "img", "image link", "picture"},
}


def _csv_header_role(header):
    """Return the core ReferenceCard field a CSV header maps to, or None (-> extended)."""
    h = str(header or "").strip().lower()
    for role, aliases in _CSV_CORE_ALIASES.items():
        if h in aliases:
            return role
    return None


def _csv_price(v):
    """Parse a price cell to a float, tolerating $ and thousands separators."""
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


@app.route("/reference/upload_csv", methods=["POST"])
def reference_upload_csv():
    """
    Create a custom game catalog from an uploaded CSV, for TCGs not on tcgcsv.

    Row 1 is the header: each column name becomes an entry field for that game.
    Recognized headers (name / number / set / rarity / price / url / image) map to
    the core reference fields; every other column is kept as a per-game
    extendedData column. The uploaded file's name (without extension) becomes the
    Game name shown in the import dropdown and used by Database Match.

    Stored in the same ReferenceSync / ReferenceCard tables as downloaded games,
    under a synthetic negative category_id so it never collides with tcgcsv IDs.
    Re-uploading a file with the same game name replaces that game's rows.
    """
    import csv, io

    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"status": "error", "message": "No CSV file was uploaded."}), 400

    raw_name = os.path.splitext(os.path.basename(file.filename))[0].strip()
    game = _normalize_game_name(raw_name)
    if not game:
        return jsonify({"status": "error", "message": "Could not read a game name from the file name."}), 400

    try:
        text = file.read().decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(text)))
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Could not read the CSV: {exc}"}), 400
    if not rows:
        return jsonify({"status": "error", "message": "The CSV is empty."}), 400

    headers = [str(h or "").strip() for h in rows[0]]
    if not any(headers):
        return jsonify({"status": "error", "message": "The first row must contain column headers."}), 400
    roles = [_csv_header_role(h) for h in headers]

    # Reuse this game's category if it was uploaded before (replace it); refuse to
    # clobber a real tcgcsv download; otherwise mint a new synthetic negative id.
    existing_cid, _ = _resolve_category_for_game(game)
    if existing_cid is not None and existing_cid >= 0:
        return jsonify({"status": "error",
                        "message": f"“{game}” already exists as a downloaded tcgcsv catalog. "
                                   f"Rename your file, or delete that catalog first."}), 409
    if existing_cid is not None:
        category_id = existing_cid
        ReferenceCard.query.filter_by(category_id=category_id).delete()
        db.session.flush()
    else:
        min_cid = db.session.query(db.func.min(ReferenceCard.category_id)).scalar()
        min_sync = db.session.query(db.func.min(ReferenceSync.category_id)).scalar()
        floor = min(0, min_cid if min_cid is not None else 0, min_sync if min_sync is not None else 0)
        category_id = floor - 1

    # Synthetic unique product_ids (negative), below anything already stored.
    min_pid = db.session.query(db.func.min(ReferenceCard.product_id)).scalar()
    pid = min(0, min_pid if min_pid is not None else 0) - 1

    set_group_ids, next_group, added = {}, 1, 0
    for row in rows[1:]:
        if not any(str(c or "").strip() for c in row):
            continue
        core = {"name": "", "number": "", "set_name": "", "rarity": "",
                "market_price": None, "url": "", "image_url": ""}
        extended = {}
        for idx, header in enumerate(headers):
            if not header:
                continue
            val = str(row[idx]).strip() if idx < len(row) and row[idx] is not None else ""
            role = roles[idx]
            if role == "market_price":
                core["market_price"] = _csv_price(val)
            elif role:
                core[role] = val
            else:
                extended[header] = val

        set_name = core["set_name"] or ""
        if set_name not in set_group_ids:
            set_group_ids[set_name] = next_group
            next_group += 1

        _reference_upsert({
            "product_id":   pid,
            "category_id":  category_id,
            "group_id":     set_group_ids[set_name],
            "game":         game,
            "set_name":     set_name,
            "name":         core["name"],
            "clean_name":   core["name"].lower(),
            "number":       core["number"],
            "rarity":       core["rarity"],
            "image_url":    core["image_url"],
            "url":          core["url"],
            "market_price": core["market_price"],
            "extended":     extended,
        })
        pid -= 1
        added += 1

    if added == 0:
        db.session.rollback()
        return jsonify({"status": "error", "message": "The CSV had headers but no data rows."}), 400

    rs = ReferenceSync.query.filter_by(category_id=category_id).first()
    if rs is None:
        rs = ReferenceSync(category_id=category_id)
        db.session.add(rs)
    rs.game = game
    rs.status = "ok"
    rs.remote_updated = None
    db.session.flush()
    _reference_recount(category_id)
    db.session.commit()
    _REF_FIELDS_CACHE.pop(category_id, None)   # rebuild derived fields from the new rows

    fields = [h for h in headers if h]
    return jsonify({
        "status": "success",
        "game": game,
        "category_id": category_id,
        "added": added,
        "fields": fields,
        "message": f"Imported {added} card(s) for “{game}” with {len(fields)} field(s).",
    })


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
    bad = _reject_if_bomb(up)
    if bad:
        return bad
    record_id = (request.form.get("record_id") or "").strip()

    if up and up.filename:
        img = _imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
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
        card = _imread(abs_path)
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
    # Candidate groups come from one GROUP BY over the derived keys —
    # name_key/serial_key are `_get_name/_get_serial(data) or None` by the
    # mapper events and is_catalog mirrors _is_catalog_only, so this can never
    # exclude a row the loop below would have kept. Only members of groups
    # that share (name, serial) at least twice are hydrated; edition
    # subdivision and every original check still run in the loop.
    from sqlalchemy import func as _f, tuple_ as _sa_tuple
    not_cat = _f.coalesce(ScanRecord.is_catalog, False) == False  # noqa: E712
    keyed = (ScanRecord.name_key.isnot(None), ScanRecord.serial_key.isnot(None))
    dup_keys = [tuple(k) for k in
                (db.session.query(ScanRecord.name_key, ScanRecord.serial_key)
                 .filter(not_cat, *keyed)
                 .group_by(ScanRecord.name_key, ScanRecord.serial_key)
                 .having(_f.count(ScanRecord.id) > 1).all())]
    records = []
    for i in range(0, len(dup_keys), 400):   # stay under the bound-param limit
        records.extend(
            ScanRecord.query
            .filter(not_cat, *keyed,
                    _sa_tuple(ScanRecord.name_key, ScanRecord.serial_key)
                    .in_(dup_keys[i:i + 400]))
            .all())
    # Same ordering the old full-table query had: newest first.
    records.sort(key=lambda r: (r.scan_date or datetime.min, r.id), reverse=True)

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
# --------------------------------------------------------------------------- #
# PDF import: on-demand, one-page-at-a-time rasterisation.
#
# A multi-page binder scan can be dozens of full-resolution pages, so rendering
# the WHOLE PDF up front (the old /pdf_extract_pages) was slow, memory-heavy, and
# gave the page no way to show progress — it looked like it hung, and effectively
# never got past the first page on large files. Instead we now:
#   * /pdf_open        — save the upload, return only the page COUNT (instant),
#   * /pdf_render_page — rasterise exactly ONE page on demand (front = odd page,
#                        back = even page), returned with a URL, and
#   * /pdf_close       — drop the saved PDF when the batch finishes/exits.
# The front-end walks every page in order, rendering each just before it needs
# it and showing a loading bar while it does.
# --------------------------------------------------------------------------- #
_PDF_BATCH_ID_RE = _re.compile(r"^[0-9_]{1,40}$")


def _pdf_batch_upload_path(batch_id):
    """Absolute path to a saved PDF upload, or None if the batch id is invalid.
    The id is generated server-side and matched against a strict pattern so it
    can never be used to reach outside the temp PDF folder."""
    if not batch_id or not _PDF_BATCH_ID_RE.match(str(batch_id)):
        return None
    return os.path.join(app.config["TEMP_PDF_FOLDER"], f"upload_{batch_id}.pdf")


@app.route("/pdf_open", methods=["POST"])
def pdf_open():
    """
    Accept a PDF upload, save it, and report how many pages it has — without
    rendering anything yet. Rendering happens one page at a time via
    /pdf_render_page so a large multi-page scan doesn't stall the UI.
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
            page_count = doc.page_count
        finally:
            doc.close()
    except Exception as e:
        try:
            os.remove(pdf_path)
        except OSError:
            pass
        return jsonify({"status": "error", "message": f"Could not read PDF: {e}"}), 500

    if page_count < 1:
        try:
            os.remove(pdf_path)
        except OSError:
            pass
        return jsonify({"status": "error", "message": "PDF has no pages"}), 400

    if page_count > MAX_PDF_PAGES:
        try:
            os.remove(pdf_path)
        except OSError:
            pass
        return jsonify({"status": "error",
                        "message": f"PDF has {page_count} pages, over the {MAX_PDF_PAGES}-page limit. "
                                   f"Split it, or raise MAX_PDF_PAGES."}), 413

    return jsonify({
        "status":     "success",
        "message":    f"PDF opened — {page_count} page(s).",
        "batch_id":   batch_id,
        "page_count": page_count,
    })


@app.route("/pdf_render_page", methods=["POST"])
def pdf_render_page():
    """
    Rasterise a single page (1-based `index`) of a previously opened PDF and
    return its image URL. Rendered pages are cached on disk, so re-requesting a
    page (e.g. front then back of the same pair) is cheap. Peak memory is a
    single page's pixmap.
    """
    ensure_dirs()

    if fitz is None:
        return jsonify({
            "status": "error",
            "message": "PDF support isn't installed on the server. Run: pip install PyMuPDF"
        }), 500

    data     = request.get_json(silent=True) or {}
    batch_id = str(data.get("batch_id") or "").strip()
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "A numeric page index is required"}), 400

    pdf_path = _pdf_batch_upload_path(batch_id)
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({
            "status":  "error",
            "message": "This PDF batch is no longer available — please re-select the PDF.",
        }), 404

    page_filename = f"pdfpage_{batch_id}_{index:03d}.png"
    page_path     = os.path.join(app.config["TEMP_PDF_FOLDER"], page_filename)

    # Serve the cached render if this page was already rasterised.
    if not os.path.exists(page_path):
        try:
            doc = fitz.open(pdf_path)
            try:
                if index < 1 or index > doc.page_count:
                    return jsonify({
                        "status":  "error",
                        "message": f"Page {index} is out of range (PDF has {doc.page_count}).",
                    }), 400
                page = doc.load_page(index - 1)
                # Native embedded-scan resolution (clamped), matching the old
                # whole-PDF path so cards keep full scanner detail.
                pix = page.get_pixmap(matrix=_pdf_render_matrix(page))
                pix.save(page_path)
                del pix, page
            finally:
                doc.close()
        except Exception as e:
            return jsonify({"status": "error", "message": f"Could not render page {index}: {e}"}), 500

    return jsonify({
        "status": "success",
        "page": {
            "index":    index,
            "filename": page_filename,
            "url":      url_for("temp_pdf_file", filename=page_filename),
        },
    })


@app.route("/pdf_close", methods=["POST"])
def pdf_close():
    """
    Best-effort cleanup once a PDF batch finishes or is exited: remove the saved
    upload so it doesn't linger in temp. Rendered page PNGs are left for the
    normal temp-folder cleanup. Always reports success — cleanup never blocks.
    """
    data     = request.get_json(silent=True) or {}
    batch_id = str(data.get("batch_id") or "").strip()
    pdf_path = _pdf_batch_upload_path(batch_id)
    if pdf_path:
        try:
            os.remove(pdf_path)
        except OSError:
            pass
    return jsonify({"status": "success"})


@app.route("/run_import_split", methods=["POST"])
def run_import_split():
    ensure_dirs()

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No image provided"}), 400
    bad = _reject_if_bomb(file)
    if bad:
        return bad

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
        results = _split_align_page_tiles(import_path, game, album, page, side, edge_type,
                                          v_cut1, v_cut2, h_cut1, h_cut2)
        for r in results:
            if r["status"] == "processed":
                r["url"] = url_for("temp_card_file", filename=r["filename"])
            else:
                r["url"] = url_for("temp_split_file", filename=r["filename"])

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

    # `filename` is request-body-controlled and is joined onto TEMP_SPLIT_FOLDER (read)
    # and TEMP_CARD_FOLDER (cv2.imwrite) below; os.path.join drops the base on an
    # absolute path and honours ../, so a raw name escapes the folder. secure_filename
    # strips both forms and is idempotent on the slot_<...> names _split_align_page_tiles
    # generates (verified). Same guard as move_temp_card_to_inventory (app.py:791).
    # Reassigned so the echoed url/filename point at the file actually written.
    filename = secure_filename(filename)
    if not filename:
        return jsonify({"status": "error", "message": "Invalid filename"}), 400

    if not isinstance(points, list) or len(points) != 4:
        return jsonify({"status": "error", "message": "Exactly 4 points are required"}), 400

    split_path = os.path.join(app.config["TEMP_SPLIT_FOLDER"], filename)
    if not os.path.exists(split_path):
        return jsonify({"status": "error", "message": f"Fallback image not found: {filename}"}), 404

    image = _imread(split_path)
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
    storage_type  = str(data.get("storage_type", "album") or "album").strip().lower()

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
            try:
                result = _finalize_one_card(filename, template_name, blank_fields, collection, storage_type)
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


# ====================== BACKGROUND PDF AUTO-IMPORT ======================
# A long PDF auto-import (cut -> align/crop -> identify -> file, for every page)
# runs on a daemon thread so you can navigate away and keep using the app while
# it works. Progress lives in memory and is polled by the import page. Reuses the
# same pipeline primitives as the interactive flow via the shared helpers below.

def _detect_cut_lines(bgr):
    """Server-side port of the import page's white-gap cut-line detection. Given a
    9-pocket page image (BGR), return (v1, v2, h1, h2) as 0-100 percentages for the
    two vertical and two horizontal cut lines. Any line that can't be confidently
    found falls back to even thirds. Mirrors detectCutLines()/findBandCenter() in
    import.html so the background importer places the same lines the interactive
    preview would."""
    h0, w0 = bgr.shape[:2]
    scale = min(900.0 / w0, 560.0 / h0, 1.0)
    dw = max(1, int(round(w0 * scale)))
    dh = max(1, int(round(h0 * scale)))
    small = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)

    b = small[:, :, 0].astype(np.int16)
    g = small[:, :, 1].astype(np.int16)
    r = small[:, :, 2].astype(np.int16)
    mn = np.minimum(np.minimum(r, g), b)
    mx = np.maximum(np.maximum(r, g), b)
    white = (mn > 200) & ((mx - mn) < 40)      # near-white background pixel

    col_white = white.mean(axis=0)             # fraction white per column
    row_white = white.mean(axis=1)             # fraction white per row

    def band_center(profile, lo_frac, hi_frac):
        L = len(profile)
        lo, hi = int(L * lo_frac), int(L * hi_frac)
        if hi <= lo:
            return None
        window = profile[lo:hi]
        peak = float(window.max()) if window.size else 0.0
        if peak < 0.5:                          # no genuine white gap here
            return None
        thr = max(0.6, peak - 0.15)
        best_len = best_start = cur_len = cur_start = 0
        for i in range(lo, hi):
            if profile[i] >= thr:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        if best_len == 0:
            return None
        return (best_start + best_len / 2.0) / L * 100.0

    def clamp(v):
        return min(95.0, max(5.0, round(v)))

    v1, v2 = band_center(col_white, 0.20, 0.47), band_center(col_white, 0.53, 0.80)
    h1, h2 = band_center(row_white, 0.20, 0.47), band_center(row_white, 0.53, 0.80)
    v1 = clamp(v1) if v1 is not None else 100 / 3.0
    v2 = clamp(v2) if v2 is not None else 200 / 3.0
    h1 = clamp(h1) if h1 is not None else 100 / 3.0
    h2 = clamp(h2) if h2 is not None else 200 / 3.0
    if v1 >= v2:                                # keep cut 1 < cut 2 per axis
        v1, v2 = 100 / 3.0, 200 / 3.0
    if h1 >= h2:
        h1, h2 = 100 / 3.0, 200 / 3.0
    return v1, v2, h1, h2


def _detect_cut_lines_for_path(page_image_path):
    """Auto-detect cut lines for a page image on disk; even thirds if unreadable."""
    bgr = _imread(page_image_path)
    if bgr is None:
        return (100 / 3.0, 200 / 3.0, 100 / 3.0, 200 / 3.0)
    return _detect_cut_lines(bgr)


def _split_align_page_tiles(page_image_path, game, album, page, side, edge_type,
                            v1=100 / 3.0, v2=200 / 3.0, h1=100 / 3.0, h2=200 / 3.0):
    """Cut a 9-pocket page image into 9 tiles, align/crop each to `edge_type`, save
    to the temp card folder, and return a result list (slot/filename/status). On a
    tile that can't be aligned the raw cut is still saved so the card reaches
    inventory. Shared by /run_import_split and the background PDF importer. Does
    NOT add url_for links (so it's safe to call off the request thread)."""
    safe_game  = secure_filename(game)  or "game"
    safe_album = secure_filename(album) or "album"
    safe_page  = secure_filename(str(page)) or "page"

    pil_image = Image.open(page_image_path).convert("RGB")
    pieces    = split_image_3x3(pil_image, v1, v2, h1, h2)

    results = []
    for photographed_idx, piece in enumerate(pieces, start=1):
        slot_num      = resolve_slot_number(photographed_idx, side)
        slot_filename = f"{safe_game}-{safe_album}-{safe_page}-{slot_num}-{side}.png"
        split_path    = os.path.join(app.config["TEMP_SPLIT_FOLDER"], slot_filename)
        piece.save(split_path)
        try:
            split_cv = _imread(split_path)
            if split_cv is None:
                raise ValueError("Could not read split image")
            cropped, ok = detect_and_crop_card(split_cv, edge_type)
            processed = cropped if ok else straighten_split_image(split_cv)
            cv2.imwrite(os.path.join(app.config["TEMP_CARD_FOLDER"], slot_filename), processed)
            results.append({"slot": slot_num, "filename": slot_filename, "status": "processed"})
        except Exception:
            try:
                piece.save(os.path.join(app.config["TEMP_CARD_FOLDER"], slot_filename))
                results.append({"slot": slot_num, "filename": slot_filename, "status": "processed"})
            except Exception:
                results.append({"slot": slot_num, "filename": slot_filename, "status": "fallback"})
    return results


def _finalize_one_card(filename, template_name, blank_fields, collection, storage_type):
    """Create or merge ONE inventory record from an aligned card image. Fronts are
    auto-identified; backs merge onto the matching pocket. Raises InventoryCapError
    if the cap is hit. Returns a result dict. Shared by /import_finalize_batch and
    the background PDF importer."""
    # `filename` is an element of the request-body `filenames` list on the
    # /import_finalize_batch path. Sanitize before the join: without it, this existence
    # check runs against TEMP_CARD_FOLDER/<raw> while move_temp_card_to_inventory below
    # (app.py:791) operates on TEMP_CARD_FOLDER/<sanitized>, so for any name the sanitiser
    # alters the two disagree — the check passes, then the move raises FileNotFoundError.
    # It is also an existence oracle for arbitrary paths via os.path.exists. Idempotent on
    # the server-generated slot_<...> names the PDF importer passes (verified).
    # Keep the caller's original name for the diagnostic echo: the joins use the
    # sanitized value, but an error row should identify the item the caller submitted
    # -- and agree with import_finalize_batch's own except handler, which echoes raw.
    raw_name = filename
    filename = secure_filename(filename)
    if not filename:
        return {"filename": raw_name, "status": "error", "message": "Aligned card image not found"}
    temp_image_path = os.path.join(app.config["TEMP_CARD_FOLDER"], filename)
    if not os.path.exists(temp_image_path):
        return {"filename": raw_name, "status": "error", "message": "Aligned card image not found"}

    filename_fields = parse_card_filename(filename)
    extracted = {**blank_fields, **filename_fields}
    if extracted.get("game"):
        extracted["game"] = _normalize_game_name(extracted["game"])
    if collection:
        extracted["collection"] = collection
    side = filename_fields.get("side", "front")

    game  = extracted.get("game",  "")
    album = extracted.get("album", "")
    page  = extracted.get("page",  "")
    slot  = extracted.get("slot",  "")

    if storage_type == "box":
        extracted["storage_type"] = "box"
        try:
            pnum, snum = int(page or 0), int(slot or 0)
            if pnum and snum:
                extracted["box_number"] = (pnum - 1) * 9 + snum
        except (TypeError, ValueError):
            pass
    else:
        extracted["storage_type"] = "album"

    final_relative_image_path = move_temp_card_to_inventory(filename)
    existing = find_existing_record_for_key(game, album, page, slot)

    if existing:
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
        create_kwargs = dict(template_name=template_name, extracted=extracted)
        if side == "back":
            create_kwargs["image_path"]      = "__blank__"
            create_kwargs["image_path_back"] = final_relative_image_path
        else:
            create_kwargs["image_path"] = final_relative_image_path
        matched_product, record = create_scan_record(**create_kwargs)

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

    return {
        "filename":        filename,
        "status":          "success",
        "record_id":       record.id,
        "side":            side,
        "image_url":       build_uploaded_file_url(record.image_path),
        "image_url_back":  build_uploaded_file_url(record.image_path_back) if record.image_path_back else None,
        # The record's data AFTER identification — `extracted` is the blank field
        # dict built before auto-identify ran, so returning it showed empty fields
        # for cards that had in fact been identified and saved.
        "extracted":       record.extracted_data or extracted,
        "matched_product": matched_product.product_name if matched_product else "No match",
        "identified":      bool(ident.get("identified")),
        "identified_name": (ident.get("applied") or {}).get("name", "") if ident.get("identified") else "",
        "ident_error":     ident.get("error", ""),
        "ident_source":    ident.get("source", ""),
        "ident_reason":    ident.get("reason", ""),
        "ident_score":     ident.get("score"),
        "ident_min_score": ident.get("min_score"),
        "ident_tied":      ident.get("tied", []),
        "card_type":       (ident.get("type_applied") or {}).get("value", ""),
    }


def _render_pdf_page_to_path(pdf_path, index):
    """Rasterise page `index` (1-based) of a PDF to a cached PNG; return its path."""
    page_filename = f"pdfpage_bg_{os.path.basename(pdf_path)}_{index:03d}.png"
    page_path = os.path.join(app.config["TEMP_PDF_FOLDER"], page_filename)
    if os.path.exists(page_path):
        return page_path
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(index - 1)
        pix = page.get_pixmap(matrix=_pdf_render_matrix(page))
        pix.save(page_path)
        del pix, page
    finally:
        doc.close()
    return page_path


def _next_sheet_index(name):
    """Next 'sheet/page' index for a container (max existing page + 1)."""
    key = (name or "").strip().lower()
    if not key:
        return 1
    rows = (ScanRecord.query.filter(ScanRecord.album_key == key)
            .with_entities(ScanRecord.extracted_data).all())
    mx = 0
    for (data,) in rows:
        try:
            p = int((data or {}).get("page") or 0)
        except (TypeError, ValueError):
            p = 0
        mx = max(mx, p)
    return mx + 1


# ── PDF-import job registry ──
_PDF_JOBS = {}
_PDF_JOBS_LOCK = _threading.Lock()
_PDF_JOB_PUBLIC = ("job_id", "game", "status", "total_pages", "pages_done",
                   "cards", "identified", "errors", "page_errors", "current", "error")


def _pdf_job_snapshot(job):
    return {k: job.get(k) for k in _PDF_JOB_PUBLIC}


def _run_pdf_import(flask_app, job_id, pdf_path, page_count, params):
    """Worker: render every page, cut/align/crop 9 cards, then file + auto-identify."""
    import time as _t
    with flask_app.app_context():
        job = _PDF_JOBS.get(job_id)
        if job is None:
            return
        game         = params["game"]
        album        = params["album"]
        base_page    = params["page"]
        edge_type    = params["edge_type"]
        collection   = params["collection"]
        storage_type = params["storage_type"]
        try:
            template = load_template(game)
            blank_fields = {k: "" for k in (template.get("fields", {}) or {}).keys()}
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"Could not load game '{game}': {exc}"
            return

        try:
            for index in range(1, page_count + 1):
                if job["stop"]:
                    break
                pair_idx = (index - 1) // 2
                side = "front" if (index % 2 == 1) else "back"
                try:
                    pocket_page = str(int(base_page) + pair_idx)
                except (TypeError, ValueError):
                    pocket_page = str(base_page)
                job["current"] = f"Page {index} of {page_count} ({side})"

                try:
                    page_img = _render_pdf_page_to_path(pdf_path, index)
                    v1, v2, h1, h2 = _detect_cut_lines_for_path(page_img)
                    tiles = _split_align_page_tiles(page_img, game, album, pocket_page, side,
                                                    edge_type, v1, v2, h1, h2)
                    for t in tiles:
                        try:
                            res = _finalize_one_card(t["filename"], game, blank_fields, collection, storage_type)
                            if res.get("status") == "success":
                                job["cards"] += 1
                                if res.get("identified"):
                                    job["identified"] += 1
                            else:
                                job["errors"] += 1
                        except InventoryCapError:
                            job["status"] = "cap_reached"
                            job["stop"] = True
                            break
                        except Exception:
                            db.session.rollback()
                            job["errors"] += 1
                except Exception:
                    job["page_errors"] += 1

                job["pages_done"] += 1
                _t.sleep(0.02)

            if job["status"] != "cap_reached":
                job["status"] = "stopped" if job["stop"] else "done"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
        finally:
            try:
                os.remove(pdf_path)
            except OSError:
                pass
            try:
                db.session.remove()
            except Exception:
                pass


@app.route("/import/start_pdf", methods=["POST"])
def import_start_pdf():
    """Start a background PDF auto-import and return immediately. Accepts either a
    fresh multipart upload (field 'pdf' or 'image') or a batch_id from /pdf_open."""
    ensure_dirs()
    if fitz is None:
        return jsonify({"status": "error", "message": "PDF support isn't installed on the server."}), 500

    form = request.form if len(request.form) else (request.get_json(silent=True) or {})

    pdf_path = None
    upload = request.files.get("pdf") or request.files.get("image")
    if upload and upload.filename:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pdf_path = os.path.join(app.config["TEMP_PDF_FOLDER"], f"upload_{batch_id}.pdf")
        upload.save(pdf_path)
    else:
        batch_id = str(form.get("batch_id") or "").strip()
        pdf_path = _pdf_batch_upload_path(batch_id) if batch_id else None
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({"status": "error", "message": "PDF not available — please re-select it."}), 400

    game  = str(form.get("game", "")).strip()
    album = str(form.get("album", "")).strip()
    if not game or not album:
        return jsonify({"status": "error", "message": "Game and storage are required."}), 400
    storage_type = str(form.get("storage_type", "album") or "album").strip().lower()
    edge_type    = normalize_card_edge_type(form.get("card_edge_type"))
    collection   = _clean_collection(form.get("collection"))
    if storage_type == "box":
        base_page = _next_sheet_index(album)
    else:
        try:
            base_page = int(str(form.get("page", "")).strip() or "1")
        except (TypeError, ValueError):
            base_page = 1

    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Could not read PDF: {exc}"}), 500
    if page_count < 1:
        return jsonify({"status": "error", "message": "PDF has no pages."}), 400
    if page_count > MAX_PDF_PAGES:
        return jsonify({"status": "error",
                        "message": f"PDF has {page_count} pages, over the {MAX_PDF_PAGES}-page limit. "
                                   f"Split it, or raise MAX_PDF_PAGES."}), 413

    job_id = batch_id or datetime.now().strftime("pdf_%Y%m%d_%H%M%S_%f")
    with _PDF_JOBS_LOCK:
        for j in _PDF_JOBS.values():
            if j["status"] == "running":
                return jsonify({"status": "already_running", **_pdf_job_snapshot(j)})
        job = {"job_id": job_id, "game": game, "status": "running",
               "total_pages": page_count, "pages_done": 0, "cards": 0, "identified": 0,
               "errors": 0, "page_errors": 0, "current": "Starting…", "stop": False, "error": None}
        _PDF_JOBS[job_id] = job

    params = {"game": game, "album": album, "page": base_page, "edge_type": edge_type,
              "collection": collection, "storage_type": storage_type}
    _threading.Thread(target=_run_pdf_import, args=(app, job_id, pdf_path, page_count, params),
                      daemon=True).start()
    return jsonify({"status": "started", **_pdf_job_snapshot(job)})


@app.route("/import/pdf_progress")
def import_pdf_progress():
    """Snapshot of all PDF-import jobs (running and finished), for polling."""
    with _PDF_JOBS_LOCK:
        jobs = [_pdf_job_snapshot(j) for j in _PDF_JOBS.values()]
    running = [j for j in jobs if j["status"] == "running"]
    return jsonify({"status": "ok", "jobs": jobs, "running": len(running)})


@app.route("/import/stop_pdf", methods=["POST"])
def import_stop_pdf():
    """Ask a running background PDF import to stop after the current card."""
    body = request.get_json(silent=True) or {}
    job_id = str(body.get("job_id") or "").strip()
    with _PDF_JOBS_LOCK:
        job = _PDF_JOBS.get(job_id)
        if job:
            job["stop"] = True
    return jsonify({"status": "ok"})


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

    img = _imread(img_path)
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
    img = _imread(img_path)
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
    Duplicates) are reached from the sidebar on these pages.

    `general` carries the inline controls rendered on this page (currently the
    OCR auto-accept confidence slider)."""
    return render_template("settings.html", general=_general_status())


@app.route("/settings/reference")
def reference_data_page():
    """Reference-data catalog downloader (tcgcsv). Moved here from the Inventory
    toolbar so catalog imports live alongside the other setup pages. The page's
    JS drives the existing /reference/* JSON endpoints."""
    return render_template("reference_data.html")


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
        "ximilar_fallback_enabled": _ximilar_fallback_on(),
        "ximilar_key_set": bool(get_api_key("XIMILAR_API_TOKEN")),
        "ximilar_env_forced": os.environ.get("XIMILAR_IDENTIFY_FALLBACK") is not None,
        # External identification provider selection (used when local OCR/catalog
        # lookup fails). One of: none | ximilar | cardsight.
        "identify_provider": _identify_provider(),
        "identify_provider_label": _identify_provider_label(),
        "identify_provider_env_forced": os.environ.get("IDENTIFY_PROVIDER") is not None,
        "cardsight_key_set": bool(get_api_key("CARDSIGHT_API_KEY")),
        # OCR auto-accept confidence (the Settings slider). `*_pct` values are
        # what the slider itself binds to; `env_forced` flags an install that
        # pins the threshold via the AUTO_IDENTIFY_MIN_SCORE env var and has not
        # yet overridden it from the UI.
        "auto_identify_min_score":     auto_identify_min_score(),
        "auto_identify_min_pct":       auto_identify_min_percent(),
        "auto_identify_default_pct":   int(round(AUTO_IDENTIFY_MIN_SCORE_DEFAULT * 100)),
        "auto_identify_floor_pct":     int(round(AUTO_IDENTIFY_MIN_SCORE_FLOOR * 100)),
        "auto_identify_ceil_pct":      int(round(AUTO_IDENTIFY_MIN_SCORE_CEIL * 100)),
        "auto_identify_env_forced":    (os.environ.get("AUTO_IDENTIFY_MIN_SCORE") is not None
                                        and AUTO_IDENTIFY_MIN_SCORE_KEY not in _settings_cache),
        "identify_providers": [
            {"value": "none",      "label": "None (local database only)", "key_set": True},
            {"value": "ximilar",   "label": "Ximilar",   "key_set": bool(get_api_key("XIMILAR_API_TOKEN"))},
            {"value": "cardsight", "label": "CardSight (free tier)", "key_set": bool(get_api_key("CARDSIGHT_API_KEY"))},
        ],
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


@app.route("/settings/general/ximilar_fallback", methods=["POST"])
def general_ximilar_fallback():
    body = request.get_json(silent=True) or request.form
    enabled = str(body.get("enabled", "")).lower() in ("1", "true", "yes", "on")
    # The old on/off toggle now maps onto the provider selector so the two can't
    # disagree: ON selects Ximilar, OFF selects None. (Use Settings → Identification
    # to pick CardSight instead.)
    set_identify_provider("ximilar" if enabled else "none")
    if enabled and not get_api_key("XIMILAR_API_TOKEN"):
        # Allowed, but warn: it won't do anything (and imports will report the
        # missing key) until a token is added.
        msg = ("Ximilar selected, but no API key is set — add your Ximilar "
               "API token in Settings \u2192 API Keys for it to work.")
    elif enabled:
        msg = "Ximilar identification enabled."
    else:
        msg = "External identification disabled (local database only)."
    return jsonify({"status": "success", "message": msg, "general": _general_status()})


@app.route("/settings/general/auto_identify_threshold", methods=["GET", "POST"])
def general_auto_identify_threshold():
    """Read (GET) or set (POST) the OCR auto-accept confidence — how sure a match
    must be before an identification is applied to a card automatically.

    POST body accepts `percent` (0-100) or `value` (0-1); both are clamped to the
    slider bounds. Takes effect immediately: auto_identify_min_score() is read
    fresh on every card, so no restart is needed."""
    if request.method == "GET":
        return jsonify({"status": "success", "general": _general_status()})

    body = request.get_json(silent=True) or request.form
    raw = body.get("percent", body.get("value", body.get("threshold", "")))
    if str(raw).strip() == "":
        return jsonify({"status": "error",
                        "message": "No confidence value was supplied.",
                        "general": _general_status()}), 400

    score = set_auto_identify_min_score(raw)
    pct = int(round(score * 100))
    if pct >= 85:
        hint = " Very strict — expect more cards to come in blank for manual entry."
    elif pct <= 50:
        hint = " Fairly permissive — more cards fill in automatically, but check them for mismatches."
    else:
        hint = ""
    return jsonify({
        "status": "success",
        "percent": pct,
        "value": score,
        "message": f"Auto-accept confidence set to {pct}%.{hint}",
        "general": _general_status(),
    })


@app.route("/settings/general/identify_provider", methods=["POST"])
def general_identify_provider():
    """Choose which external identification service is used when the local OCR /
    catalog lookup fails: 'none', 'ximilar', or 'cardsight'."""
    body = request.get_json(silent=True) or request.form
    provider = str(body.get("provider", "")).strip().lower()
    try:
        set_identify_provider(provider)
    except ValueError:
        return jsonify({"status": "error",
                        "message": "Invalid provider. Choose none, ximilar, or cardsight.",
                        "general": _general_status()}), 400

    label = _identify_provider_label(provider)
    if provider == "none":
        msg = "External card identification disabled — using the local database only."
    elif provider == "ximilar" and not get_api_key("XIMILAR_API_TOKEN"):
        msg = (f"{label} selected, but no API key is set — add your Ximilar API token in "
               "Settings \u2192 API Keys for it to work.")
    elif provider == "cardsight" and not get_api_key("CARDSIGHT_API_KEY"):
        msg = (f"{label} selected, but no API key is set — add your free CardSight API key in "
               "Settings \u2192 API Keys for it to work.")
    else:
        msg = f"{label} selected for card identification when the local lookup fails."
    return jsonify({"status": "success", "message": msg, "general": _general_status()})


# Self-contained settings page for choosing the identification provider. Rendered
# inline (no template file) so it works regardless of the theme templates, and is
# reachable at /settings/identify. The main General settings page can also embed
# the same control via /settings/general/identify_provider.
_IDENTIFY_SETTINGS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card Identification Service</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#f5f7fb; color:#1f2937; margin:0; padding:24px; }
  .card { max-width:720px; margin:0 auto; background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:24px 28px; box-shadow:0 10px 30px rgba(0,0,0,.05); }
  h1 { font-size:22px; margin:0 0 6px; }
  p.sub { color:#6b7280; margin:0 0 20px; }
  .opt { display:flex; align-items:flex-start; gap:12px; border:1px solid #e5e7eb; border-radius:12px; padding:14px 16px; margin-bottom:12px; cursor:pointer; }
  .opt:hover { border-color:#c7d2fe; background:#fafbff; }
  .opt.sel { border-color:#4f46e5; background:#f5f3ff; }
  .opt input { margin-top:3px; }
  .opt .body { flex:1; }
  .opt .name { font-weight:700; }
  .opt .desc { color:#6b7280; font-size:14px; margin-top:2px; }
  .pill { display:inline-block; font-size:12px; font-weight:700; padding:2px 8px; border-radius:999px; margin-left:8px; }
  .pill.ok { background:#dcfce7; color:#166534; }
  .pill.no { background:#fee2e2; color:#991b1b; }
  .row { display:flex; gap:10px; align-items:center; margin-top:18px; flex-wrap:wrap; }
  button { background:#4f46e5; color:#fff; border:0; border-radius:10px; padding:10px 18px; font-weight:700; cursor:pointer; }
  button:disabled { opacity:.6; cursor:default; }
  a { color:#4f46e5; text-decoration:none; }
  a:hover { text-decoration:underline; }
  .msg { margin-top:16px; padding:12px 14px; border-radius:10px; display:none; }
  .msg.show { display:block; }
  .msg.ok { background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; }
  .msg.err { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  .links { margin-top:20px; font-size:14px; color:#6b7280; }
  .links a { margin-right:14px; }
</style>
</head>
<body>
  <div class="card">
    <h1>Card Identification Service</h1>
    <p class="sub">Choose which external service identifies a card from its front image when the
       local OCR / catalog lookup can't match it (below the __MIN_PCT__% auto-accept confidence set on the
       <a href="/settings">Settings</a> page). This applies to both auto-import and
       the manual &ldquo;Read Card &amp; Find Matches&rdquo; on a card's detail page.</p>

    <label class="opt" data-value="none">
      <input type="radio" name="provider" value="none">
      <div class="body">
        <div class="name">None &mdash; local database only</div>
        <div class="desc">Never call an external service. Uses your imported catalog and existing records only.</div>
      </div>
    </label>

    <label class="opt" data-value="cardsight">
      <input type="radio" name="provider" value="cardsight">
      <div class="body">
        <div class="name">CardSight <span id="csPill" class="pill">&nbsp;</span></div>
        <div class="desc">Free tier: 750 identifications/month, no credit card. Covers sports cards and Pok&eacute;mon.
          Get a key at <a href="https://cardsight.ai" target="_blank" rel="noopener">cardsight.ai</a>.</div>
      </div>
    </label>

    <label class="opt" data-value="ximilar">
      <input type="radio" name="provider" value="ximilar">
      <div class="body">
        <div class="name">Ximilar <span id="xiPill" class="pill">&nbsp;</span></div>
        <div class="desc">Paid credits (free plan ~3,000 credits/month once activated; ~10 per card).
          Get a key at <a href="https://www.ximilar.com" target="_blank" rel="noopener">ximilar.com</a>.</div>
      </div>
    </label>

    <div class="row">
      <button id="saveBtn">Save selection</button>
      <a href="/identify/diagnose" target="_blank" rel="noopener">Test connection &rarr;</a>
    </div>

    <div id="msg" class="msg"></div>

    <div class="links">
      <a href="/settings/api">API Keys</a>
      <a href="/settings/general">General settings</a>
      <a href="/settings">All settings</a>
    </div>
  </div>

<script>
  var CURRENT = "__PROVIDER__";
  var XI_SET = ("__XIMILAR_SET__" === "yes");
  var CS_SET = ("__CARDSIGHT_SET__" === "yes");

  function pill(el, isSet) {
    el.textContent = isSet ? "key set" : "no key";
    el.className = "pill " + (isSet ? "ok" : "no");
  }
  pill(document.getElementById("xiPill"), XI_SET);
  pill(document.getElementById("csPill"), CS_SET);

  var radios = document.querySelectorAll('input[name="provider"]');
  function syncSelected() {
    document.querySelectorAll('.opt').forEach(function (o) {
      var r = o.querySelector('input');
      o.classList.toggle('sel', r.checked);
    });
  }
  radios.forEach(function (r) {
    if (r.value === CURRENT) r.checked = true;
    r.addEventListener('change', syncSelected);
  });
  syncSelected();

  var msg = document.getElementById('msg');
  function showMsg(text, ok) {
    msg.textContent = text;
    msg.className = 'msg show ' + (ok ? 'ok' : 'err');
  }

  document.getElementById('saveBtn').addEventListener('click', async function () {
    var sel = document.querySelector('input[name="provider"]:checked');
    if (!sel) { showMsg('Pick a provider first.', false); return; }
    var btn = this; btn.disabled = true;
    try {
      var res = await fetch('/settings/general/identify_provider', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: sel.value })
      });
      var data = await res.json();
      showMsg(data.message || (res.ok ? 'Saved.' : 'Save failed.'), res.ok && data.status === 'success');
      CURRENT = sel.value;
    } catch (e) {
      showMsg('Network error: ' + e.message, false);
    } finally {
      btn.disabled = false;
    }
  });
</script>
</body>
</html>"""


@app.route("/settings/identify")
def identify_settings_page():
    """Standalone page to pick the external identification provider. Self-contained
    so it doesn't depend on the theme templates; reachable at /settings/identify."""
    g = _general_status()
    html = (_IDENTIFY_SETTINGS_HTML
            .replace("__MIN_PCT__", str(g["auto_identify_min_pct"]))
            .replace("__PROVIDER__", g["identify_provider"])
            .replace("__XIMILAR_SET__", "yes" if g["ximilar_key_set"] else "no")
            .replace("__CARDSIGHT_SET__", "yes" if g["cardsight_key_set"] else "no"))
    return Response(html, mimetype="text/html")


# Self-contained page to set the advertised .local network name. Inline (no theme
# template dependency), reachable at /settings/network.
_NETWORK_SETTINGS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Network Name</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#f5f7fb; color:#1f2937; margin:0; padding:24px; }
  .card { max-width:720px; margin:0 auto; background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:24px 28px; box-shadow:0 10px 30px rgba(0,0,0,.05); }
  h1 { font-size:22px; margin:0 0 6px; }
  p.sub { color:#6b7280; margin:0 0 20px; }
  label.fld { display:block; font-weight:700; margin-bottom:6px; }
  .inrow { display:flex; align-items:stretch; max-width:440px; }
  .inrow input { flex:1; border:1px solid #d1d5db; border-right:0; border-radius:10px 0 0 10px; padding:10px 12px; font-size:15px; }
  .inrow .suffix { display:flex; align-items:center; padding:0 12px; border:1px solid #d1d5db; border-left:0; border-radius:0 10px 10px 0; background:#f3f4f6; color:#6b7280; font-size:15px; }
  .preview { margin-top:12px; font-size:14px; color:#374151; }
  .preview code, .note code, .warn code { background:#f3f4f6; border:1px solid #e5e7eb; border-radius:6px; padding:2px 8px; }
  .row { display:flex; gap:10px; align-items:center; margin-top:18px; flex-wrap:wrap; }
  button { background:#4f46e5; color:#fff; border:0; border-radius:10px; padding:10px 18px; font-weight:700; cursor:pointer; }
  button:disabled { opacity:.6; cursor:default; }
  .note { margin-top:18px; font-size:13px; color:#6b7280; line-height:1.55; }
  .warn { margin-top:14px; padding:10px 12px; border-radius:10px; background:#fffbeb; border:1px solid #fde68a; color:#92400e; font-size:13px; display:__ZC_WARN__; }
  a { color:#4f46e5; text-decoration:none; } a:hover { text-decoration:underline; }
  .msg { margin-top:16px; padding:12px 14px; border-radius:10px; display:none; }
  .msg.show { display:block; }
  .msg.ok { background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; }
  .msg.err { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  .links { margin-top:20px; font-size:14px; color:#6b7280; } .links a { margin-right:14px; }
</style>
</head>
<body>
  <div class="card">
    <h1>Network Name (.local)</h1>
    <p class="sub">Set the name this server advertises on your local network. Devices on the same
       Wi-Fi/LAN can then reach it at <strong>&lt;name&gt;.local</strong> &mdash; no IP address or
       router setup needed.</p>

    <label class="fld" for="nm">Advertised name</label>
    <div class="inrow">
      <input id="nm" type="text" value="__NAME__" autocomplete="off" spellcheck="false" placeholder="cardcollector">
      <span class="suffix">.local__PORTSUFFIX__</span>
    </div>
    <div class="preview">Address: <code id="prev">__SCHEME__://__NAME__.local__PORTSUFFIX__</code></div>

    <div class="warn">The <code>zeroconf</code> package isn't installed, so nothing is being advertised yet.
       Your name is saved and takes effect once you run <code>pip install zeroconf</code> and restart.</div>

    <div class="row"><button id="saveBtn">Save name</button></div>
    <div id="msg" class="msg"></div>

    <div class="note">
      Letters, numbers and hyphens only &mdash; spaces and other characters become hyphens
      (e.g. &ldquo;My Cards&rdquo; &rarr; <code>my-cards</code>). Saving re-advertises immediately on this
      network; already-open tabs may need a refresh. mDNS/<code>.local</code> only works for devices on the
      <em>same</em> local network &mdash; it doesn't cross networks or reach the internet.
    </div>

    <div class="links">
      <a href="/settings/general">General settings</a>
      <a href="/settings">All settings</a>
    </div>
  </div>
<script>
  var PORTSUFFIX = "__PORTSUFFIX__"; var SCHEME = "__SCHEME__";
  var nm = document.getElementById('nm');
  var prev = document.getElementById('prev');
  function slug(v){ return (v||'').toLowerCase().replace(/[^a-z0-9-]+/g,'-').replace(/-{2,}/g,'-').replace(/^-+|-+$/g,'').slice(0,63); }
  function updatePrev(){ prev.textContent = SCHEME + '://' + (slug(nm.value) || 'cardcollector') + '.local' + PORTSUFFIX; }
  nm.addEventListener('input', updatePrev); updatePrev();

  var msg = document.getElementById('msg');
  function showMsg(t, ok){ msg.textContent = t; msg.className = 'msg show ' + (ok ? 'ok':'err'); }
  document.getElementById('saveBtn').addEventListener('click', async function(){
    var btn = this; btn.disabled = true;
    try {
      var res = await fetch('/settings/network/name', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ name: nm.value })
      });
      var data = await res.json();
      if (res.ok && data.status === 'success') { nm.value = data.name; updatePrev(); showMsg(data.message || 'Saved.', true); }
      else { showMsg(data.message || 'Save failed.', false); }
    } catch(e){ showMsg('Network error: ' + e.message, false); }
    finally { btn.disabled = false; }
  });
</script>
</body>
</html>"""


@app.route("/settings/network")
def network_settings_page():
    """Standalone page to set the advertised .local name (self-contained)."""
    name = get_mdns_name()
    scheme = _MDNS.get("scheme", "http")
    port = _MDNS.get("port") or int(os.environ.get("PORT", "80"))
    default_port = 443 if scheme == "https" else 80
    suffix = "" if port == default_port else f":{port}"
    zc_active = _MDNS.get("zc") is not None
    html = (_NETWORK_SETTINGS_HTML
            .replace("__NAME__", name)
            .replace("__SCHEME__", scheme)
            .replace("__PORTSUFFIX__", suffix)
            .replace("__ZC_WARN__", "none" if zc_active else "block"))
    return Response(html, mimetype="text/html")


@app.route("/settings/network/name", methods=["POST"])
def network_set_name():
    """Persist and live re-advertise the .local name."""
    body = request.get_json(silent=True) or request.form
    name = set_mdns_name(str(body.get("name", "")))
    try:
        restart_mdns(name)
    except Exception:
        pass
    scheme = _MDNS.get("scheme", "http")
    port = _MDNS.get("port") or int(os.environ.get("PORT", "80"))
    default_port = 443 if scheme == "https" else 80
    url = f"{scheme}://{name}.local" + ("" if port == default_port else f":{port}")
    active = _MDNS.get("zc") is not None
    msg = (f"Saved — now advertising at {url}" if active
           else f"Saved as \u201c{name}\u201d. Install 'zeroconf' and restart to advertise it.")
    return jsonify({"status": "success", "name": name, "url": url, "message": msg})


@app.route("/identify/diagnose", methods=["GET"])
def identify_diagnose():
    """Provider-aware self-test. Reports the selected provider and validates its
    API key against the provider's lightweight endpoint (Ximilar account details
    / CardSight health). Visit /identify/diagnose in the browser."""
    provider = _identify_provider()
    result = {
        "selected_provider": provider,
        "selected_provider_label": _identify_provider_label(provider),
        "ximilar_key_set": bool(get_api_key("XIMILAR_API_TOKEN")),
        "cardsight_key_set": bool(get_api_key("CARDSIGHT_API_KEY")),
    }
    if provider == "cardsight":
        ok, detail = _cardsight_health_ok()
        result["key_ok"] = ok
        result["detail"] = detail
    elif provider == "ximilar":
        try:
            data = _ximilar_http("GET", "https://api.ximilar.com/account/v2/details/")
            result["key_ok"] = True
            result["detail"] = "Ximilar token authenticates."
            result["credits_counter"] = (data or {}).get("credits_counter")
        except urllib.error.HTTPError as e:
            result["key_ok"] = False
            result["detail"] = f"HTTP {e.code}: {e.reason}"
        except Exception as e:
            result["key_ok"] = False
            result["detail"] = str(e)
    else:
        result["key_ok"] = None
        result["detail"] = "No external provider selected (local database only)."
    return jsonify(result)
    """
    Self-test for the stored Ximilar API token. Visit /ximilar/diagnose in the
    browser. It tests the app's *stored & sanitized* token against the SAME
    endpoint your curl check uses (https://api.ximilar.com/account/v2/details/),
    and reports a safe fingerprint so you can compare it to your real token
    without exposing it. Purpose: when curl works but the app 401s, this shows
    whether the app is actually using the same token (and whether hidden/zero-width
    characters were stored).
    """
    raw   = get_api_key("XIMILAR_API_TOKEN") or ""
    clean = _ximilar_auth_token()

    def _fingerprint(s):
        if not s:
            return "(empty)"
        return (s[:4] + "…" + s[-4:]) if len(s) >= 10 else f"({len(s)} chars)"

    # Any characters in the RAW stored value that .strip() alone wouldn't remove.
    hidden = sorted({f"U+{ord(c):04X}" for c in raw if (not c.isprintable()) or c.isspace()})

    result = {
        "token_present":             bool(clean),
        "raw_length":                len(raw),
        "clean_length":              len(clean),
        "had_extra_or_hidden_chars": raw != clean,
        "hidden_or_space_codepoints": hidden,
        "clean_fingerprint":         _fingerprint(clean),
        "auth_header_preview":       f"Token {_fingerprint(clean)}",
    }

    if not clean:
        result["account_ok"] = False
        result["account_error"] = "No API token stored. Add it in Settings \u2192 API Keys."
        return jsonify(result)

    try:
        data = _ximilar_http("GET", "https://api.ximilar.com/account/v2/details/")
        result["account_ok"]      = True
        result["account_email"]   = (data or {}).get("email")
        result["credits_counter"] = (data or {}).get("credits_counter")
        result["message"] = ("The stored token authenticates correctly. If card "
                             "identification still 401s, the issue is specific to the "
                             "Collectibles/TCG endpoint access on this account, not the token.")
    except urllib.error.HTTPError as e:
        result["account_ok"]    = False
        result["account_error"] = f"HTTP {e.code}: {e.reason}"
        if e.code in (401, 403):
            result["message"] = ("The stored token was REJECTED, even though your curl test "
                                 "passed — so the value saved in the app is not the same token. "
                                 "Re-copy it into Settings \u2192 API Keys. Compare 'clean_fingerprint' "
                                 "here against the first/last 4 characters of your real token.")
    except Exception as e:
        result["account_ok"]    = False
        result["account_error"] = str(e)

    return jsonify(result)


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


def _set_held(ids, value):
    """Set held=value on the given record ids. Writing through extracted_data
    keeps it the source of truth; the mapper event resyncs the is_held column so
    the Inventory (held) and Sold (not held) lists update accordingly."""
    ids = [int(i) for i in ids if str(i).strip().isdigit()]
    if not ids:
        return 0
    changed = 0
    for r in ScanRecord.query.filter(ScanRecord.id.in_(ids)).all():
        data = dict(r.extracted_data or {})
        if bool(_held_from(data)) == bool(value):
            continue
        data["held"] = bool(value)
        if not value:
            # Marking sold — stamp a sale date (unless an integration sale already
            # recorded one) so the sale lands in the right reporting period.
            data.setdefault("sold_at", datetime.utcnow().isoformat())
        else:
            data.pop("sold_at", None)   # back to held: drop the stale sale date
        r.extracted_data = data   # reassign -> row dirty -> before_update resyncs column
        changed += 1
    if changed:
        db.session.commit()
    return changed


@app.route("/inventory/mark_sold", methods=["POST"])
def inventory_mark_sold():
    """Mark record(s) sold: held -> False, moving them to the Sold page."""
    n = _set_held(_archive_ids_from_request(), False)
    return jsonify({"status": "success", "sold": n,
                    "message": f"Marked {n} record(s) sold."})


@app.route("/inventory/mark_held", methods=["POST"])
def inventory_mark_held():
    """Restore record(s) to Held (unsold), moving them back to Inventory."""
    n = _set_held(_archive_ids_from_request(), True)
    return jsonify({"status": "success", "held": n,
                    "message": f"Restored {n} record(s) to inventory."})


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
    is_held               BOOLEAN DEFAULT TRUE,
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


# app_settings keys that are never written into a migration bundle.
# FLASK_SECRET_KEY signs the session cookie, so anyone holding it can mint a cookie for
# any uid and become an administrator. It is also the one setting the target install has
# no reason to inherit: carrying it across would carry live session cookies with it,
# which is the opposite of what a migration wants. The target generates its own on first
# start (see the __main__ block) and everyone signs in once.
# The marketplace tokens and mailbox password in shop_connections/email_monitors are NOT
# redacted — re-authenticating every shop is a real migration failure, and those are what
# the bundle is for. They are covered by the administrator gate on the export/download
# routes and by the manifest's "treat this bundle as a secret" note.
_MIGRATION_REDACTED_SETTINGS = frozenset({"FLASK_SECRET_KEY"})


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


def _stream_table_jsonl(model, fh, batch=2000, skip=None):
    """Write every row of `model` to an open text file as JSON lines, using
    keyset iteration so memory stays flat for very large tables. `skip`, if given,
    is called with each row and omits it when it returns True; skipped rows are
    not counted, so the manifest's row_counts reports what the file contains."""
    n = 0
    last_id = 0
    has_int_pk = hasattr(model, "id")
    if has_int_pk:
        while True:
            rows = (model.query.filter(model.id > last_id)
                    .order_by(model.id).limit(batch).all())
            if not rows:
                break
            last_id = rows[-1].id          # set before filtering: an all-skipped
            for r in rows:                 # batch must still advance the cursor
                if skip and skip(r):
                    continue
                fh.write(json.dumps(_row_to_dict(r), default=str) + "\n")
                n += 1
            db.session.expunge_all()   # release hydrated objects
    else:
        for r in model.query.all():
            if skip and skip(r):
                continue
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
        skip = None
        if model is AppSetting:
            skip = lambda row: getattr(row, "key", None) in _MIGRATION_REDACTED_SETTINGS
        with open(os.path.join(bundle_dir, f"{stem}.jsonl"), "w", encoding="utf-8") as fh:
            counts[stem] = _stream_table_jsonl(model, fh, skip=skip)

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
            "FLASK_SECRET_KEY is deliberately excluded from app_settings: it signs session "
            "cookies, and the target install generates its own on first start. Everyone "
            "signs in again after the migration — that is expected, not a fault.",
        ],
    }
    with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    out_dir = os.path.join(app.config["UPLOAD_FOLDER"], "migration_exports")
    os.makedirs(out_dir, exist_ok=True)
    # Prune stale bundles so exported secrets don't accumulate on disk.
    _prune_migration_bundles(out_dir)
    tar_path = os.path.join(out_dir, f"ccim_migration_{stamp}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=os.path.basename(bundle_dir))
    shutil.rmtree(work, ignore_errors=True)
    return tar_path, manifest


# Migration bundles are meant to be downloaded promptly and discarded; anything
# older than this is pruned on the next export so exported secrets don't linger.
MIGRATION_BUNDLE_TTL_SECONDS = 3600


def _prune_migration_bundles(out_dir, ttl_seconds=MIGRATION_BUNDLE_TTL_SECONDS, keep=None):
    """Delete migration bundles in out_dir older than ttl_seconds. Best-effort.
    `keep` (a basename) is never deleted — used when pruning during a download so
    the bundle being served can't be removed out from under the request."""
    try:
        cutoff = datetime.now().timestamp() - ttl_seconds
        for name in os.listdir(out_dir):
            if not (name.startswith("ccim_migration_") and name.endswith(".tar.gz")):
                continue
            if keep and name == keep:
                continue
            p = os.path.join(out_dir, name)
            try:
                if os.path.getmtime(p) < cutoff:
                    os.remove(p)
            except OSError:
                pass
    except OSError:
        pass


@app.route("/settings/upgrade")
def upgrade_page():
    return render_template("upgrade.html", capacity=_capacity_status())


@app.route("/settings/upgrade/status")
def upgrade_status():
    return jsonify({"status": "success", "capacity": _capacity_status()})


@app.route("/settings/upgrade/export", methods=["POST"])
def upgrade_export():
    # The route map only requires the 'upgrade' resource here, and the auto-created
    # Editor role holds edit on every resource — so without this the bundle (every
    # marketplace token, the mailbox password) is reachable by a non-admin.
    denied = _require_admin(_BUNDLE_ADMIN_MSG)
    if denied:
        return denied
    include_reference = str(request.form.get("include_reference", "")).lower() in ("1", "true", "yes", "on")
    try:
        tar_path, manifest = _build_migration_bundle(include_reference=include_reference)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Export failed: {exc}"}), 500
    size = os.path.getsize(tar_path)
    log_security_event("migration_bundle_exported", actor=_current_user(),
                       target_name=os.path.basename(tar_path))
    return jsonify({
        "status": "success",
        "message": "Migration bundle created.",
        "download_url": url_for("upgrade_download", name=os.path.basename(tar_path)),
        "filename": os.path.basename(tar_path),
        "size_bytes": size,
        "size_human": _human_size(size),
        "manifest": manifest,
    })


@app.route("/settings/upgrade/download/<name>")
def upgrade_download(name):
    """Serve a migration bundle. Three independent layers stop a crafted name: the
    <name> converter refuses any path with slashes at routing time,
    basename(secure_filename(...)) strips anything that survives, and the resource
    map gates the whole /settings/upgrade prefix ('upgrade' resource, seg[1]) — no
    per-request prefix test to defeat. That map entry is NOT sufficient on its own,
    though: 'upgrade' is an ordinary grantable resource, so administrator status is
    required separately below. Bundles are pruned here too so a single old one
    can't linger past its TTL."""
    denied = _require_admin(_BUNDLE_ADMIN_MSG)
    if denied:
        return denied
    safe = os.path.basename(secure_filename(name))
    out_dir = os.path.join(app.config["UPLOAD_FOLDER"], "migration_exports")
    _prune_migration_bundles(out_dir, keep=safe)
    return _no_sniff(send_from_directory(out_dir, safe))


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
        # drop_all took the FTS sync triggers with the base tables; recreate
        # them (and rebuild the now-stale search index) without waiting for
        # the next restart's migrations.
        migrate_add_search_fts()
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
    return render_template("storage_settings.html", slots=_storage_status(),
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


# Slots whose contents are served over HTTP: /uploads/<path> reads UPLOAD_FOLDER and
# /temp_cards, /temp_split, /temp_pdf read the temp subfolders — all of them to ANY
# signed-in user, which uploaded_file documents as deliberate ("every card image is
# served through it"). So wherever these two roots point, everything underneath them
# becomes downloadable. Relocating them is not merely a storage preference; it moves a
# disclosure boundary. "roi" is excluded because no route serves it: templates are read
# through load_template, which confines them with _within_dir.
_HTTP_SERVED_SLOTS = ("uploads", "temp")


def _storage_target_conflict(slot, target):
    """Reject a storage destination that would publish files which are not card images.
    Returns an operator-facing error string, or None when the target is acceptable.

    One invariant, checked from both directions: a served root must not contain the
    application source or the database, and the database must not be placed inside a
    served root. Without it, slot=uploads&path=. repoints UPLOAD_FOLDER at BASE_DIR and
    GET /uploads/inventory.db hands the whole database — password hashes, API keys,
    marketplace tokens — to any signed-in user.

    Deliberately a containment test rather than a blocklist of scary directories: the
    question is not whether the target looks dangerous, it is whether anything that must
    stay private ends up underneath something the web server will serve."""
    roots = app.config.get("STORAGE_ROOTS", {}) or {}
    db_path = roots.get("db", "")
    if slot in _HTTP_SERVED_SLOTS:
        # _within_dir(base, target) is "target is inside base", so the target root is
        # the base here: we are asking what the new root would come to contain.
        if _within_dir(target, BASE_DIR):
            return ("That location contains the application's own files, which would "
                    "become downloadable by any signed-in user. Pick a dedicated folder.")
        if db_path and _within_dir(target, db_path):
            return ("That location contains the database, which would become "
                    "downloadable by any signed-in user. Pick a dedicated folder.")
    elif slot == "db":
        for served in _HTTP_SERVED_SLOTS:
            root = roots.get(served, "")
            if root and _within_dir(root, target):
                return (f"That location is inside the {served} folder, whose contents are "
                        "downloadable by any signed-in user. Pick a folder outside it.")
    return None


def _move_folder_slot(slot, new_path):
    """Relocate a folder slot (uploads/temp/roi): copy its owned subfolders to
    the new root, delete the originals, then repoint the live + persisted
    config. Applies immediately — every path is read from app.config."""
    roots = app.config.get("STORAGE_ROOTS", {})
    old_root = roots.get(slot, "")
    new_root = _resolve_storage_path(new_path)

    if os.path.abspath(old_root) == os.path.abspath(new_root):
        return {"status": "success", "message": "Location unchanged.", "needs_restart": False}

    # Before makedirs, so a refused target doesn't leave an empty directory behind.
    conflict = _storage_target_conflict(slot, new_root)
    if conflict:
        return {"status": "error", "message": conflict}

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

    # The other direction of the same invariant: don't drop the database inside a
    # folder the web server hands out. Checked before makedirs so a refused target
    # doesn't leave an empty directory behind.
    conflict = _storage_target_conflict("db", new_abs)
    if conflict:
        return {"status": "error", "message": conflict}

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
    # Relocating where the app keeps its data is an operator action, not an editing
    # one. The route map only requires the grantable 'storage' resource, and the
    # setup wizard seeds the Editor role with edit on every resource — so without
    # this a non-admin could repoint UPLOAD_FOLDER and read whatever lands under it.
    denied = _require_admin("Administrator access is required to move a storage location.")
    if denied:
        return denied
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
        # The slot name is a STORAGE_SLOTS member, not caller text. The new path is
        # deliberately not recorded: it is operator-supplied and would put an
        # arbitrary string into the log the admin viewer renders.
        log_security_event("storage_root_changed", actor=_current_user(),
                           target_name=slot)
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
    bad = _reject_if_bomb(file)
    if bad:
        return bad

    np_buf    = np.frombuffer(file.read(), np.uint8)
    query_img = _imdecode(np_buf, cv2.IMREAD_COLOR)
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
# Synchronous card-identification endpoint (returns name/number/set directly),
# used as a fallback when the local OCR can't confidently identify a card.
XIMILAR_TCG_ID_URL      = "https://api.ximilar.com/collectibles/v2/tcg_id"
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
        img = _imread(abs_path)
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


def _ximilar_auth_token():
    """Return the stored Ximilar token, cleaned of copy-paste artifacts that are
    a common cause of 401s: surrounding whitespace/newlines, wrapping quotes, an
    accidental leading 'Token ' (the app adds that prefix itself), and any
    invisible/zero-width/control characters (BOM, zero-width space, non-breaking
    space, stray control bytes) that a web-form paste can inject — these survive a
    plain .strip() and make a token that works in curl fail from the app."""
    tok = (get_api_key("XIMILAR_API_TOKEN") or "").strip().strip('"').strip("'").strip()
    if tok.lower().startswith("token "):
        tok = tok[len("token "):].strip()
    tok = "".join(ch for ch in tok if ch.isprintable() and not ch.isspace())
    return tok


def _ximilar_http(method, url, payload=None):
    """Small JSON HTTP helper for the Ximilar async request API."""
    data = None
    headers = {
        "Authorization": f"Token {_ximilar_auth_token()}",
        "Accept":        "application/json",
        "User-Agent":    "CardCollectorInventoryManager/1.0",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=XIMILAR_CONNECT_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ximilar_identify_enabled():
    """The fallback will actually run only when its toggle is on AND a token is
    set. Use _ximilar_fallback_on() alone to detect the on-but-no-key case."""
    return _ximilar_fallback_on() and bool(get_api_key("XIMILAR_API_TOKEN"))


def _ximilar_identify_card_ex(image_path):
    """Like _ximilar_identify_card, but returns (result_or_None, error_or_None) so
    callers can tell a config/network/image failure (error is a human message)
    from a genuine "Ximilar answered but couldn't identify the card" (result and
    error are both None). Never raises.
    """
    if not get_api_key("XIMILAR_API_TOKEN"):
        return None, ("No Ximilar API key is set. Add your Ximilar API token in "
                      "Settings \u2192 API Keys.")
    src = _image_source_for_grading(image_path)   # downscaled base64 (small payload)
    if not src:
        return None, "This card has no readable front image to send to Ximilar."
    try:
        resp = _ximilar_http("POST", XIMILAR_TCG_ID_URL, {"records": [src]})
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return None, (f"Ximilar refused the request (HTTP {e.code}). This is usually one of: "
                          "the API token is wrong, OR the token is valid but the account has no "
                          "active plan / 0 API credits (card identification costs ~10 credits each). "
                          "Open /ximilar/diagnose to check — if it shows account_ok:true with "
                          "credits_counter:0, activate a plan / add credits in the Ximilar dashboard.")
        if e.code == 402:
            return None, ("Ximilar returned HTTP 402 — the account is out of API credits or has "
                          "no active plan for this service. Check your plan/credits in the Ximilar dashboard.")
        return None, f"Couldn't reach Ximilar: HTTP Error {e.code}: {e.reason}"
    except Exception as e:
        return None, f"Couldn't reach Ximilar: {e}"

    records = (resp or {}).get("records") or []
    if not records:
        return None, None
    rec = records[0] if isinstance(records[0], dict) else {}

    ident = None
    for obj in (rec.get("_objects") or []):
        if isinstance(obj, dict) and isinstance(obj.get("_identification"), dict):
            ident = obj["_identification"]
            break
    if ident is None and isinstance(rec.get("_identification"), dict):
        ident = rec["_identification"]
    if not isinstance(ident, dict):
        return None, None

    best = ident.get("best_match")
    if not isinstance(best, dict) or not best:
        return None, None

    name   = str(best.get("name") or "").strip()
    number = str(best.get("card_number") or "").strip()
    if not name and not number:
        return None, None

    return {
        "name":        name,
        "number":      number,
        "set":         str(best.get("set") or "").strip(),
        "set_code":    str(best.get("set_code") or "").strip(),
        "rarity":      str(best.get("rarity") or "").strip(),
        "series":      str(best.get("series") or "").strip(),
        "year":        best.get("year"),
        "subcategory": str(best.get("subcategory") or best.get("Subcategory") or "").strip(),
    }, None


def _ximilar_identify_card(image_path):
    """Send one front image to Ximilar's TCG identification endpoint and return a
    normalized {name, number, set, set_code, rarity, series, year, subcategory}
    dict — or None on any failure. Best-effort: never raises. (Thin wrapper over
    _ximilar_identify_card_ex for callers that don't need the error detail.)
    """
    result, _ = _ximilar_identify_card_ex(image_path)
    return result


def _apply_ximilar_identification(record, xi, category_id):
    """Apply a Ximilar identification onto `record` (in memory; caller commits).

    Prefers a local reference-catalog match seeded by Ximilar's accurate name +
    number — that path fills canonical fields plus market price and type via the
    normal apply. If the card's set isn't synced (or isn't in the catalog), it
    falls back to writing Ximilar's own fields directly. Returns applied updates
    or {}.
    """
    # 1) Rich path: reuse the reference matcher/apply with Ximilar's clean read.
    if category_id and card_ocr is not None:
        xi_ocr = {"name_guess": xi.get("name", ""), "number_guess": xi.get("number", "")}
        try:
            cands = _reference_candidates_for_ocr(category_id, xi_ocr, limit=3)
        except Exception:
            cands = []
        top = cands[0] if cands else None
        # Same auto-accept bar as local OCR (Settings slider), so raising it makes
        # the provider-seeded catalog re-match stricter too instead of quietly
        # applying at a fixed 60%.
        if top is not None and float(top.get("score", 0) or 0) >= auto_identify_min_score():
            applied = _apply_ocr_candidate(record, top)
            if applied:
                return applied

    # 2) Direct fill from Ximilar's fields (accurate identity even without a
    #    catalog hit). Underscores/casing on game are left to import normalization.
    updates = {}
    for key, val in (("name",       xi.get("name")),
                     ("set_number", xi.get("number")),
                     ("set",        xi.get("set")),
                     ("rarity",     xi.get("rarity"))):
        if str(val or "").strip():
            updates[key] = str(val).strip()
    if not updates:
        return {}
    merged = {**(record.extracted_data or {}), **updates}
    record.extracted_data = merged
    matched = match_product_from_extracted(merged)
    if matched:
        record.matched_product_id = matched.id
    return updates


def _ximilar_identify_candidates(record, category_id):
    """(candidates, error) for the manual-identify picker.

    Returns reference candidates (rich: set/rarity/price, applied via
    reference_product_id) when Ximilar's read resolves a synced catalog card;
    otherwise a single raw candidate (source="ximilar") carrying name/number/set/
    rarity so it can still be applied. `error` carries a human-readable message
    for every non-success outcome (no key, image/network error, or Ximilar
    couldn't identify the card) so the UI never fails silently."""
    if not get_api_key("XIMILAR_API_TOKEN"):
        return [], ("No Ximilar API key is set. Add your Ximilar API token in "
                    "Settings \u2192 API Keys to identify cards with Ximilar.")

    xi, xi_err = _ximilar_identify_card_ex(record.image_path)
    if xi_err:
        return [], xi_err
    if not xi:
        return [], "Ximilar read the front image but couldn't confidently identify this card."

    cands = []
    if category_id and card_ocr is not None:
        try:
            refs = _reference_candidates_for_ocr(
                category_id,
                {"name_guess": xi.get("name", ""), "number_guess": xi.get("number", "")},
                limit=3,
            )
        except Exception:
            refs = []
        _min = auto_identify_min_score()
        for r in refs:
            if float(r.get("score", 0) or 0) >= _min:
                cands.append({**r, "via": "ximilar"})

    if not cands:
        cands.append({
            "source":          "ximilar",
            "via":             "ximilar",
            "name":            xi.get("name", ""),
            "serial":          xi.get("number", ""),
            "set":             xi.get("set", ""),
            "rarity":          xi.get("rarity", ""),
            "game":            (record.extracted_data or {}).get("game", ""),
            "score":           0.95,
            "serial_match":    False,
            "name_similarity": 1.0,
        })
    return cands, None


# ============================================================================ #
# CardSight AI — image-based identification (free tier: 750 calls/month).
#   Auth:   header  X-API-Key: <32-char alphanumeric key>
#   Ident:  POST https://api.cardsight.ai/v1/identify/card  (multipart 'image')
#   Health: GET  https://api.cardsight.ai/v1/health
# One endpoint covers sports + Pokémon (and MTG as it rolls out). Response:
#   { success, detections:[ { confidence, card:{ name, number, setName,
#     releaseName, year, fields:[{key,value}], ... } } ] }
# ============================================================================ #
CARDSIGHT_IDENTIFY_URL = "https://api.cardsight.ai/v1/identify/card"
CARDSIGHT_HEALTH_URL   = "https://api.cardsight.ai/v1/health"
CARDSIGHT_TIMEOUT      = 40


def _cardsight_auth_key():
    """Stored CardSight key, cleaned of whitespace/quotes/invisible characters.
    CardSight keys are 32-char alphanumeric and their docs send them WITHOUT
    hyphens, so we strip hyphens too."""
    key = (get_api_key("CARDSIGHT_API_KEY") or "").strip().strip('"').strip("'").strip()
    key = key.replace("-", "")
    return "".join(ch for ch in key if ch.isprintable() and not ch.isspace())


def _downscaled_jpeg_bytes(path_value, max_side=1600, jpeg_quality=90):
    """Return JPEG bytes for a stored image (downscaled for a small upload), or
    None. Used for multipart providers like CardSight. Local files are decoded
    and re-encoded; external URLs are fetched as-is."""
    relative = normalize_to_upload_relative(path_value)
    if not relative or relative == "__blank__":
        return None
    if relative.startswith("http://") or relative.startswith("https://"):
        try:
            with urllib.request.urlopen(relative, timeout=30) as r:
                return r.read()
        except Exception:
            return None
    abs_path = os.path.join(app.config["UPLOAD_FOLDER"], relative)
    if not os.path.exists(abs_path):
        return None
    try:
        img = _imread(abs_path)
        if img is not None:
            h, w = img.shape[:2]
            scale = max_side / float(max(h, w)) if max(h, w) > max_side else 1.0
            if scale < 1.0:
                img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if ok:
                return buf.tobytes()
    except Exception:
        pass
    try:
        with open(abs_path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _multipart_encode(field_name, filename, content_type, data):
    """Minimal multipart/form-data encoder (one file part) so we stay dependency
    free. Returns (content_type_header, body_bytes)."""
    boundary = "----CCIMBoundary" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    crlf = b"\r\n"
    body = b"".join([
        b"--", boundary.encode(), crlf,
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode(), crlf,
        f"Content-Type: {content_type}".encode(), crlf, crlf,
        data, crlf,
        b"--", boundary.encode(), b"--", crlf,
    ])
    return f"multipart/form-data; boundary={boundary}", body


def _cardsight_identify_card_ex(image_path):
    """Identify one front image with CardSight. Returns (normalized_dict|None,
    error|None) with the same shape the Ximilar path uses. Never raises."""
    key = _cardsight_auth_key()
    if not key:
        return None, ("No CardSight API key is set. Add your CardSight API key in "
                      "Settings \u2192 API Keys (free tier: 750 identifications/month).")
    data = _downscaled_jpeg_bytes(image_path)
    if not data:
        return None, "This card has no readable front image to send to CardSight."

    content_type, body = _multipart_encode("image", "card.jpg", "image/jpeg", data)
    req = urllib.request.Request(
        CARDSIGHT_IDENTIFY_URL, data=body, method="POST",
        headers={
            "X-API-Key":    key,
            "Content-Type": content_type,
            "Accept":       "application/json",
            "User-Agent":   "CardCollectorInventoryManager/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=CARDSIGHT_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return None, (f"CardSight rejected the API key (HTTP {e.code}). Check the key in "
                          "Settings \u2192 API Keys — it's the 32-character key from your CardSight "
                          "dashboard.")
        if e.code == 402:
            return None, ("CardSight returned HTTP 402 — this account is out of API calls for the "
                          "period. Check your CardSight plan/usage.")
        if e.code == 429:
            return None, ("CardSight rate limit reached (HTTP 429) — you may have used the free "
                          "monthly calls, or sent requests too quickly. Try again later.")
        return None, f"Couldn't reach CardSight: HTTP Error {e.code}: {e.reason}"
    except Exception as e:
        return None, f"Couldn't reach CardSight: {e}"

    if not isinstance(payload, dict):
        return None, "CardSight returned an unexpected response."
    # Some responses include an explicit error/message; surface it.
    if payload.get("error"):
        return None, f"CardSight: {payload.get('error')}"
    detections = payload.get("detections") or []
    if not detections:
        return None, None
    det  = detections[0] if isinstance(detections[0], dict) else {}
    card = det.get("card") or {}

    name   = str(card.get("name") or "").strip()
    number = str(card.get("number") or "").strip()
    if not name and not number:
        return None, None

    rarity = ""
    for f in (card.get("fields") or []):
        if isinstance(f, dict) and str(f.get("key") or f.get("name") or "").upper() == "RARITY":
            rarity = str(f.get("value") or "").strip()
            break

    return {
        "name":        name,
        "number":      number,
        "set":         str(card.get("setName") or "").strip(),
        "set_code":    "",
        "rarity":      rarity,
        "series":      str(card.get("releaseName") or "").strip(),
        "year":        card.get("year"),
        "subcategory": "",
        "confidence":  str(det.get("confidence") or "").strip(),
    }, None


def _cardsight_health_ok():
    """(ok, detail) — validate the CardSight key against its health endpoint."""
    key = _cardsight_auth_key()
    if not key:
        return False, "No CardSight API key is set."
    req = urllib.request.Request(
        CARDSIGHT_HEALTH_URL, method="GET",
        headers={"X-API-Key": key, "Accept": "application/json",
                 "User-Agent": "CardCollectorInventoryManager/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
        return True, "CardSight API key is valid."
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return False, str(e)


def _cardsight_debug(image_path):
    """Raw CardSight identify exchange for troubleshooting: returns the HTTP status
    and the parsed JSON (or body text / error). Used by /cloud_identify?debug=1."""
    key = _cardsight_auth_key()
    if not key:
        return {"error": "No CardSight API key is set."}
    data = _downscaled_jpeg_bytes(image_path)
    if not data:
        return {"error": "No front-image bytes could be read for this record."}
    out = {"sent_image_bytes": len(data), "key_fingerprint": (key[:4] + "…" + key[-4:]) if len(key) >= 10 else f"({len(key)} chars)"}
    content_type, body = _multipart_encode("image", "card.jpg", "image/jpeg", data)
    req = urllib.request.Request(
        CARDSIGHT_IDENTIFY_URL, data=body, method="POST",
        headers={"X-API-Key": key, "Content-Type": content_type, "Accept": "application/json",
                 "User-Agent": "CardCollectorInventoryManager/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=CARDSIGHT_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", "replace")
        out["http_status"] = 200
        try:
            out["json"] = json.loads(raw)
        except Exception:
            out["text"] = raw[:3000]
    except urllib.error.HTTPError as e:
        out["http_status"] = e.code
        out["reason"] = e.reason
        try:
            out["body"] = e.read().decode("utf-8", "replace")[:3000]
        except Exception:
            pass
    except Exception as e:
        out["error"] = str(e)
    return out


# --------------------------------------------------------------------------- #
# Unified provider dispatch — both the import auto-identify and the manual
# inventory-detail identify go through these, honouring the selected provider.
# --------------------------------------------------------------------------- #
# Applying a normalized identity ({name, number, set, rarity}) is provider-
# agnostic, so both providers reuse the same apply function.
_apply_external_identification = _apply_ximilar_identification


def _external_identify_card_ex(image_path):
    """Identify a front image using the SELECTED provider. Returns
    (normalized_dict|None, error|None). Provider 'none' -> (None, None)."""
    provider = _identify_provider()
    if provider == "cardsight":
        return _cardsight_identify_card_ex(image_path)
    if provider == "ximilar":
        return _ximilar_identify_card_ex(image_path)
    return None, None


def _external_identify_candidates(record, category_id):
    """(candidates, error) for the Cloud Identification button, using the
    selected provider.

    The result comes ENTIRELY from the cloud provider — the local reference
    catalog is deliberately not consulted. Cloud Identification is the "ask the
    service directly" action, kept independent of the reference database so it
    is a genuine second opinion when the local lookup is wrong or has no data
    for the game. (`category_id` is accepted for call-signature compatibility
    and intentionally unused.)
    """
    provider = _identify_provider()
    if provider == "none":
        return [], ("No external identification service is selected. Choose Ximilar or CardSight "
                    "in Settings \u2192 General.")
    label = _identify_provider_label(provider)

    xi, err = _external_identify_card_ex(record.image_path)
    if err:
        return [], err
    if not xi:
        return [], f"{label} read the front image but couldn't confidently identify this card."

    return [{
        "source":          provider,      # "ximilar" | "cardsight"
        "via":             provider,
        "provider_label":  label,
        "name":            xi.get("name", ""),
        "serial":          xi.get("number", ""),
        "set":             xi.get("set", ""),
        "rarity":          xi.get("rarity", ""),
        "game":            (record.extracted_data or {}).get("game", ""),
        # The provider reports an identification, not a similarity score; this is
        # a display value only and never feeds the reference-data ranking rules.
        "score":           1.0,
        "serial_match":    False,
        "name_similarity": 1.0,
    }], None


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

# ====================== SHIPPING (labels + tracking) ======================
# Imported at module level so shipping_models' tables are registered on
# db.metadata before init_db()'s create_all() runs. The /shipping/* routes
# inherit the Shops permission via the "shipping": "shops" entry in
# _resource_for_path's map.
import shipping_models  # noqa: F401  (registers Order/OrderItem/Shipment)
from shipping_routes import shipping_bp, ensure_order_from_sale, start_tracking_poller

app.register_blueprint(shipping_bp)

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


EBAY_TEMPLATE_KEY = "EBAY_DESCRIPTION_TEMPLATE"

# Default eBay listing description. Uses {{token}} placeholders filled per item.
# Any entry field works as a token (e.g. {{name}}, {{set}}, {{number}}, {{rarity}},
# {{game}}, {{edition}}, {{holographic}}, {{language}}); plus computed tokens
# {{title}}, {{condition}}, {{price}}, {{image}}, {{images}}, {{image_url}}.
_DEFAULT_EBAY_TEMPLATE = """<div style="font-family:Arial,Helvetica,sans-serif;max-width:820px;margin:auto;color:#222;">
  <h1 style="font-size:22px;margin:0 0 10px;">{{title}}</h1>
  <div style="text-align:center;margin:12px 0;">{{image}}</div>
  <table style="border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;">
    <tr><td style="padding:7px 10px;border:1px solid #ddd;background:#f7f7f7;font-weight:bold;width:180px;">Game</td><td style="padding:7px 10px;border:1px solid #ddd;">{{game}}</td></tr>
    <tr><td style="padding:7px 10px;border:1px solid #ddd;background:#f7f7f7;font-weight:bold;">Set</td><td style="padding:7px 10px;border:1px solid #ddd;">{{set}}</td></tr>
    <tr><td style="padding:7px 10px;border:1px solid #ddd;background:#f7f7f7;font-weight:bold;">Card Number</td><td style="padding:7px 10px;border:1px solid #ddd;">{{number}}</td></tr>
    <tr><td style="padding:7px 10px;border:1px solid #ddd;background:#f7f7f7;font-weight:bold;">Rarity</td><td style="padding:7px 10px;border:1px solid #ddd;">{{rarity}}</td></tr>
    <tr><td style="padding:7px 10px;border:1px solid #ddd;background:#f7f7f7;font-weight:bold;">Condition</td><td style="padding:7px 10px;border:1px solid #ddd;">{{condition}}</td></tr>
  </table>
  <p style="line-height:1.5;">{{name}} &mdash; a great addition to any collection. Ships securely in a protective sleeve and top-loader with tracking.</p>
  <p style="color:#888;font-size:12px;border-top:1px solid #eee;padding-top:8px;">Listed with Card Collector Inventory Manager.</p>
</div>"""


def _ebay_template():
    return get_setting(EBAY_TEMPLATE_KEY, "") or _DEFAULT_EBAY_TEMPLATE


def _ebay_context(record):
    """Build the {{token}} -> value map for one record. Text values are stored raw
    and HTML-escaped at substitution time; image tokens are pre-built safe HTML."""
    from markupsafe import escape as _esc
    data = record.extracted_data or {}
    ctx = {}
    for k, v in data.items():
        if str(k).startswith("__"):
            continue
        if isinstance(v, (str, int, float)):
            ctx[str(k)] = v

    name = _record_display_name(record)
    game = data.get("game", "")
    serial = _get_serial(data)
    title = name
    if game:
        title = f"{name} - {game}"
    if serial:
        title = f"{title} #{serial}"
    grade = ((data.get("grading") or {}).get("front") or {}).get("label") or ""
    price = _record_market_price(record)

    try:
        urls, _b64 = _record_image_sources(record)
    except Exception:
        urls = []

    ctx.update({
        "name": name,
        "title": title,
        "condition": grade or ctx.get("condition", ""),
        "grade": grade,
        "price": (format(price, ",.2f") if isinstance(price, (int, float)) and price else (price or "")),
        "image_url": urls[0] if urls else "",
    })
    img_style = 'max-width:100%;height:auto;border-radius:6px;margin:4px;'
    ctx["image"] = (f'<img src="{_esc(urls[0])}" alt="{_esc(name)}" style="{img_style}">' if urls else "")
    ctx["images"] = "".join(f'<img src="{_esc(u)}" alt="{_esc(name)}" style="{img_style}">' for u in urls)
    return ctx


_TOKEN_RE = _re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
_EBAY_HTML_TOKENS = {"image", "images"}


def _render_tokens(template, ctx):
    """Replace {{token}} with ctx values. Non-HTML tokens are HTML-escaped so a
    value can never break the markup; {{image}}/{{images}} are inserted as-is."""
    from markupsafe import escape as _esc

    def repl(m):
        key = m.group(1)
        if key not in ctx:
            return ""
        if key in _EBAY_HTML_TOKENS:
            return str(ctx[key])
        return str(_esc(str(ctx[key])))

    return _TOKEN_RE.sub(repl, template)


def _render_ebay_description(record):
    """Rendered HTML description for a record, or None to fall back to the default."""
    try:
        return _render_tokens(_ebay_template(), _ebay_context(record))
    except Exception:
        return None


def _ebay_template_tokens(sample_record=None):
    """List of available tokens for the editor UI: computed ones + fields present
    on a sample record."""
    base = ["title", "name", "condition", "grade", "price", "image", "images", "image_url",
            "game", "set", "number", "rarity", "edition", "holographic", "language"]
    seen = set(base)
    extra = []
    if sample_record is not None:
        for k in (sample_record.extracted_data or {}).keys():
            k = str(k)
            if not k.startswith("__") and k not in seen:
                extra.append(k)
                seen.add(k)
    return base + sorted(extra)


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
    # eBay supports a full HTML description — render the (customizable) listing
    # template filled with this item's details. Falls back to the plain text above.
    if marketplace == "ebay":
        rendered = _render_ebay_description(record)
        if rendered:
            description = rendered

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


def _ebay_sample_record():
    from sqlalchemy import func as _f, and_
    return (ScanRecord.query.filter(and_(
        _f.coalesce(ScanRecord.is_catalog, False) == False,   # noqa: E712
        _f.coalesce(ScanRecord.is_archived, False) == False,  # noqa: E712
    )).order_by(ScanRecord.scan_date.desc()).first())


_EBAY_TEMPLATE_PAGE_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>eBay listing template</title>
<link href="https://cdn.jsdelivr.net/npm/grapesjs/dist/css/grapes.min.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/grapesjs-preset-webpage/dist/grapesjs-preset-webpage.min.css" rel="stylesheet">
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2937;background:#f5f7fb;margin:0;padding:16px;}
 .bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px;}
 .bar h1{font-size:19px;margin:0;margin-right:auto;}
 button{background:#4f46e5;color:#fff;border:0;border-radius:9px;padding:9px 15px;font-weight:700;cursor:pointer;font-size:14px;}
 button.sec{background:#eef2ff;color:#4338ca;} button.warn{background:#fef3c7;color:#92400e;}
 #gjs{border:1px solid #d1d5db;border-radius:10px;overflow:hidden;}
 .msg{margin:10px 0;padding:9px 12px;border-radius:9px;display:none;font-size:14px;}
 .msg.show{display:block;} .msg.ok{background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;} .msg.err{background:#fef2f2;color:#991b1b;border:1px solid #fecaca;}
 .hint{color:#6b7280;font-size:13px;margin:8px 0;line-height:1.5;}
 .links{margin-top:14px;font-size:14px;color:#6b7280;} .links a{margin-right:14px;}
 #rawWrap{display:none;} #raw{width:100%;height:520px;box-sizing:border-box;border:1px solid #d1d5db;border-radius:10px;padding:12px;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;}
 .modal{position:fixed;inset:0;background:rgba(0,0,0,.5);display:none;align-items:center;justify-content:center;z-index:9999;}
 .modal.show{display:flex;} .modal .box{background:#fff;border-radius:12px;width:min(720px,92vw);max-height:88vh;overflow:hidden;display:flex;flex-direction:column;}
 .modal .box header{padding:12px 16px;font-weight:700;border-bottom:1px solid #eee;display:flex;justify-content:space-between;align-items:center;}
 .modal iframe{border:0;width:100%;height:70vh;}
</style></head><body>
 <div class=bar>
   <h1>eBay listing template</h1>
   <button id=save>Save</button>
   <button id=previewBtn class=sec>Preview with data</button>
   <button id=htmlBtn class=sec>Edit HTML</button>
   <button id=reset class=warn>Reset to default</button>
 </div>
 <p class=hint>Drag <b>Blocks</b> (right panel) onto the canvas: text, images, columns, and the <b>Item tokens</b> that fill from each entry (photo, name, set, price...). Move blocks by dragging; edit text by double-clicking. Drop an image block and upload or drag your own logo or banner. Everything is saved as the HTML description used for eBay listings.</p>
 <div id=msg class=msg></div>
 <div id=gjs></div>
 <div id=rawWrap>
   <p class=hint>Raw HTML (advanced). Edits here apply back to the visual editor when you switch back.</p>
   <textarea id=raw spellcheck=false></textarea>
 </div>
 <div class=links><a href="/settings">All settings</a><a href="/inventory/builder">Builder</a></div>

 <div class=modal id=pvModal><div class=box>
   <header><span>Preview (filled with a sample item)</span><button class=sec id=pvClose>Close</button></header>
   <iframe id=pvFrame sandbox="allow-same-origin"></iframe>
 </div></div>

<script src="https://cdn.jsdelivr.net/npm/grapesjs"></script>
<script src="https://cdn.jsdelivr.net/npm/grapesjs-preset-webpage"></script>
<script>
 var msg=document.getElementById('msg');
 function show(t,ok){msg.textContent=t;msg.className='msg show '+(ok?'ok':'err');setTimeout(function(){msg.className='msg';},4000);}
 var DETAILS_TABLE='<table style="border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;">'
   +'<tr><td style="padding:7px 10px;border:1px solid #ddd;background:#f7f7f7;font-weight:bold;width:170px;">Game</td><td style="padding:7px 10px;border:1px solid #ddd;">{{game}}</td></tr>'
   +'<tr><td style="padding:7px 10px;border:1px solid #ddd;background:#f7f7f7;font-weight:bold;">Set</td><td style="padding:7px 10px;border:1px solid #ddd;">{{set}}</td></tr>'
   +'<tr><td style="padding:7px 10px;border:1px solid #ddd;background:#f7f7f7;font-weight:bold;">Card Number</td><td style="padding:7px 10px;border:1px solid #ddd;">{{number}}</td></tr>'
   +'<tr><td style="padding:7px 10px;border:1px solid #ddd;background:#f7f7f7;font-weight:bold;">Rarity</td><td style="padding:7px 10px;border:1px solid #ddd;">{{rarity}}</td></tr>'
   +'<tr><td style="padding:7px 10px;border:1px solid #ddd;background:#f7f7f7;font-weight:bold;">Condition</td><td style="padding:7px 10px;border:1px solid #ddd;">{{condition}}</td></tr></table>';
 var editor=null, rawMode=false;

 function buildTemplate(){
   if(rawMode) return document.getElementById('raw').value;
   var css=(editor.getCss()||'').trim();
   var html=editor.getHtml();
   return css ? ('<style>'+css+'</style>'+html) : html;
 }
 function loadIntoEditor(t){
   t=(t||'').trim(); var css='';
   if(t.toLowerCase().indexOf('<style>')===0){var end=t.toLowerCase().indexOf('</style>'); if(end>0){css=t.slice(7,end); t=t.slice(end+8);}}
   editor.setComponents(t);
   if(css){try{editor.setStyle(css);}catch(e){}}
 }
 async function api(url,payload){var r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload||{})});return r.json();}

 function startEditor(tokens, template){
   editor=grapesjs.init({container:'#gjs',height:'620px',fromElement:false,storageManager:false,
     assetManager:{upload:'/shops/ebay/template/asset',uploadName:'files',autoAdd:true,credentials:'same-origin'},
     plugins:['grapesjs-preset-webpage'],pluginsOpts:{'grapesjs-preset-webpage':{}}});
   var bm=editor.BlockManager;
   bm.add('tok-item-photo',{label:'Item Photo',category:'Item tokens',content:'<div style="text-align:center;margin:8px 0;">{{image}}</div>'});
   bm.add('tok-all-photos',{label:'All Photos',category:'Item tokens',content:'<div>{{images}}</div>'});
   bm.add('tok-details',{label:'Details Table',category:'Item tokens',content:DETAILS_TABLE});
   bm.add('tok-title',{label:'Title',category:'Item tokens',content:'<h1 style="font-size:22px;">{{title}}</h1>'});
   bm.add('tok-price',{label:'Price',category:'Item tokens',content:'<div style="font-size:18px;font-weight:bold;">${{price}}</div>'});
   (tokens||[]).forEach(function(t){ if(['image','images','image_url','title','price'].indexOf(t)>=0)return;
     bm.add('tok-'+t,{label:'{{'+t+'}}',category:'Item tokens',content:'<span>{{'+t+'}}</span>'}); });
   loadIntoEditor(template);
 }

 (async function(){
   var d=await (await fetch('/shops/ebay/template/data',{headers:{'X-Requested-With':'XMLHttpRequest'}})).json();
   var template=d.template||'';
   if(typeof grapesjs==='undefined'){
     document.getElementById('gjs').style.display='none';
     document.getElementById('htmlBtn').style.display='none';
     document.getElementById('previewBtn').style.display='none';
     rawMode=true; document.getElementById('rawWrap').style.display='block';
     document.getElementById('raw').value=template;
     show('Visual editor could not load (offline?). Editing raw HTML instead.',false);
   } else { startEditor(d.tokens||[], template); }
 })();

 document.getElementById('save').addEventListener('click',async function(){var d=await api('/shops/ebay/template/save',{template:buildTemplate()});show(d.status==='success'?'Saved.':(d.message||'Save failed.'),d.status==='success');});
 document.getElementById('reset').addEventListener('click',async function(){if(!confirm('Reset to the default template? Your custom design will be replaced.'))return;var d=await api('/shops/ebay/template/reset',{});if(d.status==='success'){if(rawMode){document.getElementById('raw').value=d.template;}else{loadIntoEditor(d.template);}show('Reset to default.',true);}});
 document.getElementById('htmlBtn').addEventListener('click',function(){var raw=document.getElementById('raw'),wrap=document.getElementById('rawWrap'),gjs=document.getElementById('gjs');if(!rawMode){raw.value=buildTemplate();wrap.style.display='block';gjs.style.display='none';rawMode=true;this.textContent='Visual editor';}else{loadIntoEditor(raw.value);wrap.style.display='none';gjs.style.display='';rawMode=false;this.textContent='Edit HTML';}});
 document.getElementById('previewBtn').addEventListener('click',async function(){var d=await api('/shops/ebay/template/preview',{template:buildTemplate()});document.getElementById('pvFrame').srcdoc=d.html||'';document.getElementById('pvModal').classList.add('show');});
 document.getElementById('pvClose').addEventListener('click',function(){document.getElementById('pvModal').classList.remove('show');});
</script></body></html>"""


@app.route("/shops/ebay/template")
def ebay_template_page():
    return Response(_EBAY_TEMPLATE_PAGE_HTML, mimetype="text/html")


@app.route("/shops/ebay/template/data")
def ebay_template_data():
    sample = _ebay_sample_record()
    return jsonify({"status": "ok", "template": _ebay_template(),
                    "is_default": not bool(get_setting(EBAY_TEMPLATE_KEY, "")),
                    "tokens": _ebay_template_tokens(sample),
                    "has_sample": sample is not None})


@app.route("/shops/ebay/template/save", methods=["POST"])
def ebay_template_save():
    body = request.get_json(silent=True) or request.form
    tpl = str(body.get("template", ""))
    if len(tpl) > 200_000:
        return jsonify({"status": "error", "message": "Template too large (200 KB max)."}), 400
    set_setting(EBAY_TEMPLATE_KEY, tpl)
    return jsonify({"status": "success"})


@app.route("/shops/ebay/template/reset", methods=["POST"])
def ebay_template_reset():
    set_setting(EBAY_TEMPLATE_KEY, "")   # empty -> the built-in default is used
    return jsonify({"status": "success", "template": _DEFAULT_EBAY_TEMPLATE})


@app.route("/shops/ebay/template/preview", methods=["POST"])
def ebay_template_preview():
    body = request.get_json(silent=True) or {}
    tpl = str(body.get("template", "")) or _ebay_template()
    sample = _ebay_sample_record()
    if sample is not None:
        ctx = _ebay_context(sample)
    else:
        ctx = {"title": "Charizard - Pokemon #4", "name": "Charizard", "game": "Pokemon",
               "set": "Base Set", "number": "4", "rarity": "Holo Rare", "condition": "Near Mint",
               "price": "350.00", "image": "", "images": "", "image_url": ""}
    try:
        html = _render_tokens(tpl, ctx)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Template error: {exc}"}), 400
    return jsonify({"status": "ok", "html": html})


@app.route("/shops/ebay/template/asset", methods=["POST"])
def ebay_template_asset():
    """Store images dragged/uploaded in the visual editor; return GrapesJS-shaped
    JSON ({data:[{src}]}) so they appear in the asset manager with a real URL."""
    ensure_dirs()
    files = request.files.getlist("files")
    if not files:
        one = request.files.get("file") or request.files.get("files[]")
        files = [one] if one else []
    sub = os.path.join(app.config["UPLOAD_FOLDER"], "ebay_assets")
    os.makedirs(sub, exist_ok=True)
    saved = []
    for f in files:
        if not f or not f.filename:
            continue
        bad = _reject_if_bomb(f)
        if bad:
            return bad
        ext = os.path.splitext(secure_filename(f.filename))[1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
            return jsonify({"status": "error",
                            "message": "Only image files (PNG, JPG, GIF, WebP, BMP) are allowed."}), 415
        try:
            f.stream.seek(0)
            real = _peek_image_size(f.stream)
        except Exception:
            real = None
        finally:
            try:
                f.stream.seek(0)
            except Exception:
                pass
        if real is None:
            return jsonify({"status": "error", "message": "That file isn't a readable image."}), 415
        name = datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + ext
        f.save(os.path.join(sub, name))
        saved.append({"src": url_for("uploaded_file", filename=f"ebay_assets/{name}")})
    return jsonify({"data": saved})


@app.route("/shops/ebay/listing_csv", methods=["POST"])
def ebay_listing_csv():
    """eBay bulk-listing CSV (File Exchange style). Description is the rendered
    template per item. Accepts {record_ids:[...]} or Builder {groups:[...]}.
    Category / ConditionID are left for you to set for your account/category."""
    import csv as _csv
    import io as _io
    body = request.get_json(silent=True) or {}
    ids = list(body.get("record_ids") or [])
    if not ids and body.get("groups"):
        ids = [i for _l, i in _builder_flatten(body["groups"])]
    ids = [int(i) for i in ids if str(i).strip().lstrip("-").isdigit()]
    if not ids:
        return jsonify({"status": "error", "message": "No records specified."}), 400

    recs = {r.id: r for r in ScanRecord.query.filter(ScanRecord.id.in_(ids)).all()}
    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["Action(SiteID=US|Country=US|Currency=USD|Version=1193)", "CustomLabel",
                "Category", "Title", "Description", "ConditionID", "PicURL",
                "Quantity", "Format", "StartPrice", "Duration"])
    for i in ids:
        r = recs.get(i)
        if not r:
            continue
        p = _build_payload(r, "ebay")
        w.writerow([_csv_safe(c) for c in [
            "Add",
            p.get("sku", ""),
            "",                                   # Category — set for your account
            (p.get("title") or "")[:80],
            p.get("description", ""),             # rendered HTML template (kept as-is)
            "",                                   # ConditionID — set per category
            ";".join(p.get("image_urls") or []),
            p.get("quantity", 1),
            "FixedPrice",
            p.get("price") if p.get("price") is not None else "",
            "GTC",
        ]])
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="ebay_listings_{stamp}.csv"'})


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
    # Catalog rows drop out in SQL (is_catalog mirrors _is_catalog_only by
    # derivation); the "empty slot" flag has no derived column, so that check
    # stays in Python on the narrowed set.
    from sqlalchemy import func as _f
    records = (ScanRecord.query
               .filter(_f.coalesce(ScanRecord.is_catalog, False) == False)  # noqa: E712
               .order_by(ScanRecord.scan_date.desc(), ScanRecord.id.desc()).all())
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

    host_changed = False
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
        # Compare before the write, or the old value is already gone.
        if field.get("host") and str(cfg.get(k, "") or "") != submitted:
            host_changed = True
        cfg[k] = submitted

    conn.config = cfg
    if host_changed:
        # Saving a secret and saving a destination are different acts. Only an
        # administrator can connect a shop (/shops/test is _require_admin-gated, and it
        # is the sole thing that sets enabled=True), but this route is shops:edit — so
        # without this a non-admin could keep the stored token, repoint store_domain at
        # a host they control, and inherit the administrator's "connected" state. Every
        # replay site then sends the token to them: push, pull, unlist, sale sync.
        # Un-connecting here revokes that inheritance at the source, which is why this
        # is the fix rather than adding _require_admin to each replay route — the next
        # route someone adds is covered too.
        conn.enabled = False
        conn.status = "disconnected"
        conn.status_detail = "Destination changed — test the connection again before syncing."
        conn.connected_at = None
    conn.updated_at = datetime.utcnow()
    db.session.commit()
    msg = f"{MARKETPLACES[marketplace]['label']} settings saved."
    if host_changed:
        msg += " The destination changed, so the shop was disconnected — test it again to re-enable syncing."
    return jsonify({"status": "success", "message": msg})


@app.route("/shops/test/<marketplace>", methods=["POST"])
def shops_test(marketplace):
    denied = _require_admin()
    if denied:
        return denied
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


# Session key holding the one-time OAuth `state` nonce for the eBay handshake.
# It lives in the session (a signed cookie) rather than the database because it
# must identify THIS BROWSER, not this installation: the whole point is to prove
# the callback belongs to a flow that this session started.
_EBAY_STATE_KEY = "ebay_oauth_state"


@app.route("/shops/ebay/connect")
def shops_ebay_connect():
    conn = _get_connection("ebay", create=True)
    provider = get_provider("ebay", conn, persist=_shop_persist)
    if provider._need("client_id", "ru_name"):
        return redirect(url_for("shops_page") + "?ebay_error=Set+App+ID+and+RuName+first")
    # A fresh unguessable nonce per handshake. The previous value was the constant
    # "ebay", which authenticates nothing: every installation sent the same one, so
    # a callback carrying it proved only that the attacker had read this source.
    state = _secrets.token_urlsafe(32)
    session[_EBAY_STATE_KEY] = state
    return redirect(provider.authorize_url(state=state))


@app.route("/shops/ebay/callback")
def shops_ebay_callback():
    code = request.args.get("code", "")
    err = request.args.get("error_description") or request.args.get("error")
    if err:
        # The handshake ended at eBay's end; drop the nonce rather than leave a live
        # one waiting in the session.
        session.pop(_EBAY_STATE_KEY, None)
        return redirect(url_for("shops_page") + "?ebay_error=" + urllib.parse.quote(err))
    if not code:
        return redirect(url_for("shops_page") + "?ebay_error=No+authorization+code+returned")

    # Only exchange a code that came back from a handshake THIS session started.
    # Without this the callback accepted any code from anyone: an attacker who
    # obtained an authorization code for their own eBay account only had to get a
    # signed-in shops:edit user to load this URL, and the app would store the
    # attacker's tokens as the installation's connection — after which every push
    # sent this inventory to the attacker's store.
    #
    # pop, not get: a callback URL is single-use, so replaying the same one cannot
    # connect a second time. compare_digest because this is an equality check on a
    # secret. SESSION_COOKIE_SAMESITE is "Lax", and the return from eBay is a
    # top-level GET navigation, so the session cookie is still sent — which is what
    # makes a session-held nonce workable here at all.
    expected = session.pop(_EBAY_STATE_KEY, "")
    got = request.args.get("state", "")
    if not expected or not got or not _hmac.compare_digest(str(expected), str(got)):
        return redirect(url_for("shops_page") + "?ebay_error=" + urllib.parse.quote(
            "This eBay sign-in didn't start from this browser. Open the Shops page and connect again."))

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
    # `enabled`, not just `conn`. This is the one token-replay route that tested only
    # for the row's existence: push (shops_push), pull (shops_pull) and the sale
    # fan-out (_mark_record_sold) all require enabled, and a disabled connection is
    # exactly what "an administrator has not vouched for this destination" means now
    # that changing store_domain clears the flag. Without this line, disconnecting on
    # host change would still leave end_listing willing to send the token there.
    if not conn or not conn.enabled:
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
    writer.writeheader()   # fieldnames are a module constant, so the header is literal
    for row in rows:
        # Product Line / Set Name / Product Name come from extracted_data via
        # _tcgplayer_rows, so every cell is user-influenced. Keys are left alone so
        # extrasaction="ignore" still matches on fieldnames.
        writer.writerow({k: _csv_safe(v) for k, v in row["csv"].items()})
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

    created_events = []
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
        created_events.append(ev)

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

    # Fan the sale out into a shippable order. Idempotent per (source, order_id);
    # lands in "needs address" unless the email carried a usable address block.
    # Wrapped in try on purpose: a shipping problem must never break the sale
    # pipeline that decrements inventory and delists on other marketplaces.
    try:
        ensure_order_from_sale(parsed, source=source, sale_events=created_events)
    except Exception as exc:
        app.logger.warning("Could not create an order from %s: %s", parsed.get("order_id"), exc)

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
    # Refuse a line break before it is stored, so the operator sees why. Every one of
    # these reaches an IMAP command, and imaplib validates none of them -- see
    # email_monitor._imap_reject, which repeats the check at the point of use because
    # rows saved before this existed still have to be handled.
    bad = email_monitor._imap_reject(
        username=f.get("username", ""), password=f.get("password", ""),
        folder=f.get("folder", ""),
        **{"sender filter": f.get("sender_filter", ""),
           "subject filter": f.get("subject_filter", "")})
    if bad:
        return jsonify({"status": "error", "message": bad}), 400
    old_host, old_port = (m.host or ""), (m.port or 993)
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
    # Same shape as a shop's store_domain: /shops/email/test is _require_admin-gated
    # and is the only thing that sets enabled=True, but this route is shops:edit and
    # kept the stored password on a blank submission. Repointing host or port would
    # otherwise inherit that blessing, and the background poller -- which runs with no
    # user at all -- would log in to the new server with the saved mailbox password.
    host_changed = (m.host or "") != old_host or (m.port or 993) != old_port
    if host_changed:
        m.enabled = False
        m.status = "disconnected"
        m.status_detail = "Server changed — test the connection again before polling."
    db.session.commit()
    msg = "Email monitor settings saved."
    if host_changed:
        msg += " The server changed, so polling was disabled — test it again to re-enable."
    return jsonify({"status": "success", "message": msg})


@app.route("/shops/email/test", methods=["POST"])
def shops_email_test():
    denied = _require_admin()
    if denied:
        return denied
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
    denied = _require_admin()
    if denied:
        return denied
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
      • a composite reference_cards(category_id, number) for the OCR number
        match, which always binds category_id (nothing ever filters by game)
    All are IF NOT EXISTS, so this is a cheap no-op on every start after the first.
    A final PRAGMA optimize lets SQLite refresh stats only when worthwhile.
    """
    from sqlalchemy import text

    statements = [
        "CREATE INDEX IF NOT EXISTS idx_scan_scan_date ON scan_records(scan_date)",
        "CREATE INDEX IF NOT EXISTS idx_scan_template ON scan_records(template_used)",
        "CREATE INDEX IF NOT EXISTS idx_scan_matched_product ON scan_records(matched_product_id)",
        "CREATE INDEX IF NOT EXISTS idx_refcard_category_number ON reference_cards(category_id, number)",
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
        "is_held":       "BOOLEAN",
    }
    added = []
    with db.engine.begin() as conn:
        for name, decl in new_cols.items():
            if name not in existing:
                conn.exec_driver_sql(f"ALTER TABLE scan_records ADD COLUMN {name} {decl}")
                added.append(name)

    # Existing rows predate "Held"; they're all still held. Backfill NULLs to
    # true so the column is accurate immediately (queries also coalesce NULL->held).
    if "is_held" in added:
        with db.engine.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE scan_records SET is_held = 1 "
                "WHERE is_held IS NULL AND COALESCE(is_catalog, 0) = 0")
    # Indexes (names match SQLAlchemy's so a fresh DB's create_all doesn't dupe).
    # game_key/album_key are served by the composite prefixes; the is_* booleans
    # are only queried through coalesce() (unindexable), so neither gets its own
    # index — migrate_drop_unused_indexes() removes them from older installs.
    index_sql = [
        "CREATE INDEX IF NOT EXISTS ix_scan_records_name_key ON scan_records(name_key)",
        "CREATE INDEX IF NOT EXISTS ix_scan_records_card_type_key ON scan_records(card_type_key)",
        "CREATE INDEX IF NOT EXISTS ix_scan_records_dup_hash ON scan_records(dup_hash)",
        "CREATE INDEX IF NOT EXISTS idx_scan_hot ON scan_records(game_key, is_catalog, is_archived, scan_date)",
        "CREATE INDEX IF NOT EXISTS idx_scan_album_hot ON scan_records(album_key, is_catalog, is_archived)",
    ]
    with db.engine.begin() as conn:
        for sql in index_sql:
            conn.exec_driver_sql(sql)

    if added:
        _backfill_scan_columns()


def migrate_drop_unused_indexes():
    """
    Drop indexes that older installs carry but that no query can use — verified
    with EXPLAIN QUERY PLAN against every query site (see the 2026-08-06 DB
    audit). Three groups:
      • idx_scan_game / idx_scan_album — json_extract expression indexes whose
        only consumer (the /inventory Python fallback) now filters the
        normalized game_key/album_key columns instead;
      • ix_scan_records_{game_key,album_key} — shadowed by the idx_scan_hot /
        idx_scan_album_hot composite prefixes;
      • ix_scan_records_is_* — every query wraps these booleans in coalesce(),
        which SQLite cannot seek an index for;
      • idx_refcard_game_number — nothing filters reference_cards by game;
        every lookup binds category_id (idx_refcard_category_number replaces it).
    DROP INDEX IF EXISTS, so this is a no-op on fresh databases and on every
    start after the first.
    """
    drops = [
        "idx_scan_game",
        "idx_scan_album",
        "ix_scan_records_game_key",
        "ix_scan_records_album_key",
        "ix_scan_records_is_finalized",
        "ix_scan_records_is_catalog",
        "ix_scan_records_is_archived",
        "ix_scan_records_is_held",
        "idx_refcard_game_number",
    ]
    with db.engine.begin() as conn:
        for name in drops:
            conn.exec_driver_sql(f"DROP INDEX IF EXISTS {name}")


def migrate_add_serial_key_column():
    """
    Add scan_records.serial_key (normalized duplicate-group serial, NULL when
    absent) plus idx_scan_dupe(name_key, serial_key) to installs that predate
    them, then backfill once. Fresh databases get both from create_all().
    Lets /duplicates find its candidate groups with one GROUP BY instead of
    hydrating and grouping every record in Python.
    """
    from sqlalchemy import inspect

    inspector = inspect(db.engine)
    if "scan_records" not in inspector.get_table_names():
        return  # fresh DB — create_all() already made the column

    existing = {c["name"] for c in inspector.get_columns("scan_records")}
    added = False
    if "serial_key" not in existing:
        with db.engine.begin() as conn:
            conn.exec_driver_sql(
                "ALTER TABLE scan_records ADD COLUMN serial_key VARCHAR(120)")
        added = True
    with db.engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS idx_scan_dupe "
            "ON scan_records(name_key, serial_key)")
    if added:
        _backfill_scan_columns()


def migrate_add_search_fts():
    """
    Full-text search backing (FTS5, trigram tokenizer) so the substring
    searches stay index-served as the collection grows:

      • scan_search — external-content FTS over scan_records.extracted_data.
        The inventory / analytics / CSV-export / Select-All search all match a
        LIKE pattern against the serialized JSON; a trigram FTS5 table serves
        that exact LIKE from an index (same pattern semantics, ASCII-case-
        insensitive — identical to the lower()/LIKE scan it replaces).
      • ref_search — the same over reference_cards.name for the OCR matchers.

    Sync is by AFTER INSERT/UPDATE/DELETE triggers on the base tables, so
    every write path is covered, ORM or raw SQL. Self-healing: when the row
    counts diverge from the base table (e.g. a factory reset dropped and
    recreated the base tables, taking the triggers with them), the index is
    rebuilt from the content table. The reset route also re-runs this
    migration directly so search works again before the next restart.

    On a SQLite without FTS5 or the trigram tokenizer (< 3.34) everything
    here is skipped; _search_condition helpers then fall back to the plain
    LIKE scan, so search keeps working, just unindexed.
    """
    specs = [
        ("scan_search", "scan_records", "extracted_data"),
        ("ref_search", "reference_cards", "name"),
    ]
    try:
        with db.engine.begin() as conn:
            for fts, base, col in specs:
                conn.exec_driver_sql(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts} USING fts5("
                    f"{col}, content='{base}', content_rowid='id', tokenize='trigram')")
                conn.exec_driver_sql(
                    f"CREATE TRIGGER IF NOT EXISTS {fts}_ai AFTER INSERT ON {base} BEGIN "
                    f"INSERT INTO {fts}(rowid, {col}) VALUES (new.id, new.{col}); END")
                conn.exec_driver_sql(
                    f"CREATE TRIGGER IF NOT EXISTS {fts}_ad AFTER DELETE ON {base} BEGIN "
                    f"INSERT INTO {fts}({fts}, rowid, {col}) VALUES ('delete', old.id, old.{col}); END")
                conn.exec_driver_sql(
                    f"CREATE TRIGGER IF NOT EXISTS {fts}_au AFTER UPDATE OF {col} ON {base} BEGIN "
                    f"INSERT INTO {fts}({fts}, rowid, {col}) VALUES ('delete', old.id, old.{col}); "
                    f"INSERT INTO {fts}(rowid, {col}) VALUES (new.id, new.{col}); END")
                n_base = conn.exec_driver_sql(f"SELECT count(*) FROM {base}").fetchone()[0]
                # count(*) on an external-content FTS table is answered from the
                # CONTENT table, so it can never detect a stale/empty index. The
                # _docsize shadow table counts what is actually indexed.
                n_fts = conn.exec_driver_sql(
                    f"SELECT count(*) FROM {fts}_docsize").fetchone()[0]
                if n_base != n_fts:
                    conn.exec_driver_sql(f"INSERT INTO {fts}({fts}) VALUES ('rebuild')")
        _FTS_READY.clear()   # re-probe: tables may have just appeared
    except Exception as exc:
        try:
            app.logger.warning("FTS5 search index unavailable (%s); "
                               "search falls back to table scans.", exc)
        except Exception:
            pass


def init_db():
    """
    Create/upgrade the database schema and load persisted settings. Every step
    is idempotent, so running it on an up-to-date database is a cheap no-op.

    Called only from the __main__ block. Importing this module must stay free
    of database side effects: test harnesses and `flask shell` rely on import
    being inert, and the import-time SECRET_KEY / SESSION_COOKIE_SECURE code
    is deliberately written around the database not being ready at import. A
    WSGI server importing this module never runs this — `python app.py` is the
    only supported launch path (see README, "Running the app").
    """
    with app.app_context():
        db.create_all()
        migrate_add_image_path_back_column()
        migrate_add_display_image_path_column()
        migrate_add_type_reference_region_column()
        migrate_add_scan_scaling_columns()
        migrate_add_serial_key_column()
        migrate_add_performance_indexes()
        migrate_drop_unused_indexes()
        migrate_add_search_fts()
        optimize_database()
        load_settings()   # load API keys/settings; one-time seed from .env


def _lan_ip():
    """Best-effort primary LAN IPv4 of this machine (no packet is actually sent)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


MDNS_NAME_KEY = "MDNS_HOSTNAME"        # app_settings key
_MDNS = {"zc": None, "port": None, "scheme": "http"}   # live advertisement state (serving process)


def _slug_hostname(raw, default="cardcollector"):
    """Sanitize free text into a valid single mDNS/DNS label: lowercase, digits and
    hyphens only, no leading/trailing hyphen, max 63 chars. Empty -> default."""
    s = (raw or "").strip().lower()
    s = _re.sub(r"[^a-z0-9-]+", "-", s)
    s = _re.sub(r"-{2,}", "-", s).strip("-")
    return s[:63] or default


def get_mdns_name():
    """The configured advertised name (sanitized), defaulting to 'cardcollector'."""
    return _slug_hostname(get_setting(MDNS_NAME_KEY, "cardcollector"))


def set_mdns_name(raw):
    """Persist a new advertised name (sanitized); returns the value stored."""
    name = _slug_hostname(raw)
    set_setting(MDNS_NAME_KEY, name)
    return name


def start_mdns(host_label=None, port=5005, scheme="http"):
    """Advertise this server on the local network as <host_label>.local via mDNS,
    so it's reachable at a stable name on whatever LAN it's started on — no router
    setup or per-device hosts edits. Records the live handle so the name can be
    changed at runtime (see restart_mdns). Returns the Zeroconf handle or None.

    No-op if the optional 'zeroconf' package isn't installed. Note: mDNS only
    resolves for devices ON THE SAME local network; it is not reachable across
    different networks or over the internet (use a tunnel/VPN for that)."""
    host_label = _slug_hostname(host_label) if host_label else get_mdns_name()
    _MDNS["port"] = port
    _MDNS["scheme"] = scheme
    try:
        from zeroconf import Zeroconf, ServiceInfo
    except Exception:
        print("[mDNS] Optional 'zeroconf' package not installed — skipping the .local name.")
        print("[mDNS] Enable a stable %s://%s.local address with:  pip install zeroconf" % (scheme, host_label))
        return None
    import socket
    ip = _lan_ip()
    svc = "_https._tcp.local." if scheme == "https" else "_http._tcp.local."
    default = 443 if scheme == "https" else 80
    try:
        info = ServiceInfo(
            svc,
            f"{host_label}.{svc}",
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties={"path": "/"},
            server=f"{host_label}.local.",
        )
        zc = Zeroconf()
        zc.register_service(info)
        _MDNS["zc"] = zc
        shown = f"{scheme}://{host_label}.local" + ("" if port == default else f":{port}")
        print(f"[mDNS] Also reachable on this network at:  {shown}   (IP {ip})")
        return zc
    except Exception as exc:
        print(f"[mDNS] Could not start the .local advertisement: {exc}")
        return None


def restart_mdns(new_name):
    """Re-advertise under a new (sanitized) name on the current port, live. Returns
    the name now in effect. Safe to call even if zeroconf isn't installed."""
    name = _slug_hostname(new_name)
    zc = _MDNS.get("zc")
    if zc is not None:
        try:
            zc.close()
        except Exception:
            pass
        _MDNS["zc"] = None
    port = _MDNS.get("port") or int(os.environ.get("PORT", "80"))
    start_mdns(name, port, scheme=_MDNS.get("scheme", "http"))
    return name


def _port_bindable(port):
    """True if we can bind the given TCP port on all interfaces right now."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# SPKI SHA-256 of keypairs that are known to be public and must never be served.
# The 2048-bit RSA key below was committed here in 65e455c4 (2026-07-14) and removed in
# 0787925 (2026-07-29) by a delete-commit, NOT a history rewrite — so it stays recoverable
# from the published history (`git show 65e455c4:certs/cardcollector.key`) and is
# compromised permanently. Its certificate is valid until 2036-07-11 and
# _ensure_self_signed_cert used to reuse any existing pair unconditionally, so without
# this check every clone still holding that pair would serve the public key for another
# decade. Fingerprinting the keypair rather than the file means a re-encoded or
# reformatted copy still matches. This is a digest of the PUBLIC key: not itself secret.
_COMPROMISED_SPKI_SHA256 = frozenset({
    "70a937e51a82f344b24305c7b57c2e2bb7e016f8a66aa8ab3113a55e0816f427",
})


def _spki_sha256(key_pem):
    """SHA-256 over the DER SubjectPublicKeyInfo of a PEM private key, or None if the
    bytes are not a readable private key."""
    from cryptography.hazmat.primitives import serialization
    try:
        key = serialization.load_pem_private_key(key_pem, password=None)
    except Exception:
        return None
    return _hashlib.sha256(key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo)).hexdigest()


def _ensure_self_signed_cert(cert_dir, hostnames, ips):
    """Generate (once) and reuse a persistent self-signed certificate covering the
    given hostnames + IPs, so the app can be served over HTTPS on the LAN — which
    browsers require before they'll allow camera (getUserMedia) access from any
    machine other than the server itself. A pair whose key is in
    _COMPROMISED_SPKI_SHA256 is discarded and regenerated rather than reused.
    Returns (cert_path, key_path)."""
    import datetime as _dt
    import ipaddress
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "cardcollector.crt")
    key_path = os.path.join(cert_dir, "cardcollector.key")
    if os.path.exists(cert_path) and os.path.exists(key_path):
        try:
            with open(key_path, "rb") as f:
                fingerprint = _spki_sha256(f.read())
        except OSError:
            fingerprint = None
        # Only a POSITIVE match discards the pair. An unreadable or unparseable key gives
        # None, which is not in the set, so it is reused exactly as before — a transient
        # read error must not churn a legitimate certificate on every boot.
        if fingerprint not in _COMPROMISED_SPKI_SHA256:
            return cert_path, key_path
        print("[https] SECURITY: this certificate's private key was published in the "
              "repository's git history and is public. Discarding it.")
        print("[https] Generating a fresh keypair — devices that accepted the old "
              "certificate will show the one-time warning again.")
        for stale in (key_path, cert_path):
            try:
                os.remove(stale)
            except OSError as exc:
                # Fail loudly rather than serve it: the caller falls back to plain HTTP,
                # which is safer than HTTPS whose key anyone can download.
                raise RuntimeError(f"refusing to serve a known-public key, and {stale} "
                                   f"could not be removed ({exc})")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cn = hostnames[0] if hostnames else "cardcollector.local"
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    san = []
    for h in dict.fromkeys(hostnames):
        try:
            san.append(x509.DNSName(h))
        except Exception:
            pass
    for ip in dict.fromkeys(ips):
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass
    now = _dt.datetime.utcnow()
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(days=1))
            .not_valid_after(now + _dt.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    try:
        os.chmod(key_path, 0o600)   # private key must not be world-readable
    except OSError:
        pass
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


if __name__ == "__main__":
    # Complete any deferred cleanup from a previous DB relocation now that we're
    # (re)starting on the current configured database.
    process_pending_deletions()

    init_db()

    with app.app_context():
        # Stable session secret so logins survive restarts and are shared across the
        # reloader's parent/child processes. Env SECRET_KEY wins; otherwise a value
        # is generated once and stored in app_settings.
        _sk = os.environ.get("SECRET_KEY")
        if not _sk:
            # An existing install's stored key still wins, so upgrading to the
            # import-time file does NOT invalidate anyone's session.
            _sk = get_setting("FLASK_SECRET_KEY", "")
            if not _sk:
                # Adopt the key already loaded at import instead of minting a second
                # one, so the file and app_settings hold the same value from here on
                # and it stops mattering which launch path a given start took.
                _sk = app.config["SECRET_KEY"]
                try:
                    set_setting("FLASK_SECRET_KEY", _sk)
                except Exception:
                    pass
        app.config["SECRET_KEY"] = _sk

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
    # Optional background tracking refresh; self-guarded, a no-op unless
    # SHIPPING_POLL_BACKGROUND=1. "Refresh tracking" remains the primary path.
    start_tracking_poller(app)

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
    # Port: default to 80 so the URL needs no ":port" suffix. Binding to 80 needs
    # admin/root (ports below 1024 are privileged); if that's not available or 80
    # is already in use, fall back so the app still starts. Override with the PORT
    # env var (e.g. PORT=5005) to pin a specific port.
    # HTTPS is required for the live camera (getUserMedia) to work on machines other
    # than the server, so it's ON by default; a self-signed certificate is generated
    # automatically. Disable with USE_HTTPS=0 to serve plain HTTP. Default port is
    # 443 for HTTPS, 80 for HTTP.
    USE_HTTPS = os.environ.get("USE_HTTPS", "1").strip().lower() in ("1", "true", "yes", "on")
    scheme = "https" if USE_HTTPS else "http"
    default_port = 443 if USE_HTTPS else 80

    PORT = int(os.environ.get("PORT", str(default_port)))
    if not _port_bindable(PORT):
        fallback = int(os.environ.get("PORT_FALLBACK", "8443" if USE_HTTPS else "5005"))
        if PORT != fallback:
            print(f"[port] Couldn't bind port {PORT} — it needs admin/root privileges, "
                  f"or it's already in use.")
            print(f"[port] Falling back to {fallback}. To use {PORT}: launch with elevated "
                  f"privileges (e.g. 'sudo'), or set PORT to a free port.")
            PORT = fallback

    with app.app_context():
        _mdns_name = get_mdns_name()

    # Set up HTTPS (self-signed) if requested; fall back to HTTP on any failure.
    ssl_context = None
    if USE_HTTPS:
        try:
            cert_dir = os.path.join(app.root_path, "certs")
            hostnames = [f"{_mdns_name}.local", _mdns_name, "localhost"]
            ips = [_lan_ip(), "127.0.0.1"]
            ssl_context = _ensure_self_signed_cert(cert_dir, hostnames, ips)
            print(f"[https] Serving over HTTPS with a self-signed certificate ({cert_dir}).")
            print("[https] Browsers show a one-time 'not secure' warning for self-signed "
                  "certs — accept it once per device to enable the camera.")
        except Exception as exc:
            if isinstance(exc, ImportError):
                print("[https] HTTPS needs the 'cryptography' package. Install it with:")
                print("[https]     pip install cryptography      (or: pip install -r requirements.txt)")
            print(f"[https] Could not set up HTTPS ({exc}); serving over HTTP instead.")
            print("[https] Note: the live camera won't work on other devices over HTTP.")
            USE_HTTPS = False
            scheme = "http"
            default_port = 80

    # Serving HTTPS ourselves is proof the cookie can be Secure, so turn it on. This
    # only ever raises the flag: assigning `ssl_context is not None` outright would
    # also LOWER it, silently discarding an explicit SESSION_COOKIE_SECURE=1 whenever
    # this launcher fell back to plain HTTP. Refusing to honour a security setting the
    # operator deliberately set is worse than the broken login it would cause, and the
    # broken login is at least visible. Plain-HTTP modes keep the import-time default
    # of off unless that variable says otherwise.
    if ssl_context is not None:
        app.config["SESSION_COOKIE_SECURE"] = True

    def _visit_url(host):
        return f"{scheme}://{host}" + ("" if PORT == default_port else f":{PORT}")

    print(f" • Visit: {_visit_url('127.0.0.1')}  (or  {_visit_url(_mdns_name + '.local')}  on this LAN)")
    if not USE_HTTPS:
        print(" • Running over HTTP — the live camera won't work on other devices. "
              "HTTPS is the default; unset USE_HTTPS=0 to restore it.")

    # The Werkzeug debug reloader is OFF by default: its interactive debugger allows
    # remote code execution, which is dangerous on a LAN-exposed server. Turn it on
    # only for local development with FLASK_DEBUG=1.
    DEBUG = os.environ.get("FLASK_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")

    # Advertise mDNS from the process that actually serves requests: the reloader
    # child when debug is on (WERKZEUG_RUN_MAIN==true), or the sole process when off.
    _mdns = None
    if (os.environ.get("WERKZEUG_RUN_MAIN") == "true") or not DEBUG:
        _mdns = start_mdns(_mdns_name, PORT, scheme=scheme)

    try:
        app.run(host="0.0.0.0", port=PORT, debug=DEBUG, threaded=True, ssl_context=ssl_context)
    finally:
        if _mdns is not None:
            try:
                _mdns.close()
            except Exception:
                pass
