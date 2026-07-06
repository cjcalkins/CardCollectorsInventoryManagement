from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, redirect
import os
import re as _re
import cv2
import json
import shutil
import numpy as np
from datetime import datetime
from PIL import Image
from werkzeug.utils import secure_filename
from models import (db, Product, ScanRecord, ShopConnection, Listing, EmailMonitor,
                    SaleEvent, ReferenceCard, ReferenceSync)
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

# tcgcsv_sync downloads a game's catalog from tcgcsv.com into ReferenceCard rows
# so OCR results can be matched to a real card and used to auto-fill entry data.
# Imported optionally so the app still boots if the module/urllib access is
# unavailable; the /reference routes report a clear error instead of crashing.
try:
    import tcgcsv_sync
except Exception:
    tcgcsv_sync = None

app = Flask(__name__, template_folder="templates")

# ====================== CONFIG ======================
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///inventory.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ROI_TEMPLATE_FOLDER"] = "templates/roi"
app.config["TEMP_IMPORT_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "import_pages")
app.config["TEMP_SPLIT_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "temp_split")
app.config["TEMP_CARD_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "temp_cards")
app.config["INVENTORY_IMAGE_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "inventory_cards")
app.config["TEMP_PDF_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "temp_pdf_pages")

db.init_app(app)


# ====================== DIRECTORY SETUP ======================
def ensure_dirs():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["ROI_TEMPLATE_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"], "debug"), exist_ok=True)
    os.makedirs(app.config["TEMP_IMPORT_FOLDER"], exist_ok=True)
    os.makedirs(app.config["TEMP_SPLIT_FOLDER"], exist_ok=True)
    os.makedirs(app.config["TEMP_CARD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["INVENTORY_IMAGE_FOLDER"], exist_ok=True)
    os.makedirs(app.config["TEMP_PDF_FOLDER"], exist_ok=True)


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
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
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
        if warped.shape[1] > warped.shape[0]:       # ensure portrait
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
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


def create_scan_record(image_path, template_name, extracted, image_path_back=None):
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
    (see tcgcsv_sync.normalize_product). Keyed on the upstream productId."""
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


def _reference_candidates_for_ocr(category_id, ocr_result, limit=8):
    """
    Build scored reference-card candidates for an OCR result within one game.
    Pre-narrows by collector number (then name) so we never fuzzy-score an entire
    game, then reuses card_ocr's scorer. Each candidate is tagged source
    "reference" and carries product_id + rich fields for auto-fill.
    """
    if not category_id or card_ocr is None:
        return []

    number = (ocr_result.get("number_guess") or "").strip()
    name   = (ocr_result.get("name_guess") or "").strip()
    base   = ReferenceCard.query.filter(ReferenceCard.category_id == category_id)

    narrowed = None
    if number:
        variants = _collector_number_variants(number) or {number}
        narrowed = base.filter(ReferenceCard.number.in_(list(variants))).limit(300).all()
    if not narrowed and name:
        first = (name.split() or [name])[0]
        narrowed = base.filter(ReferenceCard.name.ilike(f"%{first}%")).limit(500).all()
    if not narrowed:
        narrowed = base.limit(500).all()

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


# ====================== PAGE ROUTES ======================
@app.route("/")
def index():
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
    "game", "album", "page", "slot",
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

    # Fetch base query — JSON key filters are done in Python for SQLite compatibility
    # (.astext is PostgreSQL-only; SQLite stores JSON as plain text)
    query = ScanRecord.query.order_by(ScanRecord.scan_date.desc())

    if f_template:
        query = query.filter(ScanRecord.template_used == f_template)
    if search:
        query = query.filter(ScanRecord.extracted_data.cast(db.Text).ilike(f"%{search}%"))

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

    # Python-side filtering for game / album (SQLite-safe)
    all_records = all_records_raw
    if f_game:
        all_records = [
            r for r in all_records
            if str((r.extracted_data or {}).get("game", "")).strip() == f_game
        ]
    if f_album:
        all_records = [
            r for r in all_records
            if str((r.extracted_data or {}).get("album", "")).strip() == f_album
        ]

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
@app.route("/import_single_card", methods=["POST"])
def import_single_card():
    """
    Add ONE card to inventory from a front image (required) and an optional
    back image.

    Each image is best-effort auto-aligned (deskewed + cropped) the same way
    the search-by-image flow is, falling back to the raw photo when a card
    outline can't be found — so a good capture is never made worse. A single
    blank inventory record is created for the chosen Game with the image(s)
    attached; its fields are filled in by hand afterwards, or via the OCR
    "Identify" tool on the card's detail page.

    Form fields:
      game         — Game/template name (required)
      album        — optional album name to file the card under
      front_image  — the card front (required)
      back_image   — the card back  (optional)
    """
    ensure_dirs()

    game  = (request.form.get("game")  or "").strip()
    album = (request.form.get("album") or "").strip()
    edge_type = normalize_card_edge_type(request.form.get("card_edge_type"))
    if not game:
        return jsonify({"status": "error", "message": "Game is required"}), 400

    front = request.files.get("front_image")
    if not front or not front.filename:
        return jsonify({"status": "error", "message": "A front image is required"}), 400
    back = request.files.get("back_image")

    try:
        template = load_template(game)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Could not load game '{game}': {exc}"}), 400

    aligned_flags = {}

    def _save_side(file_storage, suffix):
        """Detect + crop the card (best-effort, based on the selected edge type)
        and save the image into the inventory folder. Records whether the crop
        succeeded in `aligned_flags`. Returns the upload-relative path, or None
        if the upload was missing/unreadable."""
        if not file_storage or not file_storage.filename:
            return None
        np_buf = np.frombuffer(file_storage.read(), np.uint8)
        img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img, cropped = detect_and_crop_card(img, edge_type)   # falls back to raw
        aligned_flags[suffix] = cropped
        final_name = f"single_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        relative_path = normalize_to_upload_relative(
            os.path.join("inventory_cards", final_name)
        )
        absolute_path = os.path.join(app.config["UPLOAD_FOLDER"], relative_path)
        cv2.imwrite(absolute_path, img)
        return relative_path

    try:
        front_path = _save_side(front, "front")
        if not front_path:
            return jsonify({"status": "error", "message": "Could not read the front image"}), 400
        back_path = _save_side(back, "back")   # may be None
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Could not process image: {exc}"}), 500

    # Blank starting values for every field this Game defines, plus the
    # identity fields the rest of the app reads. `game` is stored with
    # underscores turned into spaces to match how the 9-pocket import files it,
    # so single and batch cards of the same game group together in Inventory.
    blank_fields = {k: "" for k in (template.get("fields", {}) or {}).keys()}
    extracted = {**blank_fields, "game": game.replace("_", " ")}
    if album:
        extracted["album"] = album

    matched_product, record = create_scan_record(
        image_path=front_path,
        template_name=game,
        extracted=extracted,
        image_path_back=back_path,
    )

    # Let the UI mention when a card outline couldn't be found and the raw
    # photo was kept instead (so the user can retry or crop manually later).
    not_detected = [side for side, ok in aligned_flags.items() if not ok]
    if not_detected:
        message = ("Card added to inventory. Couldn't detect a "
                   f"{edge_type}-edged card in the {', '.join(not_detected)} "
                   "image, so the original photo was kept for that side.")
    else:
        message = "Card added to inventory — detected and cropped to the card."

    return jsonify({
        "status":         "success",
        "message":        message,
        "record_id":      record.id,
        "edge_type":      edge_type,
        "front_aligned":  aligned_flags.get("front", False),
        "back_aligned":   aligned_flags.get("back", False) if back_path else None,
        "image_url":      build_uploaded_file_url(record.image_path),
        "image_url_back": build_uploaded_file_url(record.image_path_back) if record.image_path_back else None,
        "detail_url":     url_for("inventory_detail", record_id=record.id),
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
JUSTTCG_API_KEY    = os.environ.get("JUSTTCG_API_KEY", "")

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
        "x-api-key":  JUSTTCG_API_KEY,
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
            "x-api-key":  JUSTTCG_API_KEY,
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
        ocr = card_ocr.ocr_card_front(abs_path)
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
        "available": tcgcsv_sync is not None,
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
    if tcgcsv_sync is None:
        return jsonify({"status": "error", "message": "tcgcsv sync module unavailable."}), 503

    # Serve from cache if fetched within the last 30 minutes.
    now = datetime.utcnow()
    cached = _REF_CATEGORIES_CACHE
    if cached["data"] and cached["at"] and (now - cached["at"]).total_seconds() < 1800:
        cats = cached["data"]
    else:
        try:
            raw = tcgcsv_sync.get_categories()
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Could not reach tcgcsv: {exc}"}), 502
        cats = sorted(
            [{"category_id": c.get("categoryId"),
              "name": c.get("displayName") or c.get("name") or "",
              "popularity": c.get("popularity", 0)} for c in raw],
            key=lambda c: c.get("popularity", 0), reverse=True,
        )
        _REF_CATEGORIES_CACHE["data"] = cats
        _REF_CATEGORIES_CACHE["at"] = now

    return jsonify({"status": "ok", "categories": cats,
                    "last_updated": tcgcsv_sync.get_last_updated()})


@app.route("/reference/groups/<int:category_id>")
def reference_groups(category_id):
    """List a category's groups (sets) — the work items the client loops over."""
    if tcgcsv_sync is None:
        return jsonify({"status": "error", "message": "tcgcsv sync module unavailable."}), 503
    try:
        groups = tcgcsv_sync.get_groups(category_id)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Could not reach tcgcsv: {exc}"}), 502
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
    if tcgcsv_sync is None:
        return jsonify({"status": "error", "message": "tcgcsv sync module unavailable."}), 503

    body = request.get_json() or {}
    try:
        category_id = int(body.get("category_id"))
        group_id    = int(body.get("group_id"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "category_id and group_id are required."}), 400
    category_name = str(body.get("category_name") or "").strip()
    group_name    = str(body.get("group_name") or "").strip()

    try:
        cards = tcgcsv_sync.fetch_group_cards(category_id, category_name, group_id, group_name)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"tcgcsv fetch failed: {exc}"}), 502

    for rec in cards:
        _reference_upsert(rec)

    # Ensure a ReferenceSync row exists and refresh its counts.
    rs = ReferenceSync.query.filter_by(category_id=category_id).first()
    if rs is None:
        rs = ReferenceSync(category_id=category_id)
        db.session.add(rs)
    rs.game = category_name or rs.game
    rs.status = "ok"
    rs.remote_updated = tcgcsv_sync.get_last_updated() or rs.remote_updated
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

            # Zoom of 2.0 ≈ 144 DPI, plenty of resolution for the 3x3 splitter
            # while keeping file sizes and processing time reasonable.
            zoom = 2.0
            matrix = fitz.Matrix(zoom, zoom)

            pages = []
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=matrix)
                page_filename = f"pdfpage_{batch_id}_{page_index + 1:03d}.png"
                page_path = os.path.join(app.config["TEMP_PDF_FOLDER"], page_filename)
                pix.save(page_path)

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

    # Auto-import mode: skip the per-tile straighten/alignment pass entirely and
    # file the RAW cut images as the final card images. Everything downstream
    # (the finalize/import step, front/back merging, etc.) is unchanged.
    skip_align = request.form.get("skip_align", "").strip().lower() in ("1", "true", "yes", "on")

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

            if skip_align:
                # Bypass alignment: the raw cut piece IS the final card image.
                card_path = os.path.join(app.config["TEMP_CARD_FOLDER"], slot_filename)
                piece.save(card_path)
                results.append({
                    "slot": slot_num, "filename": slot_filename,
                    "status": "processed",
                    "url": url_for("temp_card_file", filename=slot_filename),
                })
                continue

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
                results.append({
                    "slot": slot_num, "filename": slot_filename,
                    "status": "fallback",
                    "url": url_for("temp_split_file", filename=slot_filename),
                })

        processed_count = sum(1 for r in results if r["status"] == "processed")
        fallback_count  = sum(1 for r in results if r["status"] == "fallback")

        if skip_align:
            message = f"Split completed. {processed_count} raw cut image(s) ready (alignment skipped)."
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

                result = {
                    "filename":        filename,
                    "status":          "success",
                    "record_id":       record.id,
                    "side":            side,
                    "image_url":       build_uploaded_file_url(record.image_path),
                    "image_url_back":  build_uploaded_file_url(record.image_path_back) if record.image_path_back else None,
                    "extracted":       extracted,
                    "matched_product": matched_product.product_name if matched_product else "No match",
                }
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

    records = ScanRecord.query.all()
    scored  = []

    for record in records:
        if not record.image_path or record.image_path == "__blank__":
            continue
        img_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            normalize_to_upload_relative(record.image_path),
        )
        if not os.path.exists(img_path):
            continue
        ref_img = cv2.imread(img_path)
        if ref_img is None:
            continue
        score = _match_score(query_desc, _orb_descriptors(ref_img))
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

XIMILAR_API_TOKEN       = os.environ.get("XIMILAR_API_TOKEN", "")
XIMILAR_REQUEST_URL     = "https://api.ximilar.com/account/v2/request/"
XIMILAR_CONNECT_TIMEOUT = 40      # seconds per individual HTTP call
XIMILAR_POLL_INTERVAL   = 2.0     # seconds between status polls
XIMILAR_POLL_MAX_WAIT   = 120     # seconds to wait for a single job to finish

# Naming schemes accepted by the /condition endpoint's `mode` field.
XIMILAR_CONDITION_MODES = {"ebay", "tcgplayer", "cardmarket", "ximilar"}


def _image_source_for_grading(path_value):
    """
    Turn a stored image_path into a Ximilar record source dict.

    Returns:
      {"_url": "https://..."}  for images stored as external URLs (Ximilar fetches them)
      {"_base64": "..."}       for local files (encoded from disk)
      None                     when there is no usable image on this side
    """
    relative = normalize_to_upload_relative(path_value)
    if not relative or relative == "__blank__":
        return None

    if relative.startswith("http://") or relative.startswith("https://"):
        return {"_url": relative}

    abs_path = os.path.join(app.config["UPLOAD_FOLDER"], relative)
    if not os.path.exists(abs_path):
        return None
    try:
        with open(abs_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return None
    return {"_base64": encoded}


def _ximilar_http(method, url, payload=None):
    """Small JSON HTTP helper for the Ximilar async request API."""
    data = None
    headers = {
        "Authorization": f"Token {XIMILAR_API_TOKEN}",
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
    if not XIMILAR_API_TOKEN:
        raise RuntimeError(
            "XIMILAR_API_TOKEN is not set. Add it to your environment (or .env file) "
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


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        migrate_add_image_path_back_column()
        migrate_add_display_image_path_column()

    ensure_dirs()

    # Optional background sale-email polling (off by default; "Check now" always works).
    if os.environ.get("EMAIL_MONITOR_BACKGROUND") == "1":
        start_email_poller()

    print("Card Collector Inventory Manager")
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
