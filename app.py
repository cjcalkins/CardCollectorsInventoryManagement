from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, redirect
import os
import re as _re
import cv2
import pytesseract
import json
import shutil
import numpy as np
from datetime import datetime
from PIL import Image
from werkzeug.utils import secure_filename
from models import db, Product, ScanRecord

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


# ====================== PATH HELPERS ======================
def normalize_to_upload_relative(path_value):
    if not path_value:
        return ""

    normalized = str(path_value).replace("\\", "/")
    upload_prefix = app.config["UPLOAD_FOLDER"].replace("\\", "/").rstrip("/") + "/"

    if normalized.startswith(upload_prefix):
        normalized = normalized[len(upload_prefix):]

    return normalized.lstrip("/")


def build_uploaded_file_url(path_value):
    relative_path = normalize_to_upload_relative(path_value)
    if not relative_path:
        return None
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

    return {
        "game": parts[0],
        "album": "-".join(parts[1:-2]),
        "page": parts[-2],
        "slot": parts[-1],
    }


def get_record_value(record, key, default=""):
    extracted = record.extracted_data or {}
    value = extracted.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def build_album_index():
    records = ScanRecord.query.order_by(ScanRecord.scan_date.desc()).all()
    album_map = {}

    for record in records:
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


# ====================== OCR PREPROCESSING ======================
def window_levels(gray, low, high):
    gray_f = gray.astype(np.float32)
    stretched = (gray_f - low) * (255.0 / max(1, high - low))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched


def upscale_for_ocr(img, scale=3):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def clean_binary(img):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    return cleaned


def preprocess_roi_for_ocr_candidates(roi):
    """
    Build a prioritised list of preprocessed image candidates for OCR.
    Candidates are ordered from most-likely-to-succeed to least, so that
    the early-exit threshold in preprocess_and_ocr_best fires as soon as
    possible and avoids running all 30+ variants on every field.

    Priority order:
      1. CLAHE on raw gray         — fast, effective on clean prints
      2. Otsu threshold variants   — sharp binary, good for high-contrast text
      3. Windowed + CLAHE          — handles uneven lighting
      4. Adaptive threshold        — fallback for low-contrast regions
      5. Raw windowed              — rarely needed, kept for robustness
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    candidates = []

    # ── Tier 1: CLAHE on raw gray (fastest, most reliable on clean card text) ──
    clahe_base = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    candidates.append(("gray_clahe", clahe_base))

    windows = [(100, 110), (95, 115), (90, 120), (105, 118)]
    for low, high in windows:
        leveled       = window_levels(gray, low, high)
        leveled_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(leveled)
        blackhat      = cv2.morphologyEx(
            leveled_clahe,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        )
        enhanced = cv2.addWeighted(leveled_clahe, 0.80, blackhat, 0.35, 0)

        # ── Tier 2: Otsu — sharp binary, prioritised before adaptive ──
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu     = clean_binary(otsu)
        otsu_inv = clean_binary(otsu_inv)
        candidates.append((f"lvl_{low}_{high}_otsu",     otsu))
        candidates.append((f"lvl_{low}_{high}_otsu_inv", otsu_inv))

        # ── Tier 3: windowed + CLAHE enhanced ──
        candidates.append((f"lvl_{low}_{high}_clahe", leveled_clahe))
        candidates.append((f"lvl_{low}_{high}_enh",   enhanced))

        # ── Tier 4: adaptive threshold ──
        th = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        th_inv = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 2
        )
        th     = clean_binary(th)
        th_inv = clean_binary(th_inv)
        candidates.append((f"lvl_{low}_{high}_th",  th))
        candidates.append((f"lvl_{low}_{high}_inv", th_inv))

        # ── Tier 5: raw windowed — rarely needed ──
        candidates.append((f"lvl_{low}_{high}", leveled))

    return candidates


def run_tesseract_on_candidate(candidate_img, psm=7, oem=3, whitelist=""):
    up = upscale_for_ocr(candidate_img, scale=3)
    config_str = f"--oem {oem} --psm {psm}"
    if whitelist:
        config_str += f" -c tessedit_char_whitelist={whitelist}"

    text = pytesseract.image_to_string(up, config=config_str).strip()
    text = " ".join(text.split())

    data = pytesseract.image_to_data(up, config=config_str, output_type=pytesseract.Output.DICT)
    confs = [
        float(c)
        for c in data.get("conf", [])
        if str(c).replace(".", "", 1).isdigit() and float(c) >= 0
    ]

    mean_conf = sum(confs) / len(confs) if confs else -1.0
    return text, mean_conf, up


# Confidence threshold (0–100) at which OCR is considered good enough to stop
# trying further preprocessing candidates. Saves 60–80% of Tesseract calls on
# clean card images. Lower this if accuracy drops on difficult scans.
OCR_CONFIDENCE_THRESHOLD = 75.0

# Maximum number of preprocessing candidates to try per field before giving up.
# Full candidate set is ~30. Capping at 12 covers the most effective variants
# without exhausting all permutations on every field.
OCR_MAX_CANDIDATES = 12


def preprocess_and_ocr_best(roi, field_name="field", psm=7, oem=3, whitelist=""):
    candidates = preprocess_roi_for_ocr_candidates(roi)
    best = {"text": "", "conf": -1.0, "variant": "none", "image": None}

    debug_dir = os.path.join(app.config["UPLOAD_FOLDER"], "debug")
    os.makedirs(debug_dir, exist_ok=True)

    for variant_name, candidate_img in candidates[:OCR_MAX_CANDIDATES]:
        text, conf, up = run_tesseract_on_candidate(
            candidate_img, psm=psm, oem=oem, whitelist=whitelist
        )

        debug_path = os.path.join(debug_dir, f"{field_name}_{variant_name}.jpg")
        cv2.imwrite(debug_path, up)

        if text.strip() and conf > best["conf"]:
            best["text"]    = text
            best["conf"]    = conf
            best["variant"] = variant_name
            best["image"]   = up

        # Early exit — confidence is good enough, no need to try more variants
        if best["conf"] >= OCR_CONFIDENCE_THRESHOLD:
            break

    # Fallback: if no candidate produced text at all, run the first one regardless
    if best["image"] is None and candidates:
        variant_name, candidate_img = candidates[0]
        text, conf, up = run_tesseract_on_candidate(
            candidate_img, psm=psm, oem=oem, whitelist=whitelist
        )
        best["text"]    = text
        best["conf"]    = conf
        best["variant"] = variant_name
        best["image"]   = up

    if best["image"] is not None:
        best_path = os.path.join(debug_dir, f"{field_name}_BEST_{best['variant']}.jpg")
        cv2.imwrite(best_path, best["image"])

    return best


def load_template(template_name="product_label"):
    template_path = os.path.join(app.config["ROI_TEMPLATE_FOLDER"], f"{template_name}.json")
    if not os.path.exists(template_path):
        template_path = os.path.join(app.config["ROI_TEMPLATE_FOLDER"], "product_label.json")

    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_fields_to_image_space(fields_dict, original_w, original_h, preview_w=None, preview_h=None):
    """
    Convert ROI field coordinates to absolute pixel values against the actual image.

    Percentage format (preferred):
      x_pct, y_pct, w_pct, h_pct — floats 0.0–100.0
      Converted directly using original_w / original_h. preview dimensions ignored.

    Legacy pixel format (backwards-compatible):
      x, y, w, h — pixel values relative to preview display size.
      Scaled to image space using preview_w / preview_h ratio.
    """
    if not isinstance(fields_dict, dict):
        return {}

    normalized = {}
    for field, coords in fields_dict.items():
        if not isinstance(coords, dict):
            continue

        if "x_pct" in coords:
            try:
                x_pct = max(0.0, min(100.0, float(coords.get("x_pct", 0))))
                y_pct = max(0.0, min(100.0, float(coords.get("y_pct", 0))))
                w_pct = max(0.1, min(100.0, float(coords.get("w_pct", 10))))
                h_pct = max(0.1, min(100.0, float(coords.get("h_pct", 10))))

                x = int(round(x_pct / 100.0 * original_w))
                y = int(round(y_pct / 100.0 * original_h))
                w = max(1, int(round(w_pct / 100.0 * original_w)))
                h = max(1, int(round(h_pct / 100.0 * original_h)))
            except Exception:
                continue
        else:
            sx = sy = 1.0
            if preview_w and preview_h:
                try:
                    pw = float(preview_w)
                    ph = float(preview_h)
                    if pw > 0 and ph > 0:
                        sx = float(original_w) / pw
                        sy = float(original_h) / ph
                except Exception:
                    pass

            x = max(0, int(round(float(coords.get("x", 0)) * sx)))
            y = max(0, int(round(float(coords.get("y", 0)) * sy)))
            w = max(1, int(round(float(coords.get("w", 100)) * sx)))
            h = max(1, int(round(float(coords.get("h", 50)) * sy)))

        normalized[field] = {
            "x": x, "y": y, "w": w, "h": h,
            "config": coords.get("config", {}),
        }

    return normalized


def ocr_with_custom_fields(image_path, fields_dict):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}

    h_img, w_img = img.shape[:2]
    results = {}

    for field, coords in fields_dict.items():
        x, y, w, h = coords["x"], coords["y"], coords["w"], coords["h"]
        x2 = min(x + w, w_img)
        y2 = min(y + h, h_img)

        if x2 - x <= 10 or y2 - y <= 10:
            results[field] = ""
            continue

        roi = img[y:y2, x:x2]
        field_config = coords.get("config", {})
        psm = field_config.get("psm", 7)
        oem = field_config.get("oem", 3)
        whitelist = field_config.get("whitelist", "")

        best = preprocess_and_ocr_best(roi, field_name=field, psm=psm, oem=oem, whitelist=whitelist)
        results[field] = best["text"]
        results[f"{field}__ocr_conf"] = round(best["conf"], 2) if best["conf"] >= 0 else -1
        results[f"{field}__ocr_variant"] = best["variant"]

    return results


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


def create_scan_record(image_path, template_name, extracted, normalized_fields):
    matched_product = match_product_from_extracted(extracted)

    record = ScanRecord(
        image_path=normalize_to_upload_relative(image_path),
        template_used=template_name,
        extracted_data={**extracted, "__roi_fields_used": normalized_fields},
        matched_product_id=matched_product.id if matched_product else None,
    )
    db.session.add(record)
    db.session.commit()

    return matched_product, record


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


_HOLOGRAPHIC_OPTIONS = ("None", "Regular", "Reverse", "Shiny Text", "Special")


def _get_holographic(data: dict) -> str:
    """Normalise holographic to one of the known option strings. Absence → 'None'."""
    raw = str(data.get("holographic", "")).strip()
    return raw if raw in _HOLOGRAPHIC_OPTIONS else "None"


def _get_finalized(data: dict) -> bool:
    """Normalise finalized to a strict bool. Absence → False."""
    raw = data.get("finalized", False)
    return raw is True or str(raw).strip().lower() == "true"


def build_inventory_group_info(records: list) -> dict:
    """
    Group inventory records that represent the same physical card so the
    inventory table can display a stacked Qty badge.

    Matching criteria — ALL five must be equal:
      - name         (case-insensitive, first non-empty of _NAME_KEYS)
      - serial       (case-insensitive, first non-empty of _SERIAL_KEYS)
      - edition      (normalised string via _get_edition)
      - holographic  (normalised string via _get_holographic)
      - finalized    must be True for ALL records in the group

    Records that are missing name or serial, or where finalized is False,
    are never grouped — they appear as individual rows (count = 1).

    Returns a dict keyed by record.id:
      {
        record_id: {
          "count":     int,          # total records in the group
          "all_ids":   [int, ...],   # every record.id in the group
        },
        ...
      }
    Records not part of any multi-record group still get an entry with
    count=1 and all_ids=[record.id] so the template never needs a fallback.
    """
    # First pass: build candidate groups
    groups_map: dict = {}
    ungrouped_ids: list = []

    for record in records:
        data   = record.extracted_data or {}
        name   = _get_name(data)
        serial = _get_serial(data)

        if not name or not serial or not _get_finalized(data):
            ungrouped_ids.append(record.id)
            continue

        edition    = _get_edition(data)
        holographic = _get_holographic(data)
        key = (name, serial, edition, holographic)
        groups_map.setdefault(key, []).append(record.id)

    # Second pass: build the output dict
    result: dict = {}

    for ids in groups_map.values():
        if len(ids) == 1:
            # Only one record matched this key — treat as ungrouped
            ungrouped_ids.append(ids[0])
        else:
            for rid in ids:
                result[rid] = {"count": len(ids), "all_ids": ids}

    for rid in ungrouped_ids:
        result[rid] = {"count": 1, "all_ids": [rid]}

    return result


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
        )
    return dict(
        build_inventory_url=build_inventory_url,
        build_uploaded_file_url=build_uploaded_file_url,
    )


# ====================== PAGE ROUTES ======================
@app.route("/")
def index():
    return render_template("index.html", templates=get_template_names())


# Keys that are rendered as dedicated static columns or are internal/OCR metadata.
# They are excluded from the dynamic entry-field columns.
_STATIC_ENTRY_KEYS = frozenset({
    "game", "album", "page", "slot",
    "__roi_fields_used",
})
_INTERNAL_KEY_PREFIXES = ("__ocr_", "__")
_INTERNAL_KEY_SUFFIXES = ("__ocr_conf", "__ocr_variant")


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


@app.route("/inventory")
def inventory():
    page       = request.args.get("page", 1, type=int)
    per_page   = request.args.get("per_page", 50, type=int)
    search     = request.args.get("search", "").strip()
    f_game     = request.args.get("game", "").strip()
    f_album    = request.args.get("album", "").strip()
    f_template = request.args.get("template", "").strip()

    per_page = min(per_page, 200)

    query = ScanRecord.query.order_by(ScanRecord.scan_date.desc())

    if f_game:
        query = query.filter(ScanRecord.extracted_data["game"].astext == f_game)
    if f_album:
        query = query.filter(ScanRecord.extracted_data["album"].astext == f_album)
    if f_template:
        query = query.filter(ScanRecord.template_used == f_template)
    if search:
        query = query.filter(ScanRecord.extracted_data.cast(db.Text).ilike(f"%{search}%"))

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    # Discover dynamic entry fields from ALL filtered records (not just current page)
    # so the column set is stable across pages. Cap the scan to 500 rows for performance.
    all_sample = query.limit(500).all()
    entry_fields = discover_entry_fields(all_sample)

    # Build stacking groups for the current page only (all_ids references are
    # page-scoped; the Qty badge and bulk-delete still work correctly because
    # delete_scans accepts any list of IDs regardless of page).
    group_info = build_inventory_group_info(pagination.items)

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

    # Build query — same logic as /inventory, but no pagination
    query = ScanRecord.query.order_by(ScanRecord.scan_date.desc())
    if f_game:
        query = query.filter(ScanRecord.extracted_data["game"].astext == f_game)
    if f_album:
        query = query.filter(ScanRecord.extracted_data["album"].astext == f_album)
    if f_template:
        query = query.filter(ScanRecord.template_used == f_template)
    if search:
        query = query.filter(ScanRecord.extracted_data.cast(db.Text).ilike(f"%{search}%"))

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
        "tcg_url":  ("Price URL", lambda r: str(((r.extracted_data or {}).get("tcgplayer") or {}).get("url", ""))),
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
    """
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

    # Determine prev/next IDs using the same ordering as the inventory list:
    # scan_date DESC, id DESC (ties broken by id so the order is fully stable).
    #
    # "Previous" = the immediately preceding row in that ordering
    #              (scan_date > current OR same date with id > current)
    # "Next"     = the immediately following row
    #              (scan_date < current OR same date with id < current)

    prev_record = (
        ScanRecord.query
        .filter(
            db.or_(
                ScanRecord.scan_date > record.scan_date,
                db.and_(
                    ScanRecord.scan_date == record.scan_date,
                    ScanRecord.id > record.id,
                ),
            )
        )
        .order_by(ScanRecord.scan_date.asc(), ScanRecord.id.asc())
        .first()
    )

    next_record = (
        ScanRecord.query
        .filter(
            db.or_(
                ScanRecord.scan_date < record.scan_date,
                db.and_(
                    ScanRecord.scan_date == record.scan_date,
                    ScanRecord.id < record.id,
                ),
            )
        )
        .order_by(ScanRecord.scan_date.desc(), ScanRecord.id.desc())
        .first()
    )

    return render_template(
        "inventory_detail.html",
        record=record,
        prev_id=prev_record.id if prev_record else None,
        next_id=next_record.id if next_record else None,
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
    return render_template("albums.html", albums=album_list)


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

    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    ensure_dirs()

    safe_name = secure_filename(file.filename)
    stem, ext = os.path.splitext(safe_name)
    if not ext:
        ext = ".png"

    final_name = f"record_{record_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
    relative_path = normalize_to_upload_relative(os.path.join("inventory_cards", final_name))
    absolute_path = os.path.join(app.config["UPLOAD_FOLDER"], relative_path)
    file.save(absolute_path)

    old_path = record.image_path
    record.image_path = relative_path
    db.session.commit()

    if old_path and normalize_to_upload_relative(old_path) != relative_path:
        remove_file_if_exists(old_path)

    return jsonify({
        "status":    "success",
        "message":   "Image updated successfully",
        "image_url": build_uploaded_file_url(record.image_path),
    })


@app.route("/delete_scan/<int:record_id>", methods=["POST"])
def delete_scan(record_id):
    record = ScanRecord.query.get_or_404(record_id)
    image_path = record.image_path
    db.session.delete(record)
    db.session.commit()
    remove_file_if_exists(image_path)
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

    image_paths = [r.image_path for r in records]
    for record in records:
        db.session.delete(record)
    db.session.commit()

    for image_path in image_paths:
        remove_file_if_exists(image_path)

    return jsonify({"status": "success", "message": f"Deleted {len(records)} record(s)"})


@app.route("/add_custom_field", methods=["POST"])
def add_custom_field():
    """
    Add a custom key/value field to inventory records.
    Accepts either:
      { game, key, value }        — applies to ALL records matching the game name
      { record_ids, key, value }  — applies to explicit list of record IDs
    """
    data  = request.get_json() or {}
    key   = data.get("key",   "").strip()
    value = data.get("value", "").strip()

    if not key or not value:
        return jsonify({"status": "error", "message": "Key and value are required"}), 400

    game       = data.get("game", "").strip()
    record_ids = data.get("record_ids", [])

    if game:
        all_rows = (
            ScanRecord.query
            .with_entities(ScanRecord.id, ScanRecord.extracted_data)
            .all()
        )
        matching_ids = []
        for row_id, extracted_data in all_rows:
            if isinstance(extracted_data, dict):
                row_game = extracted_data.get("game", "")
            else:
                try:
                    row_game = json.loads(extracted_data or "{}").get("game", "")
                except (ValueError, TypeError):
                    row_game = ""
            if str(row_game).strip() == game:
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


# ====================== TEMPLATE SAVE ROUTE ======================
@app.route("/template_save", methods=["POST"])
def template_save():
    """
    Save a new percentage-based ROI template JSON file from the template builder.
    Fields must use x_pct, y_pct, w_pct, h_pct (0.0–100.0).
    """
    data   = request.get_json() or {}
    name   = data.get("name", "").strip()
    fields = data.get("fields", {})

    if not name:
        return jsonify({"status": "error", "message": "Template name is required"}), 400

    clean_name = _re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_"))
    if not clean_name:
        return jsonify({"status": "error", "message": "Template name contains no valid characters"}), 400

    if not isinstance(fields, dict) or not fields:
        return jsonify({"status": "error", "message": "At least one field is required"}), 400

    cleaned_fields = {}
    for field_name, coords in fields.items():
        if not isinstance(coords, dict):
            continue

        field_key = _re.sub(r"[^a-z0-9_]", "", str(field_name).lower().replace(" ", "_"))
        if not field_key:
            continue

        if "x_pct" not in coords:
            return jsonify({
                "status":  "error",
                "message": f"Field '{field_name}' missing percentage coordinates (x_pct, y_pct, w_pct, h_pct).",
            }), 400

        try:
            x_pct = float(coords["x_pct"])
            y_pct = float(coords["y_pct"])
            w_pct = float(coords["w_pct"])
            h_pct = float(coords["h_pct"])
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": f"Invalid coordinates for field '{field_name}'"}), 400

        cleaned_fields[field_key] = {
            "x_pct":  round(max(0.0, min(100.0, x_pct)), 4),
            "y_pct":  round(max(0.0, min(100.0, y_pct)), 4),
            "w_pct":  round(max(0.1, min(100.0, w_pct)), 4),
            "h_pct":  round(max(0.1, min(100.0, h_pct)), 4),
            "config": coords.get("config", {}),
        }

    if not cleaned_fields:
        return jsonify({"status": "error", "message": "No valid fields provided"}), 400

    ensure_dirs()
    template_path = os.path.join(app.config["ROI_TEMPLATE_FOLDER"], f"{clean_name}.json")

    try:
        with open(template_path, "w", encoding="utf-8") as f:
            json.dump({"name": clean_name, "fields": cleaned_fields}, f, indent=2)
    except OSError as exc:
        return jsonify({"status": "error", "message": f"Could not write template file: {exc}"}), 500

    return jsonify({
        "status":  "success",
        "message": f"Template '{clean_name}' saved with {len(cleaned_fields)} field(s)",
        "name":    clean_name,
        "fields":  cleaned_fields,
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


# ====================== TCGPLAYER ROUTES ======================
def _extract_tcgplayer_id(url: str):
    """Extract numeric product ID from a TCGPlayer URL."""
    match = _re.search(r"tcgplayer\.com/product/(\d+)", url or "")
    return match.group(1) if match else None


@app.route("/tcg_save_url/<int:record_id>", methods=["POST"])
def tcg_save_url(record_id):
    """Save a price-lookup URL to the record. Accepts any valid URL.
    If it's a TCGPlayer product URL the product ID is also extracted and stored."""
    record = ScanRecord.query.get_or_404(record_id)
    data   = request.get_json() or {}
    url    = data.get("url", "").strip()

    if not url:
        return jsonify({"status": "error", "message": "No URL provided"}), 400

    # Basic sanity check — must look like a URL
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url

    # Optionally extract a TCGPlayer product ID if the URL happens to be one
    product_id = _extract_tcgplayer_id(url)

    entry = {
        "url":      url,
        "full_url": url,
        "saved_at": datetime.utcnow().isoformat(),
    }
    if product_id:
        entry["product_id"] = product_id

    updated = dict(record.extracted_data or {})
    updated["tcgplayer"] = entry
    record.extracted_data = updated
    db.session.commit()

    return jsonify({
        "status":     "success",
        "message":    "URL saved",
        "url":        url,
        "product_id": product_id or "",
    })


@app.route("/tcg_clear_url/<int:record_id>", methods=["POST"])
def tcg_clear_url(record_id):
    """Remove the saved TCGPlayer URL from this record."""
    record  = ScanRecord.query.get_or_404(record_id)
    updated = dict(record.extracted_data or {})
    updated.pop("tcgplayer", None)
    record.extracted_data = updated
    db.session.commit()
    return jsonify({"status": "success", "message": "TCGPlayer link removed"})


# ====================== DUPLICATE IMAGE MANAGER ROUTES ======================
@app.route("/duplicates")
def duplicates():
    """
    Group records that share all three of:
      - name         (case-insensitive, first non-empty of _NAME_KEYS)
      - serial       (case-insensitive, first non-empty of _SERIAL_KEYS)
      - edition      (normalised string)

    Used for IMAGE deduplication only (not inventory stacking).
    All three must match. Records missing name or serial are excluded.
    Groups where all members already share the same image_path are also
    excluded — they are already resolved and won't reappear after a reload.
    """
    records = ScanRecord.query.order_by(ScanRecord.scan_date.desc()).all()

    groups_map = {}
    for record in records:
        data     = record.extracted_data or {}
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
        # Skip groups where all image paths are already identical (already resolved)
        if len(set(r.image_path for r in recs)) <= 1:
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

    All non-canonical records are repointed to the canonical image_path.
    Orphaned image files are deleted from disk after a successful commit.
    After resolution all records share one image_path, so the group is
    filtered out on the next /duplicates load.
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

    to_delete = []
    updated   = 0

    for record in records:
        if record.id == canonical_id:
            continue
        old_path = record.image_path
        if old_path and old_path != canonical_path:
            to_delete.append(
                os.path.join(app.config["UPLOAD_FOLDER"], normalize_to_upload_relative(old_path))
            )
        record.image_path = canonical_path
        updated += 1

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Database error: {exc}"}), 500

    deleted_files = 0
    for path in to_delete:
        try:
            if os.path.isfile(path):
                os.remove(path)
                deleted_files += 1
        except OSError:
            pass

    return jsonify({
        "status":  "success",
        "message": (
            f"Image from Record #{canonical_id} applied to {updated} record(s). "
            f"{deleted_files} orphaned file(s) deleted."
        ),
    })


# ====================== MAIN OCR ROUTES ======================
@app.route("/preview", methods=["POST"])
def preview():
    ensure_dirs()

    files = request.files.getlist("images")
    if not files or not files[0].filename:
        return jsonify({"error": "No image selected"}), 400

    template_name = request.form.get("template", "product_label")
    template      = load_template(template_name)

    file      = files[0]
    filename  = secure_filename(file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(image_path)

    return jsonify({
        "image_url": url_for("uploaded_file", filename=filename),
        "filename":  filename,
        "template":  template,
    })


@app.route("/run_custom_ocr", methods=["POST"])
def run_custom_ocr():
    data            = request.get_json() or {}
    filename        = data.get("image_path")
    adjusted_fields = data.get("fields", {})

    if not filename:
        return jsonify({"error": "Missing image_path"}), 400

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": f"Could not read image: {filename}"}), 400

    h_img, w_img = img.shape[:2]
    preview_meta = data.get("preview_meta", {})

    normalized_fields = normalize_fields_to_image_space(
        adjusted_fields, w_img, h_img,
        preview_meta.get("width"), preview_meta.get("height"),
    )

    extracted = ocr_with_custom_fields(image_path, normalized_fields)

    # Merge in the manually supplied location/game metadata
    for key in ("game", "album", "page", "slot"):
        val = str(data.get(key, "")).strip()
        if val:
            extracted[key] = val

    # Delete any existing record(s) occupying the same game/album/page/slot
    # so the new scan cleanly replaces whatever was there before.
    game  = extracted.get("game",  "")
    album = extracted.get("album", "")
    page  = extracted.get("page",  "")
    slot  = extracted.get("slot",  "")

    if game and album and page and slot:
        existing = [
            r for r in ScanRecord.query.all()
            if (
                str((r.extracted_data or {}).get("game",  "")).strip() == game  and
                str((r.extracted_data or {}).get("album", "")).strip() == album and
                str((r.extracted_data or {}).get("page",  "")).strip() == page  and
                str((r.extracted_data or {}).get("slot",  "")).strip() == slot
            )
        ]
        old_image_paths = [r.image_path for r in existing]
        for old in existing:
            db.session.delete(old)
        db.session.commit()
        for old_image in old_image_paths:
            if old_image and old_image != "__blank__":
                remove_file_if_exists(old_image)

    matched_product, record = create_scan_record(
        image_path=filename,
        template_name=data.get("template_name", "custom"),
        extracted=extracted,
        normalized_fields=normalized_fields,
    )

    brand        = extracted.get("brand", "").lower().strip()
    product_name = extracted.get("product_name", "").lower().strip()
    display_key  = f"{brand.title()} {product_name.title()}".strip() or "Unknown Item"

    return jsonify({
        "status":          "success",
        "key":             display_key,
        "record_id":       record.id,
        "image_url":       build_uploaded_file_url(record.image_path),
        "extracted":       extracted,
        "matched_product": matched_product.product_name if matched_product else "No match",
    })


# ====================== IMPORT SPLIT / ALIGN ROUTES ======================
@app.route("/run_import_split", methods=["POST"])
def run_import_split():
    ensure_dirs()

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No image provided"}), 400

    game  = request.form.get("game",  "").strip()
    album = request.form.get("album", "").strip()
    page  = request.form.get("page",  "").strip()

    if not game or not album or not page:
        return jsonify({"status": "error", "message": "Game, album, and page are required"}), 400

    try:
        v_cut1       = float(request.form.get("vcut1",      33))
        v_cut2       = float(request.form.get("vcut2",      66))
        h_cut1       = float(request.form.get("hcut1",      33))
        h_cut2       = float(request.form.get("hcut2",      66))
        canny_low    = int(request.form.get("cannylow",     50))
        canny_high   = int(request.form.get("cannyhigh",   200))
        approx_eps   = float(request.form.get("approxeps",   0.02))
        min_area_pct = float(request.form.get("minareapct",  0.05))
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid numeric form values"}), 400

    if not (v_cut1 < v_cut2 and h_cut1 < h_cut2):
        return jsonify({"status": "error", "message": "Cut 1 must be less than Cut 2"}), 400

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
        for idx, piece in enumerate(pieces, start=1):
            slot_filename = f"{safe_game}-{safe_album}-{safe_page}-{idx}.png"
            split_path    = os.path.join(app.config["TEMP_SPLIT_FOLDER"], slot_filename)
            piece.save(split_path)

            try:
                split_cv = cv2.imread(split_path)
                if split_cv is None:
                    raise ValueError("Could not read split image")

                processed = process_card_image(
                    split_cv,
                    canny_low=canny_low, canny_high=canny_high,
                    approx_eps=approx_eps, min_area_pct=min_area_pct,
                )

                card_path = os.path.join(app.config["TEMP_CARD_FOLDER"], slot_filename)
                cv2.imwrite(card_path, processed)

                results.append({
                    "slot": idx, "filename": slot_filename,
                    "status": "processed",
                    "url": url_for("temp_card_file", filename=slot_filename),
                })
            except Exception:
                results.append({
                    "slot": idx, "filename": slot_filename,
                    "status": "fallback",
                    "url": url_for("temp_split_file", filename=slot_filename),
                })

        processed_count = sum(1 for r in results if r["status"] == "processed")
        fallback_count  = sum(1 for r in results if r["status"] == "fallback")

        return jsonify({
            "status":  "success",
            "message": f"Import completed. {processed_count} auto-processed, {fallback_count} fallback.",
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


# ====================== BATCH OCR IMPORT ROUTE ======================
@app.route("/import_run_ocr_batch", methods=["POST"])
def import_run_ocr_batch():
    data      = request.get_json() or {}
    template_name = data.get("template_name", "product_label")
    filenames     = data.get("filenames", [])

    if not filenames or len(filenames) != 9:
        return jsonify({
            "status":  "error",
            "message": "Exactly 9 aligned card filenames are required",
        }), 400

    try:
        template = load_template(template_name)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Could not load template: {e}"}), 400

    fields  = template.get("fields", {})
    results = []

    for filename in filenames:
        temp_image_path = os.path.join(app.config["TEMP_CARD_FOLDER"], filename)

        if not os.path.exists(temp_image_path):
            results.append({"filename": filename, "status": "error", "message": "Aligned card image not found"})
            continue

        img = cv2.imread(temp_image_path)
        if img is None:
            results.append({"filename": filename, "status": "error", "message": "Could not read aligned card image"})
            continue

        h_img, w_img      = img.shape[:2]
        normalized_fields = normalize_fields_to_image_space(fields, w_img, h_img)
        extracted         = ocr_with_custom_fields(temp_image_path, normalized_fields)

        filename_fields = parse_card_filename(filename)
        extracted.update(filename_fields)

        try:
            final_relative_image_path = move_temp_card_to_inventory(filename)
            matched_product, record   = create_scan_record(
                image_path=final_relative_image_path,
                template_name=template_name,
                extracted=extracted,
                normalized_fields=normalized_fields,
            )
            results.append({
                "filename":        filename,
                "status":          "success",
                "record_id":       record.id,
                "image_url":       build_uploaded_file_url(record.image_path),
                "extracted":       extracted,
                "matched_product": matched_product.product_name if matched_product else "No match",
            })
        except Exception as e:
            results.append({"filename": filename, "status": "error", "message": str(e)})

    success_count = sum(1 for r in results if r["status"] == "success")

    return jsonify({
        "status":  "success",
        "message": f"OCR import completed for {success_count} of {len(filenames)} cards",
        "results": results,
    })


# ====================== START ======================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    ensure_dirs()

    print("OCR Inventory Scanner")
    print(" • /             — Scan page with template builder")
    print(" • /inventory    — Inventory list with server-side pagination and filters")
    print(" • /inventory/<id> — Record detail with edit, TCGPlayer link, copy-from, edition")
    print(" • /inventory/filter_options — Dropdown options API")
    print(" • /duplicates   — Duplicate image manager (name + serial + edition)")
    print(" • /albums       — Album grid view")
    print(" • /import       — 3x3 split, alignment, manual corner override, batch OCR")
    print(" • /records_summary — Copy-from dropdown API")
    print(" • /template_save   — Save new percentage-based ROI template")
    print(" • /tcg_save_url/<id>, /tcg_clear_url/<id> — TCGPlayer link management")
    print(" • Visit: http://127.0.0.1:5000")

    app.run(host="0.0.0.0", port=5005, debug=True)
