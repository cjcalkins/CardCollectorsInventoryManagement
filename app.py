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
from dotenv import load_dotenv
load_dotenv()

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
        "game": parts[0].replace("_", " "),
        "album": "-".join(parts[1:-2]).replace("_", " "),
        "page": parts[-2],
        "slot": parts[-1],
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
    # Dedicated static columns and legacy/superseded keys that must never
    # surface as dynamic ad-hoc text columns.
    "edition", "holographic", "finalized", "tcgplayer",
    "first_edition", "limited_edition",
    "empty",
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
    for (extracted_data,) in rows:
        if isinstance(extracted_data, dict):
            data = extracted_data
        else:
            try:
                data = json.loads(extracted_data or "{}")
            except (ValueError, TypeError):
                data = {}

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
    return render_template("inventory_game_select.html", games=games)


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

    # If no filter is active, show the game selection landing page.
    if not f_game and not f_album and not f_template and not search:
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
    # list can render boolean toggles and respect dropdown options.
    template_fields_config: dict = {}
    for tpl_name in get_template_names():
        try:
            tpl = load_template(tpl_name)
            for fk, fv in (tpl.get("fields") or {}).items():
                if fk not in template_fields_config and isinstance(fv, dict):
                    template_fields_config[fk] = {
                        "field_type":       fv.get("field_type", "text"),
                        "dropdown_options": fv.get("dropdown_options", []),
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
    # Python-side filtering for game/album for SQLite compatibility (.astext is PostgreSQL-only)
    query = ScanRecord.query.order_by(ScanRecord.scan_date.desc())
    if f_template:
        query = query.filter(ScanRecord.template_used == f_template)
    if search:
        query = query.filter(ScanRecord.extracted_data.cast(db.Text).ilike(f"%{search}%"))

    records = query.all()
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


@app.route("/inventory/all_ids")
def inventory_all_ids():
    """
    Return all ScanRecord IDs that match the current filter params
    (search, game, album, template).  Used by the "Select All in Filter"
    button to collect IDs across every page before a bulk operation.
    """
    search     = request.args.get("search",   "").strip()
    f_game     = request.args.get("game",     "").strip()
    f_album    = request.args.get("album",    "").strip()
    f_template = request.args.get("template", "").strip()

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

        # ── field_type: "text" | "dropdown" | "boolean" ──
        raw_field_type = str(coords.get("field_type", "text")).strip().lower()
        if raw_field_type not in ("text", "dropdown", "boolean"):
            raw_field_type = "text"

        # dropdown_options: list of non-empty strings (only meaningful for dropdown)
        raw_opts = coords.get("dropdown_options", [])
        if isinstance(raw_opts, list):
            dropdown_options = [str(o).strip() for o in raw_opts if str(o).strip()]
        else:
            dropdown_options = []

        entry = {
            "x_pct":      round(max(0.0, min(100.0, x_pct)), 4),
            "y_pct":      round(max(0.0, min(100.0, y_pct)), 4),
            "w_pct":      round(max(0.1, min(100.0, w_pct)), 4),
            "h_pct":      round(max(0.1, min(100.0, h_pct)), 4),
            "config":     coords.get("config", {}),
            "field_type": raw_field_type,
        }
        if raw_field_type == "dropdown":
            entry["dropdown_options"] = dropdown_options

        cleaned_fields[field_key] = entry

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


# ====================== TEMPLATE CONFIG (live field types) ======================
@app.route("/template_config/<template_name>")
def template_config(template_name):
    """
    Return the live field config for a given template as JSON.
    Used by the inventory detail page to detect field-type changes
    made after the page was server-rendered.
    Shape: { fieldKey: { field_type, dropdown_options? }, … }
    """
    try:
        tpl = load_template(template_name or "product_label")
        fields = tpl.get("fields", {})
        # Return only field_type and dropdown_options — omit ROI coords
        slim = {
            k: {
                "field_type":       v.get("field_type", "text"),
                "dropdown_options": v.get("dropdown_options", []),
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


# ====================== BATCH OCR IMPORT ROUTE (SSE streaming) ======================
@app.route("/import_run_ocr_batch", methods=["POST"])
def import_run_ocr_batch():
    """
    Streams Server-Sent Events so the browser can show real per-card progress.

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
            yield sse("error", {"message": f"Could not load template: {e}"})
            return

        fields  = template.get("fields", {})
        results = []

        for idx, filename in enumerate(filenames, start=1):
            temp_image_path = os.path.join(app.config["TEMP_CARD_FOLDER"], filename)

            if not os.path.exists(temp_image_path):
                result = {"filename": filename, "status": "error", "message": "Aligned card image not found"}
                results.append(result)
                yield sse("progress", {"slot": idx, "total": len(filenames), "result": result})
                continue

            img = cv2.imread(temp_image_path)
            if img is None:
                result = {"filename": filename, "status": "error", "message": "Could not read aligned card image"}
                results.append(result)
                yield sse("progress", {"slot": idx, "total": len(filenames), "result": result})
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
                result = {
                    "filename":        filename,
                    "status":          "success",
                    "record_id":       record.id,
                    "image_url":       build_uploaded_file_url(record.image_path),
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
            "message": f"OCR import completed for {success_count} of {len(filenames)} cards",
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
    print(" • /migrate_clean_legacy_fields — POST: scrub first_edition, limited_edition, boolean holographic, empty")
    print(" • /albums       — Album grid view")
    print(" • /import       — 3x3 split, alignment, manual corner override, batch OCR")
    print(" • /records_summary — Copy-from dropdown API")
    print(" • /template_save   — Save new percentage-based ROI template")
    print(" • /justtcg_fetch/<id>              — Fetch live price from JustTCG API (POST, manual trigger)")
    print(" • /tcg_save_url/<id>, /tcg_clear_url/<id> — Legacy URL save / pricing data clear")
    print(" • Visit: http://127.0.0.1:5000")

    app.run(host="0.0.0.0", port=5005, debug=True)
