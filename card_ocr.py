"""
card_ocr.py — read the FRONT image of a card and match it to existing entries.

Pipeline
--------
1. NORMALIZE (registration): find the card in the frame (it may be off-centre,
   tilted, and surrounded by a binder page), deskew it to upright, crop away the
   background, and standardise its size. This makes the top/bottom bands land on
   the same place on every card regardless of how it was scanned. Detection is
   conservative: if a card-shaped region isn't found, the original image is used
   unchanged, so this never makes a good capture worse.

2. NAME: read the top "name zone" — the top strip minus the top-right HP box and
   the top-left evolution icon. The name is the largest text there; we try two
   binarisations (Otsu suits glossy modern silver nameplates, adaptive suits flat
   older ones) and keep the tallest word on its line.

3. NUMBER: scan the FULL-WIDTH bottom band for an N/M collector number. It lives
   bottom-right on older sets and bottom-left on newer ones, so we search the
   whole width rather than a fixed corner, and accept only plausible numbers
   (N <= M). On low-resolution scans the number may be too small to read at all,
   in which case identification falls back to the name + reference catalog.

Dependencies: `pip install pytesseract` plus the Tesseract binary
(Ubuntu: `apt install tesseract-ocr`). Missing binary -> ocr_available=False,
empty guesses, no exception.
"""

import re
from collections import Counter
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
    from pytesseract import Output
    _TESS_OK = True
except Exception:
    pytesseract = None
    Output = None
    _TESS_OK = False


_NAME_WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_NUM_WL  = "0123456789/"
_NUMBER_RE  = re.compile(r"(\d{1,4})\s*/\s*(\d{1,4})")
_SETCODE_RE = re.compile(r"\b(?=[A-Z0-9]{2,6}\b)[A-Z0-9]*[A-Z][A-Z0-9]*\b")
# Words that appear in the name zone but are never the name.
_NAME_NOISE = {"hp", "stage", "basic", "evolves", "from", "pokemon", "pokmon",
               "illus", "no", "the", "ex", "gx", "restored"}


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def _to_bgr(image_or_path):
    if isinstance(image_or_path, np.ndarray):
        return image_or_path
    if isinstance(image_or_path, Image.Image):
        return cv2.cvtColor(np.array(image_or_path.convert("RGB")), cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(image_or_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_or_path}")
    return img


def _clean_name(text):
    text = re.sub(r"\s*/\s*", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Normalisation: detect, deskew, crop, standardise
# --------------------------------------------------------------------------- #
def _order_quad(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")  # tl,tr,br,bl


def normalize_card_image(bgr, out_width=1000, debug_dir=None):
    """
    Return (normalized_bgr, did_normalize). Detects the card as the largest
    colourful (saturated) region — which separates it from a white/grey binder
    page — deskews it to upright via a perspective warp, crops away the
    background, and standardises the card to `out_width` pixels wide.

    Detection runs on a downscaled copy for speed, but the warp is applied to the
    full-resolution image so no detail is lost before the final standardisation.
    Conservative: returns (original, False) if no card-shaped region is found, so
    it can be applied unconditionally without risking good captures.
    """
    try:
        H, W = bgr.shape[:2]
        # Detect on a copy no larger than ~1000px on its long side (fast morphology).
        sf = 1000.0 / max(H, W) if max(H, W) > 1000 else 1.0
        small = cv2.resize(bgr, (int(W * sf), int(H * sf)), interpolation=cv2.INTER_AREA) if sf < 1.0 else bgr
        sh, sw = small.shape[:2]

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.threshold(hsv[:, :, 1], 45, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return bgr, False

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 0.18 * sw * sh:        # detection too small to trust
            return bgr, False
        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect
        long_side, short_side = max(rw, rh), min(rw, rh)
        if short_side <= 0 or not (1.2 <= long_side / short_side <= 1.6):
            return bgr, False                          # not card-shaped -> trust original

        quad = _order_quad(cv2.boxPoints(rect)) / sf   # scale corners back to full res
        tl, tr, br, bl = quad
        Wc = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        Hc = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if Wc < 10 or Hc < 10:
            return bgr, False
        dst = np.array([[0, 0], [Wc - 1, 0], [Wc - 1, Hc - 1], [0, Hc - 1]], dtype="float32")
        warp = cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(quad.astype("float32"), dst), (Wc, Hc))
        if warp.shape[1] > warp.shape[0]:              # ensure portrait
            warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)

        scale = out_width / warp.shape[1]
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        warp = cv2.resize(warp, (out_width, int(warp.shape[0] * scale)), interpolation=interp)

        if debug_dir:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "normalized.png"), warp)
        return warp, True
    except Exception:
        return bgr, False


# --------------------------------------------------------------------------- #
# Name: largest text in the name zone (HP + evo-icon excluded)
# --------------------------------------------------------------------------- #
def _name_from_binary(im):
    """Best name candidate + its anchor height from one binarised name-zone image."""
    cfg = f"--oem 3 --psm 11 -c tessedit_char_whitelist={_NAME_WL}"
    try:
        d = pytesseract.image_to_data(im, config=cfg, output_type=Output.DICT)
    except Exception:
        return "", 0.0

    words = []  # (text, height, left, width, center_y)
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        if len(t) >= 3 and t.lower() not in _NAME_NOISE:
            h = d["height"][i]
            words.append((t, h, d["left"][i], d["width"][i], d["top"][i] + h / 2.0))
    if not words:
        return "", 0.0

    max_h = max(w[1] for w in words)
    tall = [w for w in words if w[1] >= 0.6 * max_h]
    ref = max(tall, key=lambda w: w[1])
    ref_cy, ref_h = ref[4], ref[1]
    line = sorted([w for w in tall if abs(w[4] - ref_cy) <= 0.7 * max_h], key=lambda w: w[2])

    anchor = min(range(len(line)), key=lambda i: abs(line[i][2] - ref[2]))
    keep = [line[anchor]]
    for j in range(anchor + 1, len(line)):
        p = line[j - 1]
        if line[j][2] - (p[2] + p[3]) <= 1.0 * max_h and line[j][1] >= 0.7 * ref_h:
            keep.append(line[j])
        else:
            break
    for j in range(anchor - 1, -1, -1):
        nx = line[j + 1]
        if nx[2] - (line[j][2] + line[j][3]) <= 1.0 * max_h and line[j][1] >= 0.7 * ref_h:
            keep.insert(0, line[j])
        else:
            break
    return _clean_name(" ".join(w[0] for w in keep)), ref_h


def _read_name(bgr, top=0.15, x0=0.10, x1=0.80):
    """Read the card name from the top zone, trying two binarisations and
    keeping whichever renders the name tallest (i.e. clearest)."""
    if not _TESS_OK:
        return "", -1.0
    H, W = bgr.shape[:2]
    crop = bgr[0:int(H * top), int(W * x0):int(W * x1)]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 5, 40, 40)

    otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adap = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 35, 10)
    cands = []
    for im in (otsu, adap):
        nm, h = _name_from_binary(im)
        if nm:
            cands.append((nm, h))
    if not cands:
        return "", 0.0
    cands.sort(key=lambda c: (c[1], -len(c[0].split())), reverse=True)
    return cands[0][0], 0.0


# --------------------------------------------------------------------------- #
# Number: full-width bottom band (bottom-left OR bottom-right), plausible only
# --------------------------------------------------------------------------- #
def _plausible_number(n, m):
    return 1 <= n <= m <= 2000


def _read_number(bgr, band_frac=0.13):
    """Return (normalized 'N/M', raw_text). Scans the full-width bottom band with
    several preprocessings/PSMs and returns the most frequently seen plausible
    N/M (robust to a single bad read). '' when nothing plausible is found."""
    if not _TESS_OK:
        return "", ""
    H = bgr.shape[0]
    band = bgr[int(H * (1.0 - band_frac)):H, :]
    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)

    hits, last_raw = [], ""
    for up in (2, 3):
        gg = cv2.resize(g, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        gg = cv2.bilateralFilter(gg, 5, 40, 40)
        variants = (
            cv2.adaptiveThreshold(gg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10),
            cv2.threshold(gg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        )
        for im in variants:
            for psm in (11, 7):
                cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={_NUM_WL}"
                try:
                    txt = pytesseract.image_to_string(im, config=cfg)
                except Exception:
                    continue
                last_raw = txt.strip() or last_raw
                for mm in _NUMBER_RE.finditer(txt.replace(" ", "")):
                    n, m = int(mm.group(1)), int(mm.group(2))
                    if _plausible_number(n, m):
                        hits.append(f"{n}/{m}")
        # Early exit once a clear winner has emerged (seen >=3 times).
        if hits:
            top, cnt = Counter(hits).most_common(1)[0]
            if cnt >= 3:
                return top, last_raw
    if hits:
        return Counter(hits).most_common(1)[0][0], last_raw
    return "", last_raw


def parse_collector_number(text):
    m = _NUMBER_RE.search(text or "")
    return f"{int(m.group(1))}/{int(m.group(2))}" if m else ""


def parse_set_code(text, drop=""):
    for tok in _SETCODE_RE.findall((text or "").upper()):
        if tok and tok != drop and not tok.isdigit():
            return tok
    return ""


# --------------------------------------------------------------------------- #
# Public: OCR a front image
# --------------------------------------------------------------------------- #
def ocr_card_front(image_or_path, normalize=True, debug_dir=None):
    """
    OCR the name and collector number from a card front.

    normalize=True (default) first deskews/crops/standardises the card, which is
    what lets fixed name/number zones work when the card is off-centre or tilted
    (e.g. photographed in a binder page).

    Returns a dict with: ocr_available, name_guess, number_guess, set_code_guess,
    raw_top, raw_bottom, conf_top, conf_bottom, normalized (bool).
    """
    bgr = _to_bgr(image_or_path)
    did_norm = False
    if normalize:
        bgr, did_norm = normalize_card_image(bgr, debug_dir=debug_dir)

    if debug_dir:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        H = bgr.shape[0]
        cv2.imwrite(os.path.join(debug_dir, "band_top.png"), bgr[0:int(H * 0.15), :])
        cv2.imwrite(os.path.join(debug_dir, "band_bottom.png"), bgr[int(H * 0.87):H, :])

    name_guess, conf_top = _read_name(bgr)
    number_guess, raw_bottom = _read_number(bgr)
    set_code_guess = parse_set_code(raw_bottom, drop=number_guess.replace("/", ""))

    return {
        "ocr_available": _TESS_OK,
        "name_guess": name_guess,
        "number_guess": number_guess,
        "set_code_guess": set_code_guess,
        "raw_top": name_guess,
        "raw_bottom": raw_bottom,
        "conf_top": conf_top,
        "conf_bottom": 100.0 if number_guess else -1.0,
        "normalized": did_norm,
    }


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def _norm_serial(s):
    s = str(s or "").strip()
    m = _NUMBER_RE.search(s)
    return f"{int(m.group(1))}/{int(m.group(2))}" if m else s.lower()


def _norm_name(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def match_ocr_to_records(ocr_result, candidates, name_base=0.65, serial_bonus=0.5,
                         threshold=0.45, limit=5):
    """
    Name-first scoring with a strong number bonus:
        score = min(1, name_base * name_ratio + (serial_bonus if serial matches))
    so a good name identifies a card even when the number can't be read, and an
    exact number match wins decisively when it can. Returns candidates (copied,
    with score/serial_match/name_similarity added) filtered to >= threshold.
    """
    ocr_name = _norm_name(ocr_result.get("name_guess"))
    ocr_serial = _norm_serial(ocr_result.get("number_guess"))
    has_serial = bool(ocr_serial)

    scored = []
    for cand in candidates:
        cand_name = _norm_name(cand.get("name"))
        cand_serial = _norm_serial(cand.get("serial"))
        serial_match = bool(has_serial and ocr_serial == cand_serial)
        name_ratio = (SequenceMatcher(None, ocr_name, cand_name).ratio()
                      if ocr_name and cand_name else 0.0)
        score = min(1.0, name_base * name_ratio + (serial_bonus if serial_match else 0.0))
        out = dict(cand)
        out["score"] = round(score, 3)
        out["serial_match"] = serial_match
        out["name_similarity"] = round(name_ratio, 3)
        scored.append(out)

    scored.sort(key=lambda c: c["score"], reverse=True)
    return [c for c in scored if c["score"] >= threshold][:limit]
