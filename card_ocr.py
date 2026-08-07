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
   the top-left evolution icon. The name is the largest text there, so we OCR the
   zone and keep the tallest recognised line after dropping stage/rarity/UI tokens
   (e.g. "Basic", "STAGE 1", "HP").

3. NUMBER: scan the two bottom CORNERS for an N/M collector number, skipping the
   noisy centre of the bottom band (flavour text, weakness/resistance/retreat).
   The number lives bottom-right on older sets and bottom-left on newer ones, so
   both corners are read, and only plausible numbers (N <= M) are accepted. The
   result is zero-padded to the set-total width (e.g. "28/162" -> "028/162"). On
   low-resolution scans the number may be too small to read at all, in which case
   identification falls back to the name + reference catalog.

OCR engine: RapidOCR (PP-OCRv5 *mobile* models via ONNX Runtime) — high accuracy
on real card captures (glossy nameplates, angled/curved text, low-res collector
numbers) while staying light enough for a Raspberry Pi. RapidOCR does its own text
detection + recognition, so this module just crops the name/number zones, runs the
shared engine, and applies the domain rules (name = largest text, plausible N/M
only).

Install:  pip install rapidocr onnxruntime
The PP-OCRv5 mobile det/rec models (~a few MB each) download automatically on the
first OCR call and are then cached; if the engine can't be built (no package, or
models can't be fetched offline) -> ocr_available=False, empty guesses, no
exception. The engine is configurable via environment variables (see below) so it
can be pinned to a bundled version or pointed at pre-downloaded models for fully
offline / air-gapped deployment.

Environment overrides (all optional):
    RAPIDOCR_OCR_VERSION  default "PP-OCRv5"   (e.g. "PP-OCRv6", "PP-OCRv4")
    RAPIDOCR_MODEL_TYPE   default "mobile"     (e.g. "server")
    RAPIDOCR_ENGINE       default "onnxruntime"
    RAPIDOCR_REC_LANG     default "en"         (recognition language; e.g. "latin", "ch")
    RAPIDOCR_DET_LANG     default "ch"         (v5 detection ships language-agnostic as "ch")
    RAPIDOCR_NUM_THREADS  default unset        (ONNX Runtime intra-op threads; e.g. "4" on a Pi)
    RAPIDOCR_USE_CLS      default "0"          ("1" to enable 180 deg text-line orientation)
"""

import logging
import os
import re
from collections import Counter
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class ImageDecodeError(ValueError):
    """A file exists but no decoder could read it. Distinct from
    FileNotFoundError, which this module used to raise for both cases —
    callers that tell "missing" from "corrupt" apart could not."""


# Inference failures are swallowed per crop so one bad band can't abort a scan,
# but a broken install fails EVERY crop — and the old code returned empty text
# while ocr_available() still said True, which reads exactly like a blank card.
# The counter lets a single ocr_card_front() call tell "found nothing" from
# "the engine raised on every read" and report it.
_ENGINE_ERRORS = {"count": 0, "last": ""}

# RapidOCR (PP-OCRv5 mobile via ONNX Runtime). Imported optionally so this module
# — and the whole app — still boots on installs that haven't run
# `pip install rapidocr onnxruntime` yet; the engine simply reports unavailable.
try:
    from rapidocr import RapidOCR, EngineType, ModelType, OCRVersion, LangDet, LangRec
    _RAPIDOCR_OK = True
except Exception:
    RapidOCR = None
    EngineType = ModelType = OCRVersion = LangDet = LangRec = None
    _RAPIDOCR_OK = False


def _env(name, default):
    v = os.environ.get(name)
    v = v.strip() if v else ""
    return v or default


# Enable 180 deg text-line orientation classification (extra model + time). Off by
# default: the pipeline already deskews/uprights the card, so cls rarely helps.
_USE_CLS = _env("RAPIDOCR_USE_CLS", "0").lower() in ("1", "true", "yes", "on")

# The shared engine is built lazily on first use (constructing it may trigger a
# one-time model download) and cached. `_ENGINE_FAILED` latches so we don't retry
# a hopeless build (e.g. offline with no cached models) on every card.
_ENGINE = None
_ENGINE_FAILED = False


def _build_engine_params():
    """RapidOCR config dict. Defaults to PP-OCRv5 *mobile* det+rec on ONNX Runtime;
    every axis is overridable by environment variable for edge/offline tuning.
    Note: PP-OCRv5 detection ships only as a language-agnostic 'ch' model, so the
    detector language stays 'ch' while recognition defaults to English ('en')."""
    ver        = _env("RAPIDOCR_OCR_VERSION", "PP-OCRv5")
    model_type = _env("RAPIDOCR_MODEL_TYPE", "mobile")
    engine     = _env("RAPIDOCR_ENGINE", "onnxruntime")
    rec_lang   = _env("RAPIDOCR_REC_LANG", "en")
    det_lang   = _env("RAPIDOCR_DET_LANG", "ch")

    params = {
        "Global.log_level":   "error",
        "Det.engine_type":    EngineType(engine),
        "Det.lang_type":      LangDet(det_lang),
        "Det.model_type":     ModelType(model_type),
        "Det.ocr_version":    OCRVersion(ver),
        "Rec.engine_type":    EngineType(engine),
        "Rec.lang_type":      LangRec(rec_lang),
        "Rec.model_type":     ModelType(model_type),
        "Rec.ocr_version":    OCRVersion(ver),
    }
    threads = _env("RAPIDOCR_NUM_THREADS", "")
    if threads:
        try:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = int(threads)
        except ValueError:
            pass
    return params


def _get_engine():
    """Return the shared RapidOCR engine, building it once on first use. Returns
    None (never raises) if RapidOCR isn't installed or the engine can't be built,
    so callers degrade to ocr_available=False instead of crashing."""
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_FAILED or not _RAPIDOCR_OK:
        return None
    try:
        _ENGINE = RapidOCR(params=_build_engine_params())
    except Exception:
        _ENGINE_FAILED = True
        _ENGINE = None
    return _ENGINE


def ocr_available():
    """True if the OCR engine is (or can be) initialised. Kept as a callable for
    external callers that want to probe readiness without OCR-ing a card."""
    return _get_engine() is not None


def _ocr_lines(crop, min_side=64):
    """Run the shared engine on a BGR crop and return one tuple per detected text
    line: (text, height_px, x_left_px, score). Returns [] when OCR is unavailable
    or nothing is found. Short crops are upscaled first so thin name/number bands
    read reliably on the mobile model."""
    engine = _get_engine()
    if engine is None or crop is None or getattr(crop, "size", 0) == 0:
        return []
    img = crop
    h = img.shape[0]
    if 0 < h < min_side:
        sf = float(min_side) / h
        img = cv2.resize(img, (max(1, int(img.shape[1] * sf)), int(h * sf)),
                         interpolation=cv2.INTER_CUBIC)
    try:
        res = engine(img, use_cls=_USE_CLS)
    except Exception as exc:
        # Keep degrading (one bad crop must not abort the scan) but RECORD it:
        # returning [] silently is indistinguishable from "no text here".
        _ENGINE_ERRORS["count"] += 1
        _ENGINE_ERRORS["last"] = f"{type(exc).__name__}: {exc}"
        log.warning("OCR engine raised on a crop: %s", exc)
        return []
    txts   = getattr(res, "txts", None)
    boxes  = getattr(res, "boxes", None)
    scores = getattr(res, "scores", None)
    if not txts or boxes is None:
        return []
    out = []
    for i, t in enumerate(txts):
        try:
            b = np.asarray(boxes[i], dtype="float32")
            hgt = float(b[:, 1].max() - b[:, 1].min())
            xl  = float(b[:, 0].min())
        except Exception:
            hgt, xl = 0.0, 0.0
        sc = float(scores[i]) if scores is not None and i < len(scores) else 0.0
        out.append(((t or "").strip(), hgt, xl, sc))
    return out


_NUMBER_RE  = re.compile(r"(\d{1,4})\s*/\s*(\d{1,4})")
_SETCODE_RE = re.compile(r"\b(?=[A-Z0-9]{2,6}\b)[A-Z0-9]*[A-Z][A-Z0-9]*\b")
# Words that appear in the name zone but are never the card name: evolution stage
# badges, rarity/mechanic tags, HP, and other UI text. RapidOCR reads these badges
# cleanly, so an exact match catches them reliably. A *light* fuzzy pass remains as
# a safety net for the occasional OCR slip, but at a high similarity threshold:
# the old 0.70 threshold produced false positives on legitimate short Pokemon names
# (e.g. "Staryu" ~ "vstar" = 0.73, which wrongly discarded the name), so it is
# tightened here to only reject near-identical mangles.
_NAME_NOISE = {"hp", "stage", "stage1", "stage2", "basic", "evolves", "from",
               "pokemon", "pokmon", "illus", "no", "the", "ex", "gx", "restored",
               "mega", "break", "vmax", "vstar", "vunion", "tag", "team", "lv",
               "item", "supporter", "stadium", "energy", "legend"}

# Similarity at/above which a token is treated as a mangled noise word. High on
# purpose (see above) so real names survive; clean badge reads still match exactly.
_NAME_NOISE_FUZZ = 0.86


def _is_name_noise(token):
    """True if `token` is (or closely resembles) a non-name UI/stage/rarity word.
    Exact matches are always rejected; a high-threshold fuzzy pass additionally
    catches near-identical OCR mangles without nuking legitimate short names."""
    t = re.sub(r"[^a-z0-9]", "", str(token or "").lower())
    if not t:
        return True
    if t in _NAME_NOISE:
        return True
    # Only fuzzy-reject short tokens (badges/tags are short); real names rarely
    # collide at this threshold, and requiring len>=3 avoids nuking single letters.
    if len(t) <= 6:
        for w in _NAME_NOISE:
            if len(w) >= 3 and SequenceMatcher(None, t, w).ratio() >= _NAME_NOISE_FUZZ:
                return True
    return False


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def _to_bgr(image_or_path):
    if isinstance(image_or_path, np.ndarray):
        return image_or_path
    if isinstance(image_or_path, Image.Image):
        return cv2.cvtColor(np.array(image_or_path.convert("RGB")), cv2.COLOR_RGB2BGR)
    path = str(image_or_path)
    img = cv2.imread(path)
    if img is None:
        # OpenCV 4.13's PNG reader rejects images whose ancillary chunks
        # (iTXt / zTXt / iCCP / eXIf metadata) exceed libpng's ~8 MB per-chunk cap,
        # returning None. Pillow decodes them fine, so fall back to it; the pixel
        # ceiling (Image.MAX_IMAGE_PIXELS, set by the host app) still guards bombs.
        try:
            with Image.open(path) as im:
                img = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    if img is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such image: {image_or_path}")
        raise ImageDecodeError(f"Could not decode image: {image_or_path}")
    _guard_pixel_ceiling(img, path)
    return img


def _guard_pixel_ceiling(img, path=""):
    """Reject absurdly large decodes. cv2.imread's own cap is 2**30 pixels, so
    a 30000x30000 upload sails through it and allocates ~2.7 GB as BGR — an OOM
    on the Raspberry Pi this module targets — while the Pillow fallback path
    honours Image.MAX_IMAGE_PIXELS (set by the host app). Apply the same ceiling
    to whatever OpenCV returns so both paths agree."""
    cap = getattr(Image, "MAX_IMAGE_PIXELS", None)
    if not cap:
        return
    h, w = img.shape[:2]
    if h * w > cap:
        raise ImageDecodeError(
            f"Image is too large to process ({w}x{h} = {h * w} pixels, "
            f"limit {cap}): {path or 'image'}")


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
def _strip_name_noise(text):
    """Drop stage/rarity/UI tokens (incl. OCR-mangled ones) from a raw name line,
    e.g. 'sic Panpour' -> 'Panpour'."""
    toks = [t for t in re.split(r"\s+", str(text or "").strip())
            if t and not _is_name_noise(t)]
    return _clean_name(" ".join(toks))


def _read_name(bgr, top=0.14, x0=0.10, x1=0.80):
    """Read the card name from the top zone with RapidOCR. The recogniser reads
    whole text lines, so instead of re-assembling words we take each detected line,
    strip stage/rarity/UI tokens (incl. OCR-mangled ones, e.g. the evolution badge
    to the left of the name), and keep the *tallest* remaining line — the name is
    the largest text in the zone. Ties break on letter count, then confidence.
    Returns (name, confidence 0..100); ('', 0.0) if nothing usable is read."""
    H, W = bgr.shape[:2]
    crop = bgr[int(H * 0.02):int(H * top), int(W * x0):int(W * x1)]

    best = None   # ((height, alpha_len, score), cleaned_text, score)
    for text, hgt, _x, sc in _ocr_lines(crop, min_side=96):
        cleaned = _strip_name_noise(text)
        alpha = len(re.sub(r"[^A-Za-z]", "", cleaned))
        if alpha < 3:
            continue
        key = (hgt, alpha, sc)
        if best is None or key > best[0]:
            best = (key, cleaned, sc)

    if best is None:
        return "", 0.0
    return best[1], round(best[2] * 100.0, 1)


# --------------------------------------------------------------------------- #
# Number: full-width bottom band (bottom-left OR bottom-right), plausible only
# --------------------------------------------------------------------------- #
def _plausible_number(n, m):
    return 1 <= n <= m <= 2000


def _pad_serial(nm):
    """Zero-pad the collector number so the printed number matches the set-total
    width used by catalogs: '28/162' -> '028/162', '4/102' -> '004/102', while
    '56/64' stays '56/64'. Left unchanged if it isn't a plain numeric N/M (e.g.
    promo codes like 'SV107/SV122', which _NUMBER_RE doesn't match)."""
    m = _NUMBER_RE.fullmatch((nm or "").replace(" ", ""))
    if not m:
        return nm or ""
    n, d = int(m.group(1)), int(m.group(2))
    width = len(str(d))          # match numerator width to the denominator's
    return f"{n:0{width}d}/{d}"


def _best_number(hits):
    """Vote for the most-seen N/M, folding trailing-zero artifacts (a rarity dot
    or set symbol read as a '0'): 'N/M0' votes merge into 'N/M' when both appear."""
    c = Counter(hits)
    for cand in list(c):
        n, m = cand.split("/")
        if m.endswith("0") and len(m) > 1 and f"{n}/{m[:-1]}" in c:
            c[f"{n}/{m[:-1]}"] += c.pop(cand)
    return c.most_common(1)[0][0]


# Scores this close to the best read count as tied, so the frequency vote in
# _read_number decides between them. RapidOCR line scores are 0..1.
_NUMBER_TIE_BAND = 0.05


def _read_number(bgr, band_top=0.90, band_bottom=0.965, corner_frac=0.34):
    """Return (normalized 'N/M', raw_text, confidence). The collector number lives in a bottom
    CORNER — bottom-left on modern sets, bottom-right on older ones — so we OCR
    only the two corners and skip the centre of the bottom band (where the flavour
    text and the weakness/resistance/retreat row live, which used to inject spurious
    digits). RapidOCR reads each corner as text lines; we regex every N/M out of
    them, keep only plausible ones (1 <= N <= M <= 2000) with their confidence, and
    pick the highest-confidence value — ties broken by frequency, with the
    trailing-zero artifact fold (a rarity dot / set symbol read as '0') preserved."""
    H, W = bgr.shape[:2]
    y0, y1 = int(H * band_top), int(H * band_bottom)
    xL, xR = int(W * corner_frac), int(W * (1.0 - corner_frac))
    regions = (bgr[y0:y1, 0:xL], bgr[y0:y1, xR:W])   # bottom-left, bottom-right

    scored, raw_parts = [], []
    for band in regions:
        for text, _h, _x, sc in _ocr_lines(band, min_side=64):
            if text:
                raw_parts.append(text)
            for mm in _NUMBER_RE.finditer(text.replace(" ", "")):
                n, m = int(mm.group(1)), int(mm.group(2))
                if _plausible_number(n, m):
                    scored.append((f"{n}/{m}", sc))

    raw = " ".join(raw_parts).strip()
    if not scored:
        return "", raw, -1.0

    # "Ties" within a tolerance, not on exact float equality: OCR scores from
    # separate lines/corners essentially never match to 1e-6, so the documented
    # frequency vote (and its trailing-zero fold) was dead code — one spurious
    # high-confidence read outranked a number seen repeatedly. _NUMBER_TIE_BAND
    # keeps the vote for reads the engine rates about equally.
    top_score = max(sc for _nm, sc in scored)
    tied = [nm for nm, sc in scored if (top_score - sc) <= _NUMBER_TIE_BAND]
    best = tied[0] if len(set(tied)) == 1 else _best_number(tied)
    # Report the best score among the reads that produced the winner, not a
    # fabricated 100.0.
    conf = max((sc for nm, sc in scored if nm == best), default=top_score)
    return _pad_serial(best), raw, float(conf)


def parse_set_code(text, drop=""):
    for tok in _SETCODE_RE.findall((text or "").upper()):
        if tok and tok != drop and not tok.isdigit():
            return tok
    return ""


# --------------------------------------------------------------------------- #
# Public: OCR a front image
# --------------------------------------------------------------------------- #
def ocr_card_front(image_or_path, normalize=True, debug_dir=None, game=None, type_refs=None):
    """
    OCR the name and collector number from a card front.

    normalize=True (default) first deskews/crops/standardises the card, which is
    what lets fixed name/number zones work when the card is off-centre or tilted
    (e.g. photographed in a binder page).

    `game` (optional) enables per-TCG "type" detection from the small type icon
    (e.g. a Pokemon energy symbol). `type_refs` (optional, prepared via
    prepare_type_references) makes type detection use a reference-icon library by
    template matching; without it, a built-in colour heuristic is used.

    Returns a dict with: ocr_available, name_guess, number_guess, set_code_guess,
    raw_top, raw_bottom, conf_top, conf_bottom, type_guess, type_confidence,
    normalized (bool).
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

    # Build the engine once; if unavailable, name/number come back empty (the
    # readers degrade gracefully) but type detection — which is colour/template
    # based, not OCR — still runs.
    available = _get_engine() is not None

    errors_before = _ENGINE_ERRORS["count"]
    name_guess, conf_top = _read_name(bgr)
    number_guess, raw_bottom, conf_bottom = _read_number(bgr)
    set_code_guess = parse_set_code(raw_bottom, drop=number_guess.replace("/", ""))
    type_guess, type_conf, type_method = detect_card_type(bgr, game, type_refs)
    engine_errors = _ENGINE_ERRORS["count"] - errors_before

    # A blank read because the engine threw on every crop is a broken install,
    # not a blank card — say so instead of leaving the caller to guess.
    ocr_error = _ENGINE_ERRORS["last"] if engine_errors else ""
    if engine_errors:
        log.error("OCR engine failed on %d crop(s) this scan: %s",
                  engine_errors, ocr_error)

    return {
        "ocr_available": available,
        "name_guess": name_guess,
        "number_guess": number_guess,
        "set_code_guess": set_code_guess,
        "raw_top": name_guess,
        "raw_bottom": raw_bottom,
        "conf_top": conf_top if name_guess else -1.0,
        "conf_bottom": conf_bottom if number_guess else -1.0,
        "type_guess": type_guess,
        "type_confidence": type_conf,
        # How the type was decided: "reference_match" (per-game icon library),
        # "color_heuristic" (built-in guess — the library was empty/absent),
        # or "none". The old result labelled a colour guess identically to a
        # template match.
        "type_method": type_method,
        "normalized": did_norm,
        # Degraded-mode signals. engine_errors > 0 with empty guesses means the
        # install is broken, not that the card is blank.
        "engine_errors": engine_errors,
        "ocr_error": ocr_error,
    }


# --------------------------------------------------------------------------- #
# Card "type" detection from the type icon (e.g. Pokemon energy type)
# --------------------------------------------------------------------------- #
# Different TCGs mark a card's "type" with a small coloured icon: Pokemon energy
# (Fire/Water/Grass/...), etc. We isolate the most icon-like coloured blob in the
# card's top-right corner — next to the HP, and inset so the gold border and the
# dark HP digits are excluded — and classify it by colour signature. This is a
# best-effort VISUAL guess returned with a 0..1 confidence, not a certainty:
# colour-distinct types (Water, Grass, Psychic, Water) read most reliably, while
# red-orange types (Fire/Fighting) and neutral ones (Colorless/Metal/Darkness)
# are inherently harder and come back at lower confidence (or blank).
#
# Only colour is used (no shipped icon templates), so treat it as a suggestion
# the user can confirm. The design is a per-game registry so other TCGs' type
# schemes (Magic colours, Yu-Gi-Oh attributes, ...) can be added later.

def _pokemon_type_for_hsv(h, s, v):
    """Map a dominant OpenCV-HSV colour (h in 0..179) to a Pokemon energy type."""
    if h <= 12 or h >= 168:
        return "Fire"                       # red
    if 13 <= h <= 21:
        return "Fighting" if v < 165 else "Fire"   # brown(darker) vs orange
    if 22 <= h <= 34:
        return "Lightning"                  # yellow (note: gold border is similar)
    if 35 <= h <= 85:
        return "Grass"                      # green
    if 86 <= h <= 130:
        return "Water"                      # blue / cyan
    if 131 <= h <= 150:
        return "Psychic"                    # purple / violet
    if 151 <= h <= 167:
        return "Fairy"                      # pink / magenta
    return ""


def detect_pokemon_energy_type(bgr):
    """
    Best-effort Pokemon energy type from the top-right type icon.
    Returns (type_name, confidence 0..1); ('', 0.0) when nothing usable.
    """
    try:
        H, W = bgr.shape[:2]
        # Top-right corner, inset to skip the gold frame at the very edges.
        x0, x1 = int(0.70 * W), int(0.975 * W)
        y0, y1 = int(0.015 * H), int(0.10 * H)
        box = bgr[y0:y1, x0:x1]
        if box.size == 0:
            return "", 0.0

        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        # Coloured icon pixels: saturated and not dark (drops HP digits & shadows).
        colored = (S > 80) & (V > 70)
        if float(colored.mean()) < 0.04:
            return "", 0.0                  # essentially no coloured icon found

        hues = hsv[:, :, 0][colored].astype(np.int32)
        # Dominant hue via a coarse histogram peak (robust to a few stray pixels).
        hist = np.bincount(hues // 5, minlength=36)
        peak_h = int(np.argmax(hist)) * 5 + 2
        # Concentration of hue around the peak (circular distance) -> confidence.
        near = np.abs(((hues - peak_h + 90) % 180) - 90) <= 10
        purity = float(near.mean())
        med_s = float(np.median(S[colored]))
        med_v = float(np.median(V[colored]))

        t = _pokemon_type_for_hsv(peak_h, med_s, med_v)
        if not t:
            return "", 0.0
        conf = 0.35 + 0.55 * purity
        if t in ("Fire", "Fighting", "Lightning"):   # red-orange/gold ambiguity
            conf *= 0.75
        return t, round(max(0.0, min(1.0, conf)), 2)
    except Exception:
        return "", 0.0


# game-name substring -> detector. Extend here for other TCGs.
_TYPE_DETECTORS = {
    "pokemon": detect_pokemon_energy_type,
    "pokémon": detect_pokemon_energy_type,
}

# Regions where a card's "type" marker lives. Corner regions hold a small icon
# (isolated as the largest coloured blob); the "header" region is the full-width
# top band, matched as a whole design — this is how card kinds with different
# header layouts are told apart (e.g. a Pokemon Trainer/Supporter header vs an
# Energy header vs a normal Pokemon name/HP header). Each preset gives
# coords (x0, x1, y0, y1 as fractions) and a mode ('icon' | 'band').
_TYPE_REGIONS = {
    "top_right": {"coords": (0.70, 0.975, 0.015, 0.10), "mode": "icon"},
    "top_left":  {"coords": (0.025, 0.30, 0.015, 0.10), "mode": "icon"},
    "header":    {"coords": (0.03, 0.97, 0.012, 0.115), "mode": "band"},
}
_DEFAULT_TYPE_REGION = "top_right"
# Back-compat alias (used by the built-in Pokemon colour heuristic).
_TYPE_ICON_REGION = _TYPE_REGIONS["top_right"]["coords"]

# Feature sizes: square icons vs wide header bands (keep the band wide so the
# header layout isn't squashed away).
_ICON_SIZE = 64
_BAND_SIZE = (160, 48)   # (width, height)


def normalize_type_region(region):
    """Coerce a region value to a supported preset, defaulting to top-right."""
    r = (region or "").strip().lower().replace("-", "_")
    return r if r in _TYPE_REGIONS else _DEFAULT_TYPE_REGION


def region_mode(region):
    """'icon' (corner blob) or 'band' (full header) for a region."""
    return _TYPE_REGIONS.get(normalize_type_region(region),
                             _TYPE_REGIONS[_DEFAULT_TYPE_REGION])["mode"]


def _extract_type_region(bgr, region=_DEFAULT_TYPE_REGION):
    """
    Crop the card's type marker for `region`.

    • icon regions (top_left/top_right): isolate the largest saturated colour
      blob — a small corner icon. Returns None if no coloured icon is found.
    • band regions (header): return the whole top band as-is (the header design).

    Returns a BGR crop, or None.
    """
    try:
        spec = _TYPE_REGIONS.get(normalize_type_region(region),
                                 _TYPE_REGIONS[_DEFAULT_TYPE_REGION])
        x0, x1, y0, y1 = spec["coords"]
        mode = spec["mode"]
        H, W = bgr.shape[:2]
        box = bgr[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        if box.size == 0:
            return None
        if mode == "band":
            return box.copy()

        # icon mode: isolate the largest saturated blob (drops HP text; the inset
        # already excludes the border).
        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        mask = (((S > 80) & (V > 70)).astype(np.uint8)) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return None
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[idx]
        if area < 30:
            return None
        pad = 2
        yA, yB = max(0, y - pad), min(box.shape[0], y + h + pad)
        xA, xB = max(0, x - pad), min(box.shape[1], x + w + pad)
        crop = box[yA:yB, xA:xB].copy()
        return crop if crop.size else None
    except Exception:
        return None


# Back-compat name kept for any external callers.
def _square_pad(img):
    """Letterbox an icon to a square (white border) so wide card-captured icons
    and tight uploaded ones compare consistently after resizing."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _features(img, mode="icon"):
    """
    Fixed-size grayscale (contrast-normalised) + hue histogram for matching.
    Icons are letterboxed to a square; header bands keep their wide aspect so the
    header layout is preserved. Query and reference must use the same mode (they
    always do — they share a region).
    """
    if mode == "band":
        resized = cv2.resize(img, _BAND_SIZE, interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(_square_pad(img), (_ICON_SIZE, _ICON_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    sat_mask = (hsv[:, :, 1] > 60).astype(np.uint8)
    hist = cv2.calcHist([hsv], [0], sat_mask, [30], [0, 180])
    cv2.normalize(hist, hist)
    return gray, hist


def prepare_type_references(raw_refs, size=64):
    """
    Precompute match features for a game's reference images.
    `raw_refs`: list of {'type_name', 'region', 'image'(BGR)}. Returns a list with
    cached 'gray'/'hist' features, region and mode (unreadable entries skipped).
    """
    out, skipped = [], []
    for r in raw_refs or []:
        img = r.get("image")
        if img is None or getattr(img, "size", 0) == 0:
            skipped.append(r.get("type_name", "?"))
            continue
        region = normalize_type_region(r.get("region"))
        mode = region_mode(region)
        try:
            g, h = _features(img, mode)
        except Exception as exc:
            skipped.append(f"{r.get('type_name', '?')} ({type(exc).__name__})")
            continue
        out.append({
            "type_name": r.get("type_name", ""),
            "region": region, "mode": mode, "gray": g, "hist": h,
        })
    if skipped:
        # Silently dropping these left an operator with a library that looked
        # installed but matched nothing (and, when ALL of them fail, silently
        # demoted type detection to the colour heuristic).
        log.warning("Skipped %d unreadable type reference(s): %s",
                    len(skipped), ", ".join(str(s) for s in skipped[:10]))
    return out


def match_card_types(bgr, references, size=64):
    """
    Match a card against prepared references, each of which knows the region its
    marker lives in (a top corner icon, or the full header band). The card's
    marker is extracted and feature-ised once per region in use, and every
    reference is scored against the marker from its own region. Corner matches
    blend shape+colour evenly; header matches weight shape (structure) higher,
    since header colours are similar across kinds. The winner's confidence is
    tempered by its margin over the runner-up type.

    Returns (best_type, score 0..1); ('', 0.0) when nothing matches.
    """
    if not references:
        return "", 0.0

    # Extract + feature-ise the card's marker once per region actually used.
    region_feats = {}
    for r in references:
        reg = r.get("region", _DEFAULT_TYPE_REGION)
        if reg not in region_feats:
            crop = _extract_type_region(bgr, reg)
            try:
                region_feats[reg] = _features(crop, region_mode(reg)) if crop is not None else None
            except Exception:
                region_feats[reg] = None

    best_by_type = {}
    for r in references:
        feats = region_feats.get(r.get("region", _DEFAULT_TYPE_REGION))
        if feats is None:
            continue
        qg, qh = feats
        try:
            shape = max(0.0, float(cv2.matchTemplate(qg, r["gray"], cv2.TM_CCOEFF_NORMED)[0, 0]))
            color = max(0.0, float(cv2.compareHist(qh, r["hist"], cv2.HISTCMP_CORREL)))
        except Exception:
            continue
        if r.get("mode") == "band":
            s = 0.7 * shape + 0.3 * color      # header: structure dominates
        else:
            s = 0.5 * shape + 0.5 * color
        t = r["type_name"]
        if s > best_by_type.get(t, -1.0):
            best_by_type[t] = s
    if not best_by_type:
        return "", 0.0

    ranked = sorted(best_by_type.items(), key=lambda kv: kv[1], reverse=True)
    best_t, best_s = ranked[0]
    conf = best_s
    if len(ranked) >= 2:                        # temper by separation from runner-up
        margin = best_s - ranked[1][1]
        conf = best_s * (0.6 + 0.4 * min(1.0, margin / 0.15))
    return best_t, round(max(0.0, min(1.0, conf)), 2)


def detect_card_type(bgr, game, type_refs=None):
    """
    Detect a card's "type".

    If `type_refs` (prepared reference icons for this game) are supplied, match
    the card's icon against them in each reference's designated corner — the
    general, per-game mechanism that works for any TCG once a library exists.
    Otherwise fall back to the built-in colour heuristic (currently Pokemon).

    Returns (type_name, confidence 0..1, method) where method is
    "reference_match", "color_heuristic", or "none". The method is returned
    because an EMPTY type_refs list is falsy: a game whose icon library failed
    to load silently got colour guesses that looked exactly like template
    matches, with no signal anywhere that the library was missing.
    """
    if type_refs:
        name, conf = match_card_types(bgr, type_refs)
        return name, conf, "reference_match"

    g = (game or "").strip().lower()
    if not g:
        return "", 0.0, "none"
    for key, fn in _TYPE_DETECTORS.items():
        if key in g:
            name, conf = fn(bgr)
            return name, conf, "color_heuristic"
    return "", 0.0, "none"


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
