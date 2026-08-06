"""
shipping_providers.py
=====================

Label/postage connectors for the "Shipping" tab. One provider class per
service, speaking each service's real REST API over urllib — same
no-extra-dependencies, never-raise, always-return-{ok,message} contract as
shop_providers.py.

Reality check (researched 2026-07)
---------------------------------
* TCGTracking (tcgtracking.com) — free account, self-serve API key from the
  dashboard's API tab (a payment method must be on file first). Its public API
  is exactly three endpoints: create order, track order, regenerate PDF. It
  prints a *trackable envelope* carrying a USPS Intelligent Mail Barcode and
  gives free Informed Visibility scan events. An IMB is tracking, NOT postage —
  you still put a stamp on the envelope. Optional PIP insurance up to $50.

* EasyPost (easypost.com) — self-serve; sign up and take a Test key and a
  Production key. This is what actually *buys postage*: rate-shop across
  carriers, buy a label, get a tracking code. Labels are requested as PDF (not
  the PNG default) so batch printing can merge either provider's output.

Why both, and not TCGTracking alone: TCGTracking's own dashboard can buy
EasyPost e-postage, but that is a dashboard-only feature — it is not exposed on
their public API, which has no rate, postage, or EasyPost endpoint. So the only
way to buy postage from code is to talk to EasyPost directly with your own key.
The two are complementary, not redundant:

    plain envelope, stamp on it, just want tracking  -> tcgtracking
    parcel/bubble mailer, need real postage + label  -> easypost

Credentials live in ShopConnection.config (JSON), reusing the same table and
secret-masking the Shops tab already uses. The Shops page only iterates over
shop_providers.MARKETPLACES, so these rows never appear there.
"""

import base64
import json
import urllib.request
import urllib.parse
import urllib.error

from shop_providers import _http, DEFAULT_TIMEOUT

TCGTRACKING_BASE = "https://tcgtracking.com/api/v1"
EASYPOST_BASE = "https://api.easypost.com/v2"

# US states + DC + territories, and CA provinces — the only destinations
# TCGTracking accepts. Validated locally so a typo costs zero API calls.
_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
    "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "AS", "GU", "MP", "PR", "VI", "AA", "AE", "AP",
}
_CA_PROVINCES = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}

# TCGTracking envelope/label sizes (size id -> label shown in the UI).
ENVELOPE_SIZES = [
    ("10", '#10 Envelope — 9.5" × 4.125"'),
    ("9", '#9 Envelope — 8.875" × 3.875"'),
    ("8", '#8 Envelope — 5.75" × 3.625"'),
    ("7", '#7 Envelope — 7.25" × 3.25"'),
    ("6_75", '#6¾ Envelope — 6.5" × 3.625"'),
    ("6", '#6 Envelope — 6.5" × 2.25"'),
    ("8_5x11_single_left", '8.5×11 Left Window'),
    ("8_5x11_single_right", '8.5×11 Right Window'),
    ("8_5x11_dual", '8.5×11 Dual Window'),
    ("L64", '6×4 Label'),
    ("L64p", '6×4 Label + Packing Slip'),
    ("L42", '4×2 Label'),
    ("L1135", '3.5×1.1 Label'),
]
_ENVELOPE_SIZE_IDS = {s for s, _ in ENVELOPE_SIZES}


# ──────────────────────────────────────────────────────────────────────────────
# UI metadata — drives the connector cards on the Shipping page.
# ──────────────────────────────────────────────────────────────────────────────
SHIPPING_PROVIDERS = {
    "tcgtracking": {
        "label": "TCGTracking",
        "icon": "fa-solid fa-envelope-circle-check",
        "color": "#1F7A4C",
        "available": True,
        "buys_postage": False,
        "blurb": "Print trackable envelopes with a USPS Intelligent Mail Barcode and get free "
                 "Informed Visibility scan events. Tracking only — the envelope still needs a stamp.",
        "fields": [
            {"key": "shipper_number", "label": "Shipper number", "placeholder": "12345", "type": "text"},
            {"key": "api_key", "label": "API key", "placeholder": "from the dashboard's API tab",
             "type": "password", "secret": True},
            {"key": "envelope_size", "label": "Default envelope size", "type": "select",
             "options": [s for s, _ in ENVELOPE_SIZES]},
            {"key": "insure_by_default", "label": "Insure by default (PIP, up to $50)", "type": "checkbox"},
            {"key": "from_name", "label": "Return name", "type": "text", "optional": True},
            {"key": "from_company", "label": "Return company", "type": "text", "optional": True},
            {"key": "from_address", "label": "Return street", "type": "text", "optional": True},
            {"key": "from_address2", "label": "Return street 2", "type": "text", "optional": True},
            {"key": "from_city", "label": "Return city", "type": "text", "optional": True},
            {"key": "from_state", "label": "Return state", "type": "text", "optional": True},
            {"key": "from_zip", "label": "Return ZIP", "type": "text", "optional": True},
        ],
        "help": "Sign in at tcgtracking.com, open the API tab, add a payment method, then generate a key. "
                "Leave the return address blank to use your dashboard default. Destinations are limited to "
                "the US and Canada.",
    },
    "easypost": {
        "label": "EasyPost",
        "icon": "fa-solid fa-truck-fast",
        "color": "#164DFF",
        "available": True,
        "buys_postage": True,
        "blurb": "Buy real postage. Rate-shop USPS and other carriers, purchase a label, and track it.",
        "fields": [
            {"key": "mode", "label": "Mode", "type": "select", "options": ["test", "production"]},
            {"key": "api_key", "label": "API key", "placeholder": "EZAK… (production) / EZTK… (test)",
             "type": "password", "secret": True},
            {"key": "preferred_service", "label": "Preferred service", "type": "text",
             "placeholder": "GroundAdvantage (blank = cheapest)", "optional": True},
            {"key": "parcel_weight_oz", "label": "Default parcel weight (oz)", "type": "text",
             "placeholder": "3"},
            {"key": "parcel_length", "label": "Length (in)", "type": "text", "placeholder": "6"},
            {"key": "parcel_width", "label": "Width (in)", "type": "text", "placeholder": "4"},
            {"key": "parcel_height", "label": "Height (in)", "type": "text", "placeholder": "0.75"},
            {"key": "insure_by_default", "label": "Insure by default (full order value)", "type": "checkbox"},
            {"key": "from_name", "label": "Ship-from name", "type": "text"},
            {"key": "from_company", "label": "Ship-from company", "type": "text", "optional": True},
            {"key": "from_address", "label": "Ship-from street", "type": "text"},
            {"key": "from_address2", "label": "Ship-from street 2", "type": "text", "optional": True},
            {"key": "from_city", "label": "Ship-from city", "type": "text"},
            {"key": "from_state", "label": "Ship-from state", "type": "text"},
            {"key": "from_zip", "label": "Ship-from ZIP", "type": "text"},
            {"key": "from_phone", "label": "Ship-from phone", "type": "text", "optional": True},
        ],
        "help": "Create an account at easypost.com and copy an API key from Account Settings → API Keys. "
                "Test keys produce fake labels and cost nothing — use one until the flow looks right, then "
                "switch to production. A complete ship-from address is required.",
    },
}

SHIPPING_SECRET_FIELDS = {
    pk: {f["key"] for f in meta["fields"] if f.get("secret")}
    for pk, meta in SHIPPING_PROVIDERS.items()
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _download(url, timeout=DEFAULT_TIMEOUT):
    """
    Fetch a URL as raw bytes. Separate from _http because that decodes UTF-8,
    which would corrupt a label PDF.
    """
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), None
    except urllib.error.HTTPError as exc:
        return None, f"Label download failed — HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return None, f"Label download failed — {exc.reason}"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Label download failed — {exc}"


def _money(value, default=0.0):
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return default


# Full names -> postal codes for every value in _US_STATES/_CA_PROVINCES.
# Exists because marketplace exports write "Minnesota" as readily as "MN", and
# a blind [:2] truncation turns it into Michigan — wrong-but-valid, so it
# passes validation and prints on real postage. Keys are upper-case with
# periods stripped and whitespace collapsed (see normalize_state).
_STATE_NAME_TO_CODE = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC", "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI",
    "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY",
    "AMERICAN SAMOA": "AS", "GUAM": "GU", "NORTHERN MARIANA ISLANDS": "MP",
    "PUERTO RICO": "PR", "VIRGIN ISLANDS": "VI", "US VIRGIN ISLANDS": "VI",
    "ALBERTA": "AB", "BRITISH COLUMBIA": "BC", "MANITOBA": "MB",
    "NEW BRUNSWICK": "NB", "NEWFOUNDLAND AND LABRADOR": "NL", "NOVA SCOTIA": "NS",
    "NORTHWEST TERRITORIES": "NT", "NUNAVUT": "NU", "ONTARIO": "ON",
    "PRINCE EDWARD ISLAND": "PE", "QUEBEC": "QC", "SASKATCHEWAN": "SK", "YUKON": "YT",
}


def normalize_state(value):
    """Trim/upper a state and map full names to their postal code.

    Never blind-truncates: an unknown long value comes back trimmed (capped to
    the column's 10 chars) so _validate_destination rejects it by name instead
    of a colliding two-letter prefix shipping to the wrong state."""
    s = " ".join(str(value or "").replace(".", " ").split()).upper()
    if len(s) <= 2:
        return s
    return _STATE_NAME_TO_CODE.get(s, s[:10])


def normalize_country(value, default="US"):
    """Map the common US/CA spellings to their two-letter codes.

    "United States" was previously [:2]'d into the meaningless "UN"; other
    values keep the old trim-and-truncate behaviour."""
    c = " ".join(str(value or "").replace(".", " ").split()).upper()
    if not c:
        return default
    if c in ("US", "USA") or c.startswith("UNITED STATES"):
        return "US"
    if c in ("CA", "CAN", "CANADA"):
        return "CA"
    return c[:2]


def _clean_state(value):
    return normalize_state(value)


def _validate_destination(order):
    """
    Check an order against the rules both providers share, before any HTTP.
    Returns a list of human-readable problems (empty = good to send).
    """
    problems = []
    if not (str(order.buyer_name or "").strip() or str(order.buyer_company or "").strip()):
        problems.append("a buyer name or company")
    if not str(order.address1 or "").strip():
        problems.append("a street address")
    if not str(order.city or "").strip():
        problems.append("a city")

    state = _clean_state(order.state)
    country = str(order.country or "US").strip().upper() or "US"
    if not state:
        problems.append("a state")
    elif country == "US" and state not in _US_STATES:
        problems.append(f"a valid US state (got '{state}')")
    elif country == "CA" and state not in _CA_PROVINCES:
        problems.append(f"a valid Canadian province (got '{state}')")

    zipcode = str(order.zipcode or "").strip()
    if not zipcode:
        problems.append("a ZIP/postal code")

    if order.declared_value <= 0:
        problems.append("an order value above $0")
    return problems


class ShippingProvider:
    """
    Common interface. Every method returns a dict with `ok` + `message` and
    never raises for an expected API failure — the Flask layer turns these
    straight into JSON.
    """
    key = "base"
    buys_postage = False

    def __init__(self, connection, persist=None):
        self.conn = connection
        self.cfg = dict(connection.config or {})
        self._persist = persist or (lambda: None)

    def _save_cfg(self):
        self.conn.config = dict(self.cfg)   # new dict so SQLAlchemy sees the change
        self._persist()

    def _need(self, *keys):
        return [k for k in keys if not str(self.cfg.get(k, "")).strip()]

    def _flag(self, key):
        return str(self.cfg.get(key, "")).strip().lower() in ("1", "true", "yes", "on")

    # -- interface ------------------------------------------------------------
    def test_connection(self):
        return {"ok": False, "message": "Not implemented"}

    def quote(self, order, opts=None):
        """Rate options for an order. Providers without rates return a single fixed option."""
        return {"ok": False, "message": "Not implemented", "rates": []}

    def create_label(self, order, opts=None):
        """
        Buy/create a label. On success returns:
            {ok, message, external_id, tracking_code, tracking_url, carrier,
             service, rate_amount, insured, insurance_amount,
             label_bytes, label_format, duplicate}
        """
        return {"ok": False, "message": "Not implemented"}

    def track(self, shipment):
        """Refresh tracking. Returns {ok, message, status, percent, delivery_date, events}."""
        return {"ok": False, "message": "Not implemented", "events": []}

    def relabel(self, shipment, size=None):
        """Re-fetch the label PDF (different size where supported)."""
        return {"ok": False, "message": "Not implemented"}


# ──────────────────────────────────────────────────────────────────────────────
# TCGTracking — IMB trackable envelopes
# ──────────────────────────────────────────────────────────────────────────────
class TCGTrackingProvider(ShippingProvider):
    key = "tcgtracking"
    buys_postage = False

    def _headers(self):
        return {
            "X-Shipper-Number": str(self.cfg.get("shipper_number", "")).strip(),
            "X-API-Key": str(self.cfg.get("api_key", "")).strip(),
            "Accept": "application/json",
        }

    def _from_override(self):
        """
        Return-address override. The API demands all-or-nothing: if any from_*
        field is sent, name/company + street + city + state + zip must all be
        there. So send it only when it's complete, and otherwise fall back to
        the account default rather than half-filling it.
        """
        named = str(self.cfg.get("from_name", "")).strip() or str(self.cfg.get("from_company", "")).strip()
        street = str(self.cfg.get("from_address", "")).strip()
        city = str(self.cfg.get("from_city", "")).strip()
        state = _clean_state(self.cfg.get("from_state"))
        zipc = str(self.cfg.get("from_zip", "")).strip()
        if not (named and street and city and state and zipc):
            return {}
        out = {"from_address": street, "from_city": city, "from_state": state, "from_zip": zipc}
        if str(self.cfg.get("from_name", "")).strip():
            out["from_name"] = str(self.cfg["from_name"]).strip()
        if str(self.cfg.get("from_company", "")).strip():
            out["from_company"] = str(self.cfg["from_company"]).strip()
        if str(self.cfg.get("from_address2", "")).strip():
            out["from_address2"] = str(self.cfg["from_address2"]).strip()
        return out

    def test_connection(self):
        """
        There's no ping endpoint, so probe the track endpoint with a tracking
        number that cannot exist. Bad credentials answer 401; good credentials
        answer 404 "not found". Nothing is created either way.
        """
        missing = self._need("shipper_number", "api_key")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}"}

        url = f"{TCGTRACKING_BASE}/orders/track?" + urllib.parse.urlencode(
            {"tracking_number": "0000000X000000"})
        code, parsed, raw = _http("GET", url, headers=self._headers())

        if code == 401:
            return {"ok": False, "message": (parsed or {}).get("message")
                    or "Invalid API key or shipper number."}
        if code in (200, 404):
            return {"ok": True, "message": f"Connected as shipper {self.cfg.get('shipper_number')}."}
        if code == 429:
            return {"ok": False, "message": "Rate limited (250 requests/minute). Try again shortly."}
        if code == 0:
            return {"ok": False, "message": raw or "Could not reach tcgtracking.com."}
        return {"ok": False, "message": (parsed or {}).get("message") or f"Unexpected response (HTTP {code})."}

    def quote(self, order, opts=None):
        """
        No rate API — an IMB envelope has one fixed shape. The cost is the PIP
        insurance premium (if any) plus a per-label fee after the free tier,
        neither of which is quotable ahead of time, so report $0 and say so.
        """
        opts = opts or {}
        size = opts.get("envelope_size") or self.cfg.get("envelope_size") or "10"
        return {"ok": True, "message": "TCGTracking has no rate lookup — postage is your own stamp.",
                "rates": [{
                    "id": f"tcgtracking:{size}",
                    "carrier": "USPS",
                    "service": f"IMB trackable envelope ({size})",
                    "amount": 0.0,
                    "currency": "USD",
                    "note": "Tracking only. Add postage yourself; per-label and insurance fees "
                            "are billed by TCGTracking.",
                }]}

    def create_label(self, order, opts=None):
        opts = opts or {}
        missing = self._need("shipper_number", "api_key")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}"}

        problems = _validate_destination(order)
        if problems:
            return {"ok": False, "message": "This order still needs " + ", ".join(problems) + "."}

        country = str(order.country or "US").strip().upper() or "US"
        if country not in ("US", "CA"):
            return {"ok": False, "message": f"TCGTracking ships to the US and Canada only (got '{country}'). "
                                            f"Use EasyPost for this order."}

        size = str(opts.get("envelope_size") or self.cfg.get("envelope_size") or "10")
        if size not in _ENVELOPE_SIZE_IDS:
            return {"ok": False, "message": f"'{size}' isn't a valid envelope size."}

        insured = opts.get("insured")
        if insured is None:
            insured = self._flag("insure_by_default")

        body = {
            "source_order_id": order.external_order_id or f"CCIM-{order.id}",
            "address_line1": str(order.address1).strip(),
            "customer_city": str(order.city).strip(),
            "customer_state": _clean_state(order.state),
            "zipcode": str(order.zipcode).strip(),
            "customer_country": country,
            "envelope_size": size,
            "is_insured": bool(insured),
        }
        if str(order.buyer_name or "").strip():
            body["customer_name"] = str(order.buyer_name).strip()
        if str(order.buyer_company or "").strip():
            body["company_name"] = str(order.buyer_company).strip()
        if str(order.address2 or "").strip():
            body["address_line2"] = str(order.address2).strip()
        if order.external_order_id:
            body["ordernum"] = str(order.external_order_id)[:60]

        # Itemised beats a bare total: the API recomputes the total from items
        # anyway, and the packing-slip sizes (L64p) print the line items.
        items = []
        for it in order.items:
            name = (it.name or "").strip()[:120]
            if not name:
                continue
            items.append({
                "name": name,
                "quantity": max(int(it.qty or 1), 1),
                "price": _money(it.price, 0.0),
            })
        if items:
            body["items"] = items
        else:
            body["total_amount"] = max(order.declared_value, 0.01)

        for k, v in self._from_override().items():
            body[k] = v

        code, parsed, raw = _http("POST", f"{TCGTRACKING_BASE}/orders",
                                  headers=self._headers(), body=body)

        if code == 429:
            return {"ok": False, "message": "TCGTracking rate limit hit (250/min). Try again in a minute."}
        if code == 401:
            return {"ok": False, "message": (parsed or {}).get("message") or "Invalid API key."}
        if not isinstance(parsed, dict):
            return {"ok": False, "message": raw[:300] if raw else f"Unexpected response (HTTP {code})."}

        if not parsed.get("success"):
            msg = parsed.get("message") or "Order rejected."
            step = parsed.get("step")
            # PDF generation can fail *after* the order and tracking number
            # exist. That's recoverable — keep the ids and re-fetch the PDF.
            if step == "generate_pdf" and parsed.get("order_id"):
                got = self.relabel_by_id(parsed["order_id"], size)
                if got.get("ok"):
                    return {
                        "ok": True,
                        "message": "Label created (PDF recovered after a generation error).",
                        "external_id": str(parsed["order_id"]),
                        "tracking_code": parsed.get("tracking_number", ""),
                        "tracking_url": self._track_url(parsed.get("tracking_number", "")),
                        "carrier": "USPS", "service": f"IMB trackable envelope ({size})",
                        "rate_amount": 0.0, "insured": bool(insured),
                        "insurance_amount": min(order.declared_value, 50.0) if insured else None,
                        "label_bytes": got["label_bytes"], "label_format": "PDF",
                        "envelope_size": size, "duplicate": False,
                    }
            return {"ok": False, "message": f"{msg}" + (f" (step: {step})" if step else "")}

        order_id = parsed.get("order_id")
        tracking = parsed.get("tracking_number", "")
        duplicate = bool(parsed.get("duplicate"))

        pdf_b64 = parsed.get("pdf_base64")
        if not pdf_b64 and order_id:
            # Duplicates don't return a PDF — fetch it rather than fail.
            got = self.relabel_by_id(order_id, size)
            if not got.get("ok"):
                return {"ok": False, "message": got.get("message", "Could not fetch the label PDF.")}
            label_bytes = got["label_bytes"]
        else:
            try:
                label_bytes = base64.b64decode(pdf_b64)
            except Exception:
                return {"ok": False, "message": "The label PDF came back unreadable."}

        return {
            "ok": True,
            "message": ("This order already had a label — reusing it."
                        if duplicate else "Trackable envelope created."),
            "external_id": str(order_id or ""),
            "tracking_code": tracking,
            "tracking_url": self._track_url(tracking),
            "carrier": "USPS",
            "service": f"IMB trackable envelope ({size})",
            "rate_amount": 0.0,
            "insured": bool(insured),
            "insurance_amount": min(order.declared_value, 50.0) if insured else None,
            "label_bytes": label_bytes,
            "label_format": "PDF",
            "envelope_size": size,
            "duplicate": duplicate,
        }

    @staticmethod
    def _track_url(tracking):
        if not tracking:
            return ""
        return "https://tcgtracking.com/track.php?" + urllib.parse.urlencode({"tracking": tracking})

    def relabel_by_id(self, external_id, size=None):
        params = {}
        if size:
            params["size"] = size
        for k, v in self._from_override().items():
            # The PDF endpoint takes the same override, but as query params.
            params[k] = v
        url = f"{TCGTRACKING_BASE}/orders/{external_id}/pdf"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        code, parsed, raw = _http("GET", url, headers=self._headers())
        if not isinstance(parsed, dict) or not parsed.get("success"):
            msg = (parsed or {}).get("message") or raw[:200] or f"HTTP {code}"
            return {"ok": False, "message": f"Could not regenerate the PDF — {msg}"}
        try:
            return {"ok": True, "label_bytes": base64.b64decode(parsed["pdf_base64"]),
                    "label_format": "PDF", "tracking_code": parsed.get("tracking_number", "")}
        except Exception:
            return {"ok": False, "message": "The regenerated PDF came back unreadable."}

    def relabel(self, shipment, size=None):
        if not shipment.external_id:
            return {"ok": False, "message": "This shipment has no TCGTracking order id to reprint from."}
        return self.relabel_by_id(shipment.external_id, size or shipment.envelope_size)

    def track(self, shipment):
        missing = self._need("shipper_number", "api_key")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}", "events": []}

        # Prefer the tracking number; fall back to the source order id, which
        # keeps working even if the number was never stored.
        if shipment.tracking_code:
            params = {"tracking_number": shipment.tracking_code}
        elif shipment.order and shipment.order.external_order_id:
            params = {"order_number": shipment.order.external_order_id}
        else:
            return {"ok": False, "message": "Nothing to track this shipment by.", "events": []}

        url = f"{TCGTRACKING_BASE}/orders/track?" + urllib.parse.urlencode(params)
        code, parsed, raw = _http("GET", url, headers=self._headers())
        if code == 404:
            return {"ok": False, "message": (parsed or {}).get("message") or "Not found yet.", "events": []}
        if not isinstance(parsed, dict) or not parsed.get("success"):
            msg = (parsed or {}).get("message") or raw[:200] or f"HTTP {code}"
            return {"ok": False, "message": f"Tracking lookup failed — {msg}", "events": []}

        events = []
        for e in (parsed.get("events") or []):
            events.append({
                "at": e.get("scan_datetime", ""),
                "description": e.get("description", ""),
                "city": e.get("facility_city", ""),
                "state": e.get("facility_state", ""),
                "detail": e.get("mail_phase", "") or e.get("process_description", ""),
            })
        return {
            "ok": True,
            "message": parsed.get("message") or parsed.get("status", ""),
            "status": parsed.get("status", ""),
            "percent": parsed.get("progress_percent"),
            "delivery_date": parsed.get("delivery_date"),
            "tracking_code": parsed.get("tracking_number") or shipment.tracking_code,
            "events": events,
        }


# ──────────────────────────────────────────────────────────────────────────────
# EasyPost — real postage
# ──────────────────────────────────────────────────────────────────────────────
class EasyPostProvider(ShippingProvider):
    key = "easypost"
    buys_postage = True

    def _headers(self):
        key = str(self.cfg.get("api_key", "")).strip()
        token = base64.b64encode(f"{key}:".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}", "Accept": "application/json"}

    @staticmethod
    def _error(parsed, raw, code):
        """EasyPost nests its errors under `error`, unlike the marketplaces."""
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                subs = err.get("errors")
                if isinstance(subs, list) and subs:
                    details = []
                    for s in subs:
                        if isinstance(s, dict):
                            f = s.get("field")
                            m = s.get("message")
                            details.append(f"{f}: {m}" if f else str(m))
                    if details:
                        return f"{msg} ({'; '.join(details[:3])})" if msg else "; ".join(details[:3])
                if isinstance(msg, list):
                    return "; ".join(str(m) for m in msg[:3])
                if msg:
                    return str(msg)
            if parsed.get("message"):
                return str(parsed["message"])
        return raw[:300] if raw else f"HTTP {code}"

    def _from_address(self):
        return {
            "name": str(self.cfg.get("from_name", "")).strip() or None,
            "company": str(self.cfg.get("from_company", "")).strip() or None,
            "street1": str(self.cfg.get("from_address", "")).strip(),
            "street2": str(self.cfg.get("from_address2", "")).strip() or None,
            "city": str(self.cfg.get("from_city", "")).strip(),
            "state": _clean_state(self.cfg.get("from_state")),
            "zip": str(self.cfg.get("from_zip", "")).strip(),
            "country": "US",
            "phone": str(self.cfg.get("from_phone", "")).strip() or None,
        }

    def _to_address(self, order):
        return {
            "name": str(order.buyer_name or "").strip() or None,
            "company": str(order.buyer_company or "").strip() or None,
            "street1": str(order.address1 or "").strip(),
            "street2": str(order.address2 or "").strip() or None,
            "city": str(order.city or "").strip(),
            "state": _clean_state(order.state),
            "zip": str(order.zipcode or "").strip(),
            "country": str(order.country or "US").strip().upper() or "US",
            "email": str(order.email or "").strip() or None,
            "phone": str(order.phone or "").strip() or None,
        }

    def _parcel(self, opts=None):
        opts = opts or {}

        def num(key, cfg_key, default):
            v = opts.get(key, self.cfg.get(cfg_key, ""))
            try:
                f = float(str(v).strip())
                return f if f > 0 else default
            except (TypeError, ValueError):
                return default

        # Defaults sized for a few sleeved cards in a bubble mailer.
        return {
            "weight": num("weight_oz", "parcel_weight_oz", 3.0),   # EasyPost weight is ounces
            "length": num("length", "parcel_length", 6.0),
            "width": num("width", "parcel_width", 4.0),
            "height": num("height", "parcel_height", 0.75),
        }

    @staticmethod
    def _order_reference(order):
        """The reference stamped on every EasyPost shipment at quote time —
        also what create_label checks before buying a client-posted id."""
        return order.external_order_id or f"CCIM-{order.id}"

    def _shipment_body(self, order, opts=None):
        return {
            "shipment": {
                "to_address": self._to_address(order),
                "from_address": self._from_address(),
                "parcel": self._parcel(opts),
                # Ask for PDF up front — the default is PNG, and PDF is what
                # lets batch printing merge these with TCGTracking envelopes.
                "options": {"label_format": "PDF"},
                "reference": self._order_reference(order),
            }
        }

    def test_connection(self):
        missing = self._need("api_key")
        if missing:
            return {"ok": False, "message": "Missing: API key"}

        key = str(self.cfg.get("api_key", "")).strip()
        mode = str(self.cfg.get("mode", "test")).strip().lower() or "test"
        # Cheap, read-only probe that any key can make.
        code, parsed, raw = _http("GET", f"{EASYPOST_BASE}/addresses?page_size=1",
                                  headers=self._headers())
        if code == 401:
            return {"ok": False, "message": "EasyPost rejected that API key."}
        if code == 0:
            return {"ok": False, "message": raw or "Could not reach api.easypost.com."}
        if code != 200:
            return {"ok": False, "message": self._error(parsed, raw, code)}

        looks_test = key.upper().startswith("EZTK")
        if mode == "production" and looks_test:
            return {"ok": True, "message": "Connected, but that's a TEST key while mode is set to "
                                           "production. Test keys make fake labels."}
        if mode == "test" and not looks_test:
            return {"ok": True, "message": "Connected with a PRODUCTION key while mode is set to test. "
                                           "Buying a label will charge you for real."}
        return {"ok": True, "message": f"Connected to EasyPost ({mode} mode)."}

    def quote(self, order, opts=None):
        """Create a shipment to fetch live rates. Nothing is bought here."""
        missing = self._need("api_key", "from_address", "from_city", "from_state", "from_zip")
        if missing:
            return {"ok": False, "message": f"Set the ship-from address first (missing: "
                                            f"{', '.join(missing)}).", "rates": []}
        problems = _validate_destination(order)
        if problems:
            return {"ok": False, "message": "This order still needs " + ", ".join(problems) + ".",
                    "rates": []}

        code, parsed, raw = _http("POST", f"{EASYPOST_BASE}/shipments",
                                  headers=self._headers(), body=self._shipment_body(order, opts))
        if code not in (200, 201) or not isinstance(parsed, dict):
            return {"ok": False, "message": self._error(parsed, raw, code), "rates": []}

        rates = []
        for r in (parsed.get("rates") or []):
            rates.append({
                "id": r.get("id"),
                "carrier": r.get("carrier", ""),
                "service": r.get("service", ""),
                "amount": _money(r.get("rate")),
                "currency": r.get("currency", "USD"),
                "delivery_days": r.get("delivery_days") or r.get("est_delivery_days"),
                "retail_rate": _money(r.get("retail_rate"), None) if r.get("retail_rate") else None,
            })
        rates.sort(key=lambda x: (x["amount"] if x["amount"] is not None else 9e9))
        if not rates:
            return {"ok": False, "message": "EasyPost returned no rates — check the ship-from address, "
                                            "the parcel size, and that a carrier is enabled on your "
                                            "account.", "rates": [], "shipment_id": parsed.get("id")}
        return {"ok": True, "message": f"{len(rates)} rate(s).", "rates": rates,
                "shipment_id": parsed.get("id")}

    def _pick_rate(self, rates):
        """Preferred service if it's on offer and configured, else cheapest."""
        pref = str(self.cfg.get("preferred_service", "")).strip().lower()
        if pref:
            for r in rates:
                if (r.get("service") or "").strip().lower() == pref:
                    return r
        return rates[0] if rates else None

    def create_label(self, order, opts=None):
        opts = opts or {}

        # A rate id from a previous quote can be bought directly; otherwise
        # quote now and pick. Re-quoting is safer than trusting a stale id.
        rate_id = str(opts.get("rate_id", "")).strip()
        shipment_id = str(opts.get("shipment_id", "")).strip()

        if not (rate_id and shipment_id):
            q = self.quote(order, opts)
            if not q.get("ok"):
                return {"ok": False, "message": q.get("message", "Could not rate this order.")}
            shipment_id = q.get("shipment_id")
            chosen = self._pick_rate(q["rates"])
            if not chosen:
                return {"ok": False, "message": "No rate available to buy."}
            rate_id = chosen["id"]
        else:
            # The id pair came from the client. Two order tabs (or a crafted
            # POST) can cross them, buying real postage for order A's address
            # and recording it as order B's shipment. The quote stamped this
            # order's reference on the shipment; refuse to buy unless it
            # matches.
            code, parsed, raw = _http("GET", f"{EASYPOST_BASE}/shipments/{shipment_id}",
                                      headers=self._headers())
            if code != 200 or not isinstance(parsed, dict):
                return {"ok": False, "message": self._error(parsed, raw, code)}
            if (parsed.get("reference") or "") != self._order_reference(order):
                return {"ok": False,
                        "message": "That quote belongs to a different order — re-quote "
                                   "this order and try again."}

        insured = opts.get("insured")
        if insured is None:
            insured = self._flag("insure_by_default")
        buy_body = {"rate": {"id": rate_id}}
        insurance_amount = None
        if insured and order.declared_value > 0:
            insurance_amount = order.declared_value
            buy_body["insurance"] = f"{insurance_amount:.2f}"

        code, parsed, raw = _http("POST", f"{EASYPOST_BASE}/shipments/{shipment_id}/buy",
                                  headers=self._headers(), body=buy_body)
        if code not in (200, 201) or not isinstance(parsed, dict):
            return {"ok": False, "message": self._error(parsed, raw, code)}

        label = parsed.get("postage_label") or {}
        label_url = label.get("label_url")
        if not label_url:
            return {"ok": False, "message": "EasyPost bought the shipment but returned no label URL. "
                                            "Check the shipment in your EasyPost dashboard."}

        label_bytes, err = _download(label_url)
        if err:
            return {"ok": False, "message": err}

        sel = parsed.get("selected_rate") or {}
        tracker = parsed.get("tracker") or {}
        fmt = str(label.get("label_file_type", "")).lower()
        label_format = "PDF" if "pdf" in fmt else ("PNG" if "png" in fmt else "PDF")

        return {
            "ok": True,
            "message": f"Bought {sel.get('carrier', '')} {sel.get('service', '')} "
                       f"for ${_money(sel.get('rate')):.2f}.".strip(),
            "external_id": parsed.get("id", ""),
            "tracking_code": parsed.get("tracking_code", ""),
            "tracking_url": tracker.get("public_url", ""),
            "carrier": sel.get("carrier", ""),
            "service": sel.get("service", ""),
            "rate_amount": _money(sel.get("rate")),
            "currency": sel.get("currency", "USD"),
            "insured": bool(insurance_amount),
            "insurance_amount": insurance_amount,
            "label_bytes": label_bytes,
            "label_format": label_format,
            "duplicate": False,
        }

    def relabel(self, shipment):
        """Re-download the label EasyPost already generated."""
        if not shipment.external_id:
            return {"ok": False, "message": "This shipment has no EasyPost id to reprint from."}
        code, parsed, raw = _http("GET", f"{EASYPOST_BASE}/shipments/{shipment.external_id}",
                                  headers=self._headers())
        if code != 200 or not isinstance(parsed, dict):
            return {"ok": False, "message": self._error(parsed, raw, code)}
        label_url = (parsed.get("postage_label") or {}).get("label_url")
        if not label_url:
            return {"ok": False, "message": "That EasyPost shipment has no label yet."}
        label_bytes, err = _download(label_url)
        if err:
            return {"ok": False, "message": err}
        return {"ok": True, "label_bytes": label_bytes, "label_format": "PDF"}

    def track(self, shipment):
        if self._need("api_key"):
            return {"ok": False, "message": "Missing: API key", "events": []}
        if not shipment.tracking_code:
            return {"ok": False, "message": "This shipment has no tracking code yet.", "events": []}

        url = f"{EASYPOST_BASE}/trackers?" + urllib.parse.urlencode(
            {"tracking_code": shipment.tracking_code, "page_size": 1})
        code, parsed, raw = _http("GET", url, headers=self._headers())
        if code != 200 or not isinstance(parsed, dict):
            return {"ok": False, "message": self._error(parsed, raw, code), "events": []}

        trackers = parsed.get("trackers") or []
        if not trackers:
            return {"ok": False, "message": "EasyPost has no tracker for that code yet.", "events": []}
        t = trackers[0]

        events = []
        for d in (t.get("tracking_details") or []):
            loc = d.get("tracking_location") or {}
            events.append({
                "at": d.get("datetime", ""),
                "description": d.get("message", "") or d.get("status", ""),
                "city": loc.get("city") or "",
                "state": loc.get("state") or "",
                "detail": d.get("status_detail", "") or "",
            })
        events.reverse()   # newest first, matching TCGTracking's ordering

        status = str(t.get("status", "")).replace("_", " ").strip()
        return {
            "ok": True,
            "message": status or "Tracking updated.",
            "status": status.title() if status else "",
            "percent": None,   # EasyPost has no progress metric
            "delivery_date": (t.get("est_delivery_date") or "")[:10] or None,
            "tracking_url": t.get("public_url", ""),
            "events": events,
        }


PROVIDERS = {
    "tcgtracking": TCGTrackingProvider,
    "easypost": EasyPostProvider,
}


def get_shipping_provider(key, connection, persist=None):
    cls = PROVIDERS.get(key)
    if cls is None:
        return None
    return cls(connection, persist=persist)
