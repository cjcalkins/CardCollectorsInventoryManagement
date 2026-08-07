"""
shop_providers.py
=================

Marketplace connectors for the "Shops" tab. One provider class per marketplace
(Shopify, eBay, TCGplayer). Each speaks the real REST API of its marketplace
over urllib (no extra dependencies, matching the rest of the app).

Design notes
------------
* Credentials + cached OAuth tokens live in ShopConnection.config (JSON) in the
  local SQLite DB. Providers read that dict, and may refresh/persist tokens via
  the `persist` callback handed in by the caller.
* Every public method returns a plain dict with an "ok" boolean and a
  human-readable "message", never raising for expected API failures — the Flask
  layer turns these into JSON/flash messages.
* Local card images can't be fetched by a marketplace's servers (they live on
  127.0.0.1), so Shopify receives images as base64 attachments, while eBay is
  given any public image URLs the record already has and reports clearly when a
  listing needs hosted images.

Reality check (researched 2026):
* Shopify  — self-serve. Create a custom app in the store admin, copy the Admin
             API access token (shpat_…). Fully usable.
* eBay     — OAuth 2.0 authorization-code flow via a registered dev app + RuName;
             publishing a live offer also needs business policies + an inventory
             location on the account.
* TCGplayer— the developer API is closed to new applicants (eBay-owned). This
             connector works only if you already hold TCGplayer API keys.
"""

import json
import base64
import hmac
import hashlib
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta, timezone
# Imported as a bare name ON PURPOSE. Both Cardmarket request builders assign a local
# called `xml`, so `import xml.sax.saxutils` and a `xml.sax.saxutils.escape(...)` call
# inside them would resolve to that local -- Python marks `xml` local for the whole
# function body, so it would raise UnboundLocalError before the document is even built.
from xml.sax.saxutils import escape as _xml_escape

DEFAULT_TIMEOUT = 30
SHOPIFY_API_VERSION = "2026-01"


def _utcnow():
    """Naive UTC now (matches models.utcnow, duplicated because this module
    deliberately imports nothing framework-side). Replaces the deprecated
    datetime.utcnow; token-expiry timestamps stay naive UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ──────────────────────────────────────────────────────────────────────────────
# Redirects must not carry credentials to another origin
# ──────────────────────────────────────────────────────────────────────────────
# urllib follows up to 10 redirects silently, and the stdlib's redirect_request
# rebuilds the follow-up request keeping every header except the two content ones:
#
#     CONTENT_HEADERS = ("content-length", "content-type")
#     newheaders = {k: v for k, v in req.headers.items()
#                   if k.lower() not in CONTENT_HEADERS}
#
# So a 302 from any upstream re-sends X-Shopify-Access-Token, Authorization,
# X-API-Key or X-ManaPool-Access-Token to whatever host the Location names —
# including an http:// one. That is true even for the providers whose hostnames are
# hardcoded, because the hazard is the *response*, not the URL we chose: a
# compromised or MITM'd upstream only has to answer with a redirect.
#
# An ALLOWLIST rather than a list of credential header names, deliberately. A
# denylist has to be updated every time a provider adds a token header, and the
# person adding it is not thinking about redirects. Anything not named here is
# dropped when the origin changes, so a new header is protected by default.
_REDIRECT_SAFE_HEADERS = frozenset({
    "accept", "accept-encoding", "accept-language", "user-agent",
})


def _redirect_keeps_origin(old_url, new_url):
    """True when following old->new stays on the same scheme+host+port.

    An https->http step on the SAME host counts as a change: the credential would
    go out in cleartext, which is the thing being prevented. http->https is not a
    downgrade, but it is still a different origin, so it is treated the same way —
    losing a header on an upgrade is a failed request, not a leak."""
    a, b = urllib.parse.urlsplit(old_url), urllib.parse.urlsplit(new_url)
    return (a.scheme.lower(), a.hostname or "", a.port) == \
           (b.scheme.lower(), b.hostname or "", b.port)


class _CredentialStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drops non-benign headers when a redirect leaves the original origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None or _redirect_keeps_origin(req.full_url, new.full_url):
            return new
        # Request.add_header capitalises keys ("X-Api-Key"), so compare lowered.
        for name in list(new.headers):
            if name.lower() not in _REDIRECT_SAFE_HEADERS:
                del new.headers[name]
        return new


# Installed globally, on purpose: urlopen() uses the module-level opener, so this
# one call covers every outbound request in the process — shop_providers,
# shipping_providers and the JustTCG/Ximilar/CardSight calls in app.py — including
# any added later. Building a private opener here instead would protect this file
# and quietly leave the other nine call sites exactly as they are.
urllib.request.install_opener(
    urllib.request.build_opener(_CredentialStrippingRedirectHandler))


# ──────────────────────────────────────────────────────────────────────────────
# UI metadata — drives the connector cards / config forms on the Shops page.
# `secret: True` fields are never echoed back to the browser in plain text.
# ──────────────────────────────────────────────────────────────────────────────
MARKETPLACES = {
    "shopify": {
        "label": "Shopify",
        "icon": "fa-brands fa-shopify",
        "color": "#5E8E3E",
        "available": True,
        "oauth": False,
        "blurb": "Sync inventory to your Shopify store as products. Uses a custom-app Admin API token.",
        "fields": [
            # "host": True marks a field that decides WHERE the stored credential is
            # sent. Changing one un-connects the shop (see shops_save), because a
            # connection an administrator vouched for must not silently carry its token
            # to a different destination. Flagged on the field rather than in a list
            # elsewhere so that adding a host-bearing field to any provider makes the
            # question unavoidable — same reason "secret" lives here.
            {"key": "store_domain", "label": "Store domain", "placeholder": "your-store.myshopify.com", "type": "text", "host": True},
            {"key": "access_token", "label": "Admin API access token", "placeholder": "shpat_…", "type": "password", "secret": True},
            {"key": "location_id", "label": "Inventory location ID (optional)", "placeholder": "auto-detected", "type": "text", "optional": True},
        ],
        "help": "In Shopify admin: Settings → Apps and sales channels → Develop apps → create an app, "
                "grant write_products/read_products + write_inventory, install, then copy the Admin API access token.",
    },
    "ebay": {
        "label": "eBay",
        "icon": "fa-brands fa-ebay",
        "color": "#E53238",
        "available": True,
        "oauth": True,
        "blurb": "List items via the eBay Sell Inventory API. Connect your account with OAuth.",
        "fields": [
            {"key": "environment", "label": "Environment", "type": "select", "options": ["production", "sandbox"]},
            {"key": "client_id", "label": "App ID (Client ID)", "type": "text"},
            {"key": "client_secret", "label": "Cert ID (Client Secret)", "type": "password", "secret": True},
            {"key": "ru_name", "label": "RuName (redirect URL name)", "type": "text"},
            {"key": "merchant_location_key", "label": "Merchant location key", "type": "text", "optional": True},
            {"key": "fulfillment_policy_id", "label": "Fulfillment policy ID", "type": "text", "optional": True},
            {"key": "payment_policy_id", "label": "Payment policy ID", "type": "text", "optional": True},
            {"key": "return_policy_id", "label": "Return policy ID", "type": "text", "optional": True},
            {"key": "category_id", "label": "Default category ID", "type": "text", "optional": True},
        ],
        "help": "Register an app at developer.ebay.com, set an RuName whose redirect points to this app's "
                "/shops/ebay/callback, then click Connect. Publishing live offers also needs business "
                "policies + an inventory location on your eBay account.",
    },
    "tcgplayer": {
        "label": "TCGplayer",
        "icon": "fa-solid fa-layer-group",
        "color": "#F8991D",
        "available": True,
        "oauth": False,
        "closed_note": "TCGplayer closed its developer API to new applicants (eBay-owned). This connector "
                       "works only if you already hold TCGplayer API keys.",
        "blurb": "Sync store inventory pricing/quantity via the TCGplayer Store API (existing keys only).",
        "fields": [
            {"key": "public_key", "label": "Public Key", "type": "text"},
            {"key": "private_key", "label": "Private Key", "type": "password", "secret": True},
            {"key": "store_key", "label": "Store Key", "type": "text", "optional": True},
        ],
        "help": "If you already have TCGplayer API keys, paste the public/private keys. Store-level inventory "
                "sync additionally needs your Store Key (from the store authorization workflow).",
    },
    "manapool": {
        "label": "Mana Pool",
        "icon": "fa-solid fa-droplet",
        "color": "#2B6CB0",
        "available": True,
        "oauth": False,
        "mtg_only": True,
        "blurb": "Sync inventory to Mana Pool, a Magic: The Gathering-only US marketplace.",
        "fields": [
            {"key": "email", "label": "Mana Pool account email", "type": "text", "placeholder": "you@example.com"},
            {"key": "access_token", "label": "API Access Token", "type": "password", "secret": True},
        ],
        "help": "In your Mana Pool account settings, generate an API Access Token. Auth uses your account email "
                "plus that token. Mana Pool is Magic: The Gathering only — non-MTG cards are skipped.",
    },
    "cardmarket": {
        "label": "Cardmarket",
        "icon": "fa-solid fa-store",
        "color": "#0A2A66",
        "available": True,
        "oauth": False,
        "closed_note": "Cardmarket isn't accepting new API applications; access is limited to approved "
                       "professional sellers holding existing dedicated-app credentials.",
        "blurb": "Sync stock to Cardmarket (Europe's largest TCG marketplace) via the MKM API.",
        "fields": [
            {"key": "environment", "label": "Environment", "type": "select", "options": ["production", "sandbox"]},
            {"key": "app_token", "label": "App Token (Consumer Key)", "type": "text"},
            {"key": "app_secret", "label": "App Secret (Consumer Secret)", "type": "password", "secret": True},
            {"key": "access_token", "label": "Access Token", "type": "text"},
            {"key": "access_token_secret", "label": "Access Token Secret", "type": "password", "secret": True},
        ],
        "help": "Create a dedicated app in your Cardmarket profile (professional sellers only) to obtain the four "
                "OAuth 1.0a tokens. Listing to Cardmarket requires each card matched to a Cardmarket product id "
                "(idProduct).",
    },
    "cardtrader": {
        "label": "CardTrader",
        "icon": "fa-solid fa-right-left",
        "color": "#0EA5B5",
        "available": True,
        "oauth": False,
        "blurb": "List items on CardTrader (EU/international TCG marketplace) via a Bearer token.",
        "fields": [
            {"key": "jwt_token", "label": "API Token (JWT)", "type": "password", "secret": True},
        ],
        "help": "In CardTrader: Settings → API Access → Create New Token, then paste the JWT here. Listing "
                "requires each card matched to a CardTrader blueprint id; your inventory SKU is stored in the "
                "product's user_data_field so it can be pulled back and matched.",
    },
}

SECRET_FIELDS = {
    mk: {f["key"] for f in meta["fields"] if f.get("secret")}
    for mk, meta in MARKETPLACES.items()
}


# ──────────────────────────────────────────────────────────────────────────────
# Low-level HTTP helper
# ──────────────────────────────────────────────────────────────────────────────
def _http(method, url, headers=None, body=None, form=False, raw_body=False, timeout=DEFAULT_TIMEOUT):
    """
    Make an HTTP request. Returns (status_code, parsed_body, raw_text).
    parsed_body is a dict/list when the response is JSON, else None.
    Never raises for HTTP error statuses — the status code is returned so callers
    can branch on it. Only truly unreachable hosts raise URLError (caught here too).

    raw_body=True means `body` is already bytes (e.g. a pre-built XML document) and
    is sent as-is; the caller is responsible for the Content-Type header.
    """
    headers = dict(headers or {})
    data = None
    if body is not None:
        if raw_body:
            data = body if isinstance(body, (bytes, bytearray)) else str(body).encode("utf-8")
        elif form:
            data = urllib.parse.urlencode(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        else:
            data = json.dumps(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = resp.getcode()
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        code = exc.code
    except urllib.error.URLError as exc:
        return 0, None, f"Network error: {exc.reason}"
    except Exception as exc:  # pragma: no cover - defensive
        return 0, None, f"Request failed: {exc}"

    parsed = None
    if raw:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = None
    return code, parsed, raw


def _first_api_error(parsed, raw, fallback="Unknown error"):
    """Best-effort extraction of a readable error message from a marketplace body."""
    if isinstance(parsed, dict):
        # eBay style: {"errors":[{"message": "..."}]}
        errs = parsed.get("errors")
        if isinstance(errs, list) and errs:
            msgs = [e.get("longMessage") or e.get("message") for e in errs if isinstance(e, dict)]
            msgs = [m for m in msgs if m]
            if msgs:
                return "; ".join(msgs[:3])
        # Shopify style: {"errors": "..."} or {"errors": {field: [..]}}
        if isinstance(errs, str):
            return errs
        if isinstance(errs, dict):
            flat = []
            for k, v in errs.items():
                if isinstance(v, list):
                    flat.append(f"{k}: {', '.join(str(x) for x in v)}")
                else:
                    flat.append(f"{k}: {v}")
            if flat:
                return "; ".join(flat[:3])
        if parsed.get("error_description"):
            return parsed["error_description"]
        if parsed.get("message"):
            return parsed["message"]
    if raw:
        return raw[:300]
    return fallback


# ──────────────────────────────────────────────────────────────────────────────
# Base provider
# ──────────────────────────────────────────────────────────────────────────────
class ShopProvider:
    key = "base"

    def __init__(self, connection, persist=None):
        self.conn = connection
        self.cfg = dict(connection.config or {})
        self._persist = persist or (lambda: None)

    # -- config persistence ----------------------------------------------------
    def _save_cfg(self):
        # Reassign a new dict so SQLAlchemy's JSON change tracking fires.
        self.conn.config = dict(self.cfg)
        self._persist()

    def _need(self, *keys):
        missing = [k for k in keys if not str(self.cfg.get(k, "")).strip()]
        return missing

    # -- interface (overridden) ------------------------------------------------
    def test_connection(self):
        return {"ok": False, "message": "Not implemented"}

    def push(self, payload):
        return {"ok": False, "message": "Not implemented"}

    def end_listing(self, listing):
        return {"ok": False, "message": "Not implemented"}

    def pull(self):
        return {"ok": False, "message": "Not implemented", "items": []}


# ──────────────────────────────────────────────────────────────────────────────
# Shopify
# ──────────────────────────────────────────────────────────────────────────────
class ShopifyProvider(ShopProvider):
    key = "shopify"

    def _base(self):
        domain = str(self.cfg.get("store_domain", "")).strip()
        domain = domain.replace("https://", "").replace("http://", "").strip("/")
        return f"https://{domain}/admin/api/{SHOPIFY_API_VERSION}"

    def _headers(self):
        return {
            "X-Shopify-Access-Token": str(self.cfg.get("access_token", "")).strip(),
            "Accept": "application/json",
        }

    def test_connection(self):
        missing = self._need("store_domain", "access_token")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}"}
        code, parsed, raw = _http("GET", f"{self._base()}/shop.json", headers=self._headers())
        if code == 200 and isinstance(parsed, dict) and parsed.get("shop"):
            name = parsed["shop"].get("name", "store")
            return {"ok": True, "message": f"Connected to “{name}”.", "detail": name}
        if code == 401:
            return {"ok": False, "message": "Unauthorized — check the Admin API access token."}
        if code == 404:
            return {"ok": False, "message": "Store not found — check the store domain."}
        return {"ok": False, "message": _first_api_error(parsed, raw, "Shopify test failed")}

    def push(self, payload):
        missing = self._need("store_domain", "access_token")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}"}

        images = []
        # Prefer base64 attachments (local files); Shopify hosts them for us.
        for b64 in (payload.get("image_b64") or [])[:5]:
            images.append({"attachment": b64})
        for url in (payload.get("image_urls") or [])[:5]:
            images.append({"src": url})

        variant = {
            "price": f"{payload.get('price') or 0:.2f}",
            "sku": payload.get("sku"),
            "inventory_management": "shopify",
            "inventory_quantity": int(payload.get("quantity") or 0),
        }
        product = {
            "product": {
                "title": payload.get("title") or payload.get("sku") or "Card",
                "body_html": payload.get("description") or "",
                "vendor": payload.get("brand") or "",
                "product_type": payload.get("category") or "Trading Card",
                "tags": payload.get("tags") or "",
                "status": "active",
                "variants": [variant],
            }
        }
        if images:
            product["product"]["images"] = images

        existing_id = (payload.get("external_id") or "").strip()
        if existing_id:
            # Update existing product.
            product["product"]["id"] = int(existing_id)
            code, parsed, raw = _http(
                "PUT", f"{self._base()}/products/{existing_id}.json",
                headers=self._headers(), body=product,
            )
        else:
            code, parsed, raw = _http(
                "POST", f"{self._base()}/products.json",
                headers=self._headers(), body=product,
            )

        if code in (200, 201) and isinstance(parsed, dict) and parsed.get("product"):
            p = parsed["product"]
            domain = str(self.cfg.get("store_domain", "")).replace("https://", "").strip("/")
            handle = p.get("handle", "")
            return {
                "ok": True,
                "external_id": str(p.get("id")),
                "external_url": f"https://{domain}/products/{handle}" if handle else "",
                "status": "active",
                "message": "Product synced to Shopify.",
                "extra": {"variant_id": str((p.get("variants") or [{}])[0].get("id", ""))},
            }
        if code == 401:
            return {"ok": False, "message": "Unauthorized — check the access token."}
        return {"ok": False, "message": _first_api_error(parsed, raw, "Shopify push failed")}

    def end_listing(self, listing):
        if not listing.external_id:
            return {"ok": False, "message": "No Shopify product id on record."}
        # Archive rather than delete, so history is preserved.
        body = {"product": {"id": int(listing.external_id), "status": "archived"}}
        code, parsed, raw = _http(
            "PUT", f"{self._base()}/products/{listing.external_id}.json",
            headers=self._headers(), body=body,
        )
        if code == 200:
            return {"ok": True, "message": "Product archived on Shopify.", "status": "ended"}
        return {"ok": False, "message": _first_api_error(parsed, raw, "Shopify unlist failed")}

    def pull(self):
        missing = self._need("store_domain", "access_token")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}", "items": []}
        # Pull recent products and expose variant SKUs so the caller can match.
        code, parsed, raw = _http(
            "GET", f"{self._base()}/products.json?limit=250",
            headers=self._headers(),
        )
        if code != 200 or not isinstance(parsed, dict):
            return {"ok": False, "message": _first_api_error(parsed, raw, "Shopify pull failed"), "items": []}
        items = []
        domain = str(self.cfg.get("store_domain", "")).replace("https://", "").strip("/")
        for p in parsed.get("products", []):
            for v in p.get("variants", []):
                items.append({
                    "sku": v.get("sku") or "",
                    "external_id": str(p.get("id")),
                    "external_url": f"https://{domain}/products/{p.get('handle','')}",
                    "title": p.get("title"),
                    "price": float(v.get("price") or 0),
                    "quantity": int(v.get("inventory_quantity") or 0),
                    "status": "active" if p.get("status") == "active" else (p.get("status") or "draft"),
                })
        return {"ok": True, "message": f"Fetched {len(items)} variant(s) from Shopify.", "items": items}


# ──────────────────────────────────────────────────────────────────────────────
# eBay
# ──────────────────────────────────────────────────────────────────────────────
class EbayProvider(ShopProvider):
    key = "ebay"

    OAUTH_SCOPES = "https://api.ebay.com/oauth/api_scope/sell.inventory"

    def _is_sandbox(self):
        return str(self.cfg.get("environment", "production")).lower() == "sandbox"

    def _api_root(self):
        return "https://api.sandbox.ebay.com" if self._is_sandbox() else "https://api.ebay.com"

    def _auth_host(self):
        return "https://auth.sandbox.ebay.com" if self._is_sandbox() else "https://auth.ebay.com"

    def _marketplace_id(self):
        return self.cfg.get("marketplace_id", "EBAY_US")

    # -- OAuth -----------------------------------------------------------------
    def authorize_url(self, state=""):
        params = {
            "client_id": self.cfg.get("client_id", ""),
            "redirect_uri": self.cfg.get("ru_name", ""),
            "response_type": "code",
            "scope": self.OAUTH_SCOPES,
        }
        if state:
            params["state"] = state
        return f"{self._auth_host()}/oauth2/authorize?" + urllib.parse.urlencode(params)

    def _basic_auth_header(self):
        raw = f"{self.cfg.get('client_id','')}:{self.cfg.get('client_secret','')}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def exchange_code(self, code):
        """Exchange an authorization code for user access + refresh tokens."""
        url = f"{self._api_root()}/identity/v1/oauth2/token"
        headers = {"Authorization": self._basic_auth_header(),
                   "Content-Type": "application/x-www-form-urlencoded"}
        body = {"grant_type": "authorization_code", "code": code,
                "redirect_uri": self.cfg.get("ru_name", "")}
        c, parsed, raw = _http("POST", url, headers=headers, body=body, form=True)
        if c == 200 and isinstance(parsed, dict) and parsed.get("access_token"):
            self._store_tokens(parsed)
            return {"ok": True, "message": "eBay account connected."}
        return {"ok": False, "message": _first_api_error(parsed, raw, "eBay token exchange failed")}

    def _store_tokens(self, tok):
        now = _utcnow()
        self.cfg["access_token"] = tok.get("access_token", "")
        self.cfg["access_expires_at"] = (now + timedelta(seconds=int(tok.get("expires_in", 7200)) - 120)).isoformat()
        if tok.get("refresh_token"):
            self.cfg["refresh_token"] = tok["refresh_token"]
            self.cfg["refresh_expires_at"] = (now + timedelta(seconds=int(tok.get("refresh_token_expires_in", 47304000)))).isoformat()
        self._save_cfg()

    def _refresh_if_needed(self):
        token = self.cfg.get("access_token", "")
        exp = self.cfg.get("access_expires_at")
        if token and exp:
            try:
                if datetime.fromisoformat(exp) > _utcnow():
                    return True
            except ValueError:
                pass
        refresh = self.cfg.get("refresh_token")
        if not refresh:
            return False
        url = f"{self._api_root()}/identity/v1/oauth2/token"
        headers = {"Authorization": self._basic_auth_header(),
                   "Content-Type": "application/x-www-form-urlencoded"}
        body = {"grant_type": "refresh_token", "refresh_token": refresh, "scope": self.OAUTH_SCOPES}
        c, parsed, raw = _http("POST", url, headers=headers, body=body, form=True)
        if c == 200 and isinstance(parsed, dict) and parsed.get("access_token"):
            self._store_tokens(parsed)
            return True
        return False

    def _auth_headers(self, extra=None):
        h = {
            "Authorization": f"Bearer {self.cfg.get('access_token','')}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Content-Language": "en-US",
            "X-EBAY-C-MARKETPLACE-ID": self._marketplace_id(),
        }
        if extra:
            h.update(extra)
        return h

    def is_connected(self):
        return bool(self.cfg.get("refresh_token") or self.cfg.get("access_token"))

    def test_connection(self):
        if self._need("client_id", "client_secret", "ru_name") and not self.is_connected():
            return {"ok": False, "message": "Enter app credentials, then click Connect to authorize."}
        if not self.is_connected():
            return {"ok": False, "message": "Not authorized yet — click Connect to sign in with eBay."}
        if not self._refresh_if_needed():
            return {"ok": False, "message": "Could not refresh the eBay token — reconnect the account."}
        # A cheap authorized call: list inventory locations.
        url = f"{self._api_root()}/sell/inventory/v1/location?limit=1"
        c, parsed, raw = _http("GET", url, headers=self._auth_headers())
        if c in (200, 204):
            return {"ok": True, "message": "eBay account connected and token valid."}
        if c == 401:
            return {"ok": False, "message": "eBay token rejected — reconnect the account."}
        return {"ok": False, "message": _first_api_error(parsed, raw, "eBay test failed")}

    def push(self, payload):
        if not self.is_connected():
            return {"ok": False, "message": "Connect your eBay account first."}
        if not self._refresh_if_needed():
            return {"ok": False, "message": "eBay token expired — reconnect the account."}

        needed = self._need("merchant_location_key", "fulfillment_policy_id",
                            "payment_policy_id", "return_policy_id", "category_id")
        if needed:
            return {"ok": False,
                    "message": "eBay listing needs these set in the connector: " + ", ".join(needed) +
                               ". Create business policies + an inventory location on your eBay account."}

        image_urls = [u for u in (payload.get("image_urls") or []) if u.startswith("http")]
        if not image_urls:
            return {"ok": False,
                    "message": "eBay requires publicly hosted image URLs; this record only has local images. "
                               "Add a Photo URL to the record or host the image first."}

        sku = payload.get("sku")
        # 1) Create/replace the inventory item keyed by SKU.
        item_body = {
            "availability": {"shipToLocationAvailability": {"quantity": int(payload.get("quantity") or 1)}},
            "condition": payload.get("ebay_condition", "USED_VERY_GOOD"),
            "product": {
                "title": (payload.get("title") or sku)[:80],
                "description": payload.get("description") or payload.get("title") or sku,
                "imageUrls": image_urls[:12],
            },
        }
        url = f"{self._api_root()}/sell/inventory/v1/inventory_item/{urllib.parse.quote(sku)}"
        c, parsed, raw = _http("PUT", url, headers=self._auth_headers(), body=item_body)
        if c not in (200, 201, 204):
            return {"ok": False, "message": "Inventory item: " + _first_api_error(parsed, raw, "failed")}

        # 2) Create an offer (or reuse an existing one for this SKU).
        offer_id = (payload.get("extra") or {}).get("offer_id") or self._find_offer(sku)
        offer_body = {
            "sku": sku,
            "marketplaceId": self._marketplace_id(),
            "format": "FIXED_PRICE",
            "availableQuantity": int(payload.get("quantity") or 1),
            "categoryId": str(self.cfg.get("category_id")),
            "listingDescription": payload.get("description") or payload.get("title") or sku,
            "listingPolicies": {
                "fulfillmentPolicyId": str(self.cfg.get("fulfillment_policy_id")),
                "paymentPolicyId": str(self.cfg.get("payment_policy_id")),
                "returnPolicyId": str(self.cfg.get("return_policy_id")),
            },
            "merchantLocationKey": str(self.cfg.get("merchant_location_key")),
            "pricingSummary": {"price": {"value": f"{payload.get('price') or 0:.2f}",
                                         "currency": payload.get("currency", "USD")}},
        }
        if offer_id:
            url = f"{self._api_root()}/sell/inventory/v1/offer/{offer_id}"
            c, parsed, raw = _http("PUT", url, headers=self._auth_headers(), body=offer_body)
            if c not in (200, 204):
                return {"ok": False, "message": "Offer update: " + _first_api_error(parsed, raw, "failed")}
        else:
            url = f"{self._api_root()}/sell/inventory/v1/offer"
            c, parsed, raw = _http("POST", url, headers=self._auth_headers(), body=offer_body)
            if c not in (200, 201) or not isinstance(parsed, dict):
                return {"ok": False, "message": "Offer create: " + _first_api_error(parsed, raw, "failed")}
            offer_id = parsed.get("offerId")

        # 3) Publish the offer → live listing.
        url = f"{self._api_root()}/sell/inventory/v1/offer/{offer_id}/publish"
        c, parsed, raw = _http("POST", url, headers=self._auth_headers())
        if c in (200, 201) and isinstance(parsed, dict):
            listing_id = parsed.get("listingId", "")
            base = "https://sandbox.ebay.com" if self._is_sandbox() else "https://www.ebay.com"
            return {
                "ok": True,
                "external_id": listing_id or offer_id,
                "external_url": f"{base}/itm/{listing_id}" if listing_id else "",
                "status": "active",
                "message": "Published live eBay listing.",
                "extra": {"offer_id": offer_id, "listing_id": listing_id},
            }
        # Offer created but publish failed (often missing policies/verification).
        return {"ok": False,
                "message": "Publish: " + _first_api_error(parsed, raw, "failed"),
                "external_id": offer_id, "status": "draft",
                "extra": {"offer_id": offer_id}}

    def _find_offer(self, sku):
        url = f"{self._api_root()}/sell/inventory/v1/offer?sku={urllib.parse.quote(sku)}"
        c, parsed, raw = _http("GET", url, headers=self._auth_headers())
        if c == 200 and isinstance(parsed, dict):
            offers = parsed.get("offers") or []
            if offers:
                return offers[0].get("offerId")
        return None

    def end_listing(self, listing):
        if not self._refresh_if_needed():
            return {"ok": False, "message": "eBay token expired — reconnect."}
        offer_id = (listing.extra or {}).get("offer_id")
        if not offer_id:
            return {"ok": False, "message": "No eBay offer id on record."}
        url = f"{self._api_root()}/sell/inventory/v1/offer/{offer_id}/withdraw"
        c, parsed, raw = _http("POST", url, headers=self._auth_headers())
        if c in (200, 204):
            return {"ok": True, "message": "eBay listing ended.", "status": "ended"}
        return {"ok": False, "message": _first_api_error(parsed, raw, "eBay withdraw failed")}

    def pull(self):
        if not self.is_connected():
            return {"ok": False, "message": "Connect your eBay account first.", "items": []}
        if not self._refresh_if_needed():
            return {"ok": False, "message": "eBay token expired — reconnect.", "items": []}
        url = f"{self._api_root()}/sell/inventory/v1/offer?limit=100"
        c, parsed, raw = _http("GET", url, headers=self._auth_headers())
        if c != 200 or not isinstance(parsed, dict):
            return {"ok": False, "message": _first_api_error(parsed, raw, "eBay pull failed"), "items": []}
        items = []
        for off in parsed.get("offers", []):
            price = ((off.get("pricingSummary") or {}).get("price") or {}).get("value")
            items.append({
                "sku": off.get("sku", ""),
                "external_id": off.get("listing", {}).get("listingId") or off.get("offerId"),
                "external_url": "",
                "title": off.get("sku", ""),
                "price": float(price) if price else 0.0,
                "quantity": int(off.get("availableQuantity") or 0),
                "status": (off.get("status") or "").lower() or "draft",
                "extra": {"offer_id": off.get("offerId")},
            })
        return {"ok": True, "message": f"Fetched {len(items)} eBay offer(s).", "items": items}


# ──────────────────────────────────────────────────────────────────────────────
# TCGplayer  (existing-key holders only)
# ──────────────────────────────────────────────────────────────────────────────
class TCGplayerProvider(ShopProvider):
    key = "tcgplayer"

    API_ROOT = "https://api.tcgplayer.com"
    VERSION = "v1.39.0"

    def _bearer(self):
        """Return a valid bearer token, minting a new one via client_credentials if needed."""
        token = self.cfg.get("bearer_token", "")
        exp = self.cfg.get("bearer_expires_at")
        if token and exp:
            try:
                if datetime.fromisoformat(exp) > _utcnow():
                    return token
            except ValueError:
                pass
        if self._need("public_key", "private_key"):
            return None
        body = {
            "grant_type": "client_credentials",
            "client_id": self.cfg.get("public_key"),
            "client_secret": self.cfg.get("private_key"),
        }
        c, parsed, raw = _http("POST", f"{self.API_ROOT}/token", body=body, form=True)
        if c == 200 and isinstance(parsed, dict) and parsed.get("access_token"):
            self.cfg["bearer_token"] = parsed["access_token"]
            secs = int(parsed.get("expires_in", 1209600)) if str(parsed.get("expires_in", "")).isdigit() else 1209600
            self.cfg["bearer_expires_at"] = (_utcnow() + timedelta(seconds=max(secs - 3600, 60))).isoformat()
            self._save_cfg()
            return self.cfg["bearer_token"]
        return None

    def _headers(self, token):
        return {"Authorization": f"bearer {token}", "Accept": "application/json"}

    def test_connection(self):
        if self._need("public_key", "private_key"):
            return {"ok": False, "message": "Enter your TCGplayer public and private keys."}
        token = self._bearer()
        if not token:
            return {"ok": False, "message": "Could not obtain a TCGplayer token — keys may be invalid or revoked."}
        # Categories is a cheap authorized call available to key holders.
        c, parsed, raw = _http("GET", f"{self.API_ROOT}/{self.VERSION}/catalog/categories?limit=1",
                               headers=self._headers(token))
        if c == 200:
            return {"ok": True, "message": "TCGplayer token valid."}
        if c in (401, 403):
            return {"ok": False, "message": "TCGplayer rejected the token (access may be restricted for this key)."}
        return {"ok": False, "message": _first_api_error(parsed, raw, "TCGplayer test failed")}

    def push(self, payload):
        token = self._bearer()
        if not token:
            return {"ok": False, "message": "TCGplayer token unavailable — check your keys."}
        store_key = str(self.cfg.get("store_key", "")).strip()
        if not store_key:
            return {"ok": False, "message": "Set your Store Key to sync store inventory."}
        tcg_sku = str((payload.get("extra") or {}).get("tcg_sku_id")
                      or payload.get("tcgplayer_sku_id") or "").strip()
        if not tcg_sku:
            return {"ok": False,
                    "message": "This record has no TCGplayer SKU id. TCGplayer lists against catalog SKUs, "
                               "so match the card to a TCGplayer product/SKU first."}
        # Documented store inventory pricing endpoint (existing-key holders).
        url = f"{self.API_ROOT}/{self.VERSION}/stores/{urllib.parse.quote(store_key)}/inventory/skus/{tcg_sku}"
        body = {"price": round(float(payload.get("price") or 0), 2),
                "quantity": int(payload.get("quantity") or 0),
                "channelId": 0}
        c, parsed, raw = _http("PUT", url, headers=self._headers(token), body=body)
        if c in (200, 201, 204):
            return {"ok": True, "external_id": tcg_sku, "status": "active",
                    "message": "TCGplayer store inventory updated."}
        return {"ok": False, "message": _first_api_error(parsed, raw, "TCGplayer push failed")}

    def end_listing(self, listing):
        # Setting quantity to 0 removes it from sale without deleting catalog linkage.
        token = self._bearer()
        store_key = str(self.cfg.get("store_key", "")).strip()
        if not token or not store_key or not listing.external_id:
            return {"ok": False, "message": "Missing token / store key / SKU id."}
        url = f"{self.API_ROOT}/{self.VERSION}/stores/{urllib.parse.quote(store_key)}/inventory/skus/{listing.external_id}"
        body = {"price": listing.price or 0, "quantity": 0, "channelId": 0}
        c, parsed, raw = _http("PUT", url, headers=self._headers(token), body=body)
        if c in (200, 201, 204):
            return {"ok": True, "message": "TCGplayer quantity set to 0.", "status": "ended"}
        return {"ok": False, "message": _first_api_error(parsed, raw, "TCGplayer unlist failed")}

    def pull(self):
        token = self._bearer()
        store_key = str(self.cfg.get("store_key", "")).strip()
        if not token:
            return {"ok": False, "message": "TCGplayer token unavailable.", "items": []}
        if not store_key:
            return {"ok": False, "message": "Set your Store Key to pull store inventory.", "items": []}
        url = f"{self.API_ROOT}/{self.VERSION}/stores/{urllib.parse.quote(store_key)}/inventory/products?limit=100"
        c, parsed, raw = _http("GET", url, headers=self._headers(token))
        if c != 200 or not isinstance(parsed, dict):
            return {"ok": False, "message": _first_api_error(parsed, raw, "TCGplayer pull failed"), "items": []}
        items = []
        for row in (parsed.get("results") or []):
            items.append({
                "sku": str(row.get("skuId") or row.get("productId") or ""),
                "external_id": str(row.get("skuId") or ""),
                "external_url": "",
                "title": row.get("productName") or "",
                "price": float(row.get("price") or 0),
                "quantity": int(row.get("quantity") or 0),
                "status": "active" if (row.get("quantity") or 0) > 0 else "ended",
            })
        return {"ok": True, "message": f"Fetched {len(items)} TCGplayer item(s).", "items": items}


# ──────────────────────────────────────────────────────────────────────────────
# Mana Pool  (Magic: The Gathering only)
# ──────────────────────────────────────────────────────────────────────────────
class ManaPoolProvider(ShopProvider):
    key = "manapool"

    BASE = "https://manapool.com/api/v1"

    def _headers(self):
        return {
            "X-ManaPool-Email": str(self.cfg.get("email", "")).strip(),
            "X-ManaPool-Access-Token": str(self.cfg.get("access_token", "")).strip(),
            "Accept": "application/json",
        }

    def test_connection(self):
        missing = self._need("email", "access_token")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}"}
        code, parsed, raw = _http("GET", f"{self.BASE}/seller/orders?limit=1", headers=self._headers())
        if code == 200:
            return {"ok": True, "message": "Mana Pool credentials valid."}
        if code in (401, 403):
            return {"ok": False, "message": "Mana Pool rejected the email/token."}
        return {"ok": False, "message": _first_api_error(parsed, raw, "Mana Pool test failed")}

    def push(self, payload):
        missing = self._need("email", "access_token")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}"}
        product_id = str(payload.get("manapool_product_id") or "").strip()
        if not product_id:
            return {"ok": False,
                    "message": "Mana Pool lists against its own product id (derived from a card's MTGJSON UUID). "
                               "Match this card to a Mana Pool product first."}
        item = {
            "product_id": product_id,
            "condition": payload.get("manapool_condition", "NM"),
            "finish": "foil" if payload.get("foil") else "normal",
            "price": round(float(payload.get("price") or 0), 2),
            "quantity": int(payload.get("quantity") or 0),
        }
        code, parsed, raw = _http("POST", f"{self.BASE}/seller/inventory",
                                  headers=self._headers(), body={"items": [item]})
        if code in (200, 201):
            return {"ok": True, "external_id": product_id, "status": "active",
                    "message": "Listed on Mana Pool."}
        return {"ok": False, "message": _first_api_error(parsed, raw, "Mana Pool push failed")}

    def end_listing(self, listing):
        if not listing.external_id:
            return {"ok": False, "message": "No Mana Pool product id on record."}
        code, parsed, raw = _http("POST", f"{self.BASE}/seller/inventory", headers=self._headers(),
                                  body={"items": [{"product_id": listing.external_id, "quantity": 0}]})
        if code in (200, 201):
            return {"ok": True, "message": "Mana Pool quantity set to 0.", "status": "ended"}
        return {"ok": False, "message": _first_api_error(parsed, raw, "Mana Pool unlist failed")}

    def pull(self):
        missing = self._need("email", "access_token")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}", "items": []}
        code, parsed, raw = _http("GET", f"{self.BASE}/seller/inventory", headers=self._headers())
        if code != 200 or not isinstance(parsed, (dict, list)):
            return {"ok": False, "message": _first_api_error(parsed, raw, "Mana Pool pull failed"), "items": []}
        rows = parsed.get("items", []) if isinstance(parsed, dict) else parsed
        items = []
        for row in (rows or []):
            if not isinstance(row, dict):
                continue
            items.append({
                "sku": "",  # Mana Pool has no custom SKU field → manual matching
                "external_id": str(row.get("product_id") or ""),
                "external_url": "",
                "title": row.get("name") or "",
                "price": float(row.get("price") or 0),
                "quantity": int(row.get("quantity") or 0),
                "status": "active" if (row.get("quantity") or 0) > 0 else "ended",
            })
        return {"ok": True,
                "message": f"Fetched {len(items)} Mana Pool item(s) (no shared SKU field, so matching is manual).",
                "items": items}


# ──────────────────────────────────────────────────────────────────────────────
# Cardmarket / MKM  (OAuth 1.0a HMAC-SHA1; approved professional sellers only)
# ──────────────────────────────────────────────────────────────────────────────
class CardmarketProvider(ShopProvider):
    key = "cardmarket"

    CONDITION_MAP = {
        "gem mint": "MT", "mint": "MT", "near mint": "NM", "excellent": "EX",
        "very good": "GD", "good": "GD", "lightly played": "GD",
        "moderately played": "LP", "played": "PL", "heavily played": "PL",
        "poor": "PO", "damaged": "PO",
    }

    def _base(self):
        if str(self.cfg.get("environment", "production")).lower() == "sandbox":
            return "https://sandbox.cardmarket.com/ws/v2.0"
        return "https://apiv2.cardmarket.com/ws/v2.0"

    @staticmethod
    def _pct(value):
        # RFC 3986 percent-encoding (Python leaves _.-~ unencoded and encodes '/').
        return urllib.parse.quote(str(value), safe="~")

    def _oauth_header(self, method, url):
        """OAuth 1.0a HMAC-SHA1 Authorization header for a dedicated app (no query params)."""
        import time, uuid
        oauth = {
            "oauth_consumer_key": self.cfg.get("app_token", ""),
            "oauth_token": self.cfg.get("access_token", ""),
            "oauth_nonce": uuid.uuid4().hex,
            "oauth_timestamp": str(int(time.time())),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_version": "1.0",
        }
        param_str = "&".join(f"{self._pct(k)}={self._pct(v)}" for k, v in sorted(oauth.items()))
        base = "&".join([method.upper(), self._pct(url), self._pct(param_str)])
        key = f"{self._pct(self.cfg.get('app_secret',''))}&{self._pct(self.cfg.get('access_token_secret',''))}"
        sig = base64.b64encode(hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()).decode()
        oauth["oauth_signature"] = sig
        header = f'OAuth realm="{url}", ' + ", ".join(
            f'{self._pct(k)}="{self._pct(v)}"' for k, v in oauth.items()
        )
        return {"Authorization": header, "Accept": "application/json"}

    def test_connection(self):
        missing = self._need("app_token", "app_secret", "access_token", "access_token_secret")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}"}
        url = f"{self._base()}/output.json/account"
        code, parsed, raw = _http("GET", url, headers=self._oauth_header("GET", url))
        if code == 200 and isinstance(parsed, dict) and parsed.get("account"):
            acct = parsed["account"]
            username = acct.get("username") or (acct.get("name") or {}).get("firstName", "seller")
            return {"ok": True, "message": f"Connected to Cardmarket as {username}."}
        if code in (401, 403):
            return {"ok": False, "message": "Cardmarket rejected the OAuth signature — check all four tokens."}
        return {"ok": False, "message": _first_api_error(parsed, raw, "Cardmarket test failed")}

    def push(self, payload):
        missing = self._need("app_token", "app_secret", "access_token", "access_token_secret")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}"}
        product_id = str(payload.get("cardmarket_product_id") or "").strip()
        if not product_id:
            return {"ok": False,
                    "message": "Cardmarket lists against a catalog product id (idProduct). "
                               "Match this card to a Cardmarket product first."}
        sku = payload.get("sku", "")
        cond = payload.get("cardmarket_condition", "NM")
        is_foil = "true" if payload.get("foil") else "false"
        price = f"{float(payload.get('price') or 0):.2f}"
        count = int(payload.get("quantity") or 1)
        # Every interpolated TEXT value is escaped, not just the one that is currently
        # attacker-controlled. product_id comes from extracted_data, which /update_scan
        # lets an inventory:edit user write freely, so without this a value like
        # "1</idProduct><idLanguage>7</idLanguage><comments>" rewrites a document that
        # is then signed with the operator's OAuth credentials -- the signature covers
        # the OAuth parameters, not the body.
        #
        # sku is app-generated (SHOP_SKU_PREFIX + record.id) and cond is a CONDITION_MAP
        # *value* rather than the caller's grade label, so neither is reachable today.
        # They are escaped anyway because escaping an app-generated string costs nothing,
        # and the alternative is that whoever changes where sku comes from has to
        # rediscover this. count/price are already int()/float()-coerced above and need
        # no escaping; the remaining tags are literals.
        #
        # Escaping rather than int()-coercing product_id: escaping cannot reject a value
        # Cardmarket would have accepted, and it is the same treatment the string fields
        # need anyway. A hostile id survives as inert text and the API refuses it.
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<request><article>"
            f"<idProduct>{_xml_escape(product_id)}</idProduct>"
            "<idLanguage>1</idLanguage>"
            f"<comments>{_xml_escape(str(sku))}</comments>"
            f"<count>{count}</count>"
            f"<price>{price}</price>"
            f"<condition>{_xml_escape(str(cond))}</condition>"
            f"<isFoil>{is_foil}</isFoil>"
            "<isSigned>false</isSigned><isPlayset>false</isPlayset>"
            "</article></request>"
        )
        url = f"{self._base()}/output.json/stock"
        headers = self._oauth_header("POST", url)
        headers["Content-Type"] = "text/xml"
        code, parsed, raw = _http("POST", url, headers=headers, body=xml.encode("utf-8"), raw_body=True)
        if code in (200, 201) and isinstance(parsed, dict):
            inserted = parsed.get("inserted") or [{}]
            art = inserted[0].get("idArticle") if (inserted and isinstance(inserted[0], dict)) else None
            return {"ok": True, "external_id": str(art or product_id), "status": "active",
                    "message": "Stock added on Cardmarket.",
                    "extra": {"id_product": product_id, "id_article": str(art or "")}}
        return {"ok": False, "message": _first_api_error(parsed, raw, "Cardmarket push failed")}

    def end_listing(self, listing):
        id_article = (listing.extra or {}).get("id_article") or listing.external_id
        if not id_article:
            return {"ok": False, "message": "No Cardmarket article id on record."}
        # id_article reaches here from the DB, but its provenance runs back to the user:
        # push() stores external_id as str(art or product_id), so when Cardmarket does
        # not return an idArticle the fallback is the caller's own product_id. Escaped
        # for that path. count is int()-coerced inline.
        xml = ('<?xml version="1.0" encoding="UTF-8"?>'
               f"<request><article><idArticle>{_xml_escape(str(id_article))}</idArticle>"
               f"<count>{int(listing.quantity or 1)}</count></article></request>")
        url = f"{self._base()}/output.json/stock"
        headers = self._oauth_header("DELETE", url)
        headers["Content-Type"] = "text/xml"
        code, parsed, raw = _http("DELETE", url, headers=headers, body=xml.encode("utf-8"), raw_body=True)
        if code in (200, 204):
            return {"ok": True, "message": "Removed from Cardmarket stock.", "status": "ended"}
        return {"ok": False, "message": _first_api_error(parsed, raw, "Cardmarket unlist failed")}

    def pull(self):
        missing = self._need("app_token", "app_secret", "access_token", "access_token_secret")
        if missing:
            return {"ok": False, "message": f"Missing: {', '.join(missing)}", "items": []}
        url = f"{self._base()}/output.json/stock"
        code, parsed, raw = _http("GET", url, headers=self._oauth_header("GET", url))
        if code != 200 or not isinstance(parsed, dict):
            return {"ok": False, "message": _first_api_error(parsed, raw, "Cardmarket pull failed"), "items": []}
        items = []
        for art in (parsed.get("article") or []):
            if not isinstance(art, dict):
                continue
            price = art.get("price")
            items.append({
                "sku": art.get("comments", "") or "",   # we stored CCIM-<id> here on push
                "external_id": str(art.get("idArticle") or ""),
                "external_url": "",
                "title": (art.get("product") or {}).get("enName", ""),
                "price": float(price) if price is not None else 0.0,
                "quantity": int(art.get("count") or 0),
                "status": "active",
                "extra": {"id_article": str(art.get("idArticle") or "")},
            })
        return {"ok": True, "message": f"Fetched {len(items)} Cardmarket article(s).", "items": items}


# ──────────────────────────────────────────────────────────────────────────────
# CardTrader  (Bearer JWT)
# ──────────────────────────────────────────────────────────────────────────────
class CardTraderProvider(ShopProvider):
    key = "cardtrader"

    BASE = "https://api.cardtrader.com/api/v2"

    CONDITION_MAP = {
        "gem mint": "Mint", "mint": "Mint", "near mint": "Near Mint",
        "excellent": "Slightly Played", "very good": "Slightly Played",
        "lightly played": "Slightly Played", "good": "Moderately Played",
        "moderately played": "Moderately Played", "played": "Played",
        "heavily played": "Heavily Played", "poor": "Poor", "damaged": "Poor",
    }

    def _headers(self):
        return {
            "Authorization": f"Bearer {str(self.cfg.get('jwt_token','')).strip()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def test_connection(self):
        if self._need("jwt_token"):
            return {"ok": False, "message": "Enter your CardTrader API token (JWT)."}
        code, parsed, raw = _http("GET", f"{self.BASE}/info", headers=self._headers())
        if code == 200:
            return {"ok": True, "message": "CardTrader token valid."}
        if code in (401, 403):
            return {"ok": False, "message": "CardTrader rejected the token."}
        return {"ok": False, "message": _first_api_error(parsed, raw, "CardTrader test failed")}

    def push(self, payload):
        if self._need("jwt_token"):
            return {"ok": False, "message": "Enter your CardTrader API token (JWT)."}
        blueprint_id = str(payload.get("cardtrader_blueprint_id") or "").strip()
        if not blueprint_id:
            return {"ok": False,
                    "message": "CardTrader lists against a blueprint id. Match this card to a "
                               "CardTrader blueprint first."}
        body = {
            "blueprint_id": int(blueprint_id),
            "price": round(float(payload.get("price") or 0), 2),
            "quantity": int(payload.get("quantity") or 1),
            "user_data_field": payload.get("sku", ""),   # our CCIM-<id> for round-trip matching
            "properties": {
                "condition": payload.get("cardtrader_condition", "Near Mint"),
                "mtg_foil": bool(payload.get("foil")),
            },
        }
        existing = (payload.get("extra") or {}).get("product_id") or payload.get("external_id")
        if existing:
            code, parsed, raw = _http("PUT", f"{self.BASE}/products/{existing}",
                                      headers=self._headers(), body=body)
        else:
            code, parsed, raw = _http("POST", f"{self.BASE}/products",
                                      headers=self._headers(), body=body)
        if code in (200, 201) and isinstance(parsed, dict):
            pid = parsed.get("id") or (parsed.get("resource") or {}).get("id") or existing
            return {"ok": True, "external_id": str(pid), "status": "active",
                    "external_url": f"https://www.cardtrader.com/cards/{blueprint_id}",
                    "message": "Listed on CardTrader.",
                    "extra": {"product_id": str(pid), "blueprint_id": blueprint_id}}
        return {"ok": False, "message": _first_api_error(parsed, raw, "CardTrader push failed")}

    def end_listing(self, listing):
        pid = (listing.extra or {}).get("product_id") or listing.external_id
        if not pid:
            return {"ok": False, "message": "No CardTrader product id on record."}
        code, parsed, raw = _http("DELETE", f"{self.BASE}/products/{pid}", headers=self._headers())
        if code in (200, 204):
            return {"ok": True, "message": "Removed from CardTrader.", "status": "ended"}
        return {"ok": False, "message": _first_api_error(parsed, raw, "CardTrader unlist failed")}

    def pull(self):
        if self._need("jwt_token"):
            return {"ok": False, "message": "Enter your CardTrader API token (JWT).", "items": []}
        code, parsed, raw = _http("GET", f"{self.BASE}/products/export", headers=self._headers())
        if code != 200 or not isinstance(parsed, (list, dict)):
            return {"ok": False, "message": _first_api_error(parsed, raw, "CardTrader pull failed"), "items": []}
        rows = parsed if isinstance(parsed, list) else parsed.get("products", [])
        items = []
        for p in rows:
            if not isinstance(p, dict):
                continue
            price = p.get("price")
            if isinstance(price, dict):
                price = (price.get("cents", 0) or 0) / 100.0
            items.append({
                "sku": p.get("user_data_field", "") or "",   # our CCIM-<id> round-trips here
                "external_id": str(p.get("id") or ""),
                "external_url": "",
                "title": (p.get("expansion") or {}).get("name", "") or p.get("name_en", ""),
                "price": float(price or 0),
                "quantity": int(p.get("quantity") or 0),
                "status": "active" if (p.get("quantity") or 0) > 0 else "ended",
                "extra": {"product_id": str(p.get("id") or "")},
            })
        return {"ok": True, "message": f"Fetched {len(items)} CardTrader product(s).", "items": items}


PROVIDERS = {
    "shopify": ShopifyProvider,
    "ebay": EbayProvider,
    "tcgplayer": TCGplayerProvider,
    "manapool": ManaPoolProvider,
    "cardmarket": CardmarketProvider,
    "cardtrader": CardTraderProvider,
}


def get_provider(marketplace, connection, persist=None):
    cls = PROVIDERS.get(marketplace)
    if not cls:
        return None
    return cls(connection, persist=persist)
