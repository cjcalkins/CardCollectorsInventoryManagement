"""
tcgcsv_sync.py — download a game's catalog from tcgcsv.com and cache it locally
as ReferenceCard rows, so OCR results can be matched to a real card and used to
auto-fill entry data (name, collector number, set, rarity, TCGplayer URL, ...).

tcgcsv.com mirrors TCGplayer's category -> group -> product -> price data,
refreshed once per day (~20:00 UTC). JSON endpoints:
    categories:   https://tcgcsv.com/tcgplayer/categories
    groups:       https://tcgcsv.com/tcgplayer/{categoryId}/groups
    products:     https://tcgcsv.com/tcgplayer/{categoryId}/{groupId}/products
    prices:       https://tcgcsv.com/tcgplayer/{categoryId}/{groupId}/prices
    last updated: https://tcgcsv.com/last-updated.txt

Politeness rules straight from tcgcsv's own docs, honoured here:
  * a descriptive User-Agent is REQUIRED (generic/missing UAs may be blocked),
  * keep ~100ms between requests (we use 120ms),
  * data changes only once a day — re-sync only when last-updated.txt is newer,
  * fetch server-side only; their CORS blocks browser fetch/XHR. This module is
    called from Flask routes, never from the page directly.

A card's collector number and rarity live in the product's `extendedData`
list (keys "Number" and "Rarity"); everything else is on the product itself.
"""

import json
import time
import urllib.request
import urllib.error

BASE = "https://tcgcsv.com"
# Identify ourselves clearly per tcgcsv guidance. Bump the version if behaviour
# changes so operators can spot our traffic.
USER_AGENT = "CardCollectorInventoryManager/1.0 (local inventory tool; tcgcsv reference sync)"
REQUEST_SLEEP = 0.12  # seconds between upstream requests (>100ms as asked)


# --------------------------------------------------------------------------- #
# Low-level fetch
# --------------------------------------------------------------------------- #
def _get_text(url, timeout=30):
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _get_json(url, timeout=30):
    return json.loads(_get_text(url, timeout=timeout))


def _results(url, timeout=30):
    """tcgcsv wraps most collections as {results:[...]}; return the list."""
    data = _get_json(url, timeout=timeout)
    return data.get("results", []) or []


# --------------------------------------------------------------------------- #
# Endpoint wrappers
# --------------------------------------------------------------------------- #
def get_last_updated(timeout=15):
    """tcgcsv's single last-build timestamp, or '' if unreachable."""
    try:
        return _get_text(f"{BASE}/last-updated.txt", timeout=timeout).strip()
    except Exception:
        return ""


def get_categories(timeout=30):
    """All categories (games). Raw list of tcgplayer category dicts."""
    return _results(f"{BASE}/tcgplayer/categories", timeout=timeout)


def get_groups(category_id, timeout=30):
    """All groups (sets) for a category."""
    return _results(f"{BASE}/tcgplayer/{int(category_id)}/groups", timeout=timeout)


def get_products(category_id, group_id, timeout=60):
    """All products (cards + sealed) for a group, with extendedData."""
    return _results(f"{BASE}/tcgplayer/{int(category_id)}/{int(group_id)}/products", timeout=timeout)


def get_prices(category_id, group_id, timeout=60):
    """Market-price rows for a group (join to products via productId)."""
    return _results(f"{BASE}/tcgplayer/{int(category_id)}/{int(group_id)}/prices", timeout=timeout)


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def extended_to_dict(extended_data):
    """Flatten a product's extendedData list into a {name: value} dict."""
    out = {}
    for item in extended_data or []:
        key = (item.get("name") or "").strip()
        if key:
            out[key] = item.get("value")
    return out


def is_card(extended):
    """Treat a product as a card (not sealed) if it exposes Number or Rarity."""
    return bool(extended.get("Number") or extended.get("Rarity"))


def price_index(price_rows):
    """Build {productId: representative marketPrice}. Prefers the 'Normal'
    sub-type, else the first row that carries a marketPrice."""
    by_product = {}
    for row in price_rows or []:
        by_product.setdefault(row.get("productId"), []).append(row)
    out = {}
    for pid, rows in by_product.items():
        normal = next((r for r in rows if r.get("subTypeName") == "Normal"), None)
        chosen = normal or next((r for r in rows if r.get("marketPrice") is not None), rows[0])
        out[pid] = chosen.get("marketPrice")
    return out


def normalize_product(product, category_id, category_name, group_id, group_name, market_price=None):
    """Turn a raw tcgcsv product into the flat dict used to upsert ReferenceCard.
    Returns None for non-card (sealed) products."""
    ext = extended_to_dict(product.get("extendedData"))
    if not is_card(ext):
        return None
    return {
        "category_id":  int(category_id),
        "group_id":     int(group_id),
        "product_id":   product.get("productId"),
        "game":         category_name,
        "set_name":     group_name,
        "name":         product.get("name") or "",
        "clean_name":   product.get("cleanName") or "",
        "number":       ext.get("Number") or "",
        "rarity":       ext.get("Rarity") or "",
        "image_url":    product.get("imageUrl") or "",
        "url":          product.get("url") or "",
        "market_price": market_price,
        "extended":     ext,
    }


def fetch_group_cards(category_id, category_name, group_id, group_name,
                      include_prices=True, sleep=REQUEST_SLEEP):
    """
    Download one group's card products (and optionally prices) and return a list
    of normalized ReferenceCard dicts. Sleeps between the two upstream calls to
    stay within tcgcsv's rate guidance. Raises on network/HTTP errors so the
    caller can report them per-group.
    """
    products = get_products(category_id, group_id)
    prices = []
    if include_prices:
        time.sleep(sleep)
        try:
            prices = get_prices(category_id, group_id)
        except Exception:
            prices = []  # prices are best-effort; a card is still useful without one
    pidx = price_index(prices)

    cards = []
    for p in products:
        rec = normalize_product(
            p, category_id, category_name, group_id, group_name,
            market_price=pidx.get(p.get("productId")),
        )
        if rec:
            cards.append(rec)
    return cards
