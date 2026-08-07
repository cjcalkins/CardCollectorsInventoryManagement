"""
pokemontcg_sync.py — reference-catalog adapter for the Pokémon TCG API
(https://pokemontcg.io, v2), a drop-in replacement for tcgcsv_sync.

It exposes the same interface app.py already consumes, so the existing
/reference/* routes, the ReferenceCard upsert, and the OCR matcher keep working
unchanged:

    get_categories()                         -> [{"categoryId","displayName","popularity"}]
    get_last_updated()                       -> str | None
    get_groups(category_id)                  -> [{"groupId","name", ...}]
    fetch_group_cards(cat_id, cat_name, group_id, group_name)
                                             -> [normalized card dict, ...]

pokemontcg.io uses STRING ids ("base1", "base1-4") while ReferenceCard/ReferenceSync
use INTEGER ids. We map each string id to a stable 48-bit integer (sha1 prefix);
fetch_group_cards re-derives the set string-id from the integer group_id by
matching against the live /sets list, so it is robust across restarts.

Covers the vintage WOTC sets tcgcsv is missing: Base (base1), Jungle (base2),
Fossil (base3), Base Set 2 (base4), Team Rocket (base5), Gym Heroes/Challenge
(gym1/gym2), Neo Genesis (neo1) and the rest of the Neo era, etc.

A free API key raises rate limits substantially; if POKEMONTCG_API_KEY is set in
the environment (Settings -> API Keys mirrors it there) it is sent as X-Api-Key.
"""

import os
import time
import hashlib

import requests

API_BASE = "https://api.pokemontcg.io/v2"
GAME_NAME = "Pokemon"
# Fixed synthetic category id for "Pokemon via pokemontcg.io". Large enough that
# it never collides with tcgcsv's small integer category ids.
CATEGORY_ID = 999_001
PAGE_SIZE = 250
TIMEOUT = 30
MAX_RETRIES = 4
# Ceiling on any single backoff sleep. A background sync thread must stay
# responsive; a server asking for a longer wait gets this instead, and the
# attempt budget runs out normally.
MAX_BACKOFF_SECONDS = 30

# Cache of {int group_id -> set string id} populated by get_groups()/_all_sets();
# fetch_group_cards falls back to a fresh /sets pull if a lookup misses.
_SET_ID_BY_GROUP = {}
# The set OBJECTS from the last /sets pull, so _resolve_set can answer without
# re-downloading the whole collection. Mutated in place so callers holding a
# reference (and the tests) see refreshes.
_SETS_CACHE = []


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _stable_int(s):
    """Deterministic positive 48-bit int from a string id (sha1 prefix)."""
    return int(hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:12], 16)


def _headers():
    key = os.environ.get("POKEMONTCG_API_KEY", "").strip()
    return {"X-Api-Key": key} if key else {}


def _retry_after_seconds(resp, attempt):
    """Seconds to wait before the next attempt: the server's Retry-After when
    it is a sane number, else linear backoff — CAPPED either way.

    Uncapped, a Retry-After of 86400 (a misbehaving proxy, or an aggressive
    rate limit) parked the background sync thread for a day PER ATTEMPT while
    the job sat "running" with no way to tell it apart from progress."""
    raw = (resp.headers.get("Retry-After") or "").strip()
    try:
        wait = float(raw)
    except ValueError:
        wait = 0.0
    if wait <= 0:
        wait = 1.5 * (attempt + 1)
    return min(wait, MAX_BACKOFF_SECONDS)


def _get(path, params=None):
    """GET with retry/backoff on rate limits and transient errors.

    Raises the LAST failure's own exception: an all-429 exhaustion used to
    re-raise a ConnectionError from an earlier attempt (last_exc was only set
    on RequestException), reporting a rate limit as a network outage.
    """
    url = f"{API_BASE}{path}"
    last_exc = None
    last_status = None
    last_body = ""
    for attempt in range(MAX_RETRIES):
        final_attempt = (attempt == MAX_RETRIES - 1)
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=TIMEOUT)
        except requests.RequestException as exc:
            last_exc, last_status, last_body = exc, None, ""
            if not final_attempt:      # never sleep after the last try
                time.sleep(min(1.5 * (attempt + 1), MAX_BACKOFF_SECONDS))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            last_exc = None            # this attempt's failure is the HTTP one
            last_status = resp.status_code
            last_body = (resp.text or "")[:300]
            if not final_attempt:
                time.sleep(_retry_after_seconds(resp, attempt))
            continue
        resp.raise_for_status()
        return resp.json()
    if last_exc is not None:
        raise last_exc
    detail = f"HTTP {last_status}" if last_status else "no response"
    if last_body:
        detail += f": {last_body}"
    raise RuntimeError(
        f"pokemontcg.io request failed after {MAX_RETRIES} attempts "
        f"({url}) — {detail}")


def _clean_name(name):
    out = []
    for ch in str(name or "").lower():
        out.append(ch if ch.isalnum() else "")
    return "".join(out)


def _canonical_number(number, printed_total):
    """Reconstruct the on-card N/M collector number so it matches what OCR reads
    and how the app narrows candidates. pokemontcg gives a bare N ("4"); pairing
    it with the set's printedTotal ("102") and padding N to M's width yields
    "004/102" — the same convention tcgcsv uses. Non-numeric numbers (promos like
    "SWSH001", "TG12") are returned unchanged."""
    raw = str(number or "").strip()
    if raw.isdigit() and printed_total:
        n, tot = int(raw), int(printed_total)
        return f"{n:0{len(str(tot))}d}/{tot}"
    return raw


# Preference order for choosing a single representative market price from the
# per-variant TCGplayer prices, biased by the card's rarity/edition.
_HOLO_ORDER = ["1stEditionHolofoil", "holofoil", "reverseHolofoil",
               "unlimitedHolofoil", "1stEdition", "unlimited", "normal"]
_PLAIN_ORDER = ["normal", "reverseHolofoil", "1stEdition", "unlimited",
                "holofoil", "1stEditionHolofoil", "unlimitedHolofoil"]


def _pick_market_price(tcgplayer_prices, rarity):
    if not tcgplayer_prices:
        return None
    r = str(rarity or "").lower()
    order = _HOLO_ORDER if ("holo" in r or "1st" in r or "rare" in r) else _PLAIN_ORDER
    for variant in order:
        p = (tcgplayer_prices.get(variant) or {}).get("market")
        if isinstance(p, (int, float)) and p > 0:
            return float(p)
    for entry in tcgplayer_prices.values():
        p = (entry or {}).get("market")
        if isinstance(p, (int, float)) and p > 0:
            return float(p)
    return None


# --------------------------------------------------------------------------- #
# public interface (mirrors tcgcsv_sync)
# --------------------------------------------------------------------------- #
def get_categories():
    """One category: Pokémon. Shape mirrors tcgcsv's category objects."""
    return [{"categoryId": CATEGORY_ID, "displayName": GAME_NAME,
             "name": GAME_NAME, "popularity": 1}]


def get_last_updated():
    """pokemontcg.io has no single catalog timestamp; return None."""
    return None


def _all_sets(refresh=False):
    """Fetch every set (paginated), cache both the id->set-id map and the set
    OBJECTS, and return them sorted oldest-first so vintage sets are easy to
    find.

    The objects are cached, not just the id map: _resolve_set needs a set
    object, so with only the map cached it re-downloaded the whole paginated
    /sets collection on EVERY call — about 170 full downloads per category
    sync against an API that rate-limits readily. Pass refresh=True to force a
    re-pull (used when a group id is unknown, e.g. a set added since startup).
    """
    if _SETS_CACHE and not refresh:
        return _SETS_CACHE
    sets = []
    page = 1
    while True:
        data = _get("/sets", {"page": page, "pageSize": PAGE_SIZE,
                              "orderBy": "releaseDate"})
        batch = data.get("data", []) or []
        sets.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        page += 1
    for s in sets:
        _SET_ID_BY_GROUP[_stable_int(s.get("id"))] = s.get("id")
    _SETS_CACHE[:] = sets
    return sets


def get_groups(category_id):
    """List all sets as groups. groupId is the stable int of the set string id;
    extra keys (_set_id, release_date, total) are ignored by the route but handy.

    Forces a refresh: this is the entry point a sync starts from, so a set
    released since the process started must show up. It costs one /sets pull
    per sync — unlike _resolve_set, which runs once per group and reads the
    cache this call fills."""
    groups = []
    for s in _all_sets(refresh=True):
        sid = s.get("id")
        groups.append({
            "groupId": _stable_int(sid),
            "name": s.get("name") or sid,
            "_set_id": sid,
            "release_date": s.get("releaseDate") or "",
            "printedTotal": s.get("printedTotal") or s.get("total") or 0,
        })
    return groups


def _resolve_set(group_id):
    """(set_id, set_obj) for an integer group_id, refreshing the /sets cache if
    the id isn't known yet."""
    sets = _all_sets()                      # cached after the first pull
    sid = _SET_ID_BY_GROUP.get(int(group_id))
    if sid is None:
        sets = _all_sets(refresh=True)      # unknown id: the cache may be stale
        sid = _SET_ID_BY_GROUP.get(int(group_id))
    if sid is None:
        raise RuntimeError(f"Unknown set for group_id {group_id}")
    set_obj = next((s for s in sets if s.get("id") == sid), {})
    return sid, set_obj


def normalize_card(card, set_obj, group_id):
    """Map a pokemontcg.io card object to the ReferenceCard upsert dict."""
    prices = (card.get("tcgplayer") or {}).get("prices") or {}
    rarity = card.get("rarity") or ""
    printed_total = set_obj.get("printedTotal") or set_obj.get("total") or 0
    number = _canonical_number(card.get("number"), printed_total)
    images = card.get("images") or {}
    types = card.get("types") or []

    extended = {
        "Number": number,
        "Rarity": rarity,
        "Type": ", ".join(types) if types else "",
        "HP": card.get("hp", ""),
        "Stage": ", ".join(card.get("subtypes") or []),
        "Supertype": card.get("supertype", ""),
        "pokemontcg_id": card.get("id", ""),
        "set_id": set_obj.get("id", ""),
        "series": set_obj.get("series", ""),
        "release_date": set_obj.get("releaseDate", ""),
        "printed_total": printed_total,
        "tcgplayer_prices": prices,            # all variant prices, for per-finish use
        "cardmarket": card.get("cardmarket") or {},
        "national_pokedex": card.get("nationalPokedexNumbers") or [],
    }

    return {
        "product_id":   _stable_int(card.get("id")),
        "category_id":  CATEGORY_ID,
        "group_id":     group_id,
        "game":         GAME_NAME,
        "set_name":     set_obj.get("name") or "",
        "name":         card.get("name") or "",
        "clean_name":   _clean_name(card.get("name")),
        "number":       number,
        "rarity":       rarity,
        "image_url":    images.get("large") or images.get("small") or "",
        "url":          (card.get("tcgplayer") or {}).get("url") or "",
        "market_price": _pick_market_price(prices, rarity),
        "extended":     extended,
    }


def fetch_group_cards(category_id, category_name, group_id, group_name):
    """Download every card in one set and return normalized upsert dicts."""
    set_id, set_obj = _resolve_set(group_id)
    if not set_obj:
        set_obj = {"id": set_id, "name": group_name}

    cards = []
    page = 1
    while True:
        data = _get("/cards", {
            "q": f'set.id:{set_id}',
            "page": page,
            "pageSize": PAGE_SIZE,
            "orderBy": "number",
        })
        batch = data.get("data", []) or []
        for c in batch:
            cards.append(normalize_card(c, set_obj, int(group_id)))
        if len(batch) < PAGE_SIZE:
            break
        page += 1
    return cards
