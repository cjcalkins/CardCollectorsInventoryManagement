"""
builder.py — pure selection logic for the inventory "Builder" tool.

Operates on plain card dicts (no DB / Flask deps) so it is easy to test:
    {
      "id":       <unique record id>,
      "identity": <hashable card identity, e.g. (name, set, number)>,
      "name":     "Charizard",
      "set":      "Base Set",
      "rarity":   "Rare Holo",
      "holo":     "Regular",     # holographic kind, "" / "None" if not holo
      "type":     "Fire",
    }

Three builders:

  build_pack(cards, spec)
      A single pack. spec:
        size:      total cards in the pack
        rarities:  {rarity: count, ...}   how many of each rarity
        holos:     {holo_kind: count, ...} how many holographics of each kind
        sets:      [allowed set names]     (empty = any set)
      Constrained (rarity/holo) slots are filled first via maximum bipartite
      matching so we never under-fill when inventory allows; the remaining
      "filler" slots take any remaining card from the allowed sets.

  build_set(cards, spec)
      spec:
        size:            total cards
        allow_duplicates: bool (False = one record per identity)
        types:           [allowed types]     (empty = any)
        rarities:        [allowed rarities]   (empty = any)
        sets:            [allowed sets]       (empty = any)

  build_set_of_packs(cards, spec, count)
      `count` packs, each satisfying the pack spec, using each physical record
      at most once and choosing so the number of duplicate card identities
      shared across packs is as small as possible (greedy: each slot prefers a
      record whose identity has been used in the fewest packs so far).
"""

from collections import defaultdict


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _norm(v):
    return str(v or "").strip().lower()


def _in_sets(card, sets_lower):
    if not sets_lower:
        return True
    return _norm(card.get("set")) in sets_lower


def _matches(card, kind, value):
    if kind == "rarity":
        return _norm(card.get("rarity")) == value
    if kind == "holo":
        return _norm(card.get("holo")) == value
    return True  # filler / any


# --------------------------------------------------------------------------- #
# maximum bipartite matching (Kuhn's algorithm), candidate order preserved so
# callers can bias which cards get chosen (used for duplicate minimization)
# --------------------------------------------------------------------------- #
def _match_slots(slot_candidates):
    """slot_candidates: list (per slot) of candidate card-indices, in preference
    order. Returns {slot_index: card_index} for a maximum matching."""
    card_to_slot = {}

    def augment(si, visited):
        for ci in slot_candidates[si]:
            if ci in visited:
                continue
            visited.add(ci)
            if ci not in card_to_slot or augment(card_to_slot[ci], visited):
                card_to_slot[ci] = si
                return True
        return False

    # Fill most-constrained slots (fewest candidates) first for better packing.
    order = sorted(range(len(slot_candidates)), key=lambda si: len(slot_candidates[si]))
    for si in order:
        augment(si, set())

    return {si: ci for ci, si in card_to_slot.items()}


def _constrained_slots(spec):
    """Return list of (kind, value, label) for the rarity + holo slots."""
    slots = []
    for rarity, cnt in (spec.get("rarities") or {}).items():
        for _ in range(int(cnt or 0)):
            slots.append(("rarity", _norm(rarity), str(rarity)))
    for holo, cnt in (spec.get("holos") or {}).items():
        for _ in range(int(cnt or 0)):
            slots.append(("holo", _norm(holo), f"{holo} holo"))
    return slots


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_pack(cards, spec, identity_order_key=None):
    """Build one pack of DISTINCT card identities meeting the rarity/holo counts.
    `identity_order_key(identity)->sortable` biases which identities are chosen
    (lower = preferred); used by build_set_of_packs to minimize cross-pack
    duplicates. Each physical record is used at most once."""
    sets_lower = {_norm(s) for s in (spec.get("sets") or [])}
    pool = [c for c in cards if _in_sets(c, sets_lower)]

    # Group records by identity — a pack holds each identity at most once, but an
    # identity may have several physical copies to draw from.
    by_ident = {}
    for c in pool:
        by_ident.setdefault(c["identity"], []).append(c)
    identities = list(by_ident.keys())
    if identity_order_key is not None:
        identities.sort(key=identity_order_key)
    ident_pos = {ident: i for i, ident in enumerate(identities)}

    con_slots = _constrained_slots(spec)
    size = int(spec.get("size") or len(con_slots))
    over_specified = len(con_slots) > size
    if over_specified:
        con_slots = con_slots[:size]

    # Candidate identities per constrained slot (identity qualifies if any of its
    # records matches the slot's rarity/holo). Order preserved for bias.
    slot_cands = []
    for kind, value, _label in con_slots:
        slot_cands.append([ident_pos[ident] for ident in identities
                           if any(_matches(r, kind, value) for r in by_ident[ident])])

    matched = _match_slots(slot_cands) if slot_cands else {}
    used_idents = set()
    selected = []
    shortfalls = []
    for si, (kind, value, label) in enumerate(con_slots):
        if si in matched:
            ii = matched[si]
            ident = identities[ii]
            rec = next(r for r in by_ident[ident] if _matches(r, kind, value))
            selected.append(rec)
            used_idents.add(ii)
        else:
            shortfalls.append(label)

    # filler: any remaining DISTINCT identities from the allowed sets
    filler_needed = max(0, size - len(con_slots))
    if filler_needed:
        got = 0
        for ii, ident in enumerate(identities):
            if got >= filler_needed:
                break
            if ii in used_idents:
                continue
            selected.append(by_ident[ident][0])
            used_idents.add(ii)
            got += 1
        if got < filler_needed:
            shortfalls.extend(["any"] * (filler_needed - got))

    return {
        "selected": selected,
        "filled": len(selected),
        "size": size,
        "shortfalls": shortfalls,
        "complete": len(selected) >= size and not over_specified,
        "over_specified": over_specified,
    }


def build_set(cards, spec):
    sets_lower = {_norm(s) for s in (spec.get("sets") or [])}
    types = {_norm(t) for t in (spec.get("types") or [])}
    rarities = {_norm(r) for r in (spec.get("rarities") or [])}
    size = int(spec.get("size") or 0)
    allow_dup = bool(spec.get("allow_duplicates", False))

    pool = []
    for c in cards:
        if not _in_sets(c, sets_lower):
            continue
        if types and _norm(c.get("type")) not in types:
            continue
        if rarities and _norm(c.get("rarity")) not in rarities:
            continue
        pool.append(c)

    if not allow_dup:
        seen = set()
        dedup = []
        for c in pool:
            ident = c.get("identity")
            if ident in seen:
                continue
            seen.add(ident)
            dedup.append(c)
        pool = dedup

    selected = pool[:size]
    return {
        "selected": selected,
        "filled": len(selected),
        "size": size,
        "shortfall": max(0, size - len(selected)),
        "complete": len(selected) >= size,
        "allow_duplicates": allow_dup,
    }


def build_set_of_packs(cards, spec, count):
    count = int(count or 0)
    used_ids = set()
    identity_packs = defaultdict(set)   # identity -> set of pack indices
    packs = []

    for p in range(count):
        available = [c for c in cards if c["id"] not in used_ids]
        # Prefer identities in the fewest packs so far, so the same card doesn't
        # repeat across packs unless inventory forces it.
        order_key = (lambda ident: (len(identity_packs[ident]), str(ident)))
        result = build_pack(available, spec, identity_order_key=order_key)
        for c in result["selected"]:
            used_ids.add(c["id"])
            identity_packs[c["identity"]].add(p)
        packs.append(result)

    duplicate_identities = sum(1 for pks in identity_packs.values() if len(pks) > 1)
    duplicate_slots = sum(len(pks) - 1 for pks in identity_packs.values() if len(pks) > 1)

    return {
        "packs": packs,
        "count": count,
        "duplicate_identities": duplicate_identities,
        "duplicate_slots": duplicate_slots,
        "all_complete": all(pk["complete"] for pk in packs) if packs else False,
    }
