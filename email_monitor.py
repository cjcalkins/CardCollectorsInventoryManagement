"""
email_monitor.py
================

A small IMAP client used to watch a mailbox for marketplace sale-notification
emails (primarily TCGplayer, whose API is closed, so email is the sale signal).

It is deliberately dependency-free (stdlib imaplib/email) and decoupled: it takes
a plain config dict, connects, fetches matching messages, and parses them into a
neutral sale shape. It never touches the database — the Flask layer decides what
to do with the parsed result (match to inventory, mark sold, fan-out delists).

Config dict keys:
    host, port, use_ssl, username, password, folder,
    sender_filter, subject_filter, mark_seen
"""

import os
import re
import imaplib
import ssl
import email
from email.header import decode_header, make_header

DEFAULT_TIMEOUT = 30
FETCH_LIMIT = 50

# Consecutive UID FETCH failures per message, in memory. After FETCH_SKIP_AFTER
# refusals the message is treated as permanently unfetchable and skipped (with
# a note in the result message) so it cannot block the mailbox behind it
# forever. A restart just grants another round of attempts.
_FETCH_FAILS = {}
FETCH_SKIP_AFTER = 3

_CONDITIONS = [
    "near mint", "lightly played", "moderately played", "heavily played",
    "damaged", "mint", "excellent", "good", "played", "poor",
]


# ──────────────────────────────────────────────────────────────────────────────
# IMAP command safety
# ──────────────────────────────────────────────────────────────────────────────
# imaplib does NOT validate its arguments. Read from the installed runtime,
# IMAP4._command builds the wire bytes as
#
#     data = data + b' ' + arg      ...      self.send(data + CRLF)
#
# with no check for CR or LF anywhere, and IMAP4._quote (which login uses for the
# password) escapes only backslash and double-quote. So a CRLF inside ANY of these
# values ends the current command line and starts a new one that the server executes
# as ours -- 'INBOX"\r\nZ1 DELETE "Archive' is two commands, not one folder name.
#
# All five reachable values are covered, not just the two the audit named. Note that
# IMAP4.login passes the USERNAME through unquoted, so it is the least protected of
# the lot despite looking like the most ordinary.
_IMAP_FORBIDDEN = ("\r", "\n", "\x00")


def _imap_reject(**fields):
    """Return an error string naming the first field carrying a control character
    that would break out of its IMAP command, or None when all are safe.

    Reject rather than strip: silently removing characters from a folder name
    selects a DIFFERENT mailbox than the operator asked for, and quietly reading
    the wrong inbox is a worse outcome than a visible error."""
    for name, value in fields.items():
        if any(ch in str(value or "") for ch in _IMAP_FORBIDDEN):
            return (f"The {name} contains a line break or control character, which "
                    f"cannot be sent to an IMAP server. Remove it and save again.")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# TLS
# ──────────────────────────────────────────────────────────────────────────────
# imaplib builds its SSL context with ssl._create_stdlib_context(), which IS
# ssl._create_unverified_context -- check_hostname False, verify_mode CERT_NONE.
# That applies to BOTH IMAP4_SSL (the "Use SSL" toggle) and IMAP4.starttls(), so
# neither path authenticated the server: anything on the network path could present
# a self-signed certificate and be handed the mailbox password. We pass an explicit
# verifying context instead of relying on the stdlib default.
_INSECURE_TLS_ENV = "IMAP_ALLOW_INSECURE_TLS"


def _allow_insecure_tls():
    return str(os.environ.get(_INSECURE_TLS_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def _tls_context():
    """Verifying context, unless the operator has explicitly opted out.

    The opt-out exists for a self-hosted mail server with a self-signed
    certificate, which is a real deployment rather than a hypothetical one. It is
    an environment variable rather than a form field on purpose: turning off
    certificate checking should be a deliberate act by whoever runs the server,
    not a switch an inventory:edit user can find in the UI.
    """
    if _allow_insecure_tls():
        return ssl._create_unverified_context()
    return ssl.create_default_context()


def _tls_error(host, exc):
    """Certificate failures get their own message. The generic connection error
    sends the operator looking at firewalls and ports for a problem that is
    neither."""
    return (f"TLS verification failed for {host} — {exc}. The server's certificate "
            f"is not trusted or does not match the hostname. If this is your own "
            f"mail server with a self-signed certificate, set {_INSECURE_TLS_ENV}=1 "
            f"on the app process to accept it.")


# ──────────────────────────────────────────────────────────────────────────────
# Connection
# ──────────────────────────────────────────────────────────────────────────────
def _connect(cfg):
    """Open + login an IMAP connection. Returns (conn, None) or (None, error_msg)."""
    host = str(cfg.get("host", "")).strip()
    user = str(cfg.get("username", "")).strip()
    pwd = str(cfg.get("password", "") or "")
    if not host or not user or not pwd:
        missing = [k for k, v in (("host", host), ("username", user), ("password", pwd)) if not v]
        return None, f"Missing: {', '.join(missing)}"

    # Checked here rather than only at the save route, because these values also
    # arrive from rows saved before this check existed.
    bad = _imap_reject(username=user, password=pwd)
    if bad:
        return None, bad

    try:
        port = int(cfg.get("port") or (993 if cfg.get("use_ssl", True) else 143))
    except (TypeError, ValueError):
        port = 993

    try:
        if cfg.get("use_ssl", True):
            conn = imaplib.IMAP4_SSL(host, port, timeout=DEFAULT_TIMEOUT,
                                     ssl_context=_tls_context())
        else:
            conn = imaplib.IMAP4(host, port, timeout=DEFAULT_TIMEOUT)
    except ssl.SSLError as exc:
        return None, _tls_error(host, exc)
    except Exception as exc:
        return None, f"Could not connect to {host}:{port} — {exc}"

    if not cfg.get("use_ssl", True):
        # This used to be `except Exception: pass  # continue plaintext`. Every way
        # STARTTLS can fail -- not offered, refused, or a certificate we do not
        # trust -- is indistinguishable from an attacker stripping it, and the
        # next thing on this socket is the mailbox password. Fail closed.
        try:
            conn.starttls(ssl_context=_tls_context())
        except ssl.SSLError as exc:
            _safe_logout(conn)
            return None, _tls_error(host, exc)
        except Exception as exc:
            _safe_logout(conn)
            if _allow_insecure_tls():
                return None, (f"Could not start TLS with {host}:{port} — {exc}. "
                              f"{_INSECURE_TLS_ENV} disables certificate checking, "
                              f"not the requirement for an encrypted connection.")
            return None, (f"{host}:{port} would not upgrade to an encrypted "
                          f"connection ({exc}), so the mailbox password was not "
                          f"sent. Enable SSL and use the TLS port (usually 993).")

    try:
        conn.login(user, pwd)
    except imaplib.IMAP4.error as exc:
        try:
            conn.logout()
        except Exception:
            pass
        return None, f"Login failed — {exc}"
    except Exception as exc:
        return None, f"Login error — {exc}"

    return conn, None


def _safe_logout(conn):
    try:
        conn.logout()
    except Exception:
        pass


def test_imap(cfg):
    """Verify credentials and folder access. Returns {ok, message, count}."""
    conn, err = _connect(cfg)
    if err:
        return {"ok": False, "message": err}
    folder = str(cfg.get("folder", "INBOX") or "INBOX")
    bad = _imap_reject(folder=folder)
    if bad:
        _safe_logout(conn)
        return {"ok": False, "message": bad}
    try:
        typ, data = conn.select(f'"{folder}"', readonly=True)
        if typ != "OK":
            return {"ok": False, "message": f"Folder '{folder}' not found."}
        count = int(data[0]) if data and data[0] else 0
        return {"ok": True, "message": f"Connected — '{folder}' has {count} message(s).", "count": count}
    except Exception as exc:
        return {"ok": False, "message": f"Select failed — {exc}"}
    finally:
        _safe_logout(conn)


def _search_criteria(cfg):
    crit = []
    sender = str(cfg.get("sender_filter", "") or "").strip()
    subject = str(cfg.get("subject_filter", "") or "").strip()
    if sender:
        crit += ["FROM", f'"{sender}"']
    if subject:
        crit += ["SUBJECT", f'"{subject}"']
    if not crit:
        crit = ["ALL"]
    return crit


def mailbox_top_uid(cfg):
    """Highest existing UID in the configured folder, or None on any failure.

    Used to bootstrap the poll cursor when the monitor has never run
    (last_uid == 0): without this, the oldest-first fetch window would walk
    the mailbox's ENTIRE history and process every historical sale
    notification as a new sale. Read-only; no flags are touched."""
    conn, err = _connect(cfg)
    if err:
        return None
    folder = str(cfg.get("folder", "INBOX") or "INBOX")
    if _imap_reject(folder=folder):
        _safe_logout(conn)
        return None
    try:
        typ, _ = conn.select(f'"{folder}"', readonly=True)
        if typ != "OK":
            return None
        typ, data = conn.uid("SEARCH", None, "ALL")
        if typ != "OK":
            return None
        uids = [int(u) for u in (data[0].split() if data and data[0] else [])]
        return max(uids) if uids else 0
    except Exception:
        return None
    finally:
        _safe_logout(conn)


def fetch_sale_emails(cfg, since_uid=0, limit=FETCH_LIMIT):
    """
    Fetch messages matching the sender/subject filters with UID > since_uid.
    Returns {ok, message, emails: [parsed...], max_uid}. Optionally flags them
    \\Seen when cfg['mark_seen'] is true.
    """
    conn, err = _connect(cfg)
    if err:
        return {"ok": False, "message": err, "emails": [], "max_uid": since_uid}

    folder = str(cfg.get("folder", "INBOX") or "INBOX")
    # The search filters are checked here too: they reach conn.uid("SEARCH", ...)
    # below, which is the same unvalidated _command path as select().
    bad = _imap_reject(folder=folder,
                       **{"sender filter": cfg.get("sender_filter", ""),
                          "subject filter": cfg.get("subject_filter", "")})
    if bad:
        _safe_logout(conn)
        return {"ok": False, "message": bad, "emails": [], "max_uid": since_uid}
    mark_seen = bool(cfg.get("mark_seen", True))
    emails, max_uid = [], since_uid
    try:
        typ, _ = conn.select(f'"{folder}"', readonly=not mark_seen)
        if typ != "OK":
            return {"ok": False, "message": f"Folder '{folder}' not found.", "emails": [], "max_uid": since_uid}

        typ, data = conn.uid("SEARCH", None, *_search_criteria(cfg))
        if typ != "OK":
            return {"ok": False, "message": "IMAP search failed.", "emails": [], "max_uid": since_uid}

        uids = [int(u) for u in (data[0].split() if data and data[0] else [])]
        uids = sorted(u for u in uids if u > int(since_uid or 0))
        # Cap work per poll with the OLDEST window: max_uid is a high-water
        # mark the caller persists, so taking the newest slice would advance
        # it past every older unfetched message and orphan them permanently.
        # The remainder is picked up on the next poll.
        uids = uids[:limit]

        skipped = []
        for uid in uids:
            typ, msgdata = conn.uid("FETCH", str(uid), "(RFC822)")
            if typ != "OK" or not msgdata or not msgdata[0]:
                # Stop, don't skip: a later success would push max_uid past
                # this message and it would never be fetched again. Return
                # what we have; this UID is retried on the next poll — unless
                # the server has now refused it FETCH_SKIP_AFTER times, in
                # which case it is permanently unfetchable (oversized/corrupt)
                # and stopping forever would block every email behind it.
                fails = _FETCH_FAILS.get(uid, 0) + 1
                _FETCH_FAILS[uid] = fails
                if fails < FETCH_SKIP_AFTER:
                    break
                _FETCH_FAILS.pop(uid, None)
                skipped.append(uid)
                max_uid = max(max_uid, uid)
                continue
            _FETCH_FAILS.pop(uid, None)
            raw = msgdata[0][1]
            parsed = _parse_message(raw)
            parsed["uid"] = uid
            emails.append(parsed)
            max_uid = max(max_uid, uid)
            if mark_seen:
                try:
                    conn.uid("STORE", str(uid), "+FLAGS", "(\\Seen)")
                except Exception:
                    pass

        note = (f" Skipped {len(skipped)} unfetchable message(s) "
                f"(uid {', '.join(map(str, skipped))}).") if skipped else ""
        return {"ok": True, "message": f"Fetched {len(emails)} message(s).{note}",
                "emails": emails, "max_uid": max_uid}
    except Exception as exc:
        return {"ok": False, "message": f"Fetch error — {exc}", "emails": [], "max_uid": since_uid}
    finally:
        _safe_logout(conn)


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────
def _decode(value):
    try:
        return str(make_header(decode_header(value or "")))
    except Exception:
        return value or ""


def _html_to_text(html):
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</(p|div|tr|li|h[1-6])>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"'))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _extract_body(msg):
    plain, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if part.get("Content-Disposition", "").startswith("attachment"):
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                chunk = payload.decode(charset, errors="replace")
            except Exception:
                continue
            if ctype == "text/plain":
                plain += chunk + "\n"
            elif ctype == "text/html":
                html += chunk + "\n"
    else:
        try:
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace") if payload else msg.get_payload()
        except Exception:
            body = msg.get_payload()
        if msg.get_content_type() == "text/html":
            html = body or ""
        else:
            plain = body or ""
    if not plain and html:
        plain = _html_to_text(html)
    return plain


def _parse_message(raw):
    msg = email.message_from_bytes(raw) if isinstance(raw, (bytes, bytearray)) else email.message_from_string(raw)
    subject = _decode(msg.get("Subject", ""))
    sender = _decode(msg.get("From", ""))
    date = msg.get("Date", "")
    body = _extract_body(msg)
    sale = parse_tcgplayer_sale(subject, body)
    return {
        "subject": subject, "from": sender, "date": date,
        "order_id": sale["order_id"], "items": sale["items"],
        # {} when the email carries no recognisable address block. The shipping
        # layer treats that as "needs address" rather than guessing one.
        "address": parse_shipping_address(body),
        "needs_review": sale["needs_review"], "excerpt": body[:600],
    }


def parse_shipping_address(text):
    """
    Best-effort extraction of a US/CA shipping address from a sale email.

    Returns {} rather than guessing. The shipping layer treats a missing address
    as "needs address" and asks the user, which is a far better failure than
    buying postage to a hallucinated street.

    Recognises the common layout:

        Shipping Address:
        John Smith
        123 Main St
        Apt 4
        New York, NY 10001

    The "City, ST ZIP" line is the anchor — it's the only line with a reliable
    shape — so we find that first and read the name/street lines above it.
    """
    if not text:
        return {}

    lines = [ln.strip() for ln in text.splitlines()]

    # Find where the address block starts.
    start = None
    for i, ln in enumerate(lines):
        if re.search(r"\b(shipping|ship\s*to|delivery|recipient)\s*(address)?\s*:?\s*$", ln, re.I):
            start = i + 1
            break
    if start is None:
        return {}

    # Take a short window; an address never runs long, and reading further
    # risks swallowing the next section of the email.
    window = [ln for ln in lines[start:start + 8]]

    csz_re = re.compile(
        r"^(?P<city>[A-Za-z .'\-]{2,40}),?\s+"
        r"(?P<state>[A-Za-z]{2})\.?\s+"
        r"(?P<zip>\d{5}(?:-\d{4})?|[A-Za-z]\d[A-Za-z]\s*\d[A-Za-z]\d)$"
    )

    anchor, m = None, None
    for i, ln in enumerate(window):
        if not ln:
            continue
        m = csz_re.match(ln)
        if m:
            anchor = i
            break
    if anchor is None or not m:
        return {}

    # Everything above the anchor, ignoring blanks, is name + street lines.
    above = [ln for ln in window[:anchor] if ln]
    if len(above) < 2:
        return {}   # need at least a name and a street

    out = {
        "name": above[0][:200],
        "address1": above[1][:200],
        "city": m.group("city").strip()[:120],
        "state": m.group("state").upper(),
        "zip": m.group("zip").replace(" ", "")[:12],
        "country": "US",
    }
    if len(above) >= 3:
        out["address2"] = above[2][:200]
    # A Canadian postal code means it isn't a US ZIP.
    if not re.match(r"^\d{5}", out["zip"]):
        out["country"] = "CA"
    return out


def parse_tcgplayer_sale(subject, text):
    """
    Best-effort extraction of {order_id, items[]} from a TCGplayer sale email.
    Email layouts vary, so this is conservative: when no line items can be
    confidently parsed, needs_review is True and the caller should keep the raw
    text for manual handling rather than guessing.
    """
    blob = f"{subject}\n{text}"

    order_id = ""
    for pat in (
        r"order\s*(?:number|no\.?|#)\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9\-]{4,})",
        r"order\s+([A-Z0-9]{6,}\-[A-Z0-9]+)",
        r"#\s*([0-9]{6,})",
    ):
        m = re.search(pat, blob, re.I)
        if m:
            order_id = m.group(1).strip()
            break

    items = []
    # Common shapes: "2 x Card Name (Set) - Near Mint", "Qty 1  Card Name", "1× Card Name"
    line_patterns = [
        r"(?:qty[:\s]*)?(\d{1,3})\s*[x×]\s*(.+)",
        r"(?:qty|quantity)[:\s]+(\d{1,3})\s+(.+)",
    ]
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        for pat in line_patterns:
            m = re.match(pat, line, re.I)
            if not m:
                continue
            qty = int(m.group(1))
            name_raw = m.group(2).strip(" -—:\t")
            items.append(_split_item(name_raw, qty))
            break

    # A product/SKU id if the email exposes one.
    if items:
        mid = re.search(r"(?:sku|product\s*id|tcgplayer\s*id)[:#\s]*(\d{4,})", blob, re.I)
        if mid:
            items[0].setdefault("tcgplayer_id", mid.group(1))

    return {"order_id": order_id, "items": items, "needs_review": len(items) == 0}


def _split_item(name_raw, qty):
    foil = "foil" in name_raw.lower()
    condition = ""
    for c in _CONDITIONS:
        if re.search(rf"\b{re.escape(c)}\b", name_raw, re.I):
            condition = c
            break
    set_name = ""
    ms = re.search(r"\(([^)]+)\)", name_raw)
    if ms:
        set_name = ms.group(1).strip()

    name = re.sub(r"\([^)]*\)", "", name_raw)             # drop (set)
    if condition:
        name = re.sub(rf"\b{re.escape(condition)}\b", "", name, flags=re.I)
    name = re.sub(r"(?i)\bfoil\b", "", name)
    name = re.sub(r"\s*[-—|,]\s*$", "", name).strip(" -—|,:\t")
    name = re.sub(r"\s{2,}", " ", name).strip()

    item = {"name": name, "qty": qty, "foil": foil}
    if set_name:
        item["set"] = set_name
    if condition:
        item["condition"] = condition
    return item
