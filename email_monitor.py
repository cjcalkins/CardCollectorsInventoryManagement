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

import re
import imaplib
import email
from email.header import decode_header, make_header

DEFAULT_TIMEOUT = 30
FETCH_LIMIT = 50

_CONDITIONS = [
    "near mint", "lightly played", "moderately played", "heavily played",
    "damaged", "mint", "excellent", "good", "played", "poor",
]


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

    try:
        port = int(cfg.get("port") or (993 if cfg.get("use_ssl", True) else 143))
    except (TypeError, ValueError):
        port = 993

    try:
        if cfg.get("use_ssl", True):
            conn = imaplib.IMAP4_SSL(host, port, timeout=DEFAULT_TIMEOUT)
        else:
            conn = imaplib.IMAP4(host, port, timeout=DEFAULT_TIMEOUT)
            try:
                conn.starttls()
            except Exception:
                pass  # server may not support STARTTLS; continue plaintext
    except Exception as exc:
        return None, f"Could not connect to {host}:{port} — {exc}"

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
        uids = uids[-limit:]  # cap work; newest window

        for uid in uids:
            typ, msgdata = conn.uid("FETCH", str(uid), "(RFC822)")
            if typ != "OK" or not msgdata or not msgdata[0]:
                continue
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

        return {"ok": True, "message": f"Fetched {len(emails)} message(s).",
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
        "needs_review": sale["needs_review"], "excerpt": body[:600],
    }


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
