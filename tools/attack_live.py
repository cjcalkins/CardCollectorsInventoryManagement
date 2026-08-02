#!/usr/bin/env python3
"""Attack the migration-bundle gate in whatever app.py you point it at.

Usage: tools/attack_live.py [/path/to/app.py]

WHY THIS EXISTS SEPARATELY FROM a pasted-body test: a test file with the
aad3368 handler pasted into it tests the old code while reporting on the new
one — the exact "evidence that never examined the thing it claimed to" failure
this repo keeps producing. This version AST-extracts `uploaded_file` from the
target file every run, so the code under test is always the code at that commit.

CLIENT CHOICE IS LOAD-BEARING — uses the Werkzeug test client, which passes the
raw path through. `requests`/`curl` normalize '..' client-side and would report
every traversal payload blocked. Verified: the handler receives the traversal
verbatim.

KNOWN-ANSWER SELF-TEST
----------------------
    aad3368:app.py   must report BYPASS (7 payloads leak the bundle)

That commit is the pre-fix tree — a permanent ancestor of main, sourced with
`git show` from THIS FILE's own checkout, so the control cannot be orphaned by
branch cleanup the way a workspace path or a fixture file can. The self-test
runs on every invocation and cannot be skipped: a control that comes back clean
means the attack lost its teeth, and that exits 2 with nothing reported rather
than printing a green.

DOES NOT SEE — read this before treating a green as coverage
-----------------------------------------------------------
  * Anything but the payload list below. It is a fixed 12 probes plus one
    control image; a traversal encoding nobody thought of is not covered. Two of
    the twelve exist only because an earlier probe normalized them away
    client-side and reported them blocked.
  * The real app's request pipeline. The handler is re-registered on a bare
    Flask app with stubbed auth (`inventory:edit`, no `upgrade`), so before_request
    hooks, session handling, the real login gate and any WSGI middleware are all
    absent. It answers "does this handler body leak", not "does the deployed app".
  * Any route but `uploaded_file`. The other send_from_directory routes
    (`/temp_split`, `/temp_cards`, `/temp_pdf`) are untouched by this tool.
  * A leak that needs real bundle contents. The bundle here is a two-line fake;
    detection is a substring match on it.

CONFIDENCE: HIGH on the bypass direction. Its control fires loudly — 7 of 12
payloads leak at aad3368 and 0 at main — so a run that reports clean has
demonstrably been able to report dirty on the same code path minutes earlier.
Lower confidence that a clean run means the gate is airtight: see the payload
list caveat above.
"""
import ast
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _args import (EX_CONTROL, EX_FINDING, EX_OK, git_show, repo_root,
                   require, single_target)

import posixpath


def _load_flask():
    """Bind flask into module globals, or refuse (exit 2) if it is absent.

    Deferred rather than imported at module scope so that -h/--help and a
    mistyped flag are answered by the argument handler. At module scope an
    absent flask turned `--help` into "REFUSING TO REPORT: needs flask" with
    exit 2 -- the code reserved for "I could not prove I discriminate" given
    as the answer to a question that needed no proof of anything.

    These have to be module globals, not locals: attack() runs the extracted
    handler with `ns = dict(globals())`, so the handler resolves Response, g
    and send_from_directory out of this module's namespace.
    """
    require("flask", "attack_live.py")
    global Flask, Response, g, send_from_directory
    from flask import Flask, Response, g, send_from_directory

# FULL 40-char SHA. An abbreviation is resolved through git's ref namespace before
# it is treated as an object, so a branch named `aad3368` would silently win and the
# ambiguity warning would go to a stderr this tool discards. `_args.git_show` refuses
# an unpinned ref before it reads anything -- this constant is written out in full so
# that refusal never fires, not so it can be skipped.
CONTROL_REF = "aad3368b7a6128a5066a29622531da4c2cde7cf0"

BUNDLE = b"FLASK_SECRET_KEY=super-secret-signing-key\nMAILBOX_PASSWORD=hunter2\n"
BUNDLE_NAME = "ccim_migration_20260730_120000.tar.gz"

PAYLOADS = [
    ("direct (gate must fire)",        f"/uploads/migration_exports/{BUNDLE_NAME}"),
    ("raw ..",                         f"/uploads/x/../migration_exports/{BUNDLE_NAME}"),
    ("raw .. from real subdir",        f"/uploads/cards/../migration_exports/{BUNDLE_NAME}"),
    ("url-encoded ..%2f",              f"/uploads/x/..%2fmigration_exports/{BUNDLE_NAME}"),
    ("fully encoded %2e%2e%2f",        f"/uploads/x/%2e%2e%2fmigration_exports/{BUNDLE_NAME}"),
    ("double-encoded %252e%252e%252f", f"/uploads/x/%252e%252e%252fmigration_exports/{BUNDLE_NAME}"),
    ("backslash variant",              f"/uploads/x/..\\migration_exports/{BUNDLE_NAME}"),
    ("leading slash in param",         f"/uploads//migration_exports/{BUNDLE_NAME}"),
    ("mixed case dir",                 f"/uploads/Migration_Exports/{BUNDLE_NAME}"),
    ("dot-slash prefix",               f"/uploads/./migration_exports/{BUNDLE_NAME}"),
    # QA found these two live on aad3368; their first probe normalized them away
    ("double traversal a/b/../..",     f"/uploads/a/b/../../migration_exports/{BUNDLE_NAME}"),
    ("mixed traversal x/.././",        f"/uploads/x/.././migration_exports/{BUNDLE_NAME}"),
    ("control: ordinary card image",   "/uploads/cards/card.png"),
]


# ---- the stubs the extracted handler closes over -----------------------------
def _auth_disabled():
    return False


def _current_user():
    class R:            # signed in, inventory:edit, NO upgrade permission
        is_admin = False
        permissions = {"inventory": "edit"}

    class U:
        role = R()
    return U()


def _role_allows(role, resource, need):
    if role is None:
        return False
    if getattr(role, "is_admin", False):
        return True
    rank = {"none": 0, "view": 1, "edit": 2}
    have = (getattr(role, "permissions", {}) or {}).get(resource, "none")
    return rank.get(have, 0) >= rank.get(need, 1)


def _forbidden(msg):
    return Response(msg, status=403)


def _no_sniff(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


def _prune_migration_bundles(*a, **k):
    return None


def _upload_root():
    """A throwaway UPLOAD_FOLDER holding one bundle and one ordinary card."""
    root = tempfile.mkdtemp(prefix="attack_live_")
    os.makedirs(os.path.join(root, "migration_exports"))
    os.makedirs(os.path.join(root, "cards"))
    with open(os.path.join(root, "migration_exports", BUNDLE_NAME), "wb") as fh:
        fh.write(BUNDLE)
    with open(os.path.join(root, "cards", "card.png"), "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + b"fake-image")
    return root


def extract_handler(src):
    """AST-extract uploaded_file's source from `src`, or None."""
    fn_src = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "uploaded_file":
            fn_src = ast.get_source_segment(src, node)
    if fn_src is None:
        return None
    # strip the @app.route decorator lines; we re-register it ourselves
    return "\n".join(l for l in fn_src.splitlines()
                     if not l.strip().startswith("@"))


def attack(src, verbose):
    """Run every payload against uploaded_file as it exists in `src`.

    Returns (leaked_labels, card_image_status), or None if the handler is not
    in this source at all.
    """
    fn_src = extract_handler(src)
    if fn_src is None:
        return None

    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = _upload_root()

    ns = dict(globals())
    ns["app"] = app                 # the handler reads app.config — bind OUR app,
    exec(fn_src, ns)                # not whatever a previous run left behind
    app.route("/uploads/<path:filename>")(ns["uploaded_file"])

    if verbose:
        print(f"\n{'payload':34s} {'status':6s} leaked?")
    leaks = []
    with app.test_client() as c:
        for label, url in PAYLOADS:
            r = c.get(url)
            leaked = BUNDLE in r.data
            if leaked:
                leaks.append(label)
            if verbose:
                print(f"  {label:32s} {str(r.status_code):6s} "
                      f"{'*** LEAKED ***' if leaked else 'no'}")
        img = c.get("/uploads/cards/card.png")
    return leaks, img.status_code


def self_test():
    """The attack must find the bypass at the pre-fix commit.

    Unobtainable control, or a control that comes back clean, exits 2 having
    reported nothing. A clean control means the attack no longer discriminates,
    which is indistinguishable from a fixed target by output alone.
    """
    src = git_show(CONTROL_REF, "app.py")
    if src is None:
        print(f"SELF-TEST UNAVAILABLE: cannot read {CONTROL_REF}:app.py from "
              f"{repo_root()}.")
        print("Refusing to report: nothing has demonstrated this attack can "
              "still find a bypass.")
        sys.exit(EX_CONTROL)
    got = attack(src, verbose=False)
    if got is None:
        print(f"SELF-TEST UNAVAILABLE: no uploaded_file() at {CONTROL_REF}.")
        sys.exit(EX_CONTROL)
    leaks, _img = got
    if not leaks:
        print(f"SELF-TEST FAILED: {CONTROL_REF} is the PRE-FIX tree and no "
              f"payload leaked.")
        print("The attack has lost its teeth. Refusing to report on anything "
              "else — a clean run would prove nothing.")
        sys.exit(EX_CONTROL)
    print(f"self-test OK -- {CONTROL_REF}:app.py leaks via {len(leaks)} "
          f"payload(s), so the attack still discriminates")


USAGE = """usage: attack_live.py [/path/to/app.py]

Re-registers uploaded_file() from the target app.py on a bare Flask app and
fires 12 traversal/encoding payloads plus one control image at it, checking
whether a migration bundle's bytes come back.

With no argument it attacks the app.py of the checkout this tool lives in, and
prints which file that resolved to — a green always names what it examined.

The known-answer self-test runs on EVERY invocation: the attack must still find
the bypass at aad3368 before it will report on anything.

exit 0  self-test passed and no payload leaked the bundle
exit 1  a bypass, a broken gate, or a target that could not be read/parsed
exit 2  the control was unobtainable or came back clean — NOTHING was reported
exit 64 usage error
"""


def main():
    # argv first, then the dependency, then the control. Each stage answers a
    # question the next one cannot: a bad flag is not a missing flask, and a
    # missing flask is not a failed self-test.
    target = single_target(sys.argv[1:], USAGE,
                           os.path.join(repo_root(), "app.py"))
    _load_flask()
    self_test()

    try:
        with open(target, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
    except OSError as exc:
        print(f"RESULT: target named but unreadable: {exc}")
        return EX_FINDING
    print(f"handler extracted live from {target}")

    got = attack(src, verbose=True)
    if got is None:
        print(f"RESULT: no uploaded_file() in {target} — nothing was attacked")
        return EX_FINDING
    leaks, img_status = got

    print()
    print(f"card image serves: {img_status} "
          f"({'OK' if img_status == 200 else 'BROKEN - gate too wide'})")
    print()
    if leaks:
        print(f"RESULT: BYPASS via {len(leaks)} payload(s): {leaks}")
        return EX_FINDING
    if img_status != 200:
        print("RESULT: no leak, but the gate broke ordinary card images")
        return EX_FINDING
    print(f"RESULT: none of the {len(PAYLOADS) - 1} payloads returned bundle "
          f"content; card images unaffected")
    return EX_OK


if __name__ == "__main__":
    sys.exit(main())
