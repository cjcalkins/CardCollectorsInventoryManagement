#!/usr/bin/env python3
"""Verify the RBAC resource map against Director's final ledger.

Usage:  tools/verify_rbac_branch.py [/path/to/app.py]

Extracts _resource_for_path from the target app.py verbatim (AST -> exec, not a
reimplementation), then asserts the resolved resource for every route in the
ledger. Also re-enumerates all route registrations across the repo to confirm no
mutating route is left unmapped.

Ledger source: Director 2026-07-30T11:39:26Z table, corrected 11:42:02Z
(setup not setup_select; reset in the inner settings dict; force-edit = 2 routes).

KNOWN-ANSWER SELF-TEST
----------------------
    aad3368   must FAIL — a tree with the F3 fix and the item-8 guards absent

Materialized with `git archive` from this tool's own checkout, so the control is
a permanent ancestor of main rather than a worktree anyone could prune. It used to
point at .scratch/wt-rbac-review/app.py, and the old code SKIPPED the self-test
when that path was missing — pruning a directory would have silently downgraded a
merge-gating control to a no-op. A control that cannot be obtained now exits 2.

DOES NOT SEE — read this before treating a green as coverage
-----------------------------------------------------------
  * Runtime enforcement. It proves the resource MAP resolves each path to the
    right resource; it does not prove the gate consults the map, that the gate is
    reached, or that a request is actually refused. That needs a live request.
  * Routes registered anywhere but a module-level `@x.route(...)` decorator or
    `add_url_rule` with a literal first argument — a route built from an f-string
    or registered in a loop is invisible to the enumerator.
  * Blueprints that are never registered. `/shipping/*` is enumerated and mapped
    defensively, but the blueprint is unwired today, so those routes 404 rather
    than being gated. A PASS on them is a statement about the map, not the app.
  * Anything outside the ledger. The ledger is a fixed list a human wrote; a new
    route is only checked by the residual-unmapped sweep, which catches "no
    resource at all" and not "the wrong resource".

CONFIDENCE: HIGH on the ledger and residual sweep — both have been run against
the negative control repeatedly and the residual sweep is what found the unmapped
routes in the first place. MEDIUM on the "adjacent findings present in source?"
block: those are substring checks over a source window, and all three have
misreported in BOTH directions before being scoped (see the comment there).
"""
import ast, os, re, sys, glob, subprocess, tarfile, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _args import (EX_CONTROL, EX_FINDING, EX_OK, pinned, repo_root,
                   single_target)

USAGE = """usage: verify_rbac_branch.py [/path/to/app.py]

Verifies the RBAC resource map in the target app.py against Director's ledger,
re-enumerates every mutating route to confirm none is left unmapped, and checks
the template guards and item-8 control guards in the tree beside it.

With no argument it checks the app.py of the checkout this tool lives in, and
prints which file that resolved to — a green always names what it examined.

exit 0  self-test passed and the target verified clean
exit 1  one or more ledger/guard/route failures, or the target could not be read
exit 2  the negative control was unobtainable or came back clean — NOTHING was
        reported
exit 64 usage error
"""

# The control lives in the checkout this file sits in, NOT the cwd and NOT $HOME.
CONTROL_REPO = repo_root()

# (path, expected resource) — expected AFTER the branch lands
EXPECTED = [
    # nine ordinary inventory buttons
    ("/update_scan/5", "inventory"), ("/update_scan_image/5", "inventory"),
    ("/delete_scan/5", "inventory"), ("/delete_scans", "inventory"),
    ("/add_custom_field", "inventory"), ("/grade_condition/5", "inventory"),
    ("/realign_record_image/5", "inventory"), ("/ocr_apply/5", "inventory"),
    ("/wrong_match/5", "inventory"),
    # templates (Chris-approved field-type routes + delete + view side)
    ("/update_field_type", "templates"), ("/update_field_hidden", "templates"),
    ("/template_delete", "templates"), ("/templates", "templates"),
    ("/template_config/x", "templates"),
    # pricing
    ("/save_tcgplayer_link/5", "pricing"),
    ("/collections/price", "pricing"), ("/collections/prices", "pricing"),
    # import
    ("/types", "import"), ("/types/add", "import"), ("/types/delete/5", "import"),
    # admin
    ("/setup", "__admin__"), ("/setup/select", "__admin__"),
    ("/migrate_clean_legacy_fields", "__admin__"),
    ("/settings/reset", "__admin__"), ("/settings/reset/confirm", "__admin__"),
    # shipping folded into shops (dormant blueprint, defensive)
    ("/shipping", "shops"), ("/shipping/label/5", "shops"),
    ("/shipping/orders/create", "shops"), ("/shipping/track/refresh", "shops"),
    # must NOT change: /uploads stays open (card images), gated only on the
    # migration_exports subpath by separate logic, not by the resource map
    ("/uploads/card.png", None), ("/temp_cards/x.png", None),
    # regression guards: previously-correct entries must be untouched
    ("/inventory/archive", "inventory"), ("/shops/test/ebay", "shops"),
    ("/settings/api/save", "api_keys"), ("/settings/upgrade/export", "upgrade"),
    ("/settings/users/create", "__admin__"), ("/tcg_save_url/5", "pricing"),
]

MUT = {"POST", "PUT", "PATCH", "DELETE"}
PUBLIC = ("/auth/", "/static/")


def load_resource_fn(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    ns = {}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == "_resource_for_path":
            exec(ast.get_source_segment(src, n), ns)
    if "_resource_for_path" not in ns:
        sys.exit("could not extract _resource_for_path from " + path)
    return ns["_resource_for_path"], src


def all_routes(repo):
    out = []
    for f in sorted(glob.glob(os.path.join(repo, "*.py"))):
        tree = ast.parse(open(f).read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                            and dec.func.attr == "route" and dec.args):
                        methods = ["GET"]
                        for kw in dec.keywords:
                            if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
                                methods = [e.value for e in kw.value.elts]
                        out.append((os.path.basename(f), dec.args[0].value, methods,
                                    node.name, node.lineno))
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_url_rule" and node.args):
                methods = ["GET"]
                for kw in node.keywords:
                    if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
                        methods = [e.value for e in kw.value.elts]
                out.append((os.path.basename(f), node.args[0].value, methods,
                            "add_url_rule", node.lineno))
    return out


def run_checks(app_path, repo, quiet=False):
    """Run every check against one tree. Returns the list of failure strings."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        fails = _run_checks_inner(app_path, repo)
    return fails


def _run_checks_inner(APP, REPO):
    rfp, src = load_resource_fn(APP)
    fails = []

    print("=== ledger check ===")
    for path, want in EXPECTED:
        got = rfp(path)
        ok = got == want
        if not ok:
            fails.append(f"{path}: expected {want!r}, got {got!r}")
        print(f"  {'PASS' if ok else 'FAIL':4s} {path:36s} -> {got!r}  (want {want!r})")

    print("\n=== residual unmapped mutating routes (app.py = live) ===")
    residual = 0
    for f, rule, methods, fn, ln in all_routes(REPO):
        if not set(m.upper() for m in methods) & MUT:
            continue
        p = re.sub(r"<[^>]+>", "X", rule)
        if p == "/favicon.ico" or any(p.startswith(x) for x in PUBLIC):
            continue
        if rfp(p) is None:
            residual += 1
            tag = "LIVE" if f == "app.py" else "dormant"
            print(f"  {tag:8s} {rule:44s} ({f}:{ln} {fn})")
            if f == "app.py":
                fails.append(f"unmapped live mutating route: {rule}")
    if not residual:
        print("  (none — every mutating route resolves to a resource)")

    # The eBay force-edit fix must live in the GATE, not merely somewhere in the
    # file: the strings "ebay/connect"/"ebay/callback" already appear in the route
    # decorators, so a naive substring search false-passes on unmodified main.
    # Scope the search to a window around the `need = "edit" ...` assignment.
    print("\n=== adjacent findings present in source? ===")
    lines = src.splitlines()
    need_idx = next((i for i, l in enumerate(lines) if re.search(r'need\s*=\s*"edit"', l)), None)
    gate_window = "\n".join(lines[max(0, need_idx - 25):need_idx + 15]) if need_idx else ""
    # These three checks have misreported in BOTH directions. Getting them right:
    #  - The literal paths live in a MODULE-LEVEL constant (app.py:2802), not in the
    #    gate body, so searching the gate window for "ebay/connect" false-FAILS.
    #    Searching the whole file false-PASSES, because those strings also appear in
    #    the route decorators. Correct check = constant referenced in the gate AND
    #    the constant itself defines both paths.
    #  - The comment is written "mutating *methods*" with emphasis asterisks, so
    #    r"mutating (method|action)" never matches. Allow the asterisks.
    const_re = re.search(
        r"_EBAY_OAUTH_WRITE_PATHS\s*=\s*\(([^)]*)\)", src)
    const_body = const_re.group(1) if const_re else ""
    checks = {
        "gate consults _EBAY_OAUTH_WRITE_PATHS":
            "_EBAY_OAUTH_WRITE_PATHS" in gate_window,
        "constant defines /shops/ebay/connect":
            "/shops/ebay/connect" in const_body,
        "constant defines /shops/ebay/callback":
            "/shops/ebay/callback" in const_body,
        "gate comment re mutating methods vs actions":
            bool(re.search(r"mutating\s+\*?(method|action)", gate_window, re.I)),
    }
    for label, ok in checks.items():
        print(f"  {'PASS' if ok else 'CHECK':5s} {label}")
        if not ok:
            fails.append(f"missing: {label}")

    print("\n=== template guards ===")
    guards = [
        ("templates/_settings_sidebar.html", "type_references_page", "import"),
        ("templates/settings.html", "type_references_page", "import"),
        ("templates/import.html", "type_references_page", "import"),
        ("templates/_settings_sidebar.html", "templates_page", "templates"),
        ("templates/_settings_sidebar.html", "reset_page", None),  # admin check
    ]
    for tpl, endpoint, resource in guards:
        full = os.path.join(REPO, tpl)
        body = open(full).read() if os.path.exists(full) else ""
        # find the line with the endpoint, check for a can_view/is_admin guard nearby
        hit = False
        lines = body.splitlines()
        for i, line in enumerate(lines):
            if endpoint in line:
                window = "\n".join(lines[max(0, i - 3):i + 2])
                if resource:
                    hit = f"can_view('{resource}')" in window or f'can_view("{resource}")' in window
                else:
                    hit = "is_admin" in window or "can_edit" in window
                break
        label = f"{tpl} -> {endpoint} guarded on {resource or 'admin'}"
        print(f"  {'PASS' if hit else 'FAIL':4s} {label}")
        if not hit:
            fails.append(f"unguarded: {label}")

    fails.extend(check_item8(REPO))

    print("\n" + ("=" * 60))
    if fails:
        print(f"RESULT: {len(fails)} problem(s)")
        for f in fails:
            print("  - " + f)
    else:
        print("RESULT: all ledger entries, guards and adjacent findings verified")
    return fails


# The negative control is sourced from git, NOT from a .scratch worktree. It used to
# point at .scratch/wt-rbac-review/app.py, which made a merge-gating control depend on
# a directory anyone could prune — and the old code SKIPPED the self-test when the path
# was missing, so pruning it would have silently downgraded a mandatory control to a
# no-op and let this script certify with nothing behind it. aad3368 is an ancestor of
# main, so `git archive` reconstructs it as long as the repo exists.
#
# FULL 40-char SHA. `git archive` resolves through the ref namespace exactly as
# `git show` does, so an abbreviated control would be shadowed by a branch of that
# name and this script would certify against whatever that branch points at. This is
# the one control here that does NOT go through `_args.git_show`, so materialize()
# calls `_args.pinned()` itself -- a rule enforced in only some of the paths that
# reach git is a rule enforced by luck.
NEGATIVE_CONTROL_REF = "aad3368b7a6128a5066a29622531da4c2cde7cf0"   # F3 fix + item-8 guards absent


def materialize(ref):
    """Extract `ref` into a temp dir. Returns (app_py_path, repo_root) or raises.

    NO SHELL AND NO PIPE, deliberately. This used to be
    `git archive REF | tar -x -C DEST` under `shell=True`, and a pipeline's
    returncode is the LAST command's -- so `proc.returncode` reported tar's
    status and never git's. Measured on this repo:

        git archive <absent ref>            rc 128   <- the real answer
        git archive <absent ref> | tar -x   rc 2     <- what the check saw

    It happened to refuse anyway, because tar also fails on an empty stream and
    because the `app.py` check below is a second backstop. That is the shape
    this whole directory exists to reject: the failure was caught by luck, and
    a git failure that emits a valid tar prefix before dying would have come
    back rc 0 with a partially materialized control. `git archive -o` writes
    the file itself, so the returncode inspected is the one that matters, and
    `tarfile` removes the dependency on an external tar.
    """
    pinned(ref)
    # -C the checkout this TOOL lives in, not the tree under test: the target may
    # be a bare app.py copied somewhere with no history at all, and the control
    # has to be obtainable independently of what it is being pointed at.
    dest = tempfile.mkdtemp(prefix=f"nc-{ref}-")
    tar_path = os.path.join(dest, "control.tar")
    proc = subprocess.run(["git", "-C", CONTROL_REPO, "archive",
                           "--format=tar", "-o", tar_path, ref],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git archive {ref} failed (rc {proc.returncode}): "
                           f"{proc.stderr.strip()[:200]}")
    with tarfile.open(tar_path) as tf:
        # `filter="data"` becomes the default in 3.14 and warns when omitted from
        # 3.12; it is backported as far as 3.10.12/3.11.4, so hasattr is what keeps
        # this working on an interpreter that predates it. The point is not safety
        # here -- this archive is git's own output of this repo, which carries no
        # symlinks and no absolute paths, so `data` extracts byte-identically. The
        # point is that the omitted argument prints a DeprecationWarning to stderr
        # on a newer interpreter, and a stray stderr line in this directory is a
        # line someone has to rule out as a finding.
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(dest, **kw)
    os.remove(tar_path)
    app_py = os.path.join(dest, "app.py")
    if not os.path.exists(app_py):
        raise RuntimeError(f"{ref} contains no app.py")
    return app_py, dest


def self_test():
    """Director's standing rule (2026-07-30): a verifier's clean report only counts
    if the SAME verifier has been shown to fail when it should. So before certifying
    anything, run against a tree where the work is known-absent and require failures.
    If the negative control comes back clean, the checker is broken and certifies
    nothing — refuse rather than report a pass.

    A control that cannot be OBTAINED is the same failure as a control that comes back
    clean: either way nothing has demonstrated this checker can fail. Both refuse."""
    try:
        nc_app, nc_repo = materialize(NEGATIVE_CONTROL_REF)
    except Exception as exc:
        print(f"SELF-TEST UNAVAILABLE: could not materialize the negative control "
              f"({NEGATIVE_CONTROL_REF}): {exc}")
        print("Without a control, a clean report here demonstrates nothing about this")
        print("checker's ability to fail. Refusing to certify.")
        sys.exit(EX_CONTROL)

    nc_fails = run_checks(nc_app, nc_repo, quiet=True)
    if not nc_fails:
        print("SELF-TEST FAILED: the negative control (a tree where the fix is")
        print("absent) reported CLEAN. This checker cannot detect what it is")
        print("supposed to detect; its output is worthless. Refusing to certify.")
        sys.exit(EX_CONTROL)
    print(f"self-test OK — negative control {NEGATIVE_CONTROL_REF} reports "
          f"{len(nc_fails)} expected failure(s), so the checker discriminates\n")


def main():
    # Resolve BEFORE the self-test so a mistyped path is a usage-shaped answer
    # rather than a two-minute control run followed by a traceback.
    target = os.path.abspath(
        single_target(sys.argv[1:], USAGE, os.path.join(repo_root(), "app.py")))
    if not os.path.isfile(target):
        print(f"RESULT: target named but unreadable: {target}")
        return EX_FINDING
    repo = os.path.dirname(target)

    self_test()

    print(f"verifying {target}\n")
    fails = run_checks(target, repo)
    return EX_FINDING if fails else EX_OK




# --- item 8: control guards + toast string (Director spec 2026-07-30T19:08) ---
# Every one of these routes resolves to need="edit", so the predicate must be
# can_edit(...), not can_view(...). can_view would leave the control visible to a
# view-only user and the 403 would persist — the defect item 8 exists to remove.
ITEM8_GUARDS = [
    ("templates/inventory.html",        "update_field_type",   "can_edit('templates')"),
    ("templates/inventory_detail.html", "update_field_hidden", "can_edit('templates')"),
    ("templates/inventory_detail.html", "save_tcgplayer_link", "can_edit('pricing')"),
    ("templates/shops.html",            "oauth",               "can_edit('shops')"),
]


def check_item8(repo):
    """Returns list of failure strings; empty when item 8 has landed."""
    out = []
    print("\n=== item 8: control guards ===")
    for tpl, anchor, guard in ITEM8_GUARDS:
        p = os.path.join(repo, tpl)
        body = open(p).read() if os.path.exists(p) else ""
        lines = body.splitlines()
        hit = False
        for i, line in enumerate(lines):
            if anchor in line:
                window = "\n".join(lines[max(0, i - 12):i + 4])
                if guard.replace("'", '"') in window or guard in window:
                    hit = True
                    break
        print(f"  {'PASS' if hit else 'FAIL':4s} {tpl} :: {anchor} guarded by {guard}")
        if not hit:
            out.append(f"unguarded control: {tpl}::{anchor} needs {guard}")

    print("\n=== item 8: toast no longer claims a save that did not happen ===")
    p = os.path.join(repo, "templates/inventory_detail.html")
    body = open(p).read() if os.path.exists(p) else ""
    bad = "Saved, but couldn't update" in body or "Saved, but couldn&#39;t update" in body
    print(f"  {'FAIL' if bad else 'PASS':4s} legacy \"Saved, but couldn't update\" string absent")
    if bad:
        out.append("toast still says 'Saved' when nothing saved (inventory_detail.html)")
    return out

if __name__ == "__main__":
    sys.exit(main())
