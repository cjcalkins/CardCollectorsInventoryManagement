#!/usr/bin/env python3
"""Parse-check a Jinja template: JINJA first, then its inline <script> bodies.

Run me on every template you touch, including comment-only diffs.

WHY BOTH PARSES LIVE IN ONE TOOL
--------------------------------
On 2026-08-02 a comment-only change to templates/import.html broke main. The
comment documented the deref hazard around a permission guard and, in doing so,
contained a literal Jinja statement tag as prose -- an "if can_edit" block
written out with real {%...%} delimiters. (Safe to describe here: this is a .py
file, which Jinja never parses. Inside a template it is not.)

Jinja parses statement tags with no idea that `//` means anything, so the whole
template failed to load and /import 500'd. `node --check` passed -- it is a valid
JS comment. Every content check passed -- the content was correct. Only the Jinja
parser could see it.

The parses were separate tools then, so a human could choose to run one and skip
the other, and skipped the one that mattered on exactly the diff shape where it
felt safest. They are not separable any more.

The general form, worth remembering past this instance: A COMMENT THAT DOCUMENTS
A MECHANISM, WRITTEN INSIDE A FILE THAT MECHANISM PROCESSES, INVOKES IT. The `//`
marks it as prose to the reader and to node, and to nothing else in the chain.

KNOWN-ANSWER SELF-TEST
----------------------
`--self-test` runs this checker against two commits in this repo's own history
and refuses to report on anything unless both answers come out right:

    7238040  templates/import.html   must FAIL   (the commit that broke main)
    eb86bd4  templates/import.html   must PASS   (the fix)

Both are permanent ancestors of main (7238040 via merge a933269), so the controls
cannot rot the way a fixture file would. A checker nobody has watched fail is not
evidence; this one proves it discriminates before it certifies anything.

USAGE
    tools/check_template_parse.py --self-test
    tools/check_template_parse.py templates/*.html
Exit 0 all clear, 1 a file failed, 2 a control failed (nothing was reported).
"""
import os
import re
import subprocess
import sys
import tempfile

SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S | re.I)

# Jinja is not JS. {{ expr }} yields a VALUE so it becomes an identifier token;
# {% stmt %} is control flow yielding nothing, so it is DELETED. Substituting a
# token for statement tags instead once produced `J"J": J,J` inside an object
# literal and reported a parse failure that was in this script, not the template.
EXPR_RE = re.compile(r"\{\{.*?\}\}", re.S)
STMT_RE = re.compile(r"\{%.*?%\}", re.S)

# The specific hazard above gets its own named check rather than relying on the
# Jinja parser to happen to reject it -- some tags are individually well-formed
# and would parse while still meaning something nobody intended.
COMMENT_TAG_RE = re.compile(r"^\s*//.*(\{%|\{\{)")

SELF_TEST_CASES = [("7238040", "templates/import.html", False),
                   ("eb86bd4", "templates/import.html", True)]


def node_check(js):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(js)
        path = fh.name
    try:
        p = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        return p.returncode == 0, (p.stderr or "").strip().split("\n")[0]
    except FileNotFoundError:
        return None, "node not installed"
    finally:
        os.unlink(path)


def jinja_check(src):
    try:
        import jinja2
    except ImportError:
        return None, "jinja2 not installed"
    try:
        jinja2.Environment().parse(src)
        return True, ""
    except Exception as exc:
        return False, "%s: %s" % (type(exc).__name__, exc)


def check_source(src, label, out=True):
    """Return True if src is clean. Prints findings when out is set."""
    ok = True
    jok, jerr = jinja_check(src)
    if jok is None:
        print("REFUSING TO REPORT: %s" % jerr)
        sys.exit(2)
    if out:
        print("\n%s: JINJA %s%s" % (label, "PARSE OK" if jok else "PARSE FAIL",
                                    "" if jok else " - " + jerr))
    ok = ok and jok

    tagged = [(i, l.strip()) for i, l in enumerate(src.split("\n"), 1)
              if COMMENT_TAG_RE.search(l)]
    if tagged:
        ok = False
        if out:
            print("  JINJA TAG INSIDE A // COMMENT (breaks the template):")
            for i, l in tagged:
                print("    :%d  %s" % (i, l[:80]))

    blocks = [b for _, b in SCRIPT_RE.findall(src) if b.strip()]
    if out:
        print("  %d inline block(s) with a body" % len(blocks))
    for i, body in enumerate(blocks, 1):
        js = EXPR_RE.sub("J", STMT_RE.sub("", body))
        nok, nerr = node_check(js)
        if nok is None:
            print("REFUSING TO REPORT: %s" % nerr)
            sys.exit(2)
        if out:
            print("  block %d (%d lines): %s%s"
                  % (i, body.count("\n") + 1, "PARSE OK" if nok else "PARSE FAIL",
                     "" if nok else " - " + nerr))
        ok = ok and nok
    return ok


def git_show(commit, path):
    p = subprocess.run(["git", "show", "%s:%s" % (commit, path)],
                       capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def self_test():
    """Prove the checker discriminates before it is allowed to certify anything."""
    print("SELF-TEST -- known answers from this repo's history")
    for commit, path, want_clean in SELF_TEST_CASES:
        src = git_show(commit, path)
        if src is None:
            print("  UNOBTAINABLE: %s:%s -- cannot prove this checker still" % (commit, path))
            print("  discriminates. A control that cannot be obtained must FAIL, never skip.")
            return False
        got_clean = check_source(src, "%s:%s" % (commit, path), out=False)
        verdict = "OK  " if got_clean == want_clean else "WRONG"
        print("  %s %s:%s  expected %s, got %s"
              % (verdict, commit, path,
                 "PASS" if want_clean else "FAIL",
                 "PASS" if got_clean else "FAIL"))
        if got_clean != want_clean:
            return False
    print("  self-test OK -- fails on the commit that broke main, passes on the fix\n")
    return True


def main():
    args = [a for a in sys.argv[1:] if a != "--self-test"]
    if not self_test():
        print("Refusing to report.")
        return 2
    if not args:
        return 0
    rc = 0
    for path in args:
        with open(path, encoding="utf-8", errors="replace") as fh:
            if not check_source(fh.read(), path):
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
