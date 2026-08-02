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
    tools/check_template_parse.py --help
    tools/check_template_parse.py --self-test
    tools/check_template_parse.py templates/            (directory, walked)
    tools/check_template_parse.py templates/*.html
Exit 0 all clear, 1 a file failed, 2 a control failed (nothing was reported),
64 a usage error -- which includes a path that expanded to no templates, so a
glob that matched nothing can never come back looking clean.
"""
import os
import re
import subprocess
import sys
import tempfile

import _args   # sys.path[0] is this script's own directory, so this resolves
               # wherever the tool is invoked from and by whatever path.

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

# FULL 40-char SHAs, not abbreviations. An abbreviated control goes through git's
# ref-resolution order FIRST, so a branch or tag of the same name silently wins:
# `git show f3eb259:path` with a branch named f3eb259 present returns the BRANCH's
# file, exit 0, with the ambiguity warning on stderr only -- which this tool
# discards. Git ignores a 40-hex ref by design ("it will be ignored when you just
# specify 40-hex"), so only the full form cannot be shadowed. See PINNED_RE.
SELF_TEST_CASES = [("72380402cb22b13cc23a85bdf754b46eb7e8cdbd",
                    "templates/import.html", False),
                   ("eb86bd41c4d630d42f7538863c70ca52b1fc16c6",
                    "templates/import.html", True)]

PINNED_RE = re.compile(r"[0-9a-f]{40}\Z")


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


def self_test():
    """Prove the checker discriminates before it is allowed to certify anything."""
    print("SELF-TEST -- known answers from this repo's history")
    for commit, path, _want in SELF_TEST_CASES:
        if not PINNED_RE.match(commit):
            print("  CONTROL NOT PINNED: %r is not a full 40-char SHA, so a ref of"
                  % commit)
            print("  the same name would shadow it silently. Refusing to certify.")
            return False
    for commit, path, want_clean in SELF_TEST_CASES:
        # Against the checkout this FILE lives in. The bare `git show` this
        # replaces resolved against the CWD, so invoking the tool by absolute
        # path from outside the tree refused on a checkout where the control
        # was perfectly reachable -- and pointed at a sibling checkout of this
        # repo it would have answered from the wrong tree instead.
        src = _args.git_show(commit, path)
        if src is None:
            print("  UNOBTAINABLE: %s:%s from %s -- cannot prove this checker still"
                  % (commit, path, _args.repo_root()))
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


USAGE = """usage: check_template_parse.py [--self-test] [PATH ...]

Parse-checks Jinja templates: the Jinja parse and node --check on the inline
<script> bodies, together, plus a detector for a Jinja delimiter inside a //
comment. PATH may be a file or a directory (walked for *.html).

The known-answer self-test runs on EVERY invocation -- you cannot use this tool
without first proving it still discriminates. --self-test runs it with no targets.

exit 0  clean, and the self-test passed
exit 1  a file failed to parse, or a named target could not be read
exit 2  a control was unobtainable -- NOTHING was reported
exit 64 usage error, including paths that expanded to no templates
"""


def main():
    argv = sys.argv[1:]
    # allow_empty because --self-test legitimately runs with no targets; the
    # guarantee that a run covering nothing never exits 0 moves here rather than
    # disappearing -- see the `if not args` arm below. extra_flags declares
    # --self-test legal so _args does not reject it as an unknown option; this
    # tool still reads the flag off argv itself.
    args = _args.targets(argv, USAGE, allow_empty=True,
                         extra_flags=("--self-test",))
    # No targets is only ever legitimate under an explicit --self-test. Bare
    # `tool.py` -- or a wrapper whose glob matched nothing -- must not exit 0,
    # or checking nothing reads exactly like checking everything and finding it
    # clean. Same reason an unobtainable control fails instead of skipping.
    if not args and "--self-test" not in argv:
        print("no targets. Pass templates, or --self-test to run the controls alone.\n")
        print(USAGE)
        return _args.EX_USAGE

    if not self_test():
        print("Refusing to report.")
        return _args.EX_CONTROL
    if not args:
        return 0
    rc = 0
    for path in args:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
        except OSError as exc:
            # A target we were told to check and could not read is a failure, not
            # a skip -- the caller named it expecting an answer about it.
            print("\n%s: CANNOT READ - %s" % (path, exc))
            rc = 1
            continue
        if not check_source(src, path):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
