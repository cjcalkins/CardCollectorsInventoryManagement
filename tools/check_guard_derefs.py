#!/usr/bin/env python3
"""Catch the class of defect Director found at inventory.html:828.

A presence/absence check ("is the control hidden for users without the right?")
returns TRUE both when the guard works and when the guard has killed the page —
because absence is exactly what breaks it. This check asks the different question:

    when this guard is FALSE, does any JS still dereference the elements it removed?

Method (static, no browser):
  1. locate each {% if can_edit(...) %} ... {% endif %} block in the template
  2. collect every id="..." declared inside that block
  3. scan the whole file for derefs of those ids that are NOT null-safe
       null-safe  = `?.`  or guarded by `if (x)` / `if (!x) return`
       unsafe     = `x.addEventListener(`, `x.classList`, `x.textContent`, `x.value` ...
  4. report unsafe derefs, and whether they sit at script top level (run at load)

Usage: check_guard_derefs.py <template.html|dir> [...]   (--help for exit codes)
"""
import os, re, sys, tempfile

import _args   # sys.path[0] is this script's own directory, so this resolves
               # wherever the tool is invoked from and by whatever path.

DEREF = re.compile(
    r"\b(?P<var>[A-Za-z_$][\w$]*)\s*\.\s*"
    r"(?P<member>addEventListener|classList|textContent|innerHTML|value|checked|"
    r"disabled|style|dataset|focus|click|setAttribute|removeAttribute)\b")

ASSIGN = re.compile(r"(?:const|let|var)\s+(?P<var>[\w$]+)\s*=\s*document\.getElementById\(\s*['\"](?P<id>[^'\"]+)['\"]\s*\)")


def guarded_blocks(text):
    """Yield (start_line, end_line, guard_expr) for each {% if can_edit/can_view %} block."""
    lines = text.splitlines()
    out, stack = [], []
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r"\{%-?\s*if\s+(?P<expr>.+?)\s*-?%\}", line):
            stack.append((i, m.group("expr")))
        if re.search(r"\{%-?\s*endif\s*-?%\}", line) and stack:
            start, expr = stack.pop()
            if "can_edit" in expr or "can_view" in expr:
                out.append((start, i, expr))
    return out


def script_blocks(lines):
    """(start,end) line ranges of each <script> block — our scope boundary."""
    out, start = [], None
    for i, l in enumerate(lines, 1):
        if "<script" in l and start is None:
            start = i
        elif "</script>" in l and start is not None:
            out.append((start, i)); start = None
    return out


def run(paths, quiet=False, unreadable=None):
    """unreadable, if given, collects paths we were told to check and could not
    open. The caller must treat a non-empty list as a failure: a target that was
    named and then skipped is a hole in the answer, not an absence of findings."""
    problems = 0
    _p = (lambda *a, **k: None) if quiet else print
    for path in paths:
        try:
            text = open(path).read()
        except OSError as exc:
            _p(f"\n{path}: CANNOT READ - {exc}")
            if unreadable is not None:
                unreadable.append(path)
            continue
        lines = text.splitlines()
        blocks = guarded_blocks(text)
        scopes = script_blocks(lines)
        if not blocks:
            _p(f"\n{path}: no can_edit/can_view blocks")
            continue
        _p(f"\n=== {path} ===")

        for start, end, expr in blocks:
            body = "\n".join(lines[start - 1:end])
            ids = re.findall(r'id="([^"]+)"', body)
            if not ids:
                continue
            # An early `return` keyed on ANY guard-removed element protects the whole
            # rest of that scope — once the IIFE returns, nothing downstream runs. Credit
            # it to every variable in the scope, not just the one named in the check.
            # Use ids from EVERY guarded block in the file, not just this one: sibling
            # blocks share the guard condition, so `if (!editBtn) return` (an id from the
            # sibling block) still protects this block's elements when the guard is false.
            all_guarded_ids = set()
            for s2, e2, _ in blocks:
                all_guarded_ids.update(
                    re.findall(r'id="([^"]+)"', "\n".join(lines[s2 - 1:e2])))
            guard_vars = set()
            for i, line in enumerate(lines, 1):
                m = ASSIGN.search(line)
                if m and m.group("id") in all_guarded_ids:
                    guard_vars.add(m.group("var"))
            scope_shield = {}
            for s, e in scopes:
                for j in range(s, e + 1):
                    if re.search(r"\breturn\b", lines[j - 1]) and any(
                        re.search(r"if\s*\([^)]*!\s*" + re.escape(v) + r"\b", lines[j - 1])
                        for v in guard_vars
                    ):
                        scope_shield[(s, e)] = j
                        break

            # A deref inside `X.forEach(...)` where X = querySelectorAll(sel) and EVERY
            # element matching sel lives inside a guarded block is unreachable when the
            # guard is false: empty NodeList -> callback never invoked. Classify those
            # separately instead of calling them unsafe, or the tool cries wolf forever.
            unreachable_ranges = []
            for i, line in enumerate(lines, 1):
                qm = re.search(r"(?:const|let|var)\s+(\w+)\s*=\s*document\.querySelectorAll\("
                               r"\s*['\"]\.([\w-]+)['\"]", line)
                if not qm:
                    continue
                nlvar, cls = qm.group(1), qm.group(2)
                matches = [k for k, l2 in enumerate(lines, 1)
                           if cls in l2 and "<" in l2]
                if not matches or any(not any(s2 <= k <= e2 for s2, e2, _ in blocks)
                                      for k in matches):
                    continue                       # some element lives outside a guard
                for k, l2 in enumerate(lines, 1):
                    if f"{nlvar}.forEach(" not in l2:
                        continue
                    d = 0
                    for j2 in range(k, len(lines) + 1):
                        d += lines[j2 - 1].count("{") - lines[j2 - 1].count("}")
                        if j2 > k and d <= 0:
                            unreachable_ranges.append((k, j2, nlvar, cls))
                            break

            hits, unreachable = [], []
            for eid in ids:
                # every declaration of this id, with its enclosing <script> scope
                for i, line in enumerate(lines, 1):
                    m = ASSIGN.search(line)
                    if not m or m.group("id") != eid:
                        continue
                    var, decl = m.group("var"), i
                    scope = next(((s, e) for s, e in scopes if s <= decl <= e), (1, len(lines)))

                    # early-return null guard anywhere between decl and end of scope?
                    guard_re = re.compile(
                        r"if\s*\(\s*!\s*" + re.escape(var) + r"\b[^)]*\)\s*return|"
                        r"if\s*\([^)]*!\s*" + re.escape(var) + r"\b[^)]*\)\s*return")
                    protected_from = None
                    for j in range(decl, scope[1] + 1):
                        if guard_re.search(lines[j - 1]):
                            protected_from = j
                            break

                    for j in range(scope[0], scope[1] + 1):
                        if start <= j <= end:
                            continue
                        line_j = lines[j - 1]
                        # shadowed by a local re-declaration on this line?
                        if re.search(r"(?:const|let|var)\s+" + re.escape(var) + r"\s*=", line_j):
                            continue
                        if protected_from is not None and j > protected_from:
                            continue
                        shield = scope_shield.get(scope)
                        if shield is not None and j > shield:
                            continue
                        for mm in DEREF.finditer(line_j):
                            if mm.group("var") != var:
                                continue
                            seg = line_j[:mm.start()]
                            if ("?." in line_j[mm.start():mm.start() + len(var) + 3]
                                    or f"if ({var})" in line_j or f"if (!{var})" in line_j
                                    or f"{var} &&" in seg):
                                continue
                            rec = (j, var, mm.group("member"), eid, decl,
                                   line_j.strip()[:88])
                            if any(a <= j <= b for a, b, _, _ in unreachable_ranges):
                                unreachable.append(rec)
                            else:
                                hits.append(rec)
            if hits:
                _p(f"  guard {{% if {expr} %}} lines {start}-{end}")
                for j, var, member, eid, decl, snippet in sorted(hits):
                    problems += 1
                    _p(f"    UNSAFE  :{j}  {var}.{member}  (id={eid}, declared :{decl})")
                    _p(f"            {snippet}")
            else:
                _p(f"  guard {{% if {expr} %}} lines {start}-{end}: clean "
                      f"(no unprotected derefs)")
            if unreachable:
                rng = unreachable_ranges[0]
                _p(f"    note: {len(unreachable)} deref(s) inside "
                      f"{rng[2]}.forEach (.{rng[3]} matches only guard-removed "
                      f"elements) -> empty NodeList, callback never runs; not counted")
    return problems


# FULL 40-char SHA, never an abbreviation -- _args.git_show refuses anything else,
# so this is enforced rather than remembered. Why it has to be: git resolves a name
# through the ref namespace BEFORE treating it as an object, so a branch named
# f3eb259 would answer `git show f3eb259:path` with the BRANCH's bytes, exit 0,
# ambiguity warning on stderr only -- and this tool discards stderr.
SELF_TEST_REF  = "f3eb259e30e54505eed94b9376c0c2ad5c4fd0e3"   # item-8 regression present
SELF_TEST_FILE = "templates/inventory.html"
SELF_TEST_MIN  = 13                              # 13 reachable derefs; the other 5 are
                                                 # forEach bodies, classified not counted


def self_test():
    """Director's standing rule: a clean report only counts if this same checker has
    been shown to fail when it should. f3eb259 is an ancestor of main, so the control
    is reconstructed from git rather than from a .scratch worktree anyone could prune.
    A control that cannot be obtained is treated as a failed control, not a skip —
    silently skipping is how a mandatory check becomes a no-op."""
    # The control comes from the checkout this FILE lives in -- not $HOME (inert on
    # every other machine, and cross-tree on its author's) and not the cwd. See the
    # _args module docstring: both wrong answers shipped here, in opposite directions.
    src = _args.git_show(SELF_TEST_REF, SELF_TEST_FILE)
    if src is None:
        print(f"SELF-TEST UNAVAILABLE: cannot read {SELF_TEST_REF}:{SELF_TEST_FILE} "
              f"from this tool's own checkout {_args.repo_root()}")
        print("Refusing to certify — nothing has demonstrated this checker can fail.")
        sys.exit(_args.EX_CONTROL)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(src)
        ctl = fh.name
    n = run([ctl], quiet=True)
    os.unlink(ctl)
    if n < SELF_TEST_MIN:
        print(f"SELF-TEST FAILED: control {SELF_TEST_REF} reports {n} unsafe deref(s), "
              f"expected >= {SELF_TEST_MIN}.")
        print("The checker has lost the ability to see the defect it exists to find. "
              "Refusing to certify.")
        sys.exit(_args.EX_CONTROL)
    print(f"self-test OK — control {SELF_TEST_REF} reports {n} unsafe deref(s) "
          f"(>= {SELF_TEST_MIN}), so the checker discriminates")


USAGE = """usage: check_guard_derefs.py PATH [PATH ...]

Finds elements removed by a Jinja permission guard that JS still dereferences
without a null check. PATH may be a file or a directory (walked for *.html).

The known-answer self-test runs on every invocation.

CANNOT SEE: JS-conditional sites or class selectors. See tools/README.md.

exit 0  clean, and the self-test passed
exit 1  unsafe deref found
exit 2  control unobtainable -- NOTHING was reported
exit 64 usage error, including paths that expanded to no templates
"""


def main():
    # This tool has no self-test-only mode, so allow_empty stays False: zero
    # targets would otherwise print "no unsafe derefs" having opened nothing --
    # a green over an empty set, the exact shape of a check that cannot fail.
    # _args.targets handles -h/--help, unknown flags, directory walks and the
    # expanded-to-nothing case, all with the shared exit codes.
    paths = _args.targets(sys.argv[1:], USAGE)
    self_test()
    unreadable = []
    problems = run(paths, unreadable=unreadable)
    print()
    if unreadable:
        print(f"RESULT: {len(unreadable)} named target(s) could not be read — "
              f"this run does not cover them")
        sys.exit(1)
    if problems:
        print(f"RESULT: {problems} unsafe deref(s) of guard-removed elements — "
              f"the page throws for users the guard applies to")
        sys.exit(1)
    print("RESULT: no unsafe derefs of guard-removed elements")


if __name__ == "__main__":
    main()
