#!/usr/bin/env python3
"""Find template-literal interpolations that reach innerHTML WITHOUT an escaper.

WHY THIS EXISTS: check_guard_derefs.py answers a different question — "when a Jinja
guard is false, does JS still dereference the elements it removed?" It says nothing
about escaping. Worse, on a file with no guarded ids it examines nothing and prints
a clean RESULT unconditionally: a vacuous green. shops.html at c087367 has exactly
one can_edit block (218-223) containing ZERO ids, so the deref checker cannot fail
there no matter what the escaping looks like. Do not read its green as "sinks safe."

Method (static):
  1. find sink variables: `X.innerHTML = V` / `X.innerHTML = `...``
  2. follow one hop of assembly into those vars: `V = ...`, `V += ...`, `V.push(...)`
  3. inside the template literals that reach a sink, classify every `${expr}`:
       ESCAPED   expr is wrapped in esc(...) / escHtml(...) / encodeURIComponent(...)
       RAW       anything else — reported
  4. RAW is not automatically a bug: a call-site literal (`${title}` passed in by the
     caller) carries no attacker data. This tool reports; a human decides. It exists
     so the reviewer reads 20 classified lines instead of re-deriving them.

Self-test: at the control commit the known-vulnerable expressions must come back RAW.
A control that cannot be obtained FAILS — it never skips (Director's standing rule,
amended 2026-07-31 after my own controls were found able to vanish silently).

    c087367:templates/shops.html  must report >= 3 RAW, including `it.title`,
    `s.name` and `c.f(x)`. Merged main with all three sinks known-unescaped and
    live — QA detonated :1063 there. A permanent ancestor of main, so the control
    cannot rot the way a fixture file drifts from the defect it was cut from.

DOES NOT SEE — read this before treating a green as coverage
-----------------------------------------------------------
  * Sinks that are not `innerHTML`. `insertAdjacentHTML`, `outerHTML`,
    `document.write` and jQuery `.html()` are all outside its reach.
  * URL contexts. It asks "is this interpolation escaped", and `esc()` counts as
    escaped everywhere — but `esc()` does nothing to a `javascript:` URL reaching
    an `href`/`src`. A file can be 100% ESCAPED here and still have a scheme
    injection. That needs a scheme allowlist check, which this tool is not.
  * Assembly deeper than the fixpoint follows. It chases assignment, `+=`,
    `.push()` and one hop through a named function; data laundered through an
    array index, an object property or a callback parameter is not tracked.
  * Server-rendered Jinja. `{{ x }}` in the HTML body is Jinja's problem, not
    this tool's — it only classifies JS `${...}` interpolations.

CONFIDENCE: LOWEST of the four I wrote. Its control has been exercised far less
than the others' — it has been run in anger against shops.html and import.html and
little else, and its literal scanner is the most intricate code in this directory.
Re-run it against its control and read the classified lines yourself before
treating its output as load-bearing. Stated here rather than discovered later.

Usage: check_html_escaping.py <template.html|dir> [...]
"""
import os, re, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _args import EX_CONTROL, EX_FINDING, EX_OK, git_show, targets

ESCAPERS = ("esc(", "escHtml(", "encodeURIComponent(", "escapeHtml(")

SINK_ASSIGN = re.compile(r"(?P<lhs>[\w$.\[\]']+)\s*\.\s*innerHTML\s*=\s*(?P<rhs>.+)$")


def scan_literals(text):
    """Proper scan of template literals over the WHOLE file.

    A line-based regex cannot do this: the sinks here span multiple lines and nest
    (`<tr>${cols.map(c => `<td>${c.f(x)}</td>`)}</tr>`). Walk the text tracking
    backtick regions and ${...} code regions so each literal is attributed its OWN
    interpolations and inner literals are found as separate literals.

    Returns [(start_off, end_off, [(off, expr), ...])].
    """
    out, i, n = [], 0, len(text)
    stack = []          # open literals: [start_off, [interps]]
    code_depth = []     # brace depth of each open ${ } inside the current literal
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "`":
            if stack and not code_depth:
                s, interps = stack.pop()
                out.append((s, i, interps))
            else:
                stack.append([i, []])
            i += 1
            continue
        if stack and not code_depth and c == "$" and i + 1 < n and text[i + 1] == "{":
            code_depth.append(1)
            j, depth = i + 2, 1
            k = j
            while k < n and depth:
                if text[k] == "\\":
                    k += 2
                    continue
                if text[k] == "`":            # nested literal — handled on its own pass
                    depth_b = 0
                    k += 1
                    while k < n:
                        if text[k] == "\\":
                            k += 2
                            continue
                        if text[k] == "`" and depth_b == 0:
                            break
                        if text[k] == "$" and k + 1 < n and text[k + 1] == "{":
                            depth_b += 1
                            k += 1
                        elif text[k] == "}" and depth_b:
                            depth_b -= 1
                        k += 1
                    k += 1
                    continue
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if not depth:
                        break
                k += 1
            stack[-1][1].append((i, text[j:k]))
            code_depth.pop()
            i = j                              # re-enter to catch nested literals
            continue
        i += 1
    return out


def line_of(text, off):
    return text.count("\n", 0, off) + 1


def is_escaped(expr):
    e = expr.strip()
    return any(e.startswith(x) for x in ESCAPERS)


IDENT = re.compile(r"[A-Za-z_$][\w$]*")
ASSIGN_TO = re.compile(r"(?:(?:const|let|var)\s+)?([A-Za-z_$][\w$]*)\s*(?:\+?=|\.push\s*\()")


def run(paths, quiet=False, unreadable=None):
    _p = (lambda *a, **k: None) if quiet else print
    total_raw = 0
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            # A target we were told to check and could not read is a failure, not
            # a skip — the caller named it expecting an answer about it.
            _p("\n%s: CANNOT READ — %s" % (path, exc))
            if unreadable is not None:
                unreadable.append(path)
            continue
        lits = scan_literals(text)

        # sink variables: X.innerHTML = V
        sinks = set()
        for m in re.finditer(r"\.\s*innerHTML\s*=\s*([A-Za-z_$][\w$]*)\s*;", text):
            sinks.add(m.group(1))
        # X.innerHTML = `literal`  -> that literal reaches directly
        direct = set()
        for m in re.finditer(r"\.\s*innerHTML\s*=\s*`", text):
            for s, e, _ in lits:
                if s == text.index("`", m.start()):
                    direct.add(s)

        # attribute each literal to the identifier it is assigned/pushed/returned into
        owner, funcs = {}, {}
        for s, e, _ in lits:
            # look back to the previous statement boundary, not just the line start:
            # `const rows = d.skipped.map(s =>` can sit on the line ABOVE the literal.
            b = max(text.rfind(";", 0, s), text.rfind("{", 0, s), text.rfind("}", 0, s))
            head = text[b + 1:s]
            m = None
            for m in ASSIGN_TO.finditer(head):
                pass
            if m:
                owner[s] = m.group(1)
            elif "return" in head:
                # literal returned from a function/arrow — find its name
                fm = None
                for fm in re.finditer(r"(?:function\s+([A-Za-z_$][\w$]*)|"
                                      r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:\(|async|function))",
                                      text[:s]):
                    pass
                if fm:
                    owner[s] = fm.group(1) or fm.group(2)
                    funcs[owner[s]] = True

        # fixpoint: a literal reaches innerHTML if its owner is a sink, or if any
        # reaching literal interpolates its owner's name
        reaching = {s for s, e, _ in lits if s in direct or owner.get(s) in sinks}
        changed = True
        while changed:
            changed = False
            names = set()
            for s, e, interps in lits:
                if s in reaching:
                    for _, expr in interps:
                        names.update(IDENT.findall(expr))
            # `html += tbl(...)` — a call assigned/appended to a var that already
            # reaches a sink. The callee's returned literals reach too.
            reach_vars = sinks | {owner[s] for s in reaching if s in owner}
            for ln in text.splitlines():
                for v in reach_vars:
                    if re.search(r"\b" + re.escape(v) + r"\s*\+?=", ln):
                        names.update(re.findall(r"([A-Za-z_$][\w$]*)\s*\(", ln))
            for s, e, _ in lits:
                if s not in reaching and owner.get(s) in names:
                    reaching.add(s)
                    changed = True

        _p(f"\n=== {path} ===")
        _p(f"  innerHTML sink vars: {sorted(sinks) or '(none)'}   "
           f"literals reaching a sink: {len(reaching)}/{len(lits)}")

        raw, esc_ok = [], 0
        for s, e, interps in sorted(lits):
            if s not in reaching:
                continue
            for off, expr in interps:
                if is_escaped(expr):
                    esc_ok += 1
                else:
                    raw.append((line_of(text, off), expr.strip()))
        total_raw += len(raw)
        _p(f"  interpolations reaching innerHTML: {len(raw) + esc_ok}  "
           f"(escaped {esc_ok}, RAW {len(raw)})")
        for ln, expr in sorted(raw):
            _p(f"    RAW  :{ln}  ${{{expr[:70]}}}")
    return total_raw


# --- control -----------------------------------------------------------------
# c087367bb45 is merged main with all three shops.html sinks known-unescaped and live
# (QA detonated :1063 there). If this checker cannot see them RAW, it is broken.
# FULL 40-char SHA: an abbreviation goes through git's ref namespace first, so a
# branch of that name would be read instead, exit 0, with the ambiguity warning on a
# stderr this tool discards. `_args.git_show` refuses anything shorter before reading.
CONTROL_REF = "c087367bb459f7bf5eec3256266ae0c87d24b841"
CONTROL_FILE = "templates/shops.html"
CONTROL_MIN = 3
CONTROL_MUST_SEE = ("it.title", "s.name", "c.f(x)")


def self_test():
    src = git_show(CONTROL_REF, CONTROL_FILE)
    if src is None:
        print(f"SELF-TEST UNAVAILABLE: cannot read {CONTROL_REF}:{CONTROL_FILE} from "
              f"this tool's own checkout.")
        print("A control that cannot be obtained is a FAILED control, not a skip. "
              "Refusing to certify.")
        sys.exit(EX_CONTROL)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(src)
        ctl = fh.name
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = run([ctl], quiet=False)
    os.unlink(ctl)
    seen = buf.getvalue()
    missing = [e for e in CONTROL_MUST_SEE if e not in seen]
    if n < CONTROL_MIN or missing:
        print(f"SELF-TEST FAILED: control {CONTROL_REF}:{CONTROL_FILE} reported {n} RAW "
              f"interpolation(s); missing expected {missing or 'n/a'}.")
        print("This checker can no longer see the sinks it exists to find. "
              "Refusing to certify.")
        sys.exit(EX_CONTROL)
    print(f"self-test OK — control {CONTROL_REF}:{CONTROL_FILE} reports {n} RAW "
          f"interpolations including {', '.join(CONTROL_MUST_SEE)}")


USAGE = """usage: check_html_escaping.py <template.html|dir> [...]

Classifies every JS template-literal interpolation that reaches an innerHTML
sink as ESCAPED or RAW. PATH may be a file or a directory (walked for *.html).

The known-answer self-test runs on EVERY invocation — you cannot use this tool
without first proving it still discriminates.

exit 0  self-test passed and at least one target was classified
exit 1  a named target could not be read
exit 2  the control was unobtainable — NOTHING was reported
exit 64 usage error, including paths that expanded to no templates

THIS TOOL IS A CLASSIFIER, NOT A GATE, and its exit 0 says so. RAW does not mean
"bug" — on current main, shops.html reports 35 RAW and every one is a number, a
boolean, an already-escaped inner literal or an assembled sub-string. Wiring RAW
to exit 1 was the first thing I tried and it is wrong: it makes 1 the permanent
state of every audited file, which trains a reader to ignore it. A red that means
nothing is the same defect as a green that means nothing.

So exit 0 here means "the self-test passed and N lines were classified" — go read
them. It does NOT mean the file is free of XSS. Do not gate a merge on this
tool's status; gate it on someone having read the RAW list.
"""


def main():
    paths = targets(sys.argv[1:], USAGE)
    self_test()
    unreadable = []
    raw = run(paths, unreadable=unreadable)
    print()
    print(f"checked {len(paths) - len(unreadable)}/{len(paths)} target(s)")
    if unreadable:
        print(f"RESULT: {len(unreadable)} target(s) named but unreadable: "
              f"{', '.join(unreadable)}")
        return EX_FINDING
    print(f"RESULT: {raw} raw interpolation(s) reaching innerHTML. RAW is a finding to "
          f"judge, not automatically a bug —\n        call-site literals carry no "
          f"attacker data. Escaped interpolations are counted separately above.")
    print("        Exit 0 here means CLASSIFIED, not CLEAN. Read the RAW lines; "
          "do not gate on this status.")
    return EX_OK


if __name__ == "__main__":
    sys.exit(main())
