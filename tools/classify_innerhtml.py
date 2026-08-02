#!/usr/bin/env python3
r"""Inventory every `innerHTML` occurrence across the template set, bucketed.

(The r-prefix is load-bearing: this docstring contains `'it\'s'` and `[\s+]`,
and without it Python eats the backslash out of the first -- rendering the
example as the very thing it says the stripper mishandles -- and treats the
others as invalid escape sequences, a SyntaxWarning from 3.12 on.)

This is the tool the XSS audit sizes itself with: the per-file DYN column is
where "how much is left" comes from. It is a CENSUS, not a gate — see the exit
table in --help. Exit 0 means an inventory was emitted, never that the
templates are clean.

WHY THE CONTROL IS A PINNED COMMIT AND THE EXPECTATION IS HAND-WRITTEN
---------------------------------------------------------------------
`BASELINE` below is a hand-verified decomposition of one file. It is the known
answer, and it must stay hand-written: an expectation derived by running this
same classifier would make the self-test `classifier(x) == classifier(x)`,
which cannot fail and therefore cannot detect a regression.

What moves is its INPUT. Until this tool entered the repo it compared BASELINE
against whatever `inventory_detail.html` happened to be in the directory you
handed it, which conflated two different events into one message:

    control = working tree     the FILE changed      -> FAILED, accusing the classifier
                               the CLASSIFIER changed -> FAILED, accusing the classifier
    control = pinned commit    the FILE changed      -> control unmoved, PASSES
                               the CLASSIFIER changed -> FAILS, and it is right

The first arrangement fails spuriously the moment the audit edits its own
control file — which is `inventory_detail.html`, the largest remaining target —
and the repair that unsticks it is editing BASELINE to match, turning the
control into a copy of the subject. Pinning the input removes the spurious
failure, so nothing tempts that edit; changing what the numbers *should* be now
means moving CONTROL_REF, which is a visible act in a diff.

BUCKETS (for writes)
  static    after every quoted or backticked chunk is removed from the assigned
            expression, what remains is WHITESPACE AND `+`, AND NOTHING ELSE.
            In practice: one or more string/template literals, joined by `+` or
            by nothing at all -- no identifier, no call, no other operator, not
            even parentheses. Deliberately syntactic: a ternary over two
            literals selected by a boolean is `expr`, because the tool has no
            dataflow and cannot know the selector is a boolean. "Cannot carry
            data" is the intent, not the test.
  interp    a template literal containing ${...}
  ident     a bare identifier / member expression (`el.innerHTML = html`).
            Interpolation may have happened upstream, so these defeat a naive
            grep for '${' inside an innerHTML assignment.
  expr      anything else -- calls, .map(), ternaries, concatenations
`interp` + `ident` + `expr` is the DYN column and the review surface.

The `static` gloss is stated as the residue test rather than as a paraphrase
because two successive paraphrases of it over-promised -- "cannot carry data"
and then "literals and the operators between them", when the code accepts `+`
and no other operator. A sentence that states the mechanism can be checked
against six lines of source; a paraphrase can only be checked by writing
probes, which is how both of those were caught.

WHAT THIS TOOL CANNOT SEE, measured against templates/ at 1c534be
  * Sinks that are not `innerHTML`. `insertAdjacentHTML` appears 3 times and is
    invisible here; `outerHTML =` and `document.write` are 0 today, so the gap
    is small now and will not announce itself if that changes.
  * Computed access -- `el["inner" + "HTML"]` is not matched. 0 occurrences
    today, and the regex is textual, so a rename of that shape is silent.
  * It does not RESOLVE INTERMEDIATES. `ident` says data may have been
    assembled elsewhere; it does not say whether that assembly escaped.
  * `static` is a claim about the expression, not about the page. A literal
    with no `${}` is static however the surrounding code was reached.
  * The chunk stripper is ESCAPE-BLIND -- `'[^']*'` ends at the quote inside
    `'it\'s'` -- so a literal containing an escaped quote leaves residue and
    falls out of `static` into `expr`. It DOES NOT REACH THE CONTROL: there are
    0 backslashes in all 78 write expressions and 0 on all 78 write lines at
    CONTROL_REF, so `static 31` is unaffected. The direction is the safe one --
    the write lands IN the review surface, costing a reader a minute rather
    than hiding a write -- and that is mechanical, not six lucky samples: a
    misparse leaves the backslash itself in the residue, and `\` is not in
    `[\s+]`, so an empty residue requires every backslash to have sat inside a
    chunk the stripper matched, which is the case where it parsed correctly.
  * A ternary over two literals selected by a boolean (`:2221` in the control)
    is `expr`, not `static`. Same direction: over-classified INTO the review
    surface. Getting it right would require knowing the selector is a boolean,
    which is a type checker, and this is a line-oriented syntactic scanner.
  * Continuation accumulation stops after 40 lines, so a single assignment
    spanning more than that is truncated and may be misbucketed.
  * One file at a time, no Jinja evaluation: includes, macros and `{% %}`
    branches are text. A sink inside a branch that never renders is counted.
"""
import json
import os
import re
import sys

import _args   # sys.path[0] is this script's own directory, so this resolves
               # wherever the tool is invoked from and by whatever path.

# FULL 40-char SHA, never an abbreviation -- _args.git_show refuses anything
# else. This is the commit the decomposition below was hand-counted against;
# it is an ancestor of main, so it stays reachable after any branch cleanup.
CONTROL_REF = "651582a58ed534bb5df4c40d911ba6e4083e7416"
CONTROL_FILE = "templates/inventory_detail.html"

# HAND-DERIVED BY READING ALL 78 WRITE STATEMENTS AT CONTROL_REF, BY SOMEONE
# WHO DID NOT WRITE THIS EXTRACTOR. That split is the point: an expectation
# produced by the author of the thing it checks inherits the author's model of
# where a statement ends, which is the exact defect this file's termination
# rule fixes. The blob was verified with `git hash-object` against the
# materialised copy (42db5a91de725adc766d696d87e818059d06e427) before a line
# of it was read.
#
# THIS DICT SUPERSEDES TWO EARLIER PUBLISHED ANSWERS AND BOTH WERE WRONG:
#   44 static / 34 dynamic  -- a first-line-only regex; missed 9 template
#                              literals whose ${...} sits on a continuation
#                              line, plus 2 concatenations.
#   33 static / 45 dynamic  -- accumulated only while UNBALANCED, so a first
#                              line that closes its own literal ended the
#                              statement; see the comment above CONTINUES_RE.
# Six data-bearing writes tree-wide were called `static` under the second, two
# of them in this file (:2599 :2604). Do not read either retired pair as a
# figure to restore.
#
# BOTH COUNTS AND LINE LISTS ARE ASSERTED, AND NEITHER ALONE IS SUFFICIENT ON
# THIS FILE. :2599 and :2604 move static -> interp, and neither bucket has a
# line list, so the lists alone miss them. :2221 moves ident -> expr while
# :1734 moves expr -> interp, so `expr` is 3 before and 3 after with a
# different member, and the counts alone miss that.
BASELINE = {
    "file": "inventory_detail.html",
    "total": 82, "reads": 4, "writes": 78,
    "static": 31, "dynamic": 47,
    "interp": 39, "ident": 5, "expr": 3,
    "ident_lines": [1533, 1578, 2563, 2716, 2861],
    "expr_lines": [2221, 2311, 2317],
}

WRITE_RE = re.compile(r"\.innerHTML\s*\+?=")
OCCUR_RE = re.compile(r"\binnerHTML\b")


def _balanced(s):
    """True if brackets/quotes in s are balanced enough to end a statement."""
    depth = 0
    i = 0
    quote = None
    while i < len(s):
        c = s[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
            elif quote == "`" and c == "$" and i + 1 < len(s) and s[i + 1] == "{":
                depth += 1
                i += 1
        else:
            if c in "\"'`":
                quote = c
            elif c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
        i += 1
    return quote is None and depth <= 0


# A LINE THAT CLOSES ITS OWN LITERAL IS BALANCED, AND STILL NOT FINISHED.
# Accumulating only while unbalanced never starts on
#
#     el.innerHTML = '<div class="alert">'
#                  + escH(data.message || 'Imported.');
#
# because line one balances. `classify()` is then handed a complete string
# literal and buckets it `static` -- "cannot carry data" -- on a write that
# carries a server-supplied `data.message`. Balance is necessary for a
# statement to have ended, not sufficient.
#
# The rule below adds the sufficient part: keep going while there is no
# top-level `;` AND the next non-blank line opens with an operator that
# continues an expression. That set is not a guess about formatting -- it is
# the set for which JS's automatic semicolon insertion does NOT insert one, so
# a line opening with any of them provably belongs to the statement above it.
# `++` is excluded: it is a restricted production and ASI does terminate there.
CONTINUES_RE = re.compile(r"^\s*(\?\?|\|\||&&|\+(?![+=])|\?|:|\.(?!\.))")

MAX_CONTINUATION_LINES = 40


def _next_nonblank(lines, start):
    """Index of the first line at or after `start` with anything on it."""
    for k in range(start, len(lines)):
        if lines[k].strip():
            return k
    return None


def extract_statements(lines):
    """Yield (lineno, assigned_expression) for each innerHTML write."""
    out = []
    for idx, line in enumerate(lines):
        m = WRITE_RE.search(line)
        if not m:
            continue
        expr = line[m.end():]
        j = idx
        while j + 1 < len(lines) and j - idx < MAX_CONTINUATION_LINES:
            if not _balanced(expr):
                j += 1
                expr += "\n" + lines[j]
                continue
            if _statement_end(expr) >= 0:
                break          # a top-level ';' -- the statement really is over
            k = _next_nonblank(lines, j + 1)
            if (k is None or k - idx >= MAX_CONTINUATION_LINES
                    or not CONTINUES_RE.match(lines[k])):
                break
            expr += "\n" + "\n".join(lines[j + 1:k + 1])
            j = k
        out.append((idx + 1, _truncate_at_statement_end(expr)))
    return out


def _statement_end(expr):
    """Index of the first top-level ';' in `expr`, or -1 if there is none.

    One scanner, two callers -- the accumulator asks whether the statement has
    ended and the truncator asks where. They were about to be two copies of the
    same quote/depth walk, which is how the two `PINNED_RE` copies in this
    directory drifted.
    """
    depth = 0
    quote = None
    i = 0
    while i < len(expr):
        c = expr[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
            elif quote == "`" and c == "$" and i + 1 < len(expr) and expr[i + 1] == "{":
                depth += 1
                i += 1
        else:
            if c in "\"'`":
                quote = c
            elif c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == ";" and depth <= 0:
                return i
        i += 1
    return -1


def _truncate_at_statement_end(expr):
    """Cut at the first top-level ';' so trailing statements on the same line
    (`el.innerHTML = 'x'; return; }`) are not mistaken for part of the value."""
    i = _statement_end(expr)
    if i >= 0:
        return expr[:i].strip()
    return expr.strip().rstrip(";").strip()


IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(\.[A-Za-z_$][A-Za-z0-9_$]*)*$")


def classify(expr):
    if "${" in expr:
        return "interp"
    if IDENT_RE.match(expr):
        return "ident"
    # pure literal(s): only quoted/backticked chunks, optionally joined by +,
    # with no bare identifier operand
    stripped = re.sub(r"`[^`]*`|'[^']*'|\"[^\"]*\"", "", expr)
    if re.fullmatch(r"[\s+]*", stripped or ""):
        return "static"
    return "expr"


def analyse_source(src, name):
    lines = src.split("\n")
    total = len(OCCUR_RE.findall(src))
    writes = extract_statements(lines)
    buckets = {"static": [], "interp": [], "ident": [], "expr": []}
    for lineno, expr in writes:
        buckets[classify(expr)].append(lineno)
    return {
        "file": name,
        "total": total,
        "reads": total - len(writes),
        "writes": len(writes),
        "static": len(buckets["static"]),
        "dynamic": len(buckets["interp"]) + len(buckets["ident"]) + len(buckets["expr"]),
        "interp": len(buckets["interp"]),
        "ident": len(buckets["ident"]),
        "expr": len(buckets["expr"]),
        "ident_lines": buckets["ident"],
        "expr_lines": buckets["expr"],
    }


def analyse(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return analyse_source(fh.read(), os.path.basename(path))


def self_test():
    """Prove the classifier still reproduces a known answer before it reports.

    The control comes from a pinned commit, so a failure here can only mean the
    classifier moved -- the bytes it is measured against cannot. That is what
    lets this message name the classifier without hedging.
    """
    src = _args.git_show(CONTROL_REF, CONTROL_FILE)
    if src is None:
        print("SELF-TEST UNAVAILABLE: cannot read %s:%s from this tool's own "
              "checkout %s" % (CONTROL_REF, CONTROL_FILE, _args.repo_root()))
        print("Refusing to emit an inventory -- nothing has demonstrated this "
              "classifier still reproduces a known answer.")
        return False
    # basename of CONTROL_FILE, NOT BASELINE["file"] -- feeding the expectation
    # back in as the input would make that one key a comparison of a value with
    # itself. As written it asserts the two constants still name one file.
    got = analyse_source(src, os.path.basename(CONTROL_FILE))
    diffs = {k: (v, got[k]) for k, v in BASELINE.items() if got[k] != v}
    if diffs:
        print("SELF-TEST FAILED: the classifier no longer reproduces the "
              "hand-verified decomposition of")
        print("%s at %s. The control is a pinned commit, so its bytes did not "
              "move -- this is" % (CONTROL_FILE, CONTROL_REF[:12] + "..."))
        print("a change in the CLASSIFIER, not in the working tree.")
        for k, (exp, act) in sorted(diffs.items()):
            print("  %-10s expected %-6s got %s" % (k, exp, act))
        print("Do not edit BASELINE to match. Either the change is a fix, in "
              "which case re-count by")
        print("hand and move CONTROL_REF deliberately, or it is a regression.")
        return False
    print("self-test OK -- control %s:%s reproduces the hand-verified "
          "%s/%s/%s -- static %s, dynamic %s (interp %s, ident %s, expr %s) -- "
          "and both line lists, ident %s and expr %s, member for member"
          % (CONTROL_REF, CONTROL_FILE, got["total"], got["reads"],
             got["writes"], got["static"], got["dynamic"], got["interp"],
             got["ident"], got["expr"], got["ident_lines"], got["expr_lines"]))
    return True


USAGE = """usage: classify_innerhtml.py [DIR] [--json PATH]

Inventories every innerHTML occurrence in DIR (default: templates/ inside this
tool's own checkout) and prints a per-file table. The DYN column is the review
surface.

THIS TOOL IS A CENSUS, NOT A GATE. Its exit 0 means an inventory was emitted,
not that anything is clean -- read the table. Do not wire a wrapper to gate on
its status; gate on someone having read the DYN column.

--json PATH  also write the per-file line numbers to PATH. Off by default: a
             default write would either depend on the cwd or leave an untracked
             file in whatever clone you ran it in.

exit 0  an inventory was emitted (NOT a clean bill)
exit 1  the named directory could not be read, or holds no *.html
exit 2  the control was unobtainable, or the classifier no longer reproduces it
exit 64 usage error
"""


def main():
    argv = sys.argv[1:]
    json_path = None
    if "--json" in argv:
        i = argv.index("--json")
        if i + 1 >= len(argv):
            print("--json needs a path.\n")
            print(USAGE)
            return _args.EX_USAGE
        json_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    # single_target so a bare run always has something to look at and can never
    # take the checked-nothing-exited-0 path; it prints what it resolved to.
    tdir = _args.single_target(
        argv, USAGE, os.path.join(_args.repo_root(), "templates"))

    if not self_test():
        return _args.EX_CONTROL

    print("\ninventorying %s" % os.path.abspath(tdir))
    try:
        names = sorted(n for n in os.listdir(tdir) if n.endswith(".html"))
    except OSError as exc:
        print("CANNOT READ %s: %s" % (tdir, exc))
        return _args.EX_FINDING
    if not names:
        print("%s holds no *.html -- nothing to inventory." % tdir)
        return _args.EX_FINDING

    rows = [r for r in (analyse(os.path.join(tdir, n)) for n in names) if r["total"]]
    rows.sort(key=lambda r: -r["dynamic"])
    print("%-32s%6s%6s%6s%7s%6s%7s%6s%5s"
          % ("file", "total", "read", "write", "static", "DYN", "interp",
             "ident", "expr"))
    for r in rows:
        print("%-32s%6d%6d%6d%7d%6d%7d%6d%5d"
              % (r["file"], r["total"], r["reads"], r["writes"], r["static"],
                 r["dynamic"], r["interp"], r["ident"], r["expr"]))
    # The total row is deliberately labelled and deliberately wider than the
    # data rows. Sum the per-file rows rather than parsing this line: a field
    # index that is right for the data is off by one here, which has produced
    # two plausible wrong sums.
    t = lambda k: sum(r[k] for r in rows)
    print("%-32s%6d%6d%6d%7d%6d%7d%6d%5d"
          % ("TOTAL (%d files)" % len(rows), t("total"), t("reads"),
             t("writes"), t("static"), t("dynamic"), t("interp"), t("ident"),
             t("expr")))
    print("\n%d of %d *.html files carry an innerHTML occurrence."
          % (len(rows), len(names)))

    if json_path:
        with open(json_path, "w") as fh:
            json.dump(rows, fh, indent=1)
        print("per-file ident/expr line numbers -> %s" % os.path.abspath(json_path))
    print("Exit 0 means INVENTORY EMITTED, not clean.")
    return _args.EX_OK


if __name__ == "__main__":
    sys.exit(main())
