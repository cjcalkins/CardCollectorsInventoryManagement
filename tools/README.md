# tools/

Static checkers for defect classes that have actually shipped here. They live in
the repo, not in one person's workspace, so that a reviewer, a merger and an
author can all run the same check and get the same answer.

That placement is the point. Every one of these was written after something got
through, and for a while they existed only where their author could run them —
so a green from them was not independently reproducible, which is the same
weakness as a check that cannot fail.

## The contract every checker here follows

**A checker that has never been watched fail is not evidence.** Each one runs a
known-answer self-test *before* it will report on anything, sourcing its control
from a commit in this repo's own history — not from a fixture file that drifts
away from the defect it was cut from.

**A control that cannot be obtained must FAIL, never skip.** If the control
commit is missing or the tool's dependency is absent, the checker exits non-zero
and reports nothing. Silence must never read as clean.

**Name what the checker cannot see.** Each docstring says what is outside its
reach. A clean result is only as wide as the question the tool asks, and reading
it as broader is how several of these defects survived their first review.

## The checkers

### `check_template_parse.py` — Jinja parse + inline JS parse, together

    tools/check_template_parse.py --self-test
    tools/check_template_parse.py templates/*.html

Run on every template you touch, **including comment-only diffs**.

A comment-only change once broke `main`: a comment documenting a permission
guard contained a literal Jinja statement tag as prose, so the template failed to
load and the page 500'd. `node --check` passed — it was a valid JS comment. Every
content check passed — the content was correct. Only the Jinja parser saw it.

The two parses are deliberately in one tool. They were separate then, which let a
human run one and skip the other, on exactly the diff shape where skipping feels
safest. Also flags the specific hazard by name: a Jinja delimiter inside a `//`
comment.

Controls: `7238040:templates/import.html` must FAIL (the commit that broke main),
`eb86bd4:templates/import.html` must PASS (the fix).

Does not see: runtime behaviour. A template that parses can still throw on load —
that needs a browser.

### `check_guard_derefs.py` — permission guards that null an element still dereferenced

    tools/check_guard_derefs.py templates/*.html

A `{% if can_edit(...) %}` around an element removes it for unprivileged users. If
JS below dereferences that element without a null check, the whole script block
throws at load and every handler after it dies — silently, for exactly the users
the guard was meant to protect. Presence/absence testing cannot catch this,
because absence is what breaks it.

Control: `f3eb259` must report its known unsafe derefs.

Does not see: **JS-conditional sites.** It keys on Jinja guard blocks and the ids
inside them, so an element made conditional by a JS ternary, or referenced by
class rather than id, is outside its reach. `import.html`'s `buildTile` has four
such unguarded derefs (`img`, `.manual-adjust-btn`, `.badge`) that this tool will
not report — pointing it there and reading green is coverage that isn't there.

## Adding one

Keep the contract: a self-test with a control from history that must fail, a
non-zero exit when the control is unobtainable, and a docstring saying what the
tool is blind to.
