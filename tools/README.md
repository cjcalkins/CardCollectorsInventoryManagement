# tools/

Static checkers for defect classes that have actually shipped here. They live in
the repo, not in one person's workspace, so that a reviewer, a merger and an
author can all run the same check and get the same answer.

> **That sentence was not true of the code until this commit, and the shallow
> clone below is the standing test that keeps it true.** The files moved into the
> repo; their *control lookup* did not. `check_guard_derefs.py` read its control
> out of a hardcoded `~/.buzz/REPOS/...` and `check_template_parse.py` ran a bare
> `git show` against the **cwd**. One depth-1 clone, which has none of the
> controls, answers it better than prose can:
>
>     BEFORE   check_guard_derefs.py   EXIT=0  "self-test OK — control f3eb259
>                                               reports 13 unsafe deref(s)"
>              check_template_parse.py EXIT=2  "UNOBTAINABLE: 7238040:..."
>
> Same clone, same command shape, same missing controls — **one tool refused
> honestly and the other certified itself against a commit it does not have.** The
> only difference was how each answered "which checkout?". Both now ask
> `_args.repo_root()`, which answers from the script's own location:
>
>     AFTER    both tools               EXIT=2  "...from this tool's own checkout
>                                                /tmp/ccim-shallow2"
>
> **Run the AFTER pair in a depth-1 clone before trusting any change to control
> resolution.** It has a second half that is easy to drop: back in a full
> checkout both tools must still return 0 and report. A tool that always refuses
> passes the shallow clone for the wrong reason, and that green is
> indistinguishable from a real one.

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

**A control must be a FULL 40-character SHA, never a symbolic ref and never an
abbreviation.** A branch or tag resolves to a different commit in a different
checkout, so the control stops being the control. An *abbreviated* SHA has the
same defect and it is not obvious: git resolves a name through the ref namespace
**before** treating it as an object, so a ref of the same name silently wins.

    repo with a branch named f3eb259, control abbreviated to "f3eb259":
      git show f3eb259:templates/inventory.html   -> the BRANCH's file, exit 0
      stderr: "warning: refname 'f3eb259' is ambiguous."   <- and the tools
                                                             discard stderr
    the full 40-char form, same repo, same branch present:
      git show f3eb259e30e5...:templates/inventory.html    -> the control

Git ignores a 40-hex ref by design — *"it will be ignored when you just specify
40-hex"* — so only the full form is outside the ref namespace and cannot be
shadowed.

**Enforced in `_args.git_show`, which is the one place every control passes
through**, not in each checker. A tool cannot add an unpinned control without
going through the function that refuses them, and a tool written later inherits
the rule without anyone remembering it. It first shipped as a copy in each of
the two checkers that already obeyed it and absent from the three that did not —
the same divergence `_args.py` exists to end, repeated in miniature.

Refusal is an exit, not a `None` return, because the two are different
diagnoses. `None` means *this checkout does not have the control* — a fact about
the tree. An unpinned ref is a defect in the tool, and no caller should be able
to soften it into a missing-control message.

Worth knowing why this is enforced rather than noted: when it was demonstrated,
the shadowed lookup happened to return a *fixed* template and the guard
checker's `>= 13` threshold caught it. That was the direction the substitution
went, not a property of the check — a shadow pointing at any tree that satisfies
the threshold passes silently on the wrong bytes. A rule enforced by luck reads
exactly like one that is enforced.

**Name what the checker cannot see.** Each docstring says what is outside its
reach. A clean result is only as wide as the question the tool asks, and reading
it as broader is how several of these defects survived their first review.

**Nothing checked must never exit 0.** No targets, a glob that matched nothing, a
directory with no templates, a named file that could not be read — every one of
those exits non-zero. Exit 0 means the tool proved it discriminates *and then*
looked at something. A run over an empty set is indistinguishable from a clean
run by status alone, which is the same weakness as a check that cannot fail.

### Exit codes

| code | meaning |
|------|---------|
| 0 | clean — the self-test passed **and** at least one target was read |
| 1 | a real finding, or a named target that could not be read |
| 2 | a control was unobtainable — **nothing was reported**, this is not a pass |
| 64 | usage error: bad flag, no targets, or paths that expanded to nothing |

`2` is reserved for "I could not prove I discriminate" so a wrapper keying on
status can tell it from a mistyped argument. Both are non-zero; only 0 is a pass.

Measure that status directly, and read `$?` before anything else runs.

    tool.py … | tail -4; echo $?              <- tail's status, not the tool's
    printf '%s %s\n' "$(basename $t)" "$?"    <- basename's status; the command
                                                 substitution runs first and
                                                 clobbers $? before it expands
    rc=$(tool.py …); echo $?                  <- fine, but rc holds stdout

Both of the first two produced a wrong number against these very tools — the
first nearly a false finding on the day they landed, the second a battery of
thirteen cases that all read `EXIT=0`, including the four that had just been
made to exit 64. **Assign `rc=$?` on the very next line and print `$rc`.** A
harness that misreads the status turns every code in the table above into 0.

## The checkers

### `check_template_parse.py` — Jinja parse + inline JS parse, together

    tools/check_template_parse.py --help
    tools/check_template_parse.py --self-test
    tools/check_template_parse.py templates/          # directory, walked for *.html
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

Controls (full SHAs, per the clause above): `72380402cb22…:templates/import.html`
must FAIL (the commit that broke main), `eb86bd41c4d6…:templates/import.html`
must PASS (the fix).

Does not see: runtime behaviour. A template that parses can still throw on load —
that needs a browser.

Does not see, in the *named detector*: a `//` that is not at the start of a line.
`a = 1;  // {% if x %}` is missed by the by-name check. The Jinja parse still
fails the file, which is the division worth remembering — **the parse is the
completeness claim, the detector is only the diagnostic that names the cause.**
Any blindness listed here is a gap in the explanation, not in the coverage.

### `check_guard_derefs.py` — permission guards that null an element still dereferenced

    tools/check_guard_derefs.py templates/            # or individual files
    tools/check_guard_derefs.py templates/*.html

A `{% if can_edit(...) %}` around an element removes it for unprivileged users. If
JS below dereferences that element without a null check, the whole script block
throws at load and every handler after it dies — silently, for exactly the users
the guard was meant to protect. Presence/absence testing cannot catch this,
because absence is what breaks it.

Control: `f3eb259e30e5…:templates/inventory.html` must report >= 13 unsafe derefs.

Does not see: **JS-conditional sites.** It keys on Jinja guard blocks and the ids
inside them, so an element made conditional by a JS ternary, or referenced by
class rather than id, is outside its reach. `import.html`'s `buildTile` has four
such unguarded derefs (`img`, `.manual-adjust-btn`, `.badge`) that this tool will
not report — pointing it there and reading green is coverage that isn't there.

## Adding one

Keep the contract: a self-test with a control from history that must fail, a
non-zero exit when the control is unobtainable, a non-zero exit when the run
covered nothing, and a docstring saying what the tool is blind to.

Take argv, the exit codes and the repo from `_args`. `import _args` works from
anywhere because `sys.path[0]` is the script's own directory, and
`_args.git_show()` resolves against `_args.repo_root()` — the checkout this
file lives in. Do not reach for `$HOME` or the cwd; both shipped here and both
were wrong, in opposite directions. Declare any tool-specific flag with
`extra_flags=("--your-flag",)` rather than trimming argv at the call site.

Fetch your control **only** through `_args.git_show`. It is what enforces the
40-hex rule, so a control read by a private `subprocess.run(["git", ...])` is a
control nobody checked.

And give it a `--help`. These two shipped without one and answered a traceback —
a poor first reply from a tool whose whole purpose is to be trusted by people who
did not write it.
