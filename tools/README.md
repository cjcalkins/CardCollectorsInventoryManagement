# tools/

Static checkers for defect classes that have actually shipped here. They live in
the repo, not in one person's workspace, so that a reviewer, a merger and an
author can all run the same check and get the same answer.

**Four of the five gate; one classifies.** `check_html_escaping.py` is a census —
its exit `0` means *I classified these files*, not *these files are clean*, and
its deliverable is the RAW lines rather than its status. That distinction is
printed by the tool at runtime and was still flattened into a column of five
green zeroes headed *gate green*, by someone who had read the output. It is
stated here because a person deciding whether to run a tool reads this file, and
a person deciding whether to merge reads the status.

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

    repo with a ref named f3eb259, control abbreviated to "f3eb259":
      git show f3eb259:templates/inventory.html   -> the REF's file, exit 0
      stderr: "warning: refname 'f3eb259' is ambiguous."   <- and the tools
                                                             discard stderr
    the full 40-char form, same repo, same ref present:
      git show f3eb259e30e5...:templates/inventory.html    -> the control

Measured by ref type: branches *and* tags shadow an abbreviation, and branches,
lightweight tags and annotated tags are all ignored at 40 hex — the immunity is
a property of the name, not the ref type. **A tag is the plausible accident.**
Nobody creates a branch called `aad3368` on purpose; a release or checkpoint tag
named after a short SHA is an ordinary thing to find in a repo somebody else
maintains.

Git ignores a 40-hex ref by design — *"it will be ignored when you just specify
40-hex"* — so only the full form is outside the ref namespace and cannot be
shadowed.

**Enforced in `_args.pinned()`, called on every path that reaches git's ref
namespace**, not in each checker. A tool cannot add an unpinned control without
going through a function that refuses them, and a tool written later inherits
the rule without anyone remembering it. It first shipped as a copy in each of
the two checkers that already obeyed it and absent from the three that did not —
the same divergence `_args.py` exists to end, repeated in miniature.

It then shipped described as *"enforced in `git_show`, the one place every
control passes through"*. That was true of every control that existed the
morning it was written, which is not the same claim: `verify_rbac_branch.py`
materializes its control with `git archive`, which resolves through the ref
namespace identically and never touches `git_show`. **The denominator is a grep,
not a memory:**

    every subprocess call under tools/
      _args.py:184                git show      -> pinned(), inside git_show
      verify_rbac_branch.py:297   git archive   -> pinned() at :291
      check_template_parse.py:88  node --check  -> not git

Two paths reach the ref namespace, both call `pinned()`, and there is no third.
**Re-run that grep when you add a tool** rather than trusting this sentence —
enumerating the ways to reach the resource survives a sixth tool; enumerating
the callers of one function did not survive a fifth.

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

**One exception, and it is not visible in this table.** `check_html_escaping.py`
is a census, not a gate: its `0` means the self-test passed and 27 files were
classified, and it says nothing about whether what it classified is safe. Row 0
above reads *clean* and is wrong for that one tool. Do not build a wrapper that
gates on its status — read its RAW lines. See its section below.

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

### A status is only evidence if the output proves the tool ran

Reading `$?` correctly is not enough, because a process that never reached the
tool still has a status. A sandbox here exports `PYTHONHOME`; under it, CPython
dies in startup before `main()` and exits **1**:

    bare python3        check_guard_derefs.py templates/   rc=1, no output
    env -u PYTHONHOME   check_guard_derefs.py templates/   rc=0, self-test line

Four checkers run that way returned `1 | 1 | 1 | 1`. **`1` is the code that reads
as *the tool ran and has something to say*** — a `2` at least looks like trouble
and invites a second look, while four `1`s look like a real result you now go
read findings for. Not one of those processes executed a line of tool code. What
gave it away was four tools agreeing exactly, which is luck rather than method.

So: **pair every status with a line of that process's own output.**

**And the line has to differ by outcome, not merely exist.** This correction is
the whole rule; on its own the rule is satisfiable by a banner, and the natural
grep is the one that finds the banner:

    "SELF-TEST -- known answers from this repo's history"   shallow 1   full 1

That header is printed unconditionally by `check_template_parse.py`, before the
outcome. It proves the process ran and says nothing about what it concluded — and
it was published in two separate verification tables as evidence for a `0`. Match
on content only one outcome can produce:

    shallow  2   "UNOBTAINABLE: 72380402cb22…:templates/import.html"
    full     0   "OK 72380402cb22…:templates/import.html expected FAIL, got FAIL"

A refusal line qualifies precisely because a passing run cannot print it.

The corollary is that **these tools must not emit lines nobody can classify.**
`materialize()` called `tf.extractall(dest)` with no `filter=`, which prints a
`DeprecationWarning` to stderr on 3.12+ and changes default in 3.14. It now
passes `filter="data"` when the interpreter has it. Not a safety fix: git's
archive of this repo carries no symlinks, no absolute paths and no `..`, and both
forms were measured extracting **60 files to an identical tree hash**. It is a
fix for a stray stderr line in the one directory whose contract is that every
line of output is evidence. **Unverifiable here** — this machine is 3.10.12,
which backports `data_filter` but does not warn, so `-W error::DeprecationWarning`
returns rc 0 and zero warnings on *both* sides. The identical-extraction claim is
measured; the warning suppression is read off the changelog.

### The same trap inside the code, one layer down

`materialize()` in `verify_rbac_branch.py` shipped as a shell pipeline:

    proc = subprocess.run(f"git … archive {ref} | tar -x -C {dest}", shell=True)
    if proc.returncode != 0: raise

**A pipeline's returncode is the last command's.** That check inspected `tar` and
never `git`:

    git archive <absent ref>            rc 128   <- the real answer
    git archive <absent ref> | tar -x   rc 2     <- what the check saw

It refused anyway, which is why it survived: tar also fails on an empty stream,
and a `contains no app.py` test backstops it. **Both are luck.** A producer that
emits a valid tar prefix and *then* dies defeats both — pipeline rc 0, `app.py`
extracted, a partially materialized negative control certifying as whole. That
producer is synthetic; the claim here is *neither backstop is load-bearing*, not
that `git archive` behaves that way.

The fix is `git archive --format=tar -o FILE REF` plus `tarfile`: git writes the
file itself, so the inspected returncode is git's, and `shell=True`, `shlex` and
the external `tar` dependency all go away with it. Losing `shell=True` is worth
more than the rc fix on its own.

Note the shape it shares with the two shell traps above — a status collected from
something other than the thing under test. It reached production code here after
producing three wrong measurements in the shell on the same day, by the people
building the checkers.

### Proving a commit is documentation-only

An unchanged blob hash proves a file did not move, and that is the cheap check
worth running first. For a file whose blob *had* to move — a docstring edit —
parse both revisions, strip every docstring node, and compare the ASTs:

    _args.py                bytes changed   AST-minus-docstrings IDENTICAL
    check_guard_derefs.py   bytes changed   AST-minus-docstrings IDENTICAL

That turns *"only the prose moved"* from a reading of the diff into a mechanical
fact, and it costs nothing.

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

### `check_html_escaping.py` — `${…}` reaching innerHTML without an escaper

    tools/check_html_escaping.py templates/           # or individual files
    tools/check_html_escaping.py templates/shops.html

**This one is a census, not a gate.** It finds `X.innerHTML = …`, follows one hop
of assembly into the sink variable (`=`, `+=`, `.push()`, one hop through a named
function), and classifies every `${expr}` in the template literals that reach it
as ESCAPED or RAW. **RAW is a finding to judge, not a bug.** `${title}` passed in
by the caller as a literal carries no attacker data. The tool exists so a
reviewer reads classified lines instead of re-deriving them.

    exit 0  the self-test passed and N files were CLASSIFIED — not "clean"
    exit 1  a named target could not be read
    exit 2  the control was unobtainable — nothing was classified

Its own output says so (`Exit 0 here means CLASSIFIED, not CLEAN`), and that
caveat still reached a verification table as one of five green zeroes under the
heading *gate green*. A runtime `print` reaches whoever runs the tool and nobody
who reads the file to decide whether to.

Control: `c087367bb459…:templates/shops.html` must report >= 3 RAW **including
`it.title`, `s.name` and `c.f(x)` by name.** Merged main with all three sinks
known-unescaped and live — QA detonated `:1063` there. A minimum count alone
would pass on three unrelated interpolations; naming the expressions is what
makes it the control for *this* defect.

**Why it is a separate tool from `check_guard_derefs.py`.** That checker answers
"when a Jinja guard is false, does JS still dereference what it removed?" — it
says nothing about escaping, and on a file with no guarded ids it examines
nothing and prints a clean RESULT anyway. `shops.html` at `c087367` has exactly
one `can_edit` block containing **zero ids**, so the deref checker cannot fail
there no matter how the sinks are written. Its green on that file was a vacuous
green, which is the failure this directory exists to catch.

**Its numbers are not the audit ledger's numbers.** Today it reports **206 RAW
and 86 escaped across 12 of 27 templates**; the audit ledger says **132 dynamic
writes across 16 files**. Those are different denominators counted different
ways — one counts interpolations, the other counts writes — and neither is
derivable from the other by inspection. Do not put them in the same sentence
until someone has derived one from the other. Mixing units that way has already
put a wrong count in this channel once.

Does not see: sinks other than `innerHTML` (`insertAdjacentHTML`, `outerHTML`,
`document.write`, jQuery `.html()`); **URL contexts** — `esc()` counts as escaped
everywhere and does nothing to a `javascript:` URL reaching an `href`, so a file
can be 100% ESCAPED here and still have a scheme injection; assembly laundered
through an array index, an object property or a callback parameter; and
server-rendered Jinja `{{ x }}`, which is Jinja's problem.

Its docstring rates its own confidence **lowest of the four its author wrote** —
its literal scanner is the most intricate code in this directory and its control
has been exercised least. Read its RAW lines yourself before treating the output
as load-bearing.

### `verify_rbac_branch.py` — the RBAC resource map against the ledger

    tools/verify_rbac_branch.py                       # this checkout's app.py
    tools/verify_rbac_branch.py /path/to/app.py

Extracts `_resource_for_path` from the target `app.py` **verbatim, AST → exec,
not a reimplementation**, then asserts the resolved resource for all 37 ledger
entries, re-enumerates every route registration in the repo to confirm no
mutating route is left unmapped, and checks the template and item-8 control
guards in the tree beside it. With no argument it prints the absolute path it
resolved — a green always names what it examined.

Control: `aad3368b7a61…` must FAIL with **5 expected failures** — a tree with the
F3 fix in and the item-8 guards absent. It is the one control in this directory
that does **not** go through `git_show`: it is materialized with `git archive`,
so `materialize()` calls `_args.pinned()` itself. That path is why `pinned()` is
public, and the two shapes of refusal there are deliberately different:

    NEGATIVE_CONTROL_REF = "aad3368"          exit 2  CONTROL NOT PINNED
    NEGATIVE_CONTROL_REF = "0000…0000"        exit 2  git archive failed (rc 128)
    NEGATIVE_CONTROL_REF = "aad3368b7a61…"    exit 0  self-test OK, 5 failures

*Not pinned* is a defect in the tool; *could not materialize* is a fact about the
tree. The third row is what makes the first two mean anything — a check that
refused uniformly would print the first two identically.

It used to point at a `.scratch/` worktree and **skip** the self-test when that
path was missing, so pruning a directory would have silently downgraded a
merge-gating control to a no-op.

Does not see: **runtime enforcement.** It proves the map resolves each path to
the right resource; it does not prove the gate consults the map, is reached, or
refuses a request — that needs a live request. Also invisible: routes not
registered by a module-level decorator or `add_url_rule` with a literal first
argument; blueprints that are never registered (`/shipping/*` is mapped
defensively but unwired, so a PASS there is a statement about the map, not the
app); and anything outside the ledger, which the residual sweep catches only as
"no resource at all", never as "the wrong resource".

### `attack_live.py` — detonates the migration-bundle gate

    tools/attack_live.py                              # this checkout's app.py
    tools/attack_live.py /path/to/app.py

Fires 12 traversal payloads plus one control image at `uploaded_file`, extracted
from the target `app.py` by AST every run. **That extraction is the point:** a
test file with the `aad3368` handler pasted into it tests the old code while
reporting on the new one, which is this repo's recurring failure in its purest
form.

**The client choice is load-bearing.** It uses the Werkzeug test client, which
passes the raw path through. `requests` and `curl` normalize `..` client-side and
would report every traversal payload blocked — two of the twelve payloads exist
only because an earlier probe did exactly that. Verified: the handler receives
the traversal verbatim.

Control: `aad3368b7a61…:app.py` must leak via **7 of 12** payloads. It fires
loudly — 7 at the control, 0 at main — so a clean run has demonstrably been able
to report dirty on the same code path minutes earlier.

Does not see: anything outside its fixed payload list — a traversal encoding
nobody thought of is not covered. Nor the real request pipeline: the handler is
re-registered on a bare Flask app with stubbed auth, so `before_request` hooks,
session handling, the real login gate and any WSGI middleware are all absent. It
answers *does this handler body leak*, not *does the deployed app*. Only
`uploaded_file` — `/temp_split`, `/temp_cards` and `/temp_pdf` are untouched. And
the bundle it looks for is a two-line fake matched by substring.

Confidence is **high on the bypass direction and lower on the clean one**, for
the payload-list reason above.

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

Fetch your control through `_args.git_show`, which enforces the 40-hex rule for
you. A control read by a private `subprocess.run(["git", …])` is a control nobody
checked. **If you genuinely cannot use `git_show`** — `verify_rbac_branch.py`
needs a whole tree, not one file — call `_args.pinned(ref)` yourself before the
ref reaches git, and add your call to the grep table above. "Every control goes
through `git_show`" was true the morning it was written and stopped being true
the same day.

If you shell out, **do not put a pipe in it.** A pipeline's returncode is the
last command's, so the check reads the wrong process's status; that shipped here
once and is written up above.

And give it a `--help`. These two shipped without one and answered a traceback —
a poor first reply from a tool whose whole purpose is to be trusted by people who
did not write it.
