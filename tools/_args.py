"""Shared argv handling, exit codes and repo resolution for the checkers here.

Three of these tools grew up in one workspace and each invented its own answer
to the same three questions: where is `--help`, where is the repo the controls
come from, and what does the process exit with. Divergent answers to the last
one are the dangerous ones -- a wrapper keying on exit status has to be able to
tell "clean" from "I could not prove I discriminate" from "you typed the flag
wrong", and it can only do that if every tool agrees.

    0   clean -- the self-test passed AND at least one target was read
    1   a real finding, or a named target that could not be read
    2   a control was unobtainable -- NOTHING was reported, this is not a pass
    64  usage error, including paths that expanded to nothing

WHERE THE REPO COMES FROM, AND WHY IT IS NOT THE CWD OR $HOME
------------------------------------------------------------
Every checker here reconstructs its control from this repo's own history, so it
has to answer "which checkout?" before it can answer anything else. Two wrong
answers both shipped, and they fail in opposite directions:

  * `os.path.expanduser("~/.buzz/REPOS/...")` -- a hardcoded home path. On any
    machine but its author's the control is unobtainable and the tool exits 2
    forever. On its author's machine it is worse than that: run inside a second
    checkout, it self-tests against the FIRST one and then reports on the
    second, so the proof-of-discrimination and the thing being certified come
    from different trees.

  * bare `git show` with no `-C` -- resolves against the cwd. Run the tool by
    absolute path from anywhere outside the tree and it refuses with
    "control unobtainable" on a checkout where the control is perfectly
    reachable. A false refusal is cheap; the same mechanism pointed at a
    sibling checkout of this repo is not.

`tools/` sits at the repo root, so this file's own grandparent directory IS the
checkout -- unambiguous, and the same answer no matter who runs it or from
where. A git worktree needs no help here: `git -C <worktree>` reaches the
shared object store, so every control resolves normally.

CCIM_REPO remains as an escape hatch for a copy of `tools/` that is not inside
a checkout at all, but note what it is: an opt-in re-entry into failure mode
(B) above. The property that made (B) dangerous was not that two trees were
involved, it was that the output looked identical to a correct run -- so when
the override is in force the tools say so, once, on stderr. Cross-tree
resolution is allowed here; silent cross-tree resolution is not.

A CONTROL MUST BE A FULL 40-CHAR SHA
------------------------------------
Never `main~3`, `origin/main`, a branch, a tag -- and never an abbreviation.
The harm from resolving a control in the wrong tree is confined to trees that
lack the commit only because an immutable SHA names the same bytes everywhere
it exists at all. A ref resolves to a *different commit* per tree, so the
self-test would pass against the wrong bytes and the tool would return a wrong
answer on a checkout that has everything.

An abbreviated SHA is not exempt, and that is the part that is not obvious:
git resolves a name through the REF NAMESPACE BEFORE it treats the name as an
object, so an abbreviation is a symbolic ref the moment anything is named that.

Measured by ref type in a clone that HAS the control, every ref pointed at
main. Control bytes md5 cb1ead65; main's bytes md5 3bc72f59:

    lightweight TAG named "f3eb259"        -> 3bc72f59   SHADOWED
    branch       named "f3eb259"           -> 3bc72f59   SHADOWED
    lightweight TAG named the lc 40-hex    -> cb1ead65   ignored
    annotated   TAG named the lc 40-hex    -> cb1ead65   ignored
    lightweight TAG named the UC 40-hex    -> cb1ead65   ignored
    branch       named the 40-hex          -> cb1ead65   ignored
    stderr on every shadowed row: "warning: refname 'f3eb259' is ambiguous."
                                  <- and the callers discard stderr

Two things that matter more than the branch case usually quoted. The immunity
is a property of the NAME, not the ref type -- annotated and lightweight tags
are ignored at 40-hex exactly as branches are. And **a tag is the plausible
accident**: nobody creates a branch called `aad3368` on purpose, but a release
or checkpoint tag named after a short SHA is an ordinary thing to find in a
repo somebody else maintains. The hazard does not require anyone to do
anything strange.

Git ignores a 40-hex ref by design -- "it will be ignored when you just specify
40-hex" -- so only the full form is outside the ref namespace and cannot be
shadowed. git_show() enforces this rather than trusting each caller, because
almost every control in this repo passes through it: a tool cannot add an
unpinned control without going through the one function that refuses them.

Almost, and the exception is the reason `pinned()` is public. Reading a control
is not the only way to reach git: `verify_rbac_branch.py` materializes its
negative control with `git archive`, which resolves through the ref namespace
exactly as `git show` does and never touches `git_show()`. So it calls
`pinned()` itself. A rule enforced on only some of the paths that reach git is
a rule enforced by luck, which is the same objection that moved the check out
of the two checkers in the first place.

Do not weaken this into a warning. When it was first demonstrated the shadowed
lookup happened to return a *fixed* template and the caller's `>= 13` threshold
caught it -- that was the direction the substitution went, not a property of
the check. A shadow pointing at any tree that satisfies the threshold passes
silently on the wrong bytes.
"""
import os
import re
import subprocess
import sys

EX_OK = 0
EX_FINDING = 1
EX_CONTROL = 2
EX_USAGE = 64  # sysexits EX_USAGE. Deliberately not 2 -- see the module docstring.


_override_announced = False


def repo_root():
    """The checkout this file lives in. Never $HOME, never the cwd.

    realpath, not abspath: putting a tool on $PATH by symlinking `tools/` is an
    ordinary thing to do with something advertised as "anyone can run this",
    and abspath leaves the link in the path, so the grandparent is the link's
    directory rather than the checkout. Both sides of the CCIM_REPO comparison
    are resolved too -- an override that is a symlink to this same tree changes
    nothing, and a warning that fires on a no-op teaches people to skim it.
    """
    own = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    override = os.environ.get("CCIM_REPO")
    if not override:
        return own
    override = os.path.realpath(override)
    if override == own:
        return own
    global _override_announced
    if not _override_announced:
        _override_announced = True
        sys.stderr.write(
            "NOTE: CCIM_REPO is set -- controls are being read from %s, not\n"
            "      from this tool's own checkout %s.\n"
            "      The self-test and the audited tree are different trees.\n"
            % (override, own))
    return override


# Lowercase only, which is stricter than git needs and deliberately so. Git's
# 40-hex rule is case-insensitive -- a branch named the UPPERCASE 40-hex is also
# ignored, verified -- so refusing uppercase buys no safety, it just keeps every
# control in this repo in the one form `git rev-parse` emits. The refusal is in
# the safe direction either way.
PINNED_RE = re.compile(r"[0-9a-f]{40}\Z")


def pinned(ref):
    """Exit 2 unless `ref` is a full 40-hex SHA. See the module docstring.

    `git_show()` calls this, so a tool that reads its control the ordinary way
    inherits the rule and never has to name it. Call it directly only if you
    reach git some other way -- `git archive`, `git cat-file`, a bare
    `subprocess` -- because those resolve through the ref namespace too and
    they do not come through here.
    """
    if not PINNED_RE.match(ref or ""):
        print("CONTROL NOT PINNED: %r is not a full 40-char SHA. Git resolves a "
              "name through\nthe ref namespace before treating it as an object, so "
              "a branch or tag of that\nname would silently shadow the control and "
              "the self-test would run on the\nwrong bytes." % (ref,))
        print("Refusing to certify — the control cannot be trusted to be the control.")
        sys.exit(EX_CONTROL)
    return ref


def git_show(ref, path):
    """`git show ref:path` against this tool's own checkout.

    Refuses outright if `ref` is not a full 40-char SHA -- see the module
    docstring. That is an exit rather than a None because the two are different
    diagnoses: None means "this tree does not have the control", which is a fact
    about the checkout, while an unpinned ref is a defect in the tool itself and
    no caller should be able to soften it into a missing-control message.

    Returns the file's text, or None if the control is unobtainable -- callers
    must translate None into exit 2, never into a skip. An unpinned `ref` does
    not come back as None: it exits 2 here, because "I read something that was
    not the control" and "I could not read the control" are the same claim and
    only the second one is detectable downstream.
    """
    pinned(ref)
    p = subprocess.run(["git", "-C", repo_root(), "show", "%s:%s" % (ref, path)],
                       capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def require(module, tool):
    """Import a dependency or refuse. An absent dependency is an unobtainable
    control, not a skip: without it nothing has demonstrated the tool can fail."""
    try:
        return __import__(module)
    except ImportError:
        print("REFUSING TO REPORT: %s needs %s and it is not installed."
              % (tool, module))
        print("An absent dependency means the self-test never ran. That is a "
              "failed control, not a clean run.")
        sys.exit(EX_CONTROL)


def _split_flags(argv, usage, extra_flags):
    """-h/--help exits 0, an undeclared flag exits 64, the rest come back.

    Both entry points below go through here so there is exactly one answer to
    "what counts as a flag" in this directory. Two of these tools had their own
    copy of these nine lines and the copies had already drifted.
    """
    if "-h" in argv or "--help" in argv:
        print(usage)
        sys.exit(EX_OK)
    unknown = [a for a in argv if a.startswith("-") and a not in extra_flags]
    if unknown:
        print("unknown option(s): %s\n" % " ".join(unknown))
        print(usage)
        sys.exit(EX_USAGE)
    return [a for a in argv if a not in extra_flags]


def single_target(argv, usage, default, extra_flags=()):
    """One optional target path, for the checkers that examine a single file.

    Returns `default` when none is named -- so these tools always have
    something to look at and can never take the "checked nothing, exited 0"
    path that `targets()` guards against. A named target that cannot be read
    is the CALLER's finding (exit 1), not a usage error: naming a file is a
    question about that file, and answering "bad syntax" would be a different
    question.

    The default must name what it resolved to in the tool's own output. A green
    that does not say which file it examined is the same shape as a green over
    an empty set.
    """
    paths = _split_flags(argv, usage, extra_flags)
    if len(paths) > 1:
        print("one target at a time.\n")
        print(usage)
        sys.exit(EX_USAGE)
    return paths[0] if paths else default


def targets(argv, usage, exts=(".html",), allow_empty=False, extra_flags=()):
    """Resolve argv into a list of target paths, or exit non-zero.

    Handles -h/--help, rejects unknown flags, and walks directories for `exts`.
    Files named on the command line are NOT stat'd here -- the tool opens them
    and reports an unreadable one as a finding (exit 1), because the caller
    named it expecting an answer about it.

    `extra_flags` names the tool-specific flags this caller understands, e.g.
    ("--self-test",). They are accepted rather than rejected as unknown, and
    stripped from the returned list so they are never mistaken for a path. The
    caller still reads its own flag off argv -- this function only decides that
    the flag is legal. The alternative was for each tool to strip its flags
    before calling in, and per-call-site argv surgery is how these four
    diverged in the first place.

    `allow_empty` is for tools whose own flag (e.g. --self-test) legitimately
    runs with no targets. Everything else that resolves to nothing exits 64:
    a run over an empty set is indistinguishable by status from a clean run
    over everything, which is the same weakness as a check that cannot fail.
    Note that `allow_empty` moves that guarantee to the caller rather than
    removing it: a tool that returns [] here must still not exit 0 unless its
    own flag did the work.
    """
    paths = _split_flags(argv, usage, extra_flags)
    if not paths:
        if allow_empty:
            return []
        print("no targets given -- nothing would be checked.\n")
        print(usage)
        sys.exit(EX_USAGE)

    out = []
    for a in paths:
        if os.path.isdir(a):
            for root, _dirs, files in os.walk(a):
                out.extend(os.path.join(root, f) for f in sorted(files)
                           if f.endswith(tuple(exts)))
        else:
            out.append(a)
    if not out:
        print("%s expanded to no %s file(s) -- nothing would be checked.\n"
              % (" ".join(paths), "/".join(exts)))
        sys.exit(EX_USAGE)
    return out
