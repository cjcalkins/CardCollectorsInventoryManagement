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

A CONTROL MUST BE A PINNED SHA
------------------------------
Never `main~3`, `origin/main`, a branch or a tag. The harm from resolving a
control in the wrong tree is confined to trees that lack the commit only
because an immutable SHA names the same bytes everywhere it exists at all. A
symbolic ref resolves to a *different commit* per tree, so the self-test would
pass against the wrong bytes and the tool would return a wrong answer on a
checkout that has everything. That confinement is a property of the refs, not
of this module -- it has to be maintained by whoever adds the next control.
"""
import os
import subprocess
import sys

EX_OK = 0
EX_FINDING = 1
EX_CONTROL = 2
EX_USAGE = 64  # sysexits EX_USAGE. Deliberately not 2 -- see the module docstring.


_override_announced = False


def repo_root():
    """The checkout this file lives in. Never $HOME, never the cwd."""
    own = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    override = os.environ.get("CCIM_REPO")
    if not override or os.path.abspath(override) == own:
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


def git_show(ref, path):
    """`git show ref:path` against this tool's own checkout.

    Returns the file's text, or None if the control is unobtainable -- callers
    must translate None into exit 2, never into a skip.
    """
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
    if "-h" in argv or "--help" in argv:
        print(usage)
        sys.exit(EX_OK)
    unknown = [a for a in argv if a.startswith("-") and a not in extra_flags]
    if unknown:
        print("unknown option(s): %s\n" % " ".join(unknown))
        print(usage)
        sys.exit(EX_USAGE)

    paths = [a for a in argv if a not in extra_flags]
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
