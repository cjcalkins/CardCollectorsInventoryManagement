#!/usr/bin/env python3
"""Preflight: verify this environment can run the app BEFORE anything imports.

Run it standalone to check an install:

    python preflight.py

It also runs automatically at the top of `python app.py`, so a missing package
or system library produces a short report with the fix instead of a raw
traceback (the trigger for this file: a fresh DietPi install died at
`import cv2` with "libGL.so.1: cannot open shared object file").
Set CCIM_SKIP_PREFLIGHT=1 to bypass it.

This module must stay stdlib-only, and it must NOT import cv2 (or anything
that imports cv2, e.g. rapidocr) in-process: app.py sets
OPENCV_IO_MAX_IMAGE_PIXELS before its own `import cv2`, and importing cv2
here first would sidestep that sequencing. The imaging/OCR stack is therefore
probed in a subprocess.
"""
import json
import os
import subprocess
import sys

MIN_PYTHON = (3, 10)

# Packages requirements.txt declares as core. "launch" means app.py (or a module
# it imports unconditionally) imports it at startup; "runtime" means a feature
# imports it later — the app would start, but a core feature is broken, so a
# missing package still blocks with a clear message rather than failing weeks
# later on first use.
REQUIRED = [
    ("flask",            "Flask",            "launch"),
    ("flask_sqlalchemy", "Flask-SQLAlchemy", "launch"),
    ("sqlalchemy",       "SQLAlchemy",       "launch"),
    ("werkzeug",         "Werkzeug",         "launch"),
    ("dotenv",           "python-dotenv",    "launch"),
    ("numpy",            "numpy",            "launch"),
    ("PIL",              "Pillow",           "launch"),
    ("reportlab",        "reportlab",        "runtime: PDF financial reports"),
    ("cryptography",     "cryptography",     "runtime: HTTPS certificate generation"),
    ("requests",         "requests",         "runtime: catalog/provider APIs"),
]

# The app starts without these and disables the feature with its own message;
# preflight just says so up front.
OPTIONAL = [
    ("fitz",     "PyMuPDF", "PDF import"),
    ("zeroconf", "zeroconf", "the https://<name>.local mDNS address"),
    ("hnswlib",  "hnswlib",  "fast Search-by-Image shortlisting"),
]

# Substring of a failed import's message -> the system-level fix. OpenCV's full
# desktop build (which rapidocr forces in next to the headless build) links
# these; minimal/headless Linux images ship without them.
SHARED_LIB_FIXES = [
    ("libGL.so",         "sudo apt install -y libgl1 libglib2.0-0"),
    ("libgthread",       "sudo apt install -y libgl1 libglib2.0-0"),
    ("libglib",          "sudo apt install -y libgl1 libglib2.0-0"),
]

_PROBE_SCRIPT = r"""
import json
out = {}
for name in ("cv2", "onnxruntime", "rapidocr"):
    try:
        __import__(name)
        out[name] = None
    except BaseException as exc:   # noqa: BLE001 - report anything, never crash
        out[name] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
"""


def _remedy(error_text):
    """Map an import failure to the most likely fix."""
    for needle, fix in SHARED_LIB_FIXES:
        if needle in error_text:
            return (fix
                    + "\n    (Minimal/headless Linux images ship without OpenCV's"
                      " system libraries — see README > Requirements.)")
    if "No module named" in error_text:
        return ("pip install -r requirements.lock   (recommended)\n"
                "    If you already installed the requirements, you are probably"
                " in a different\n    virtualenv/conda env than the one you"
                " installed into.")
    return "See README > Requirements."


def _check_module(module):
    """Import `module` in-process. Returns an error string, or None if OK."""
    try:
        __import__(module)
        return None
    except BaseException as exc:  # noqa: BLE001 - a broken package can raise anything
        return "%s: %s" % (type(exc).__name__, exc)


def _probe_imaging_stack():
    """Import cv2/onnxruntime/rapidocr in a child process (see module docstring).
    Returns a dict name -> error-or-None, or None if the probe itself failed."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True, text=True, timeout=180,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def run_preflight():
    if os.environ.get("CCIM_SKIP_PREFLIGHT") == "1":
        return

    say = lambda msg: print("[preflight] " + msg, file=sys.stderr)
    problems = 0

    if sys.version_info < MIN_PYTHON:
        say("REQUIRED: Python %d.%d+ is required; this is Python %d.%d."
            % (MIN_PYTHON + sys.version_info[:2]))
        problems += 1

    for module, pip_name, role in REQUIRED:
        err = _check_module(module)
        if err is not None:
            say("REQUIRED: %s (import %s) failed: %s" % (pip_name, module, err))
            if role.startswith("runtime"):
                say("  Needed at " + role)
            say("  Fix: " + _remedy(err))
            problems += 1

    probe = _probe_imaging_stack()
    if probe is None:
        # The probe machinery itself failed; don't block on it — if cv2 is truly
        # broken, the app's own import will still stop the launch (uglier, but
        # nothing is silently skipped).
        say("note: could not probe the OpenCV/OCR stack; continuing.")
    else:
        if probe.get("cv2") is not None:
            say("REQUIRED: OpenCV (import cv2) failed: %s" % probe["cv2"])
            say("  Fix: " + _remedy(probe["cv2"]))
            problems += 1
        ocr_missing = [n for n in ("onnxruntime", "rapidocr") if probe.get(n) is not None]
        if ocr_missing:
            say("WARNING: %s failed to import — card OCR will be unavailable"
                " until fixed:" % " and ".join(ocr_missing))
            for name in ocr_missing:
                say("  %s: %s" % (name, probe[name]))
            say("  Fix: " + _remedy(probe[ocr_missing[0]]))

    for module, pip_name, feature in OPTIONAL:
        if _check_module(module) is not None:
            say("note: optional %s not installed — %s is disabled." % (pip_name, feature))

    if problems:
        say("")
        say("Startup blocked: %d required dependency problem%s (details above)."
            % (problems, "" if problems == 1 else "s"))
        say("After fixing, run:  python app.py    (or re-check with:  python preflight.py)")
        sys.exit(1)


def _elevated():
    """Best-effort 'is this an administrator/root shell'. Advisory only here — the
    binding decision lives in app.py's launcher, and this module is imported by
    app.py at import time, so nothing in it may exit on a privilege check."""
    if os.name == "nt":
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return None      # unknown, not "no"
    try:
        return os.geteuid() == 0
    except AttributeError:
        return None


def main():
    if os.environ.get("CCIM_SKIP_PREFLIGHT") == "1":
        print("[preflight] Skipped: CCIM_SKIP_PREFLIGHT=1 is set — nothing was checked.")
        return 0
    run_preflight()
    print("[preflight] All required dependencies are importable.")
    # Say it here rather than let the launcher be the first to mention it: finding
    # out about the privilege requirement from a preflight that just said "ready"
    # is a worse experience than being told in the same breath.
    if _elevated() is False:
        launch = "python app.py" if os.name == "nt" else "sudo -E python3 app.py"
        print("[preflight] This shell is NOT elevated. The app requires administrator "
              "privileges and will refuse to start.")
        print("[preflight] Ready to run, once elevated:  %s" % launch)
    else:
        print("[preflight] Ready to run: python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
