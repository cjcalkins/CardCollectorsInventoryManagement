#!/usr/bin/env bash
# run.sh - Run app.py with sudo while preserving an active conda or venv environment

set -euo pipefail

# Detect if a conda environment is active
if [[ -n "${CONDA_PREFIX:-}" && -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Detected active conda environment: ${CONDA_DEFAULT_ENV}"
    echo "CONDA_PREFIX: ${CONDA_PREFIX}"
    echo "Using PATH: ${PATH}"
    echo "Launching: sudo env \"PATH=\$PATH\" python app.py"
    sudo env "PATH=${PATH}" python app.py

# Detect if a venv (virtualenv) environment is active
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Detected active venv environment: ${VIRTUAL_ENV}"
    echo "Using PATH: ${PATH}"
    echo "Launching: sudo env \"PATH=\$PATH\" python app.py"
    sudo env "PATH=${PATH}" python app.py

else
    echo "No active conda or venv environment detected."
    echo "Falling back to system python with sudo..."
    sudo python app.py
fi
