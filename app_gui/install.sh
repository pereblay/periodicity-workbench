#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.14 python3.13 python3.12; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]] || [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.12, 3.13, or 3.14 is required. Install Python 3.14 and run ./install.sh again." >&2
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VERSION" in
  3.12|3.13|3.14) ;;
  *)
    echo "Selected interpreter is Python $PYTHON_VERSION. Use Python 3.12–3.14 for the supported PySide6 wheel." >&2
    exit 1
    ;;
esac

if [[ -x .venv/bin/python ]]; then
  VENV_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$VENV_VERSION" != "$PYTHON_VERSION" ]]; then
    echo "Replacing the existing .venv (Python $VENV_VERSION) with Python $PYTHON_VERSION."
    rm -rf .venv
  fi
fi

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

echo "Installing the native Qt app with Python $PYTHON_VERSION..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt

echo "Installation complete. Start the app with ./run.sh"
