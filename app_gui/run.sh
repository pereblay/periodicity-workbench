#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "The app environment is not installed. Run ./install.sh first." >&2
  exit 1
fi

PYTHON_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VERSION" in
  3.12|3.13|3.14) ;;
  *)
    echo "The app environment uses Python $PYTHON_VERSION. Recreate it with ./install.sh (Python 3.12–3.14)." >&2
    exit 1
    ;;
esac

exec .venv/bin/python desktop_app.py
