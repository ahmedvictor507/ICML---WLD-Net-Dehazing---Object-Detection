#!/usr/bin/env bash
# Launcher for the Dehazing GUI — always uses the dehaze_env venv.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../dehaze_env/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Virtual environment not found at $PYTHON"
    echo "Please create it: python3 -m venv $SCRIPT_DIR/../dehaze_env"
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/main.py" "$@"
