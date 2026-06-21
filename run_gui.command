#!/usr/bin/env bash
# Launch the Photo Extractor GUI.
#
# The uv-managed Python ships a _tkinter built for Tcl/Tk 8.6 but omits the
# 8.6 script library, so we point it at Homebrew's tcl-tk@8 (and load its
# matching 8.6 C libs via DYLD_LIBRARY_PATH). Prerequisite: `brew install tcl-tk@8`.
set -e
cd "$(dirname "$0")"

TK_PREFIX="$(brew --prefix tcl-tk@8 2>/dev/null || echo /opt/homebrew/opt/tcl-tk@8)"
if [ ! -f "$TK_PREFIX/lib/tcl8.6/init.tcl" ]; then
    echo "tcl-tk@8 not found. Install it with:  brew install tcl-tk@8" >&2
    exit 1
fi
export TCL_LIBRARY="$TK_PREFIX/lib/tcl8.6"
export TK_LIBRARY="$TK_PREFIX/lib/tk8.6"
export DYLD_LIBRARY_PATH="$TK_PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

exec uv run python gui.py
