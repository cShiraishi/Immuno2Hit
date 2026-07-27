#!/bin/bash
# Double-click to open Immuno2Hit (TREM2 / CD28 / IDO1) in the browser.
cd "$(dirname "$0")/backend" || exit 1
echo "Immuno2Hit — TREM2 · CD28 · IDO1"
echo "Close this window to stop the server."
echo
exec python3 app.py
