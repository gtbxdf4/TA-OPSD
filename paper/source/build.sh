#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
if [ -n "${TECTONIC:-}" ]; then
    compiler="$TECTONIC"
elif command -v tectonic >/dev/null 2>&1; then
    compiler=tectonic
elif [ -x /Applications/ChatGPT.app/Contents/Resources/tectonic/tectonic ]; then
    compiler=/Applications/ChatGPT.app/Contents/Resources/tectonic/tectonic
else
    printf '%s\n' 'Install Tectonic or set TECTONIC to its executable path.' >&2
    exit 1
fi
exec "$compiler" -X compile --outdir build --outfmt pdf --keep-logs --keep-intermediates --print --untrusted main.tex
