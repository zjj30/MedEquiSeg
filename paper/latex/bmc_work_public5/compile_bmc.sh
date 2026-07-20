#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TINYTEX_BIN="${TINYTEX_BIN:-$HOME/.TinyTeX/bin/x86_64-linux}"
OUTDIR="$ROOT/paper/latex/build_bmc_public5"
MAIN="$ROOT/paper/latex/bmc_work_public5/main_bmc.tex"
STABLE_PDF="$ROOT/output/pdf/medequiseg_bmc_medical_imaging_manuscript_public5.pdf"

if [[ ! -x "$TINYTEX_BIN/pdflatex" ]]; then
  echo "TinyTeX not found at $TINYTEX_BIN/pdflatex" >&2
  exit 1
fi

export PATH="$TINYTEX_BIN:$PATH"
export TEXINPUTS="$ROOT/paper/latex//:"
export BSTINPUTS="$ROOT/paper/latex/bmc_work_public5/bst//:"
mkdir -p "$OUTDIR" "$(dirname "$STABLE_PDF")"
cd "$ROOT"

rm -f "$OUTDIR/main_bmc.pdf"
pdflatex -halt-on-error -interaction=nonstopmode -output-directory="$OUTDIR" "$MAIN" >/dev/null
pdflatex -halt-on-error -interaction=nonstopmode -output-directory="$OUTDIR" "$MAIN" >/dev/null

PDF="$OUTDIR/main_bmc.pdf"
test -s "$PDF"
cp "$PDF" "$STABLE_PDF"
pages="$(pdfinfo "$PDF" 2>/dev/null | awk '/Pages:/ {print $2}')"
echo "OK: $STABLE_PDF (${pages:-?} pages)"
