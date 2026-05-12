#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p pdf png

echo "Compiling TikZ diagrams..."
for f in tikz/*.tex; do
    name=$(basename "$f" .tex)
    echo "  -> $name"
    pdflatex -interaction=nonstopmode -output-directory=pdf "$f" > "pdf/${name}.log" 2>&1
    # Clean up aux/log artifacts
    rm -f "pdf/${name}.aux" "pdf/${name}.log"
    # Rasterize to PNG (macOS sips; fallback to ImageMagick convert)
    if command -v sips &>/dev/null; then
        sips -s format png "pdf/${name}.pdf" \
             --resampleHeightWidthMax 2400 \
             --out "png/${name}.png" 2>/dev/null
    elif command -v convert &>/dev/null; then
        convert -density 300 -quality 95 "pdf/${name}.pdf" "png/${name}.png"
    else
        echo "    (no rasterizer found; PDF is available)"
    fi
done
echo "Done. PDFs -> pdf/   PNGs -> png/"
