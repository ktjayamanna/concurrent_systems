#!/bin/bash
set -e

cd "$(dirname "$0")"

/usr/bin/pdflatex -interaction=nonstopmode sage_plus_plus_paper.tex
/usr/bin/bibtex sage_plus_plus_paper
/usr/bin/pdflatex -interaction=nonstopmode sage_plus_plus_paper.tex
/usr/bin/pdflatex -interaction=nonstopmode sage_plus_plus_paper.tex

echo "Done: sage_plus_plus_paper.pdf"
