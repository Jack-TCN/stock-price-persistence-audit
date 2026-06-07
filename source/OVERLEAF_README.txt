Overleaf source structure
=========================

Main manuscript: main.tex
Supplementary material: supplementary_material.tex

Structure:
- sections/: manuscript and supplementary section files loaded with \input
- tables/: generated LaTeX table files
- figures/: renamed manuscript figures, including EPS versions

Recommended compiler: pdfLaTeX on Overleaf. If EPS conversion causes a compile issue, switch the figure extension in sections/04_results.tex from .eps to .pdf.
