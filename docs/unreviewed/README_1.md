# Thesis LaTeX source

Standard layout:

- **Thesis.tex** — main document (build from this directory)
- **VTU_sgr.cls** — document class
- **chapters/** — chapter sources (Introduction, Related_Work, Feature_Extraction, etc.)
- **frontmatter/** — abstract, certificate, dedication, abbreviations, bibliography
- **figures/** — images (referred to as `figures/...` in `\includegraphics`)
- **references/** — `Bibliography.bib`
- **build/** — all generated files (`.aux`, `.bbl`, `.log`, `.pdf`, etc.)

## Build

From this directory:

```bash
make
```

- If **latexmk** is installed (recommended), it is used and handles pdflatex, bibtex, index and repeated runs.
- If not, the Makefile falls back to the manual pdflatex + bibtex sequence.

Force one method:

```bash
make pdf-latexmk   # use latexmk only
make pdf-direct    # use pdflatex + bibtex only
```

Output: `build/Thesis.pdf`

## Clean

```bash
make clean
```
