# Build thesis with all outputs in build/
# Prefer latexmk if available (handles pdflatex + bibtex + index automatically).
MAIN = Thesis
BUILD = build
BIBINPUTS = references//:

# So pdflatex/bibtex are found (macOS MacTeX, Homebrew, TeX Live)
export PATH := /Library/TeX/texbin:/usr/local/texlive/2024/bin/universal-darwin:/usr/local/texlive/2023/bin/universal-darwin:/usr/local/texlive/2022/bin/universal-darwin:/opt/homebrew/opt/basictex/bin:/opt/homebrew/bin:$(PATH)

.PHONY: all clean pdf pdf-direct

all: pdf

# Use latexmk when available; otherwise run pdflatex + bibtex manually
pdf:
	@if command -v pdflatex >/dev/null 2>&1 || command -v latexmk >/dev/null 2>&1; then \
		if command -v latexmk >/dev/null 2>&1; then $(MAKE) pdf-latexmk; else $(MAKE) pdf-direct; fi; \
	else \
		echo "pdflatex not found. Install LaTeX (e.g. MacTeX) or: brew install --cask mactex-no-gui"; exit 1; \
	fi

# latexmk: one command for full build (pdflatex, bibtex, index, repeat as needed)
pdf-latexmk:
	BIBINPUTS=$(BIBINPUTS) latexmk -pdf -output-directory=$(BUILD) -interaction=nonstopmode -bibtex -auxdir=$(BUILD) -outdir=$(BUILD) $(MAIN).tex
	@echo "Output: $(BUILD)/$(MAIN).pdf"

# Manual sequence (used when latexmk is not installed)
pdf-direct:
	pdflatex -output-directory=$(BUILD) -interaction=nonstopmode $(MAIN)
	BIBINPUTS=$(BIBINPUTS) bibtex $(BUILD)/$(MAIN)
	-makeindex $(BUILD)/$(MAIN)
	-(cd $(BUILD) && makeglossaries $(MAIN)) 2>/dev/null || true
	pdflatex -output-directory=$(BUILD) -interaction=nonstopmode $(MAIN)
	pdflatex -output-directory=$(BUILD) -interaction=nonstopmode $(MAIN)
	@echo "Output: $(BUILD)/$(MAIN).pdf"

clean:
	rm -f $(BUILD)/*.aux $(BUILD)/*.bbl $(BUILD)/*.blg $(BUILD)/*.log $(BUILD)/*.out
	rm -f $(BUILD)/*.toc $(BUILD)/*.lof $(BUILD)/*.lot $(BUILD)/*.fls $(BUILD)/*.fdb_latexmk
	rm -f $(BUILD)/*.synctex.gz $(BUILD)/*.run.xml $(BUILD)/*.bcf
	rm -f $(BUILD)/*.acn $(BUILD)/*.acr $(BUILD)/*.alg $(BUILD)/*.glo $(BUILD)/*.glsdefs
	rm -f $(BUILD)/*.idx $(BUILD)/*.ilg $(BUILD)/*.ind $(BUILD)/*.ist
	rm -f $(BUILD)/*.pdf
