.PHONY : all front front-new front-old print restorethud force clean figures tables
.DEFAULT_GOAL : all

# images/models/ and images/plots/ are versioned, and `make figures` rewrites them from the notebooks' caches

chp     := $(wildcard chapters/*/*.tex)

compile := pdflatex --shell-escape --synctex=1 --output-format=pdf thesis.tex
bib := bibtex thesis

all: thesis.pdf

# title page in black on white for the hard cover: the current banner (new) or the bachelor thesis cover with the seal (old)
front-new: front-new.pdf
front-old: front-old.pdf
front: front-new front-old

# the figures of chapters 6-8 into images/models/ and images/plots/ (needs the merged
# parquets in ../nb/data/)
figures:
	@uv run python scripts/make_thud_figures.py

# every number chapters 7-8 report, printed for copying into chapters/
tables:
	@uv run python scripts/make_thud_tables.py

clean:
	rm -f thesis*.aux thesis*.bbl thesis*.blg thesis*.log thesis*.out thesis*.synctex.gz thesis*.toc thesis*.lof thesis*.lot thesis*.fls thesis*.fdb_latexmk
	rm -f front-*.aux front-*.log front-*.out

force:
	@$(compile)
	@$(bib)
	@$(compile)
	@$(compile)
	@printf "\rCompilation [Done]\n"

thesis.pdf: thesis.tex thesis.xmpdata thud.bib book.thud.tex thud.cls $(chp) $(wildcard images/*.pdf images/*/*.pdf images/*/*/*.pdf)
	@$(compile)
	@$(bib)
	@$(compile)
	@$(bib)
	@$(compile)
	@printf "\rCompilation [Done]\n"

front-%.pdf: front-%.tex thesis.tex book.thud.tex thud.cls $(wildcard banners/black/*)
	@pdflatex --shell-escape --output-format=pdf $<
	@printf "\rCompilation [Done]\n"

# the seal in vector form for front-old: logo.svg with its K100 grey (35,31,32) set to pure black, for the foil stamping plate
front-old.pdf: banners/black/logo.pdf
banners/black/logo.pdf: banners/black/logo.svg
	@sed 's/rgb(35,31,32)/rgb(0,0,0)/g' $< > banners/black/logo.black.svg
	@uv run --no-project --with svglib --with reportlab python -c "from svglib.svglib import svg2rlg; from reportlab.graphics import renderPDF; renderPDF.drawToFile(svg2rlg('banners/black/logo.black.svg'), '$@')"
	@rm banners/black/logo.black.svg
