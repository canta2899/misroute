.PHONY : all force clean
.DEFAULT_GOAL : all

# latexmk tracks the dependencies itself and does nothing when slides.pdf is up to date;
# LuaLaTeX and the vendored moloch/ come from latexmkrc
latexmk := latexmk -silent -interaction=nonstopmode -halt-on-error

all:
	@$(latexmk) slides.tex

force:
	@$(latexmk) -g slides.tex

clean:
	@latexmk -c slides.tex
	@rm slides.pdf
