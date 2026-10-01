# Slides

Beamer decks for the thesis defence and the department talk; the thesis is in `../thud/`.
Load the `moloch` skill before changing a deck's layout, fonts or colors.

- Moloch v2.2.0 is vendored in `moloch/`, built from github.com/jolars/moloch, and is
  not installed system-wide. `latexmkrc` puts it on `TEXINPUTS` and selects LuaLaTeX,
  so build from this folder with `make` (`make force`, `make clean`).
- To update Moloch, rebuild the `.sty` files from the upstream source (the skill's
  gotchas.md has the command) and replace `moloch/*.sty`.
- Fonts: Fira Sans and Fira Mono through `fontspec`.
- Accent color: the thesis `bindingcolor`, `RGB 0,90,156`, set with `\molochcolors`.
- Figures come from `../thud/images/` via `\graphicspath`. They are drawn for an A4
  page, so their labels read small on a 16:9 slide.
- `slides.tex` is the deck; it still holds the layout demo with placeholder text.
