# Thesis

LaTeX sources of the thesis, built with the UniUD `thud` class. Run every command from
this folder.

## Commands

| Command | What it does |
| --- | --- |
| `make` | builds `thesis.pdf`, only if a source, the bibliography or an image changed |
| `make force` | builds `thesis.pdf` even if nothing changed |
| `make clean` | removes the LaTeX build files |
| `make figures` | rewrites the figures of chapters 6-8 in `images/models/` and `images/plots/` |
| `make tables` | prints the numbers of chapters 7-8 |
| `make front-new` | the title page in black on white, for the hard cover |
| `make front-old` | the same, in the style of the bachelor thesis cover, with the seal |
| `make front` | both title pages |

`make figures` and `make tables` read the merged parquets in `../nb/data/` (see the
[main README](../README.md)). The figures are committed, so `make` works without them.

## Layout

- `thesis.tex`: preamble, title page data and the chapter list
- `thud.cls`, `book.thud.tex`: the class and its page layout
- `chapters/<n>/entrypoint.tex`: one folder per chapter, plus `appendix/`
- `images/`: figures; `models/` and `plots/` come from `make figures`
- `banners/`: university banners for the title page
- `scripts/`: the figure and table scripts
- `thud.bib`: bibliography

## License

[CC BY-NC-SA 4.0](LICENSE)
