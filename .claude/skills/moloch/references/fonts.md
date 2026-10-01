# Fonts

Moloch's font theme sets **sizes and weights only**; the family is the deck's choice.
With no `fontspec`, Beamer falls back to Computer Modern Sans, which has no small caps.
Choosing a family means `fontspec`, so the deck builds with LuaLaTeX (or XeLaTeX).

## Families

Beamer typesets everything in the sans family, so `\setsansfont` is the one that shows.
Tested recipe, text and math:

```latex
\usefonttheme{professionalfonts}   % let unicode-math own the math font
\usetheme{moloch}
\usepackage{fontspec}
\usepackage{unicode-math}
\setsansfont{Fira Sans}[BoldFont={Fira Sans SemiBold}]
\setmonofont{Fira Mono}
\setmathfont{Fira Math}
```

Variants, both tested:

- Metropolis look, light body: `\setsansfont{Fira Sans Light}[BoldFont={Fira Sans}]`.
- Helvetica, matching a document set in `helvet`: `\setsansfont{TeX Gyre Heros}`.

Fonts shipped with TeX Live load by name: Fira Sans (Thin to Heavy), Fira Mono, Fira
Math, TeX Gyre Heros. Check any other with `kpsewhich <File>-Regular.otf` or
`fc-list | grep -i <name>`. A missing font is a fatal `fontspec Error: The font "..."
cannot be found`.

## Sizes and weights

The base size is a class option: `\documentclass[aspectratio=169,12pt]{beamer}`
(`8pt` to `20pt`, *default* `11pt`); every size below scales with it.

| Beamer font | Moloch sets |
| --- | --- |
| `title` | `\Large`, bold |
| `subtitle` | `\large` |
| `author`, `date` | `\small` |
| `section title` | `\Large`, bold |
| `frametitle` | `\large`, bold |
| `framesubtitle` | `\small` |
| `block title`, `block title alerted` | `\normalsize`, bold |
| `caption` | `\small` (`caption name` bold) |
| `description item` | bold |
| `standout` | `\Large`, bold |

Change one with `\setbeamerfont{frametitle}{size=\Large, series=\mdseries}`. Fonts left
out of the table (`institute`, `normal text`, `itemize/enumerate body`, `footnote`,
`page number in head/foot`) keep Beamer's defaults and take the same command.

## Title case

`titleformat = regular | smallcaps | allsmallcaps | allcaps` (*default* `regular`) sets
every title at once. One title only: `titleformat title`, `titleformat subtitle`,
`titleformat section`, `titleformat frame`, `titleformat plain` (the text of standout
frames). All work in `\molochset`; all but `titleformat plain` also work in
`\usetheme[...]` (gotchas.md).

- These keys set the `shape` of the matching Beamer font, so set the shape here and
  size or series with `\setbeamerfont`.
- `smallcaps` needs a font with small caps: Fira Sans has them (tested), Computer
  Modern Sans does not. A family without them logs `Font shape ... undefined` and
  prints upright.
- `allcaps` and `allsmallcaps` change letters only; digits, punctuation and math in a
  title stay as typed.
