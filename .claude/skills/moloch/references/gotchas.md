# Gotchas

## Version in use

These references are for v2.2.0. An older Moloch (TeX Live 2025 ships v0.6.0) has none
of the v2 features, and fails with `Undefined control sequence \molochcolors`, or
`I do not know the key` for `titlepage`, `colortheme`, `standout=` values,
`frametitle margin ...`, `sectionpagenumbering`. The log names the version that loaded:
`grep "Moloch theme" <deck>.log`.

Building v2.2.0 from the source (github.com/jolars/moloch, `src/`): run
`tex beamerthememoloch.ins` in a copy of `src/*.dtx` and `src/*.ins`, and put the
resulting `.sty` files on `TEXINPUTS`.

## Where the upstream docs are wrong for v2.2.0

- `standoutnumberformat` (customization docs) does not exist; the text format of
  standout frames is `titleformat plain`.
- The `\molochcolors{theme=...}` docs list three themes; the source has five
  (`paper` and `catppuccin` too).
- The inner theme implementation page shows `standout` without values; the source
  supports `standout=alerted` and `standout=example`.

## Options in `\usetheme[...]`

LaTeX strips spaces from package options, so `frametitle margin top=1cm` reaches Moloch
as `frametitlemargintop`. Moloch defines a space-free twin for its multi-word keys,
so both spellings work in `\usetheme` and `\molochset`. The one without a twin is
`titleformat plain`: in `\usetheme` it fails with
`I do not know the key '/titleformatplain'`, so set it with `\molochset` after
`\usetheme`.

## Colors reset by a switch

`\molochcolors{theme=...}` and `variant=` rebuild the palette and drop raw
`\setbeamercolor` changes to the colors Moloch manages. See colors.md.

## Standout frames and labels

`standout` opens a group, so a `label` written after it stays local and gives
`label undefined`. Write `\begin{frame}[label=end, standout]`.

## Deprecated, with their replacements

- `numbering=none|counter|fraction`: the `page number in head/foot` template
  (layout.md).
- `background=dark`: `\molochcolors{variant=dark}`.
- `\usecolortheme{moloch-tomorrow}`, `\usecolortheme{moloch-highcontrast}`:
  `\molochcolors{theme=tomorrow}`, `theme=highcontrast`.
- `\mreducelistspacing`: `\vspace{-\topsep}`.

## Builds

- The progress bars need the total frame count from the `.aux` file, so they are wrong
  after one LuaLaTeX pass; latexmk reruns until they settle.
- Progress bars and the title separator are already excluded from TikZ
  externalization and from `memoize`.
- Speaker notes on a second screen (`\setbeameroption{show notes on second screen}`)
  turn slide text white after the first section page under XeLaTeX only; the LuaLaTeX
  build here is unaffected.
