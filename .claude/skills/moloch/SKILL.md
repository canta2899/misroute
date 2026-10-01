---
name: moloch
description: Moloch Beamer theme reference (v2.2.0). Use when writing or changing a Beamer deck that uses moloch: layout, fonts, colors, title or section pages, footer, progress bar, standout frames.
---

# Moloch

Moloch is a Beamer theme, the maintained fork of Metropolis. These references describe
**v2.2.0**. Its color system (v2.0) replaced the Metropolis one, so Metropolis answers
and pre-2.0 Moloch docs are suspect until checked against the source.

## Reference

Read the file for the request before writing any LaTeX:

| Request | File |
| --- | --- |
| slide size, margins, title page, section pages, frame titles, footer, frame numbers, progress bar, standout frames, vertical centering, bullets, figures | [layout.md](references/layout.md) |
| color themes, light and dark, accent colors, block colors | [colors.md](references/colors.md) |
| font families, sizes, weights, small caps, math font | [fonts.md](references/fonts.md) |
| anything that breaks, or behaves unlike the docs | [gotchas.md](references/gotchas.md) |

A key or template missing from these files is still not a guess: `grep` it in the
theme's `.sty` files (`kpsewhich beamerthememoloch.sty` finds the ones in use), where
every option is a `pgfkeys` entry under `/moloch/inner`, `/moloch/outer`,
`/moloch/color`, `/moloch/colors` or `/moloch/font`. Beamer's own templates, colors
and fonts are in `texdoc beamer`. On conflict, the `.sty` wins over any doc.

## Making a change

1. Pick the highest rung that does the job, and stop there:
   1. a Moloch option, in `\usetheme[...]{moloch}` for the whole deck or `\molochset{...}`
      from that point on
   2. `\molochcolors{...}` for any color it has a key for
   3. a Beamer override: `\setbeamertemplate`, `\setbeamerfont`, `\setbeamercolor`,
      `\setbeamersize`
   4. plain LaTeX inside one frame, for a one-off
2. Build with `latexmk`. Done when the log has no `!` line and no pgfkeys
   `I do not know the key` error.
3. Render each changed page and look at it:
   `pdftoppm -png -r 80 -f N -l N <deck>.pdf <dir>/p`. Done when the image shows the
   change and nothing else moved.
