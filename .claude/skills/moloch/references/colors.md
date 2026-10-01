# Colors

Moloch v2 keeps a **theme** (a palette) and a **variant** (`light` or `dark`), plus
per-variant overrides that survive every later theme or variant switch. Work through
`\molochcolors{...}` so overrides stay in that store.

## Theme and variant

| Theme | Look (light variant) |
| --- | --- |
| `default` | Metropolis: dark teal text and frame title, orange alerts and progress bar, teal examples, off-white background |
| `tomorrow` | near-black frame title, red alerts, blue examples, purple progress bar |
| `highcontrast` | black on white, Okabe-Ito accents (color-blind safe); the dark variant is white on black |
| `paper` | black on white, no frame title bar, colors from Few's *Show Me the Numbers* |
| `catppuccin` | pastel Latte (light) and Macchiato (dark), low contrast: weak on projectors |

- Whole deck: `\usetheme[colortheme=tomorrow, colortheme variant=dark]{moloch}`.
- Anywhere, between frames: `\molochcolors{theme=tomorrow, variant=dark}`. Switching
  the theme keeps the current variant, the `block` style and every override.
- `\usecolortheme{moloch-tomorrow}` and `\usecolortheme{moloch-highcontrast}` still load
  but print a deprecation warning; `background=dark` is the deprecated form of
  `variant=dark`.

## Overriding a color

`\molochcolors{<key>=<color>}` sets the key for the current variant;
`light/<key>=` and `dark/<key>=` set one variant whether it is active or not. Values are
any xcolor expression (`accent`, `accent!60!black`, `red`).

| Key | Colors |
| --- | --- |
| `normal text fg`, `normal text bg` | body text and slide background; title page text, author, date and block titles follow `normal text fg` |
| `frametitle fg`, `frametitle bg` | the frame title bar |
| `alerted text` | `\alert{}` and `alertblock` titles |
| `example text` | `exampleblock` titles and Beamer's `example` environment |
| `progressbar fg`, `progressbar bg` | every progress bar: head, foot, frame title, section page; in the light variant also the title separator, unless that is set |
| `title separator` | the rule on the title page |
| `standout fg`, `standout bg` | `[standout]` frames |
| `standout alerted fg`, `standout alerted bg` | `[standout=alerted]` frames |
| `standout example fg`, `standout example bg` | `[standout=example]` frames |

One accent for the whole deck:

```latex
\definecolor{accent}{RGB}{0,90,156}
\molochcolors{progressbar fg=accent, alerted text=accent, frametitle bg=accent}
```

## Beamer colors without a key

For a Beamer color `\molochcolors` has no key for, use
`\setbeamercolor{<name>}{fg=..., bg=...}` **after the last** `\molochcolors{theme=...}` or
`variant=` call: every switch re-runs the palette and overwrites raw `\setbeamercolor`
on colors Moloch manages (verified: a `\setbeamercolor{frametitle}` set before a
variant switch is lost; the same color through `\molochcolors` survives).

Beamer colors that show on Moloch slides: `section title`, `subsection title`,
`page number in head/foot`, `footnote`, `institute`, `thanks`, `itemize item`,
`progress bar in head/foot`, `progress bar in section page`, `block title`,
`block body`, `block title alerted`, `block title example`.

The palette's named colors (`mNormaltextFg`, `mAlertFg`, `mExampleFg`,
`mFrametitleBg`, `mProgressbarFg`, ...) are redefined by each theme; read them, never
set them. The Metropolis names `mDarkTeal`, `mLightBrown`, `mLightGreen`, `mDarkBrown`
are fixed.

## Blocks

`block = transparent | fill` (*default* `transparent`), as a theme option or in
`\molochset`. `fill` gives `block title` a background of `normal text.bg!80!fg` and the
body half that; the alert and example blocks reuse the same backgrounds and only change
the title text color. For a block background of your own, set `block body` and
`block title` with `\setbeamercolor` after `block=fill`.

## Other Beamer color themes

To use a stock color theme (`seahorse`, `crane`, ...), load the parts instead of the
whole theme, so Moloch's palette never loads:

```latex
\useoutertheme{moloch}
\useinnertheme{moloch}
\usefonttheme{moloch}
\usecolortheme{seahorse}
```

Moloch draws no title background, so themes with a white title foreground (`whale`) show
white on white on the title page.
