# Layout

Options go in `\usetheme[key=value]{moloch}` for the whole deck, or `\molochset{key=value}`
to change them from that point on (between frames). Values marked *default* are what you
get without setting the key.

## Slide size

- 16:9 is a Beamer class option, unknown to Moloch: `\documentclass[aspectratio=169]{beamer}`.
  The paper is 16 cm x 9 cm, and with Beamer's 1 cm side margins `\textwidth` is 14 cm.
  Other values: `43` (default, 12.8 x 9.6), `1610`, `149`, `141`, `54`, `32`, `2013`.
- Side margins: `\setbeamersize{text margin left=0.8cm, text margin right=0.8cm}`.
- Size figures to the box, never by width alone, so tall plots cannot overflow:
  `\includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{...}`.
  Scaling a figure scales its labels; a plot drawn for an A4 page reads small on a
  slide, and the fix is redrawing it at slide size.

## Title page

`titlepage = moloch | plain | split` (*default* `moloch`)

- `moloch`: left aligned, golden-ratio vertical spacing; title graphic on top, then
  title, subtitle, a horizontal separator rule, author, institute, date.
- `plain`: Beamer's centered layout, title graphic at the bottom.
- `split`: two columns (45% each) with a vertical rule between them: title and subtitle on
  the left; author, institute, date and title graphic on the right. Sets the `date`
  font to `\footnotesize`.

Fields are `\title`, `\subtitle`, `\author`, `\institute`, `\date`, `\titlegraphic`; an
empty field is skipped, so `\date{}` removes the date. `\thanks{...}` in a field adds a
footnote on the title page. `\maketitle` outside a frame builds its own frame, `plain`
and unnumbered.

`titleseparator linewidth = <dim>` (*default* `0.5pt`, `0pt` hides it) sets the rule
under the title (`moloch`) or between the columns (`split`). Its color is
`\molochcolors{title separator=...}`.

To restyle one part, override its template after the `titlepage` option (the option
resets all of them): `title page`, `title`, `subtitle`, `author`, `institute`, `date`,
`title graphic`, `title separator` (horizontal), `title separator vertical` (split).
Each has the variants `[moloch]`, `[plain]`, `[split]`, e.g.
`\setbeamertemplate{title graphic}[plain]`, or a body of your own:
`\setbeamertemplate{institute}{\small\insertinstitute\par}`.

## Section and subsection pages

- `sectionpage = none | simple | progressbar` (*default* `progressbar`): a frame inserted
  by every `\section`. `simple` is the centered title; `progressbar` is the title over
  a thin bar showing how far into the deck this section starts.
- `subsectionpage = none | simple | progressbar` (*default* `none`): the same for
  `\subsection`. Turning it on restyles section pages to match; with both on they can
  appear back to back, so pair it with `sectionpage=none` if that happens.
- `sectionpagenumbering = none | show` (*default* `none`): `show` counts section pages
  in the frame numbers (they never print a number themselves).
- `\section` goes between frames. Templates: `section page` (`[simple]`,
  `[progressbar]`), `progress bar in section page`. Font and color: `section title`.

## Frame titles

- The title is a full-width bar in color `frametitle` (fg and bg, see colors.md).
- `frametitle margin top | bottom | left | right = <dim>` pads the text inside the bar.
  *Defaults* come from the strut height: top and bottom `\ht\strutbox`, left and right
  0.85 of it.
- Subtitle: `\begin{frame}{Title}{Subtitle}` or `\framesubtitle{...}`; font
  `framesubtitle` (*default* `\small`).
- Case: `titleformat frame`, in fonts.md.
- `allowframebreaks` frames get `i`, `ii`, ... after the title (template
  `frametitle continuation`).

## Footer and frame numbers

The footline (template `footline`, variant `[plain]`) has the template `frame footer` on
the left, empty by default, and the template `page number in head/foot` on the right.

- Footer text on every frame: `\setbeamertemplate{frame footer}{Jane Doe, Defence}`.
- Number style, a Beamer template:
  `\setbeamertemplate{page number in head/foot}[framenumber]` (*default*, `7`),
  `[totalframenumber]` (`7 / 20`), `[appendixframenumber]` (`7 / 20` in the main part,
  backup slides counted separately after `\appendix`), or `{}` for no number.
- One frame without footer: `\begin{frame}[plain]`. Not counted:
  `\begin{frame}[noframenumbering]`.

## Progress bar

- `progressbar = none | head | frametitle | foot` (*default* `none`): a bar across every
  frame, at the top of the slide, under the frame title, or at the bottom.
- `progressbar linewidth = <dim>` (*default* `1pt`) sets both this bar and the section
  page bar.
- Colors: `progressbar fg` and `progressbar bg` in colors.md. Templates:
  `progress bar in head/foot`, `progress bar in section page`.

## Standout frames

- `\begin{frame}[standout]`: centered large text (font `standout`: `\Large`, bold) on an
  inverted background. `standout=alerted` uses the alert palette (dark orange in the
  default theme), `standout=example` the example palette (dark teal).
- `standoutnumbering = none | hide | show` (*default* `none`): `none` leaves standout
  frames out of the count; `hide` counts them without printing the footer; `show`
  counts and prints it.
- Case of the text: `titleformat plain`, in fonts.md.
- Write `label=...` before `standout` in the options (gotchas.md).

## Vertical placement

Beamer centers frame content vertically by default; Moloch redefines the `c` option so
the centering is exact (Beamer's sits too high). `\begin{frame}[t]` pins content to the
top; the class option `t` does it for every frame.

## Lists and captions

- `itemize` bullets are drawn shapes: circle, square, triangle for levels 1 to 3.
  Templates `itemize item`, `itemize subitem`, `itemize subsubitem`; replace with
  e.g. `\setbeamertemplate{itemize item}{\textbullet}`, keeping the color with
  `\usebeamercolor[fg]{itemize item}`. `enumerate` keeps Beamer's templates.
- Captions are `[numbered]` with `: ` after the label;
  `\setbeamertemplate{caption}{\insertcaption}` drops the "Figure n" prefix.

## Blocks

`block`, `alertblock` and `exampleblock` have no background by default; `block=fill`
adds one. The key and its colors are in colors.md.
