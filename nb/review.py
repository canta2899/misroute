import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json

    import marimo as mo

    from dataset import csvexport
    from dataset import reviewlib as R

    return R, csvexport, json, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Review

    One `data/<run>/<kind>` folder at a time. Click a row to open it; edits live
    in memory until **save**, which writes `samples.edited.jsonl` — never
    `samples.jsonl`. Swapping the file in is your call.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # bumped by every edit, read by every view: the rows are mutated in place,
    # so this counter is what tells marimo to re-render them
    get_edits, set_edits = mo.state(0)
    # the row under review. It is also what opens the modal: clearing it closes
    # the modal and clears the table's selection, so the same row can be
    # reopened right after
    get_selected, set_selected = mo.state(None)
    # bumped by the reload button - the loader has nothing else to depend on
    get_loads, set_loads = mo.state(0)
    return get_edits, get_loads, get_selected, set_edits, set_loads, set_selected


@app.cell(hide_code=True)
def _(R, mo, set_loads):
    # `run/kind`, not the absolute path: the row is narrow and every entry
    # shares the same data/ prefix anyway
    _choices = {f"{p.parent.name}/{p.name}": p for p in R.folders()}
    folder = mo.ui.dropdown(
        _choices, value=next(iter(_choices), None), label="folder", searchable=True
    )
    reload_button = mo.ui.button(
        label="reload",
        tooltip="re-open the folder from disk, dropping unsaved edits",
        on_change=lambda _: set_loads(lambda n: n + 1),
    )
    flagged_only = mo.ui.checkbox(label="flagged")
    mo.hstack(
        [folder, flagged_only, reload_button], justify="start", align="center", gap=2
    )
    return flagged_only, folder


@app.cell
def _(R, folder, get_loads, mo):
    get_loads()  # reloading is re-opening the folder, dropping unsaved edits
    mo.stop(folder.value is None, mo.md("**no `samples.jsonl` under `data/`**"))
    session = R.Session.open(folder.value)
    return (session,)


@app.cell(hide_code=True)
def _(R, flagged_only, get_edits, get_selected, mo, session):
    get_edits()
    _frame = R.table(session.rows, flagged_only.value)
    _ids = list(_frame["id"])
    _keep = [_ids.index(get_selected())] if get_selected() in _ids else []
    samples = mo.ui.table(
        _frame,
        selection="single",
        initial_selection=_keep,
        page_size=20,
        max_height=560,
        show_column_summaries=False,
    )
    mo.vstack(
        [
            mo.md(R.headline_view(session.folder, session.rows, len(session.dirty))),
            samples,
        ]
    )
    return (samples,)


@app.cell(hide_code=True)
def _(get_selected, samples, set_selected):
    # the table is rebuilt on every edit, which clears its selection - only the
    # human picking a row should open the modal
    _picked = samples.value
    if len(_picked) and _picked.iloc[0]["id"] != get_selected():
        set_selected(_picked.iloc[0]["id"])
    return


@app.cell(hide_code=True)
def _(apply_button, decision, mo, note, sample, session, set_edits, set_selected, turn_editors):
    mo.stop(not apply_button.value)
    _touched = session.review(
        sample, list(turn_editors.value), decision.value, note.value
    )
    set_edits(lambda v: v + 1)
    set_selected(None)  # applying closes the modal, back to the table
    mo.md(
        f"applied to `{'`, `'.join(sorted(_touched))}` (not written yet)"
        + (
            " - pulled the pair partner along, pairs regenerate as a unit"
            if len(_touched) > 1
            else ""
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    def modal(body):
        """One sample, over the table. marimo has no dialog primitive, but ui
        elements keep working inside raw html, so this is a fixed overlay."""
        return mo.Html(
            '<div style="position:fixed;inset:0;z-index:1000;'
            'background:rgba(0,0,0,0.5);display:flex;justify-content:center;'
            'align-items:flex-start;padding:2.5rem 1rem;overflow:auto">'
            '<div style="background:var(--background);color:var(--foreground);'
            "border:1px solid var(--border);border-radius:10px;"
            "box-shadow:0 12px 48px rgba(0,0,0,0.45);width:min(920px,100%);"
            'max-height:88vh;overflow:auto;padding:1.5rem">'
            f"{mo.as_html(body)}</div></div>"
        )

    return (modal,)


@app.cell(hide_code=True)
def _(mo, set_selected):
    # lives outside the modal cell on purpose: a cell that both reads a state
    # and sets it does not re-run itself, so a close button built in there
    # would clear the selection without ever closing the modal
    close_button = mo.ui.button(label="close", on_change=lambda _: set_selected(None))
    return (close_button,)


@app.cell(hide_code=True)
def _(R, close_button, get_edits, get_selected, mo, modal, session):
    get_edits()
    sample = session.sample(get_selected())
    mo.stop(sample is None, mo.md(""))

    turn_editors = mo.ui.array(
        [
            mo.ui.text_area(
                value=turn["content"],
                label=f"**{turn['role']}**",
                rows=6,
                full_width=True,
            )
            for turn in sample["turns"]
        ]
    )
    decision = mo.ui.radio(
        list(R.REVIEW_CHOICES), value=R.default_decision(sample), inline=True
    )
    note = mo.ui.text(
        value=sample.get("review_note") or "", placeholder="note", full_width=True
    )
    apply_button = mo.ui.run_button(label="apply", kind="success")

    modal(
        mo.vstack(
            [
                mo.hstack(
                    [mo.md(R.header_view(sample)), close_button],
                    justify="space-between",
                    align="start",
                ),
                mo.md("#### conversation"),
                turn_editors,
                mo.md("#### checks & judge"),
                mo.md(R.checks_view(sample)),
                mo.md(R.judge_view(sample)),
                mo.md("#### your review decision"),
                mo.md(R.pair_view(sample, session.rows)),
                decision,
                mo.md(
                    "<sub>valid: keep as-is, disagreeing with the judge · "
                    "fixed: you hand-edited the turns above, keep it · "
                    "regenerate: agree with the judge, send it back. gold/decoy "
                    "can't be changed here - they set the dataset's balance; "
                    "rewrite the text or regenerate instead.</sub>"
                ),
                mo.hstack(
                    [note, apply_button], justify="start", widths=[1, 0], gap=1
                ),
                mo.accordion({"advanced: raw record": sample}),
            ]
        )
    )
    return apply_button, decision, note, sample, turn_editors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Save and export""")
    return


@app.cell(hide_code=True)
def _(mo):
    save_button = mo.ui.run_button(label="write samples.edited.jsonl", kind="success")
    save_button # type: ignore
    return (save_button,)


@app.cell(hide_code=True)
def _(mo, save_button, session, set_edits):
    mo.stop(not save_button.value)
    _message = session.save()
    set_edits(lambda v: v + 1)
    mo.md(f"`{_message}`")
    return


@app.cell(hide_code=True)
def _(get_edits, json, mo, session):
    get_edits()
    _lines = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in session.rows
    )
    mo.download(
        _lines.encode(),
        filename="samples.edited.jsonl",
        mimetype="application/jsonl",
        label="download the reviewed jsonl",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The csv exports the rows the table currently shows. Leave the column picker
    empty for every field.
    """)
    return


@app.cell(hide_code=True)
def _(csvexport, mo, session):
    field_of_label = {
        csvexport.field_label(c): c for c in csvexport.field_choices(session.rows)
    }
    columns = mo.ui.multiselect(list(field_of_label), label="columns", full_width=True)
    columns # type: ignore
    return columns, field_of_label


@app.cell(hide_code=True)
def _(R, columns, csvexport, field_of_label, flagged_only, get_edits, mo, session):
    get_edits()
    _shown = set(R.table(session.rows, flagged_only.value)["id"])
    _rows = [r for r in session.rows if r["id"] in _shown]
    _columns = [field_of_label[label] for label in columns.value] or None
    mo.vstack(
        [
            mo.md(f"{len(_rows)} rows"),
            mo.download(
                csvexport.frame(_rows, _columns).to_csv(index=False).encode(),
                filename=f"{session.folder.parent.name}-{session.folder.name}.csv",
                mimetype="text/csv",
                label="download csv",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
