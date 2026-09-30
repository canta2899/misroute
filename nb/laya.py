import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd

    from analysis import laya as L
    from analysis import plots
    from analysis import probe as P

    pd.set_option("display.width", 170, "display.max_columns", 50)
    return L, P, mo, pd, plots


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Laya: a decision model against the coordinators and their probes

    Laya is a non-generative decision model: one encoder forward pass scores a typed
    `choice` question over a state, with no text generated. The public checkpoints are
    zero-shot on our labels. Each checkpoint (`english`, `multilingual`, `typed-decisions`) runs on a grid of prompt
    variants, `options | instructions | state`:

    - `options`: `handles` (`<key>: <handles>`), `tool` (the coordinator's tool description
      without the example), `tool_full` (the coordinator's tool description verbatim; laya
      cuts every option at 48 tokens, so the example is cut), `name` (`<name>: <handles>`)
    - `instructions`: `task` (one plain question), `rule` (the resolution rule in short
      form), `system` (the coordinator's system prompt verbatim)
    - `state`: `turns` (the json turn list), `text` (`user: ... assistant: ...`)

    The results come from `src/simulate/laya_app.py`, in `data/sim/laya/<kind>.csv`; the
    exact texts are in `data/sim/laya/variants.json`.

    `laya-multilingual-ft` is `multilingual` fine-tuned on our labels with RLCD
    (Reinforcement Learning for Calibrated Decisions: a policy gradient whose reward is a
    proper scoring rule, plus cross-entropy), then given a fitted temperature. It comes
    from `src/simulate/laya_finetune_app.py`, in `data/sim/laya/finetune/<kind>.csv`,
    `single_turn` only.

    Every accuracy is averaged over the same scenario-grouped folds as `nb/probing.py`.
    gpt-oss-20b is left out (`P.pooled_runs`), as in `nb/probing.py`.
    """)
    return


@app.cell
def _(L, mo):
    kinds = L.kinds()
    kind = mo.ui.dropdown(kinds, value=kinds[0], label="kind")
    kind
    return (kind,)


@app.cell
def _(L, P, kind):
    t = L.load(kind.value)
    ft = L.load_finetuned(kind.value)
    results = P.pooled_runs(
        {f"{m} {k}": P.result(m, k) for m, k in P.runs() if k == kind.value}
    )
    print(
        f"{kind.value} | {t['id'].nunique()} samples | checkpoints {L.checkpoints(t)} "
        f"| {t['variant'].nunique()} variants | {int(t['over_budget'].sum())} rows over budget "
        f"| runs {[d['model'] for d in results.values()]} "
        f"| fine-tuned {'yes' if ft is not None else 'no'}"
    )
    return ft, results, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prompt variants

    Accuracy of every variant, and the share of the samples sent to the expert predicted
    most (`gold` is balanced, so a router that reads the text sits near 0.2). Chance is 0.2.
    """)
    return


@app.cell
def _(L, plots, t):
    plots.plot_laya_variants(L.variant_grid(t, "acc"), L.variant_grid(t, "top_share"))
    return


@app.cell
def _(L, t):
    L.variant_table(t).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Accuracy per level of each factor, averaged over the other two factors.

    The expert predicted most is `scale_arch` in almost every cell. `scale_arch` is also
    the second option in every variant, since the options are always in the same order.
    This grid cannot tell whether the model prefers that position or that content.
    """)
    return


@app.cell
def _(L, t):
    L.factor_effects(t).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The chosen variant

    For each held-out fold, each zero-shot checkpoint uses the variant with the highest
    accuracy on the other four folds. The fine-tuned checkpoint has one variant. Every
    section below reads these out-of-fold choices: one prediction per (checkpoint, sample).
    """)
    return


@app.cell
def _(L, ft, t):
    s = L.select_all(t, ft)
    L.selection(s)
    return (s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fine-tuned laya

    `multilingual` with the `handles|task|turns` variant, trained once per held-out fold
    on the other four folds. Two scenarios of those four folds are held out of training,
    and the temperature is fitted on them. The training loop is the one in laya's
    fine-tuning notebook, run on one GPU for 4 epochs, with the option order shuffled for
    every sample in every epoch. Per held-out fold: accuracy, mean confidence and the
    fitted temperature (above 1: the trained model was overconfident).
    """)
    return


@app.cell
def _(L, ft, mo):
    mo.stop(ft is None, mo.md("No fine-tuned run for this kind."))
    L.finetune_folds(ft).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Accuracy

    Laya's checkpoints beside the text baselines, the coordinators and their probes (at the
    peak layer), per tier. The baselines and `laya-multilingual-ft` are trained on the
    other folds. The other laya checkpoints and the coordinators are zero-shot.
    """)
    return


@app.cell
def _(L, results, s):
    table = L.compare(s, results)
    return (table,)


@app.cell
def _(kind, plots, table):
    plots.plot_laya_compare(table, kind.value)
    return


@app.cell
def _(L, results, s):
    L.summary(s, results).round(3)
    return


@app.cell
def _(L, results, s):
    L.compare_wide(s, results).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Where laya routes

    Share of the samples each checkpoint sends to each expert, with the chosen variants.
    """)
    return


@app.cell
def _(L, plots, s):
    plots.plot_laya_predictions(L.prediction_shares(s))
    return


@app.cell
def _(plots, s):
    plots.plot_laya_confusions(s)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Adversarial pairs

    The two samples of a pair share most of their text and have different gold experts.
    Per router: pairs with `both`, `one` or `neither` sample right, and `collapsed` (same
    expert for both). Coordinator, probe and baselines are the mean over runs, as in
    `nb/failure.py`.
    """)
    return


@app.cell
def _(L, results, s):
    L.pair_compare(s, results).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Decoy predictions

    Per tier and router: share wrong (`error`), wrong on the decoy (`decoy`), wrong on
    another expert (`other`).
    """)
    return


@app.cell
def _(L, results, s):
    L.error_direction_compare(s, results).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Sample by sample

    Laya's accuracy on the samples a coordinator (or its probe) gets right against the
    samples it gets wrong. `gap` near 0 and `kappa` near 0: laya fails independently of
    that router. `laya_only`: share of samples only laya gets right.
    Mean over runs, with the range of `gap`.
    """)
    return


@app.cell
def _(L, results, s):
    L.dependence_summary(s, results).round(3)
    return


@app.cell
def _(L, results, s):
    L.dependence(s, results).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Confidence

    `answer_confidence` is the probability of the predicted expert. The zero-shot
    checkpoints ship no temperatures fitted to our data, so their confidence is raw.
    `laya-multilingual-ft` applies the temperature fitted on its calibration slice. `ece`: expected calibration
    error over 10 bins. `auroc`: how well confidence separates right from wrong answers
    (0.5 = not at all).
    """)
    return


@app.cell
def _(L, plots, s):
    plots.plot_laya_reliability(L.reliability(s))
    return


@app.cell
def _(L, s):
    L.calibration(s).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Samples

    One checkpoint and variant at a time. `laya-multilingual-ft` has only
    `handles|task|turns`.
    """)
    return


@app.cell
def _(ft, pd, t):
    rows = pd.concat([t, ft], ignore_index=True) if ft is not None else t
    return (rows,)


@app.cell
def _(L, mo, rows, t):
    checkpoint = mo.ui.dropdown(
        L.checkpoints(rows), value=L.checkpoints(rows)[0], label="checkpoint"
    )
    variant = mo.ui.dropdown(
        list(L.variant_grid(t).index), value=L.variant_grid(t).index[0], label="variant"
    )
    outcome = mo.ui.dropdown(["all", "right", "wrong"], value="wrong", label="outcome")
    mo.hstack([checkpoint, variant, outcome], justify="start", gap=1)
    return checkpoint, outcome, variant


@app.cell
def _(P, checkpoint, kind, outcome, rows, variant):
    _s = rows[
        (rows["checkpoint"] == checkpoint.value) & (rows["variant"] == variant.value)
    ]
    if outcome.value != "all":
        _s = _s[_s["correct"] == (outcome.value == "right")]
    _text = P.dataset(kind.value).set_index("id")["text"]
    _s.assign(text=_s["id"].map(_text))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
