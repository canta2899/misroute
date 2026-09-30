import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd

    from analysis import plots
    from analysis import probe as P

    pd.set_option("display.width", 170, "display.max_columns", 50)
    return P, mo, pd, plots


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Failure analysis: probe vs text baselines

    The probe (at its peak layer) and each baseline use the same scenario-grouped folds, so they
    can be compared sample by sample. Each sample falls in one cell:

    - `both`: both right
    - `probe_only`: only the probe is right
    - `baseline_only`: only the baseline is right
    - `neither`: both wrong
    """)
    return


@app.cell
def _(P):
    results = {f"{m} {k}": P.result(m, k) for m, k in P.runs()}
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    gpt-oss-20b is left out (`P.pooled_runs`), as in `nb/probing.py`.
    """)
    return


@app.cell
def _(P, results):
    pooled = P.pooled_runs(results)
    list(pooled)
    return (pooled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Error dependence

    Probe accuracy on the samples the baseline gets right against the samples it gets wrong,
    on `natural` and `adversarial`. If the probe reads the same thing as the baseline, `gap` is
    far below 0 and `kappa` is near 1. If it fails independently, both are near 0.

    - `kappa`: Cohen's kappa on right/wrong
    - `kappa_ceiling`: highest kappa the two accuracies allow
    - `kappa_ratio`: `kappa / kappa_ceiling`

    The coordinator is left out: it is not trained on the same labels and folds.
    `plain` is left out: the baselines make almost no errors there.

    Per run, then the mean per kind with the range of `gap` and `kappa`.
    """)
    return


@app.cell
def _(P, pooled):
    P.dependence_table(pooled).round(3)
    return


@app.cell
def _(P, pooled):
    dependence = P.dependence_summary(pooled)
    dependence.round(3)
    return (dependence,)


@app.cell
def _(dependence, plots):
    plots.plot_dependence(dependence)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Adversarial pairs

    The two samples of a pair share most of their text and have different gold experts.
    Per router: pairs with `both`, `one` or `neither` sample right, and `collapsed` (same expert
    for both samples). The baselines do not depend on the model, so their range is 0.
    """)
    return


@app.cell
def _(P, pooled):
    P.pair_table(pooled).round(3)
    return


@app.cell
def _(P, pooled):
    pairs = P.pair_summary(pooled)
    pairs.round(3)
    return (pairs,)


@app.cell
def _(pairs, plots):
    plots.plot_pairs(pairs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Decoy predictions

    Per tier and router: share of samples wrong (`error`), wrong on the decoy (`decoy`), and
    wrong on another expert (`other`). `probe_below`: runs where the probe predicts the decoy
    less often than that router.

    `decoy_share` (share of the errors on the decoy) is only a diagnostic: random errors lower it.
    """)
    return


@app.cell
def _(P, pooled):
    P.error_direction_table(pooled).round(3)
    return


@app.cell
def _(P, pooled):
    decoy = P.error_direction_summary(pooled)
    decoy.round(3)
    return (decoy,)


@app.cell
def _(decoy, plots):
    plots.plot_decoy(decoy)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Samples

    One run at a time. Filter `cell == "baseline_only"` to see what the probe gets wrong and
    the baseline gets right.
    """)
    return


@app.cell
def _(P, mo, results):
    run = mo.ui.dropdown(list(results), value=list(results)[0], label="run")
    cell = mo.ui.dropdown(["all", *P.CELLS], value="baseline_only", label="cell")
    baseline = mo.ui.dropdown(list(P.BASELINE_NAMES), value="lexical", label="baseline")
    mo.hstack([run, cell, baseline], justify="start", gap=1)
    return baseline, cell, run


@app.cell
def _(P, baseline, cell, results, run):
    _c = P.cells(results[run.value], baseline=baseline.value)
    _c if cell.value == "all" else _c[_c["cell"] == cell.value]
    return


if __name__ == "__main__":
    app.run()
