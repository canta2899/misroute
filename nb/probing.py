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
    # Probing the hidden states
    """)
    return


@app.cell
def _(P, mo):
    models = sorted({m for m, _ in P.runs()})
    model = mo.ui.dropdown(models, value=models[0], label="model")
    kind = mo.ui.dropdown(
        ["single_turn", "multi_turn"], value="single_turn", label="kind"
    )
    mo.hstack([model, kind], justify="start", gap=1)
    return kind, model


@app.cell
def _(P, kind, model):
    d = P.result(model.value, kind.value)
    peak = P.peak_layer(d)
    print(
        f"{d['label']} | {len(d['y'])} samples | {d['n_layers']} layers x {d['hidden_dim']} dims "
        f"| {d['n_unrouted']} unrouted dropped | optimal layer {peak} (best on the whole dataset)"
    )
    return d, peak


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Layer sweep
    """)
    return


@app.cell
def _(d, plots):
    plots.plot_layers(d)
    return


@app.cell
def _(d):
    d["by_layer"].round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Routers
    """)
    return


@app.cell
def _(d, plots):
    plots.plot_compare(d)
    return


@app.cell
def _(P, d):
    P.compare(d).round(3)
    return


@app.cell
def _(P, d):
    P.significance(d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confusion matrix: model vs probe at its peak layer
    """)
    return


@app.cell
def _(d, peak, plots):
    plots.plot_confusions(d, peak)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ROC (one vs rest)
    """)
    return


@app.cell
def _(P, d, kind, model, peak, plots):
    plots.plot_roc(d, *P.probabilities(model.value, kind.value, peak))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Every run at once

    One row per run: the model, the text baselines, the probe, and the probe's margin over the
    model (whole run and `adversarial`). The baselines read only the csv, so they repeat within a kind.
    """)
    return


@app.cell
def _(P):
    results = {f"{m} {k}": P.result(m, k) for m, k in P.runs()}
    return (results,)


@app.cell
def _(P, results):
    overview = P.overview(results)
    overview[[*P.ROUTERS, "margin"]].round(3)
    return (overview,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Probe vs model: McNemar

    Holm over every run at the last prompt token, two tests per run (whole run, `adversarial`).
    The reasoning-token runs are a separate family.
    """)
    return


@app.cell
def _(P):
    P.significance_family("prompt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### gpt-oss-20b is left out from here

    Its probe is less accurate than the model on both kinds (see the last section).
    Every view below, and every view in `nb/failure.py`, uses `P.pooled_runs`: 7 models, 14 runs.
    """)
    return


@app.cell
def _(P, results):
    pooled = P.pooled_runs(results)
    P.overview(pooled)[[*P.ROUTERS, "margin"]].round(3)
    return (pooled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Accuracy across tiers

    The drop from `plain` to `adversarial`, for each router. The probe uses the whole-run peak
    layer on every tier. `drop`: mean drop per kind. `probe_ahead`: runs where the probe drops less.
    """)
    return


@app.cell
def _(P, pooled):
    P.tier_summary(pooled).round(3)
    return


@app.cell
def _(P, pooled):
    P.tier_drop(pooled).round(3)
    return


@app.cell
def _(plots, pooled):
    plots.plot_tiers(pooled)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-turn drop

    `ratio` = |delta_probe| / |delta_model|. Above 1, the probe loses more accuracy than the model.
    """)
    return


@app.cell
def _(P, pooled):
    cost = P.multi_turn_cost(pooled)
    cost.round(3)
    return (cost,)


@app.cell
def _(cost, plots):
    plots.plot_multi_turn_cost(cost)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The probe as an error flag

    A sample is flagged when the probe and the model disagree.

    - `precision`: share of flagged samples the model gets wrong
    - `recall`: share of the model's errors that are flagged
    - `false_alarm`: share of the model's correct decisions that are flagged
    - `flagged_share`: share of all samples flagged, the traffic a fallback reviews
      (`base_rate * recall + (1 - base_rate) * false_alarm`)
    - `residual_error`: share of all samples the model gets wrong and the flag misses,
      the error left if the fallback fixes every flagged misroute (`base_rate * (1 - recall)`)

    `flag_correlations` relates each rate to `margin` (probe accuracy minus model accuracy).
    """)
    return


@app.cell
def _(P, pooled):
    P.across_runs(P.flags, results=pooled).round(3)
    return


@app.cell
def _(P, pooled):
    P.across_runs(P.flags, results=pooled)[
        ["base_rate", "recall", "flagged_share", "residual_error"]
    ].astype(float).agg(["min", "median", "max"]).round(3)
    return


@app.cell
def _(P, pooled):
    P.flag_correlations(pooled).round(3)
    return


@app.cell
def _(plots, pooled):
    plots.plot_flag_rates(pooled)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Per-model tables (chapter 7), one row per run. McNemar: `b` model right / probe wrong,
    `c` model wrong / probe right. `p_holm` is over every run at the last prompt token.
    """)
    return


@app.cell
def _(P):
    P.run_tables("prompt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Depth of the peak layer

    Layer counts differ, so the x axis is `layer / last layer`. One point per run, at its peak
    layer. 14 runs, gpt-oss-20b left out.
    """)
    return


@app.cell
def _(pooled):
    depth_runs = pooled
    return (depth_runs,)


@app.cell
def _(depth_runs, plots):
    plots.plot_depth_peaks(depth_runs)
    return


@app.cell
def _(P, depth_runs):
    P.depth_summary(depth_runs).round(3)
    return


@app.cell
def _(P, depth_runs):
    P.peak_depths(depth_runs)[["n_layers", "peak_layer", "peak_depth"]].round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## gpt-oss-20b: last reasoning token

    gpt-oss-20b always reasons, so it may decide after the last prompt token. These runs read the
    hidden states at the last reasoning token (single-turn only). Folds, probe and baselines are the same.
    """)
    return


@app.cell
def _(P, mo):
    reasoning_runs = P.reasoning_runs()
    reasoning_run = mo.ui.dropdown(
        {f"{m} {k}": (m, k) for m, k in reasoning_runs},
        value=f"{reasoning_runs[0][0]} {reasoning_runs[0][1]}",
        label="run",
    )
    reasoning_run
    return (reasoning_run,)


@app.cell
def _(P, reasoning_run):
    d_reasoning = P.result(*reasoning_run.value, site="reasoning")
    return (d_reasoning,)


@app.cell
def _(d_reasoning, plots):
    plots.plot_layers(d_reasoning)
    return


@app.cell
def _(d_reasoning, plots):
    plots.plot_compare(d_reasoning)
    return


@app.cell
def _(P, d_reasoning):
    P.compare(d_reasoning).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Last prompt token against last reasoning token. The reasoning row has its own Holm family
    (`P.significance_family("reasoning")`).
    """)
    return


@app.cell
def _(P, reasoning_run):
    P.site_table(*reasoning_run.value).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At each site's peak layer: how often the probe agrees with the model's choice, and a second
    probe trained on the model's choice (`decision_acc`: scored against that choice;
    `decision_vs_gold`: scored against gold).
    """)
    return


@app.cell
def _(P, reasoning_run):
    P.decision_readout(*reasoning_run.value).round(3)
    return


if __name__ == "__main__":
    app.run()
