import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve

    ROOT = Path(__file__).resolve().parents[1]
    FT_DIR = ROOT / "data" / "sim" / "modernbert"  # simulate/modernbert_app.py output
    CACHE = ROOT / "nb" / "data" / "cache"  # probe layer sweeps
    BASELINES = ROOT / "nb" / "data" / ".baselines.json"

    EXPERTS = [
        "time_to_market",
        "scale_arch",
        "resource_opt",
        "legacy_compat",
        "strict_compliance",
    ]
    MODELS = [  # size order, gpt-oss-20b left out as in chapter 8
        "gemma4-e2b-it",
        "qwen3-4b",
        "qwen3-8b",
        "gemma4-12b-it",
        "gemma4-26b-a4b-it",
        "llama3.3-70b",
        "qwen2.5-72b",
    ]
    TIERS = ["ALL", "plain", "natural", "adversarial"]
    N_FOLDS = 5
    return (
        BASELINES,
        CACHE,
        EXPERTS,
        FT_DIR,
        MODELS,
        N_FOLDS,
        TIERS,
        json,
        mo,
        np,
        pd,
        plt,
        roc_auc_score,
        roc_curve,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fine-tuned ModernBERT-large against the coordinators and their probes

    A quick look, self-contained on purpose. `modernbert-large-ft` is ModernBERT-large
    with a 5-way linear head, trained end to end with cross-entropy on the same
    scenario-grouped folds as the probes and the baselines
    (`src/simulate/modernbert_app.py`, output in `data/sim/modernbert/<kind>.csv`).

    Every accuracy is the mean over the 5 held-out folds, inside each tier. The probe
    is read at its optimal layer (the highest fold-mean accuracy). With more than one
    seed, the fine-tuned numbers are the mean over the seeds.
    """)
    return


@app.cell
def _(FT_DIR, mo):
    kinds = sorted(p.stem for p in FT_DIR.glob("*.csv"))
    mo.stop(not kinds, mo.md(f"No results yet in `{FT_DIR}`."))
    kind = mo.ui.dropdown(kinds, value=kinds[0], label="kind")
    kind
    return (kind,)


@app.cell
def _(FT_DIR, N_FOLDS, TIERS, kind, np, pd):
    ft = pd.read_csv(FT_DIR / f"{kind.value}.csv")

    def fold_mean(correct, folds, tiers):
        """Accuracy per tier, averaged over the held-out folds (never pooled)."""
        out = {}
        for t in TIERS:
            m = np.ones(len(correct), bool) if t == "ALL" else tiers == t
            out[t] = float(
                np.mean([correct[m & (folds == f)].mean() for f in range(N_FOLDS)])
            )
        return out

    ft_by_seed = pd.DataFrame(
        {
            f"seed {s}": fold_mean(
                g["correct"].to_numpy(),
                g["fold"].to_numpy(),
                g["construction"].to_numpy(),
            )
            for s, g in ft.groupby("seed")
        }
    ).T
    ft_by_fold = ft.groupby(["seed", "fold"])["correct"].mean().unstack("fold").round(3)
    ft_by_fold.columns = [f"fold {f}" for f in ft_by_fold.columns]
    ft_by_seed.round(3), ft_by_fold
    return fold_mean, ft, ft_by_seed


@app.cell
def _(FT_DIR, fold_mean, kind, pd):
    # simulate/modernbert_frozen_app.py output, if it has landed
    _path = FT_DIR / "frozen" / f"{kind.value}.csv"
    frozen_rows = (
        pd.read_csv(_path) if _path.exists() else pd.DataFrame(columns=["model"])
    )
    frozen = pd.DataFrame(
        {
            name: fold_mean(
                g["correct"].to_numpy(),
                g["fold"].to_numpy(),
                g["construction"].to_numpy(),
            )
            for name, g in frozen_rows.groupby("model")
        }
    ).T
    frozen.round(3)
    return frozen, frozen_rows


@app.cell
def _(CACHE, MODELS, fold_mean, ft, kind, np, pd):
    def load_run(model):
        z = np.load(CACHE / f"{model}_{kind.value}.npz", allow_pickle=True)
        y, folds = z["y"], z["folds"]
        correct = z["oof"] == y
        per_fold = np.stack(
            [correct[:, folds == f].mean(1) for f in range(len(set(folds)))], 1
        )
        return z, int(per_fold.mean(1).argmax())

    runs = {
        m: load_run(m) for m in MODELS if (CACHE / f"{m}_{kind.value}.npz").exists()
    }

    # sanity: the fine-tune used the same folds as the probes
    _z = next(iter(runs.values()))[0]
    _probe_fold = dict(zip(_z["ids"], _z["folds"]))
    _mismatch = int((ft["fold"] != ft["id"].map(_probe_fold)).sum())
    assert _mismatch == 0, f"{_mismatch} rows on a different fold than the probes"

    coordinator = pd.DataFrame(
        {
            m: fold_mean(z["predicted"] == z["y"], z["folds"], z["construction"])
            for m, (z, _) in runs.items()
        }
    ).T
    probe = pd.DataFrame(
        {
            m: fold_mean(z["oof"][layer] == z["y"], z["folds"], z["construction"])
            for m, (z, layer) in runs.items()
        }
    ).T
    return coordinator, probe, runs


@app.cell
def _(BASELINES, TIERS, coordinator, frozen, ft_by_seed, json, kind, np, pd, probe):
    _cache = json.loads(BASELINES.read_text())[kind.value]["per_fold"]
    baselines = pd.DataFrame(
        {name: {t: float(np.mean(v[t])) for t in TIERS} for name, v in _cache.items()}
    ).T

    summary = pd.concat(
        [
            ft_by_seed.mean().rename("modernbert-large-ft").to_frame().T,
            frozen,
            baselines,
            pd.DataFrame(
                {
                    "coordinator mean": coordinator.mean(),
                    "coordinator min": coordinator.min(),
                    "coordinator max": coordinator.max(),
                    "probe mean": probe.mean(),
                    "probe min": probe.min(),
                    "probe max": probe.max(),
                }
            ).T,
        ]
    )[TIERS].round(3)
    summary
    return (baselines,)


@app.cell
def _(TIERS, baselines, coordinator, frozen, ft_by_seed, np, plt, probe):
    _ft = ft_by_seed.mean()
    _lines = [
        ("modernbert-large-ft", _ft, "C3", "-"),
        *[(name, row, "C1", ":") for name, row in frozen.iterrows()],
        ("lexical", baselines.loc["lexical"], "0.3", "--"),
        ("tfidf", baselines.loc["tfidf"], "0.5", "--"),
        ("embedding", baselines.loc["embedding"], "0.7", "--"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    x = np.arange(len(probe))
    for ax, tier in zip(axes.flat, TIERS):
        ax.bar(x - 0.2, coordinator[tier], 0.4, label="coordinator", color="C0")
        ax.bar(x + 0.2, probe[tier], 0.4, label="probe", color="C2")
        for name, vals, color, style in _lines:
            ax.axhline(vals[tier], color=color, ls=style, lw=1.5, label=name)
        ax.set_title(tier)
        ax.set_xticks(x, probe.index, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("accuracy (fold mean)")
    axes[1, 0].set_ylabel("accuracy (fold mean)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AUC

    Macro one-vs-rest ROC AUC over the 5 sub-agents: how well the probabilities rank
    the correct sub-agent, independent of the argmax. Out-of-fold probabilities, pooled
    over the folds. The probe's come from its cached layer (`proba_layer`, the optimal
    layer when `nb/probing.py` wrote them).
    """)
    return


@app.cell
def _(EXPERTS, TIERS, frozen_rows, ft, np, pd, roc_auc_score, runs):
    P_COLS = [f"p_{e}" for e in EXPERTS]

    _order = np.argsort(EXPERTS)  # sklearn wants the labels sorted

    def auc(y, proba, tiers):
        proba = proba[:, _order]
        out = {}
        for t in TIERS:
            m = np.ones(len(y), bool) if t == "ALL" else tiers == t
            out[t] = float(
                roc_auc_score(y[m], proba[m], multi_class="ovr", labels=sorted(EXPERTS))
            )
        return out

    _ft_auc = pd.DataFrame(
        {
            f"modernbert-large-ft seed {s}": auc(
                g["gold"].to_numpy(), g[P_COLS].to_numpy(), g["construction"].to_numpy()
            )
            for s, g in ft.groupby("seed")
        }
    ).T
    _frozen_auc = pd.DataFrame(
        {
            name: auc(
                g["gold"].to_numpy(), g[P_COLS].to_numpy(), g["construction"].to_numpy()
            )
            for name, g in frozen_rows.groupby("model")
        }
    ).T
    _probe_auc = pd.DataFrame(
        {
            f"probe {m} (layer {int(z['proba_layer'])})": auc(
                z["y"], z["proba"], z["construction"]
            )
            for m, (z, _) in runs.items()
            if "proba" in z.files and list(z["experts"]) == EXPERTS
        }
    ).T
    pd.concat([_ft_auc, _frozen_auc, _probe_auc])[TIERS].round(3)
    return (P_COLS,)


@app.cell
def _(EXPERTS, P_COLS, ft, plt, roc_curve):
    _g = ft[ft["seed"] == ft["seed"].min()]
    fig_roc, _ax = plt.subplots(figsize=(6, 5))
    for _e, _c in zip(EXPERTS, P_COLS):
        _fpr, _tpr, _ = roc_curve(_g["gold"] == _e, _g[_c])
        _ax.plot(_fpr, _tpr, label=_e)
    _ax.plot([0, 1], [0, 1], color="0.6", ls=":")
    _ax.set_xlabel("false positive rate")
    _ax.set_ylabel("true positive rate")
    _ax.set_title("modernbert-large-ft: ROC per sub-agent (one-vs-rest)")
    _ax.legend(fontsize=8)
    fig_roc
    return


if __name__ == "__main__":
    app.run()
