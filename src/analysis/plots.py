"""Every figure in the analysis"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from analysis import probe as P

sns.set_theme(style="whitegrid", context="notebook", rc={"axes.grid": False})

ROUTERS = list(P.ROUTERS)
COLOR = {
    "model": "#c1442f",
    "lexical": "#8a8a8a",
    "tfidf": "#6f6a5e",
    "embedding": "#c9c0b4",
    "probe": "#2f6fc1",
}


def _fold_marks(ax, pos, height, per_fold, width, label=None):
    """One capped line per bar, from the worst fold to the best fold.

    Returns the top of that line per bar, so the caller can put its label above it.
    """
    values = np.array([np.asarray(v, float) for v in per_fold])
    if np.isnan(values).all():
        return np.asarray(height, float)
    # the bar is the mean of these same fold accuracies, so it always sits inside the line
    height = np.asarray(height, float)
    lo, hi = np.nanmin(values, axis=1), np.nanmax(values, axis=1)
    ax.errorbar(
        pos,
        height,
        yerr=[height - lo, hi - height],
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=width * 26,
        capthick=0.9,
        alpha=0.75,
        label=label,
    )
    return hi


def _site(d) -> str:
    """Title suffix naming the read site, empty for the default last prompt token."""
    return " - last reasoning token" if d.get("site") == "reasoning" else ""


def plot_layers(d, ax=None):
    """The probe layer by layer, against the model's flat accuracy."""
    hard = d["constructions"][-1]
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 4.2))
    layers = d["by_layer"].index
    # the bars are the fold-to-fold range, so a layer that only wins on one fold is visible
    for name, col, mask, style, alpha in [
        ("probe", "acc", None, "-o", 1.0),
        (f"probe ({hard})", hard, d["construction"] == hard, "--o", 0.55),
    ]:
        spread = P.layer_folds(d, mask)
        mean = d["by_layer"][col].to_numpy()
        ax.errorbar(
            layers,
            mean,
            yerr=[mean - spread.min(1), spread.max(1) - mean],
            fmt=style,
            ms=3,
            color=COLOR["probe"],
            alpha=alpha,
            elinewidth=0.8,
            capsize=2,
            label=f"{name}, fold min-max",
        )
    for name, mask, style in [
        ("model", np.ones(len(d["y"]), bool), "-"),
        (f"model ({hard})", d["construction"] == hard, "--"),
    ]:
        v = (d["predicted"] == d["y"])[mask].mean()
        ax.axhline(v, ls=style, color=COLOR["model"], lw=1.2, label=f"{name} = {v:.3f}")
    ax.axhline(P.CHANCE, ls=":", color="grey", label="chance")
    peak = P.peak_layer(d)
    ax.axvline(peak, ls="-", color=COLOR["probe"], lw=1, alpha=0.35)
    ax.plot(
        [peak],
        [d["by_layer"]["acc"].iloc[peak]],
        "X",
        ms=11,
        color=COLOR["probe"],
        mec="white",
        mew=0.8,
        label=f"optimal layer = {peak}",
    )
    ax.set(
        xlabel="layer",
        ylabel="out-of-fold accuracy",
        ylim=(0, 1.05),
        title=f"Layer sweep - {d['model']} - {d['kind']}{_site(d)}",
    )
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    return fig


ROUTER_LABEL = {"lexical": "bag-of-words", "tfidf": "tf-idf"}


def plot_compare(d, ax=None):
    """Every router on the same folds, grouped by construction."""
    table = P.compare(d)
    folds = P.compare_folds(d)
    slices = list(table.index)
    x = np.arange(len(slices))
    width = 0.8 / len(ROUTERS)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8.5, 4.2))
    for i, router in enumerate(ROUTERS):
        pos = x + i * width
        height = table[router].astype(float)
        ax.bar(
            pos,
            height,
            width,
            label=ROUTER_LABEL.get(router, router),
            color=COLOR[router],
        )
        top = _fold_marks(
            ax,
            pos,
            height,
            folds[router],
            width,
            label="fold worst-best" if i == 0 else None,
        )
        for px, h, t in zip(pos, height, np.fmax(height, top)):
            ax.text(px, t + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(P.CHANCE, ls=":", color="grey", label="chance")
    ax.set_xticks(x + width * (len(ROUTERS) - 1) / 2, slices)
    ax.set(
        ylabel="out-of-fold accuracy",
        ylim=(0, 1.1),
        title=f"Accuracy Comparison - {d['model']} - {d['kind']}{_site(d)}",
    )
    # below the axes: inside, at this font size, it covers the plain bars
    ax.legend(ncol=4, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return fig


def plot_confusions(d, layer=None):
    """The model beside the probe: same labels, same colour scale."""
    layer = P.peak_layer(d) if layer is None else layer
    labels = d["experts"]
    # an unrouted sample has no tool call, so its prediction is None: sklearn's
    # confusion_matrix cannot sort a mix of strings and NaN, so it is relabelled to a
    # string outside `labels` first, which the matrix then simply counts as a miss.
    model_pred = np.where(pd.isna(d["predicted"]), "unrouted", d["predicted"])
    pairs = [
        (f"{d['model']} routing", model_pred, "Blues"),
        (f"probe, layer {layer}", d["oof"][layer], "Greens"),
    ]
    mats = [confusion_matrix(d["y"], p, labels=labels) for _, p, _ in pairs]
    vmax = max(m.max() for m in mats)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, _, cmap), cm in zip(axes, pairs, mats):
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax,
            cmap=cmap,
            values_format="d",
            colorbar=False,
            im_kw={"vmin": 0, "vmax": vmax},
        )
        ax.set_title(f"{title}   acc={np.trace(cm) / cm.sum():.3f}")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Confusion — {d['label']}")
    fig.tight_layout()
    return fig


def plot_roc(d, proba, layer, ax=None):
    """One-vs-rest ROC of the probe at its peak layer, plus the macro average."""
    labels = d["experts"]
    Y = label_binarize(d["y"], classes=labels)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.2, 5.4))
    grid = np.linspace(0, 1, 200)
    interp = []
    for i, expert in enumerate(labels):
        fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
        interp.append(np.interp(grid, fpr, tpr))
        ax.plot(fpr, tpr, lw=1.2, label=f"{expert} (AUC {auc(fpr, tpr):.3f})")
    macro = np.mean(interp, axis=0)
    ax.plot(
        grid, macro, lw=2.5, color="black", label=f"macro (AUC {auc(grid, macro):.3f})"
    )

    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
    ax.set(
        xlabel="false positive rate",
        ylabel="true positive rate",
        title=f"Probe ROC, layer {layer} — {d['label']}",
    )
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


KIND_STYLE = {"single_turn": ("o", COLOR["probe"]), "multi_turn": ("s", COLOR["model"])}
KIND_LABEL = {"single_turn": "single-turn", "multi_turn": "multi-turn"}


def _by_kind_bars(ax, table, column, ylabel, title, signed=False, fmt=None):
    """One grouped bar per model, single-turn beside multi-turn.

    `table` is indexed by "<model> <kind>", the index `across_runs` builds, so the model
    order of `probe.runs()` is preserved instead of being sorted alphabetically.
    """
    models = list(dict.fromkeys(P.split_label(i)[0] for i in table.index))
    x = np.arange(len(models))
    width = 0.38
    for i, kind in enumerate(P.KINDS):
        pos = x + i * width
        values = [
            float(table.loc[f"{m} {kind}", column])
            if f"{m} {kind}" in table.index
            else np.nan
            for m in models
        ]
        color = (
            [COLOR["probe"] if v >= 0 else COLOR["model"] for v in values]
            if signed
            else KIND_STYLE[kind][1]
        )
        ax.bar(
            pos,
            values,
            width,
            color=color,
            edgecolor="white",
            alpha=1.0 if i == 0 else 0.55,
            label=KIND_LABEL[kind],
        )
        for px, v in zip(pos, values):
            text = format(v, fmt) if fmt else (f"{v:+.3f}" if signed else f"{v:.2f}")
            ax.text(
                px,
                v + (0.004 if v >= 0 else -0.004),
                text,
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7,
            )
    if signed:
        ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x + width / 2, models, rotation=20, ha="right", fontsize=8)
    ax.set(ylabel=ylabel, title=title)
    return ax


def plot_tiers(results, ax=None):
    """Model and probe accuracy across the three tiers, one panel per kind.

    Thin lines are single runs, thick lines the mean over runs. The drop from `plain` to
    `adversarial` is the slope, so the section's comparison, does the probe lose less than
    the coordinator as the tier gets harder, is read off the two thick lines.
    """
    t = P.tier_table(results)
    summary = P.tier_summary(results)
    tiers = list(P.CONSTRUCTIONS)
    x = np.arange(len(tiers))
    fig, axes = (
        (ax[0].figure, np.asarray(ax))
        if ax is not None
        else plt.subplots(1, len(P.KINDS), figsize=(10.0, 4.2), sharey=True)
    )
    series = [("model", "coordinator"), ("probe", "probe")]
    for ax, kind in zip(axes, P.KINDS):
        runs_ = t[t["kind"] == kind]
        for run, g in runs_.groupby(level="run", sort=False):
            g = g.droplevel("run")
            for col, _ in series:
                ax.plot(
                    x,
                    g.loc[tiers, col].astype(float),
                    "-",
                    lw=0.8,
                    alpha=0.25,
                    color=COLOR[col],
                )
        means = {
            col: summary.loc[[(kind, tr) for tr in tiers], col].astype(float).to_numpy()
            for col, _ in series
        }
        for col, label in series:
            y, other = means[col], means["probe" if col == "model" else "model"]
            ax.plot(x, y, "-o", lw=2.4, ms=6, color=COLOR[col], label=f"{label} (mean)")
            # the higher of the two lines at a tier carries its label above, the lower below
            for px, v, o in zip(x, y, other):
                ax.annotate(
                    f"{v:.3f}",
                    (px, v),
                    textcoords="offset points",
                    xytext=(0, 7 if v >= o else -14),
                    ha="center",
                    fontsize=8.5,
                    color=COLOR[col],
                )
        ax.set_xticks(x, tiers)
        ax.set_title(KIND_LABEL[kind], fontsize=10)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=11, loc="lower left")
    fig.suptitle("Accuracy per tier, coordinator against probe")
    fig.tight_layout()
    return fig


def plot_multi_turn_cost(cost, ax=None):
    """The multi-turn drop model by model: one bar for the coordinator, one for the probe.

    Each bar is the multi-turn accuracy minus the single-turn accuracy, so a bar below zero
    is accuracy lost when the request is spread across turns. The comparison the section
    makes, does the probe fall further than the coordinator, is the length of the two bars
    of one model.
    """
    models = [m for m in dict.fromkeys(m for m, _ in P.runs()) if m in cost.index]
    x = np.arange(len(models))
    width = 0.38
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 4.2))
    for i, (prefix, label) in enumerate([("model", "coordinator"), ("probe", "probe")]):
        values = cost.loc[models, f"delta_{prefix}"].astype(float).to_numpy()
        pos = x + (i - 0.5) * width
        ax.bar(pos, values, width, color=COLOR[prefix], edgecolor="white", label=label)
        for px, v in zip(pos, values):
            ax.text(
                px,
                v + (0.002 if v >= 0 else -0.002),
                f"{v:+.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8.5,
            )
    ax.axhline(0, color="black", lw=1)
    lo = float(cost[["delta_model", "delta_probe"]].astype(float).min().min())
    ax.set_ylim(lo - 0.015, 0.02)
    ax.set_xticks(x, models, fontsize=8)
    ax.set(ylabel="multi-turn drop", title="Multi-turn drop, coordinator against probe")
    ax.legend(fontsize=11, loc="lower left")
    fig.tight_layout()
    return fig


def plot_flag_rates(results, ax=None):
    """The disagreement flag run by run: F1 on one panel, false alarm rate on the other.

    Each bar is one run read on its own. F1 says how well the flag covers that run's
    misroutes and the false alarm rate what it costs on that run's correct decisions.
    Every one of these numbers is read against the coordinator of its own run, so the
    panels are not a ranking of the models against each other.
    """
    table = P.across_runs(P.flags, results=results)
    fig, axes = (
        (ax[0].figure, np.asarray(ax))
        if ax is not None
        else plt.subplots(1, 2, figsize=(12.5, 4.2))
    )
    _by_kind_bars(
        axes[0],
        table,
        "f1",
        "F1-score",
        "Flag F1-score on the coordinator's errors",
        fmt=".3f",
    )
    axes[0].set_ylim(0, float(table["f1"].astype(float).max()) + 0.12)
    _by_kind_bars(
        axes[1],
        table,
        "false_alarm",
        "false alarm rate",
        "Flag false alarm rate on the coordinator's correct decisions",
        fmt=".3f",
    )
    axes[1].set_ylim(0, float(table["false_alarm"].astype(float).max()) + 0.04)
    axes[0].legend(fontsize=8, loc="upper left", ncol=2)
    fig.suptitle("The disagreement flag on each run")
    fig.tight_layout()
    return fig


def plot_depth_peaks(results, col="acc", ax=None):
    """Every run's peak depth on one axis, with the band that holds most of them."""
    peaks = P.peak_depths(results, col)
    summary = P.depth_summary(results, col)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9.5, 3.6))
    ax.axvspan(
        summary["band_lo"],
        summary["band_hi"],
        color=COLOR["probe"],
        alpha=0.10,
        label=f"{summary['band_lo']:.1f}-{summary['band_hi']:.1f}: "
        f"{int(summary['in_band'])} of {int(summary['n_runs'])} runs",
    )
    ax.axvline(
        summary["median"],
        color=COLOR["model"],
        lw=1.4,
        label=f"median = {summary['median']:.2f}",
    )
    rng = np.random.default_rng(P.SEED)
    for kind in P.KINDS:
        sub = peaks.loc[[i for i in peaks.index if P.split_label(i)[1] == kind]]
        marker, color = KIND_STYLE[kind]
        ax.plot(
            sub["peak_depth"],
            rng.uniform(-0.35, 0.35, len(sub)),
            marker,
            ms=7,
            ls="none",
            color=color,
            alpha=0.85,
            label=KIND_LABEL[kind],
        )
    ax.set(
        xlim=(0, 1),
        ylim=(-1, 1),
        yticks=[],
        xlabel="relative depth (layer / last layer)",
        title=f"Peak depth of the probe, {int(summary['n_runs'])} runs",
    )
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


# failure analysis

BASELINE_LABEL = {
    "lexical": "bag-of-words",
    "tfidf": "tf-idf",
    "embedding": "embedding",
}
ROUTER_LABEL = {"probe": "probe", "model": "coordinator", **BASELINE_LABEL}


def plot_dependence(summary):
    """Probe accuracy where the baseline is right against where it is wrong.

    One panel per (kind, tier), one row per baseline. The filled end is the accuracy on the
    samples the baseline gets right, the hollow end the accuracy on those it gets wrong, so
    the length of a segment is how much the probe loses where the text misleads the
    baseline. A probe that failed independently of the baseline would draw a point.
    """
    tiers = list(P.DEPENDENCE_TIERS)
    names = list(P.BASELINE_NAMES)
    fig, axes = plt.subplots(len(tiers), len(P.KINDS), figsize=(11.0, 5.0), sharex=True)
    color = COLOR["probe"]
    for col, kind in enumerate(P.KINDS):
        for row, tier in enumerate(tiers):
            ax = axes[row, col]
            for i, b in enumerate(names):
                y = len(names) - 1 - i
                r = summary.loc[(kind, b, tier)]
                lo, hi = float(r["acc_baseline_wrong"]), float(r["acc_baseline_right"])
                ax.plot([lo, hi], [y, y], "-", lw=2.2, color=color, alpha=0.8)
                ax.plot(hi, y, "o", ms=6.5, color=color)
                ax.plot(lo, y, "o", ms=6.5, mfc="white", mec=color, mew=1.6)
                ax.annotate(
                    f"{lo - hi:+.3f}",
                    (lo, y),
                    textcoords="offset points",
                    xytext=(-6, -3),
                    ha="right",
                    fontsize=9.5,
                    color=color,
                )
            ax.set_yticks(
                range(len(names)),
                [BASELINE_LABEL[b] for b in reversed(names)],
                fontsize=9,
            )
            ax.set_ylim(-0.6, len(names) - 0.4)
            ax.set_title(f"{KIND_LABEL[kind]}, {tier}", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("probe accuracy")
    axes[0, 0].set_xlim(0.55, 1.0)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", color=color, label="baseline right"),
        plt.Line2D(
            [],
            [],
            ls="none",
            marker="o",
            mfc="white",
            mec=color,
            label="baseline wrong",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_pairs(summary):
    """The adversarial pairs, router by router, one column per kind.

    Top row: how many of the two requests of a pair each router classifies correctly, as
    shares of the pairs. Bottom row: collapsed pairs, where it predicts the same sub-agent
    for both requests, which caps it at one right request. Labels use the terms of the
    thesis text.
    """
    routers = ["probe", "model", *P.BASELINE_NAMES]
    labels = [ROUTER_LABEL[r] for r in routers]
    y = np.arange(len(routers))[::-1]
    fig, axes = plt.subplots(
        2, len(P.KINDS), figsize=(11.0, 6.2), sharey=True, sharex=True
    )
    segments = [
        ("both", "both requests right", 1.0),
        ("one", "one request right", 0.5),
        ("neither", "neither request right", 0.18),
    ]
    for col_i, kind in enumerate(P.KINDS):
        s = summary.loc[kind].reindex(routers).astype(float)
        top, bottom = axes[0, col_i], axes[1, col_i]
        start = np.zeros(len(routers))
        for col, label, alpha in segments:
            top.barh(
                y,
                s[col],
                left=start,
                color="#2d2d2d",
                alpha=alpha,
                height=0.62,
                label=label,
                edgecolor="white",
            )
            for py, x0, w in zip(y, start, s[col]):
                if w >= 0.08:
                    top.text(
                        x0 + w / 2,
                        py,
                        f"{w:.3f}",
                        ha="center",
                        va="center",
                        fontsize=9.5,
                        color="white" if alpha > 0.4 else "black",
                    )
            start += s[col].to_numpy()
        top.set(
            xlim=(0, 1),
            title=f"{KIND_LABEL[kind]}: requests of a pair classified correctly",
        )
        # one colour outside the legend greys: the routers are named on the y axis
        bottom.barh(y, s["collapsed"], color="#5b7fa6", height=0.62)
        for py, w in zip(y, s["collapsed"]):
            bottom.text(w + 0.01, py, f"{w:.3f}", va="center", fontsize=9.5)
        bottom.set(xlim=(0, 1), title=f"{KIND_LABEL[kind]}: collapsed pairs")
        bottom.set_xlabel("share of pairs")
        top.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_yticks(y, labels, fontsize=9)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    # on top, next to the row it describes: at the bottom it read as a key to the collapsed bars
    fig.legend(handles, labels_, loc="upper center", ncol=3, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_decoy(summary):
    """Decoy rate per router: the share of the samples of a tier on which it predicts the
    decoy, one panel per kind, one bar per tier. Labels use the terms of the thesis text."""
    routers = ["probe", "model", *P.BASELINE_NAMES]
    y = np.arange(len(routers))[::-1]
    height = 0.36
    tiers = [("natural", "#9db7d5"), ("adversarial", "#2f4f75")]
    fig, axes = plt.subplots(1, len(P.KINDS), figsize=(11.0, 3.8), sharey=True, sharex=True)
    for ax, kind in zip(axes, P.KINDS):
        s = summary.loc[kind]["decoy"].astype(float)
        for i, (tier, color) in enumerate(tiers):
            pos = y + height / 2 - i * height
            values = [s.loc[(r, tier)] for r in routers]
            ax.barh(pos, values, height=height, color=color, label=tier)
            for py, w in zip(pos, values):
                ax.text(w + 0.01, py, f"{w:.3f}", va="center", fontsize=9.5)
        ax.set(xlim=(0, 0.6), title=f"{KIND_LABEL[kind]}: decoy rate")
        ax.set_xlabel("share of samples")
    axes[0].set_yticks(y, [ROUTER_LABEL[r] for r in routers], fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig

# dataset figures


def plot_balance(V):
    """Cramér's V between design factors: 0 = orthogonal, 1 = one factor determines the
    other. The only cells allowed to be non-zero are structural."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        V, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1, square=True, ax=ax
    )
    ax.set_title("Cramér's V between design factors")
    fig.tight_layout()
    return fig


FACTOR_LABEL = {"construction": "difficulty"}


def plot_omega2(anova):
    """omega2 of every design factor on every surface feature.

    collision is left out: it is defined as the pair (gold, decoy), so its effect is not a
    new fact next to those two factors' own effects, and showing it only adds a redundant
    column.
    """
    piv = anova[anova.factor != "collision"].pivot(
        index="feature", columns="factor", values="omega2"
    )
    piv = piv.rename(columns=FACTOR_LABEL)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.heatmap(
        piv,
        annot=True,
        fmt=".3f",
        cmap="rocket_r",
        vmin=0,
        vmax=max(0.15, piv.max().max()),
        ax=ax,
    )
    ax.set_title("omega2: variance of each surface feature explained by each factor")
    fig.tight_layout()
    return fig


def plot_gold_effect(data: dict, anovas: dict, features=("pct_digit", "n_units")):
    """Gold's effect on a fixed pair of features, compared across both datasets: one row
    per dataset, one column per feature, so the same features are read side by side
    instead of each dataset picking its own strongest one."""
    kinds = list(data)
    fig, axes = plt.subplots(
        len(kinds), len(features), figsize=(6 * len(features), 3.8 * len(kinds))
    )
    axes = np.atleast_2d(axes)
    for i, kind in enumerate(kinds):
        d, anova = data[kind], anovas[kind]
        for j, feature in enumerate(features):
            ax = axes[i, j]
            sns.pointplot(
                data=d,
                x="gold",
                y=feature,
                errorbar=("ci", 95),
                linestyle="none",
                ax=ax,
            )
            ax.tick_params(axis="x", rotation=25)
            omega2 = anova.loc[
                (anova.factor == "gold") & (anova.feature == feature), "omega2"
            ].iloc[0]
            ax.set_title(f"{kind}: {feature} (omega2={omega2:.3f})")
    fig.tight_layout()
    return fig


def plot_posthoc(post, n_features=4, n_cols=2, kind=None):
    """Post-hoc pairs on `gold`, one matrix per feature: the cell is mean(row) - mean(col),
    coloured only where the pair survives Holm. Blank = the two experts are not separated.

    Laid out as a grid, not a single row: a row of N small square panels gets crushed to a
    sliver once scaled to the page width, which is what made the annotations unreadable.
    """
    features = list(pd.unique(post["feature"]))[:n_features]
    experts = sorted(set(post["a"]) | set(post["b"]))
    n_cols = min(n_cols, len(features))
    n_rows = -(-len(features) // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4.1 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, features):
        sub = post[post["feature"] == f]
        M = pd.DataFrame(np.nan, index=experts, columns=experts)
        for r in sub.itertuples():
            if r.p_holm < 0.05:
                M.loc[r.a, r.b], M.loc[r.b, r.a] = r.diff, -r.diff
        lim = np.nanmax(np.abs(M.to_numpy())) if M.notna().any().any() else 1.0
        sns.heatmap(
            M,
            annot=True,
            fmt=".3f",
            cmap="vlag",
            center=0,
            vmin=-lim,
            vmax=lim,
            square=True,
            cbar=False,
            annot_kws={"fontsize": 9},
            linewidths=0.5,
            linecolor="#eeeeee",
            ax=ax,
        )
        # heatmap draws no border on a blank (NaN) cell, so the cell edges are drawn as a grid
        ax.set_xticks(np.arange(len(experts) + 1), minor=True)
        ax.set_yticks(np.arange(len(experts) + 1), minor=True)
        ax.grid(which="minor", color="#dddddd", lw=0.5)
        ax.tick_params(which="minor", length=0)
        ax.set_title(f"{f}", fontsize=13)
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)
    for ax in axes[len(features) :]:
        ax.set_visible(False)
    title = f"Games-Howell on gold - {kind}" if kind else "Games-Howell on gold"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_interactions(models: dict[str, pd.DataFrame], value: str = "partial_omega2"):
    """The `gold` interaction gain of each multi-factor model, one column per model.

    `models` maps a short model label to a `dataset.interactions(...)` result. Cells mark
    the significant ones (Holm) with an asterisk, so the heatmap and the p_holm column
    read together.
    """
    piv = pd.DataFrame(
        {name: df.set_index("feature")[value] for name, df in models.items()}
    )
    sig = pd.DataFrame(
        {name: df.set_index("feature")["p_holm"] < 0.05 for name, df in models.items()}
    )
    annot = piv.round(3).astype(str) + sig.reindex_like(piv).map(
        lambda s: "*" if s else ""
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.heatmap(
        piv,
        annot=annot,
        fmt="",
        cmap="rocket_r",
        vmin=0,
        vmax=max(0.06, piv.max().max()),
        ax=ax,
    )
    ax.set_title(f"{value}: gold interaction variance, by model (* = p_holm < 0.05)")
    fig.tight_layout()
    return fig


def plot_ceiling(ceiling, kind=None):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ceiling.plot.bar(ax=ax, rot=0, width=0.8)
    ax.axhline(P.CHANCE, ls=":", color="grey", label="chance")
    title = (
        f"accuracy from surface features only - {kind}"
        if kind
        else "accuracy from surface features only"
    )
    ax.set(ylim=(0, 1.05), ylabel="out-of-fold accuracy", title=title)
    ax.set_xticklabels(
        [t.get_text().replace(", ", "\n") for t in ax.get_xticklabels()], fontsize=8
    )
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_shap(shap_imp, kind=None):
    """Mean absolute SHAP per feature, split by expert: which feature pushes where."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    shap_imp.plot.barh(
        ax=ax,
        stacked=True,
        width=0.8,
        color=sns.color_palette("colorblind", shap_imp.shape[1]),
    )
    title = f"feature importance - {kind}" if kind else "feature importance"
    ax.set(xlabel="mean |SHAP| (out of fold)", title=title)
    ax.legend(fontsize=8, title=None)
    fig.tight_layout()
    return fig


TIER_LABEL = {
    "ALL": "All",
    "plain": "Plain",
    "natural": "Natural",
    "adversarial": "Adversarial",
}


def plot_baselines(kinds=P.KINDS):
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 3.8), sharey=True)
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        table = pd.DataFrame(P.baselines(kind)["accuracy"]).loc[
            ["ALL", *P.CONSTRUCTIONS]
        ]
        table = table[[c for c in P.BASELINE_NAMES if c in table.columns]]
        table.index = [TIER_LABEL[i] for i in table.index]
        table.columns = [BASELINE_LABEL[c] for c in table.columns]
        table.plot.bar(
            ax=ax,
            rot=0,
            color=[
                COLOR[c] for c in P.BASELINE_NAMES if c in P.baselines(kind)["accuracy"]
            ],
        )
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=6, padding=1)
        ax.axhline(P.CHANCE, ls=":", color="grey")
        ax.set(title=f"Text-only baselines, {KIND_LABEL[kind]}", ylim=(0, 1.1))
    np.atleast_1d(axes)[0].set_ylabel("Out-of-fold accuracy")
    fig.tight_layout()
    return fig


def plot_pair_overlap(overlaps):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.histplot(data=overlaps, x="pair_overlap", hue="kind", bins=40, ax=ax)
    ax.set(title="word overlap inside an adversarial pair", xlabel="pair_overlap")
    fig.tight_layout()
    return fig


def plot_composition(kind):
    """Design balance seen directly: experts x strata and experts x personas."""
    d = P.dataset(kind)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, col in zip(axes, ["construction", "persona"]):
        sns.heatmap(
            pd.crosstab(d["gold"], d[col]),
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel(FACTOR_LABEL.get(col, col))
        ax.set_title(f"gold x {FACTOR_LABEL.get(col, col)}, {kind}")
    fig.tight_layout()
    return fig


# laya

LAYA_COLOR = {
    "laya-english": "#5b2c83",
    "laya-multilingual": "#8a5cc2",
    "laya-typed-decisions": "#c3a6e6",
    "laya-multilingual-ft": "#d6337f",
}


def plot_laya_compare(table, kind, ax=None):
    """Laya's checkpoints beside the text baselines, the coordinators and their probes.

    Laya and the baselines are one bar each, laya's with its worst and best fold. The
    coordinators and the probes are one bar for the mean over runs, with one dot per run.
    """
    slices = list(dict.fromkeys(table["slice"]))
    laya = list(dict.fromkeys(table.loc[table["family"] == "laya", "router"]))
    bars = (
        [("laya", r, r, LAYA_COLOR.get(r, "#8a5cc2")) for r in laya]
        + [("baseline", b, ROUTER_LABEL.get(b, b), COLOR[b]) for b in P.BASELINE_NAMES]
        + [
            ("coordinator", None, "coordinator (mean of runs)", COLOR["model"]),
            ("probe", None, "probe (mean of runs)", COLOR["probe"]),
        ]
    )
    x = np.arange(len(slices))
    width = 0.8 / len(bars)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(11, 4.6))
    dot_label, fold_label = "one run", "worst to best fold"
    for i, (family, router, label, color) in enumerate(bars):
        sel = table[table["family"] == family]
        if router is not None:
            sel = sel[sel["router"] == router]
        values = [sel.loc[sel["slice"] == s, "acc"].to_numpy(float) for s in slices]
        pos = x + i * width
        height = np.array([v.mean() for v in values])
        ax.bar(pos, height, width, label=label, color=color)
        top = [v.max() for v in values]
        if family == "laya":
            folds = [
                sel.loc[sel["slice"] == s, ["fold_min", "fold_max"]].to_numpy(float)[0]
                for s in slices
            ]
            top = _fold_marks(ax, pos, height, folds, width, label=fold_label)
            fold_label = None
        for px, h, v, t in zip(pos, height, values, top):
            if router is None:
                ax.scatter(
                    np.full(len(v), px),
                    v,
                    s=9,
                    color="black",
                    alpha=0.6,
                    zorder=3,
                    label=dot_label,
                )
                dot_label = None
            ax.text(
                px,
                max(h, t) + 0.015,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    ax.axhline(P.CHANCE, ls=":", color="grey", label="chance")
    ax.set_xticks(x + width * (len(bars) - 1) / 2, slices)
    ax.set(
        ylabel="accuracy", ylim=(0, 1.3), title=f"Laya against every router - {kind}"
    )
    ax.legend(ncol=4, fontsize=8, loc="upper center")
    fig.tight_layout()
    return fig


def plot_laya_predictions(shares, ax=None):
    """Share of the samples each checkpoint routes to each expert, against the gold shares."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 2.8))
    sns.heatmap(
        shares.astype(float),
        annot=True,
        fmt=".2f",
        cmap="Purples",
        vmin=0,
        vmax=1,
        cbar=False,
        ax=ax,
    )
    ax.set(xlabel="predicted expert", ylabel="", title="Where laya routes")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_laya_confusions(t):
    """One confusion matrix per checkpoint, same labels and colour scale."""
    labels = list(P.EXPERTS)
    ckpts = list(dict.fromkeys(t["checkpoint"]))
    mats = [
        confusion_matrix(
            t.loc[t["checkpoint"] == c, "gold"],
            t.loc[t["checkpoint"] == c, "predicted"],
            labels=labels,
        )
        for c in ckpts
    ]
    vmax = max(m.max() for m in mats)
    fig, axes = plt.subplots(1, len(ckpts), figsize=(5.5 * len(ckpts), 5))
    for ax, ckpt, cm in zip(np.atleast_1d(axes), ckpts, mats):
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax,
            cmap="Purples",
            values_format="d",
            colorbar=False,
            im_kw={"vmin": 0, "vmax": vmax},
        )
        ax.set_title(f"{ckpt}   acc={np.trace(cm) / cm.sum():.3f}")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_laya_reliability(rel, ax=None):
    """Accuracy against confidence per checkpoint; the diagonal is perfect calibration."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], ls=":", color="grey", label="calibrated")
    for ckpt, g in rel.groupby("checkpoint", sort=False):
        ax.plot(
            g["confidence"],
            g["acc"],
            "-o",
            ms=4,
            color=LAYA_COLOR.get(ckpt, "#8a5cc2"),
            label=ckpt,
        )
    ax.axhline(P.CHANCE, ls="--", lw=0.8, color="grey", label="chance")
    ax.set(
        xlabel="answer_confidence (bin mean)",
        ylabel="accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Reliability",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_laya_variants(acc, top_share):
    """Every prompt variant per checkpoint: accuracy, and the share of the samples sent to
    the expert predicted most (1.0 = every sample to one expert)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.32 * len(acc) + 1.6), sharey=True)
    for ax, grid, title, cmap, vmin in [
        (axes[0], acc, "accuracy", "Purples", P.CHANCE),
        (axes[1], top_share, "share of the top expert", "Greys", P.CHANCE),
    ]:
        sns.heatmap(
            grid.astype(float),
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=1,
            cbar=False,
            ax=ax,
            annot_kws={"fontsize": 8},
        )
        ax.set(title=title, xlabel="", ylabel="")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Prompt variants (options | instructions | state)")
    fig.tight_layout()
    return fig
