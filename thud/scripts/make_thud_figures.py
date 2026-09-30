"""Write the thesis figures into thud/images/.

All parts:
make figures (or uv run python scripts/make_thud_figures.py)

A single part (faster):
uv run python scripts/make_thud_figures.py dataset    one part (dataset, models, overview)

Figures are drawn by `analysis.plots`, as in the notebooks.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import dataset as dslib
from analysis import plots
from analysis import probe as P

IMAGES = Path(__file__).resolve().parents[1] / "images"


def save(fig, subdir: str, name: str) -> str:
    out = IMAGES / subdir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{subdir}/{name}.pdf"


def dataset() -> list[str]:
    sub = "plots/anova"
    frames = {kind: dslib.frame(kind) for kind in P.KINDS}
    anovas = {kind: dslib.anova(d, features) for kind, (d, features) in frames.items()}
    written = [
        save(
            plots.plot_gold_effect({k: d for k, (d, _) in frames.items()}, anovas),
            sub,
            "dataset_gold",
        )
    ]
    for kind in P.KINDS:
        d, features = frames[kind]
        written += [
            save(plots.plot_composition(kind), sub, f"dataset_composition_{kind}"),
            save(
                plots.plot_posthoc(dslib.posthoc(d, anovas[kind]), kind=kind),
                sub,
                f"dataset_posthoc_{kind}",
            ),
            save(
                plots.plot_ceiling(dslib.ceiling(d, features), kind),
                sub,
                f"dataset_ceiling_{kind}",
            ),
            save(
                plots.plot_shap(dslib.shap_importance(d, features), kind),
                sub,
                f"dataset_shap_{kind}",
            ),
        ]
    return written


def models() -> list[str]:
    sub, written = "models", []
    for model, kind in P.runs():
        d = P.result(model, kind)
        slug = f"{model}_{kind}".replace(".", "")
        written += [
            save(plots.plot_layers(d), sub, f"{slug}_layers"),
            save(plots.plot_compare(d), sub, f"{slug}_routers"),
        ]
    # the same run probed at the last reasoning token (gpt-oss-20b), the last section of
    # nb/probing.py
    for model, kind in P.reasoning_runs():
        d = P.result(model, kind, site="reasoning")
        slug = f"{model}_{kind}_reasoning".replace(".", "")
        written += [
            save(plots.plot_layers(d), sub, f"{slug}_layers"),
            save(plots.plot_compare(d), sub, f"{slug}_routers"),
        ]
    return written


def overview() -> list[str]:
    sub = "plots/overview"
    results = {f"{model} {kind}": P.result(model, kind) for model, kind in P.runs()}
    # every chapter-8 cross-run figure leaves gpt-oss-20b out
    pooled = P.pooled_runs(results)
    return [
        save(plots.plot_tiers(pooled), sub, "overview_tiers"),
        save(
            plots.plot_multi_turn_cost(P.multi_turn_cost(pooled)),
            sub,
            "overview_multi_turn",
        ),
        save(
            plots.plot_dependence(P.dependence_summary(pooled)),
            sub,
            "overview_dependence",
        ),
        save(plots.plot_pairs(P.pair_summary(pooled)), sub, "overview_pairs"),
        save(
            plots.plot_decoy(P.error_direction_summary(pooled)), sub, "overview_decoy"
        ),
        save(plots.plot_depth_peaks(pooled), sub, "overview_depth"),
    ]


PARTS = {"dataset": dataset, "models": models, "overview": overview}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "parts", nargs="*", choices=list(PARTS), help="parts to write (default: all)"
    )
    parts = parser.parse_args().parts or list(PARTS)
    written = [f for part in parts for f in PARTS[part]()]
    print("wrote:", *sorted(written), sep="\n  ")
