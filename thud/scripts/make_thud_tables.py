"""Print every number thesis chapters 7-8 report, for copying into thud/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from analysis import probe as P

pd.set_option(
    "display.width",
    250,
    "display.max_columns",
    50,
    "display.max_rows",
    100,
    "display.float_format",
    "{:.6g}".format,
)


def section(title):
    print(f"\n=================== {title} ===================")


results = {f"{m} {k}": P.result(m, k) for m, k in P.runs()}

section(
    "CHAPTER 7: per-model tables (b = model right / probe wrong, c = model wrong / probe right)"
)
print(P.run_tables("prompt"))

section("CHAPTER 7: every router per tier, probe at the run's optimal layer")
for tier in P.CONSTRUCTIONS:
    print(f"--- {tier} ---")
    print(P.across_runs(P.compare, slice_=tier, results=results)[list(P.ROUTERS)])

section("CHAPTER 8: gpt-oss-20b, last prompt token vs last reasoning token")
for model, kind in P.reasoning_runs():
    print(P.site_table(model, kind))
    print(P.decision_readout(model, kind))

# every cross-run view below leaves gpt-oss-20b out
pooled = P.pooled_runs(results)

section("CHAPTER 8: overview table (gpt-oss-20b excluded)")
print(P.overview(pooled)[[*P.ROUTERS, "margin"]])

section(
    "CHAPTER 8: accuracy per tier, probe at the whole-run peak layer (gpt-oss-20b excluded)"
)
print(P.tier_summary(pooled))
print()
print(P.tier_table(pooled)[["model", "probe", "margin"]])
print()
print(P.tier_drop(pooled))

section("CHAPTER 8: multi-turn drop table (gpt-oss-20b excluded)")
cost = P.multi_turn_cost(pooled)
print(cost)

section("CHAPTER 8: error-flag table (gpt-oss-20b excluded)")
print(
    P.across_runs(P.flags, results=pooled)[
        [
            "precision",
            "recall",
            "f1",
            "base_rate",
            "false_alarm",
            "flagged_share",
            "residual_error",
            "margin",
        ]
    ]
)
print()
print(P.flag_correlations(pooled))

section(
    "CHAPTER 8: failure analysis, error dependence per tier (gpt-oss-20b excluded, mean over runs)"
)
print(P.dependence_summary(pooled))

section(
    "CHAPTER 8: failure analysis, adversarial pairs (gpt-oss-20b excluded, mean over runs)"
)
print(P.pair_summary(pooled))

section(
    "CHAPTER 8: failure analysis, decoy predicted out of all samples (gpt-oss-20b excluded, mean over runs)"
)
print(P.error_direction_summary(pooled))

section("CHAPTER 8: depth summary (gpt-oss-20b excluded)")
print(P.depth_summary(pooled))
print()
print(P.peak_depths(pooled)[["peak_depth"]].sort_values("peak_depth"))  # pyright: ignore[reportCallIssue]
