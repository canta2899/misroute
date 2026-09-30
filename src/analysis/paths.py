"""Where the analysis reads and writes."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

NB = REPO_ROOT / "nb"
DATA = NB / "data"  # merged parquets, one per (model, kind)
CACHE = DATA / "cache"  # layer sweeps, written per layer
BASELINE_CACHE = (
    DATA / ".baselines.json"
)  # text-only baselines, keyed by sample-id hash
DATASET_DIR = REPO_ROOT / "dataset"  # the exported csvs the baselines read
LAYA = (
    REPO_ROOT / "data" / "sim" / "laya"
)  # simulate/laya_app.py output, one csv per kind
LAYA_FT = LAYA / "finetune"  # simulate/laya_finetune_app.py output, one csv per kind
MODERNBERT_FROZEN = (
    REPO_ROOT / "data" / "sim" / "modernbert" / "frozen"
)  # simulate/modernbert_frozen_app.py output, one csv per kind
