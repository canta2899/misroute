import argparse
import sys
from pathlib import Path

import pandas as pd

ID = "id"
RESPONSE = "raw_response"
PREDICTED = "predicted_expert"
HIDDEN = "hidden_states"
SHAPE_COLUMNS = ("n_layers", "hidden_dim", "hidden_dtype")


def read_model(model_dir: Path) -> pd.DataFrame:
    shards = sorted(model_dir.glob("shard_*.parquet"))
    if not shards:
        raise SystemExit(f"no shard_*.parquet in {model_dir}")
    df = pd.concat((pd.read_parquet(p) for p in shards), ignore_index=True)
    missing = [c for c in (ID, RESPONSE, HIDDEN) if c not in df.columns]
    if missing:
        raise SystemExit(f"{model_dir}: shards are missing {missing}")
    return df.drop_duplicates(subset=ID, keep="last")


def merge(
    dataset_dir: Path,
    sim_dir: Path,
    kind: str,
    out_dir: Path,
    only: list[str] | None = None,
) -> Path | None:
    csv_path = dataset_dir / f"{kind}.csv"
    if not csv_path.exists():
        print(f"skip {kind}: no {csv_path}")
        return None

    base = pd.read_csv(csv_path)
    models = sorted(p.name for p in sim_dir.iterdir() if (p / kind).is_dir())
    if only:
        models = [m for m in models if m in only]
    if not models:
        print(f"skip {kind}: no model folders with a {kind}/ under {sim_dir}")
        return None

    for model in models:
        df = read_model(sim_dir / model / kind)
        columns = {
            ID: ID,
            RESPONSE: f"{model}_response",
            HIDDEN: f"{model}_hidden_states",
            PREDICTED: f"{model}_predicted",
            **{c: f"{model}_{c}" for c in SHAPE_COLUMNS if c in df.columns},
        }
        columns = {k: v for k, v in columns.items() if k in df.columns}
        base = base.merge(df[list(columns)].rename(columns=columns), on=ID, how="left")  # pyright: ignore[reportCallIssue]
        # a model that ran on only part of the dataset leaves NaNs behind, which
        # would turn the shape columns into floats reshape() refuses
        for name in ("n_layers", "hidden_dim"):
            column = f"{model}_{name}"
            if column in base.columns:
                base[column] = base[column].astype("Int64")

        covered = int(base[f"{model}_response"].notna().sum())  # pyright: ignore[reportArgumentType]
        print(
            f"{kind}/{model}: {len(df)} shard rows, {covered}/{len(base)} samples matched"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # one model per file is the nb/probing convention (`<model>_<kind>.parquet`,
    # which is where the notebook reads the model name from); a merge of several
    # models has no single name to take, so it falls back to the kind alone
    out = out_dir / (
        f"{models[0]}_{kind}.parquet" if len(models) == 1 else f"{kind}.parquet"
    )
    base.to_parquet(out, index=False)
    print(f"{kind}: {len(base)} rows x {len(base.columns)} columns -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="folder `ds export` wrote",
    )
    ap.add_argument(
        "--sim-dir",
        type=Path,
        default=Path("data/sim"),
        help="folder holding <model>/<kind>/shard_*.parquet",
    )
    ap.add_argument(
        "--kind", choices=("single_turn", "multi_turn"), help="default: both"
    )
    ap.add_argument(
        "--models", help="comma-separated model folders to merge (default: all)"
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="destination (default: <dataset-dir>/merged)",
    )
    args = ap.parse_args()

    if not args.sim_dir.is_dir():
        sys.exit(f"no simulation output at {args.sim_dir}")
    out_dir = args.out or args.dataset_dir / "merged"

    kinds = [args.kind] if args.kind else ["single_turn", "multi_turn"]
    only = args.models.split(",") if args.models else None
    if not any(merge(args.dataset_dir, args.sim_dir, k, out_dir, only) for k in kinds):
        sys.exit("nothing merged")


if __name__ == "__main__":
    main()
