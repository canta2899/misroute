import datetime
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config
from .store import KINDS, read_json, read_jsonl, write_json

ANALYSIS_COLUMNS = [
    "id",
    "request_id",
    "kind",
    "scenario_id",
    "construction",
    "gold",
    "decoy",
    "collision",
    "persona",
    "pattern",
    "variant",
    "pivot",
    "oracle_predicted",
    "oracle_correct",
    "oracle_decidable",
    "oracle_quote",
    "oracle_constraints",
    "structural_ok",
    "duplicate_of",
    "duplicate_similarity",
    "pair_overlap",
    "review_decision",
    "regenerate",
    "n_turns",
    "turns",
    "text",
    "user_text",
]

FACTORS = [
    "gold",
    "decoy",
    "collision",
    "construction",
    "persona",
    "pattern",
    "variant",
]

FIELD_DESCRIPTIONS = {
    "id": "unique sample id",
    "request_id": "id shared by both members of an adversarial pair",
    "kind": "single_turn or multi_turn",
    "construction": "plain / natural / adversarial - how the sample was built",
    "gold": "the expert the resolution rule says must handle this query",
    "decoy": "the competing expert the query also raises, empty for plain samples",
    "collision": "gold>decoy - which mandate beats which preference, 'none' for plain samples",
    "scenario_id": "the seed technical situation this sample is built from",
    "persona": "voice the developer query is written in",
    "pattern": "prompt template family used to render the query",
    "variant": "which rendering of the pattern this row is",
    "pivot": "adversarial pairs only - the phrase that flips the binding expert",
    "generator_model": "model that wrote the sample text",
    "generated_at": "timestamp the sample was generated",
    "oracle_predicted": "expert the rubric judge assigned",
    "oracle_correct": "whether the oracle's verdict matches gold",
    "oracle_decidable": "whether the judge found the sample decidable at all",
    "oracle_quote": "quote the judge cited as evidence",
    "oracle_constraints": "constraints the judge saw present in the text",
    "oracle_model": "model that ran the oracle judge",
    "structural_ok": "passes the structural checks in checks.py",
    "structural_errors": "which structural checks failed, if any",
    "duplicate_of": "id of a near-duplicate sample, if one was found",
    "duplicate_similarity": "similarity score to that duplicate",
    "pair_overlap": "word overlap between this sample and its adversarial twin",
    "regenerate": "1 if ds repair will delete and regenerate this sample",
    "review_decision": "outcome of the manual review, empty if not reviewed",
    "review_note": "free-text note left by the reviewer",
    "n_turns": "number of turns in the conversation",
    "turns": "the conversation as json - what a model is actually prompted with",
    "text": "full transcript, one line per turn (for reading, not for parsing)",
    "user_text": "developer turns only, concatenated",
}


def render_turns(turns) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns or [])


def user_text(sample: dict) -> str:
    return " ".join(
        t["content"] for t in sample.get("turns") or [] if t["role"] == "user"
    )


def collision(sample: dict) -> str:
    decoy = sample.get("decoy")
    return f"{sample.get('gold')}>{decoy}" if decoy else "none"


def to_frame(rows: list[dict]) -> pd.DataFrame:
    columns: list[str] = []
    for row in rows:
        columns.extend(k for k in row if k != "turns" and k not in columns)
    derived = ["collision", "n_turns", "turns", "text", "user_text"]
    columns = [c for c in columns if c not in derived]
    data = [
        {
            **{k: _cell(row.get(k)) for k in columns},
            "collision": collision(row),
            "n_turns": len(row.get("turns") or []),
            "turns": _cell(row.get("turns") or []),
            "text": render_turns(row.get("turns")),
            "user_text": user_text(row),
        }
        for row in rows
    ]
    return pd.DataFrame(data, columns=[*columns, *derived])


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def frame(rows: list[dict], columns: list[str] | None = None) -> pd.DataFrame:
    out = to_frame(rows)
    if columns is not None:
        out = out[[c for c in columns if c in out.columns]]
    return out  # pyright: ignore[reportReturnType]


def write_csv(rows: list[dict], path, columns: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame(rows, columns).to_csv(path, index=False)
    return path


def field_choices(rows: list[dict]) -> list[str]:
    return list(to_frame(rows[:1]).columns) if rows else list(FIELD_DESCRIPTIONS)


def field_label(name: str) -> str:
    return f"{name} - {FIELD_DESCRIPTIONS.get(name, 'no description available')}"


def metadata(
    rows: list[dict],
    kind: str,
    columns: list[str],
    *,
    run_id: str | None = None,
    source: str | None = None,
    reports: dict | None = None,
) -> dict:
    frame_ = frame(rows, columns)
    return {
        "dataset": kind,
        "run_id": run_id,
        "source": source,
        "n_samples": len(rows),
        "n_columns": len(frame_.columns),
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "chance": round(config.CHANCE, 4),
        "experts": {e: config.EXPERTS[e]["name"] for e in config.EXPERT_IDS},
        "resolution_rule": config.RESOLUTION_RULE,
        "columns": {c: FIELD_DESCRIPTIONS.get(c, "") for c in frame_.columns},
        "factors": {
            f: dict(Counter(frame_[f].astype(str)))
            for f in FACTORS
            if f in frame_.columns
        },
        "reports": reports or {},
    }


REPORTS = ("baselines", "audit", "pairs", "duplicates", "structure")


def export(run, out: Path, *, edited: bool = False, keep_flagged: bool = False) -> list[dict]:
    """Write `<out>/<kind>.csv` and `<kind>.meta.json` for every kind of a run."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for kind in KINDS:
        source = run.samples(kind)
        if edited and source.with_name("samples.edited.jsonl").exists():
            source = source.with_name("samples.edited.jsonl")
        rows = read_jsonl(source)
        if not rows:
            continue
        kept = rows if keep_flagged else [r for r in rows if not r.get("regenerate")]
        csv_path = write_csv(kept, out / f"{kind}.csv", ANALYSIS_COLUMNS)
        reports = {
            name: read_json(run.reports(kind) / f"{name}.json")
            for name in REPORTS
            if (run.reports(kind) / f"{name}.json").exists()
        }
        meta = metadata(kept, kind, ANALYSIS_COLUMNS, run_id=run.id, source=str(source),
                        reports=reports)
        write_json(out / f"{kind}.meta.json", meta)
        written.append({"kind": kind, "rows": len(kept), "dropped": len(rows) - len(kept),
                        "csv": csv_path})
    return written
