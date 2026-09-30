"""nb/review.py (marimo) is the front end: this module holds everything that is not widgets"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import checks
from .csvexport import user_text
from .store import DATA_DIR, KINDS, read_jsonl, write_jsonl

ROLES = ("user", "assistant")
REVIEW_CHOICES = ("valid", "fixed", "regenerate")
TABLE_COLUMNS = ["!", "id", "construction", "gold", "oracle", "flags", "preview"]
PREVIEW_CHARS = 90


def folders(root: Path = DATA_DIR) -> list[Path]:
    if not root.exists():
        return []
    runs = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        run / kind
        for run in runs
        for kind in KINDS
        if (run / kind / "samples.jsonl").exists()
    ]


def flags(sample: dict) -> list[str]:
    out = []
    if sample.get("structural_ok") is False:
        out.append("invalid")
    if sample.get("duplicate_of"):
        out.append("duplicate")
    if is_weak_pair(sample):
        out.append("weak_pair")
    if not sample.get("oracle_predicted"):
        out.append("unjudged")
    elif not sample.get("oracle_decidable", True):
        out.append("undecidable")
    elif not sample.get("oracle_correct"):
        out.append("oracle_disagreement")
    if sample.get("review_decision"):
        out.append(f"review:{sample['review_decision']}")
    return out


def is_weak_pair(sample: dict) -> bool:
    overlap = sample.get("pair_overlap")
    return overlap is not None and overlap < checks.min_pair_overlap(sample["kind"])


def is_flagged(sample: dict) -> bool:
    if "regenerate" in sample:
        return bool(sample["regenerate"])
    return bool(flags(sample))


def default_decision(sample: dict) -> str | None:
    return sample.get("review_decision") or (
        "regenerate" if is_flagged(sample) else None
    )


def find_partner(rows: list[dict], sample: dict) -> dict | None:
    if sample.get("construction") != "adversarial":
        return None
    return next(
        (
            s
            for s in rows
            if s["request_id"] == sample["request_id"] and s["id"] != sample["id"]
        ),
        None,
    )


def updated_turns(sample: dict, texts: list[str]) -> list[dict]:
    turns = [
        {"role": turn["role"], "content": text.strip()}
        for turn, text in zip(sample.get("turns") or [], texts)
        if text.strip()
    ]
    return turns or sample["turns"]


def table(rows: list[dict], flagged_only: bool = False) -> pd.DataFrame:
    data = [
        [
            "⚑" if is_flagged(sample) else "",
            sample["id"],
            sample.get("construction"),
            sample.get("gold"),
            _oracle_cell(sample),
            " ".join(flags(sample)),
            _ellipsis(user_text(sample), PREVIEW_CHARS),
        ]
        for sample in rows
        if not flagged_only or is_flagged(sample)
    ]
    return pd.DataFrame(data, columns=TABLE_COLUMNS)


def _oracle_cell(sample: dict) -> str:
    predicted = sample.get("oracle_predicted")
    if not predicted:
        return ""
    mark = "✓" if sample.get("oracle_correct") else "✗"
    return f"{mark} {predicted}" + (
        "" if sample.get("oracle_decidable", True) else " ?"
    )


def _ellipsis(text: str, limit: int) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def header_view(sample: dict) -> str:
    secondary = "  ·  ".join(
        str(b)
        for b in (
            sample.get("pattern"),
            sample.get("variant"),
            sample.get("persona"),
            sample.get("scenario_id"),
        )
        if b
    )
    return (
        f"### `{sample['id']}` - {sample.get('construction')}\n<sub>{secondary}</sub>"
    )


def checks_view(sample: dict) -> str:
    ok = sample.get("structural_ok")
    structural = "ok" if ok else "**failed**"
    if sample.get("structural_errors"):
        structural += ": " + ", ".join(sample["structural_errors"])
    lines = [f"- **structural** - {structural}"]

    overlap = sample.get("pair_overlap")
    if overlap is not None:
        weak = " - **below threshold**" if is_weak_pair(sample) else ""
        lines.append(f"- **pair overlap** - {overlap:.2f}{weak}")

    dup = sample.get("duplicate_of")
    lines.append("- **duplicate** - " + (f"**of `{dup}`**" if dup else "none"))

    regen = bool(sample.get("regenerate"))
    lines.append(f"- **regenerate** - {'**yes**' if regen else 'no'}")
    return "\n".join(lines)


def judge_view(sample: dict) -> str:
    labels = [f"**gold** {sample.get('gold')}"]
    if sample.get("decoy"):
        labels.append(f"**decoy** {sample['decoy']}")
    lines = ["  ·  ".join(labels)]

    predicted = sample.get("oracle_predicted")
    if not predicted:
        lines.append("_not judged yet - run `ds audit`_")
    else:
        if not sample.get("oracle_decidable", True):
            verdict = "**undecidable**"
        else:
            verdict = "agrees" if sample.get("oracle_correct") else "**disagrees**"
        lines.append(f"**oracle** - predicted `{predicted}`, {verdict}")
        if sample.get("oracle_quote"):
            lines.append(f"> {sample['oracle_quote']}")
        if sample.get("oracle_constraints"):
            lines.append(
                "**oracle saw in play** - "
                + ", ".join(f"`{c}`" for c in sample["oracle_constraints"])
            )
    return "\n\n".join(lines)


def pair_view(sample: dict, rows: list[dict]) -> str:
    twin = find_partner(rows, sample)
    if twin is None:
        return ""
    decision = twin.get("review_decision") or "not reviewed yet"
    return (
        f"⚠️ **adversarial pair** - partner is `{twin['id']}`, currently "
        f"**{decision}** (regenerate={twin.get('regenerate')}). Marking "
        "**regenerate** here also flags the partner; **valid**/**fixed** "
        "do not - check it yourself."
    )


def headline_view(folder: Path, rows: list[dict], dirty: int = 0) -> str:
    flagged = sum(1 for r in rows if is_flagged(r))
    reviewed = sum(1 for r in rows if r.get("review_decision"))
    edits = f" - **{dirty} unsaved edits**" if dirty else ""
    return (
        f"**{len(rows)} samples** in `{folder}` - "
        f"{flagged} flagged, {reviewed} reviewed{edits}"
    )


@dataclass
class Session:
    """One folder open for review"""

    folder: Path
    rows: list[dict] = field(default_factory=list)
    dirty: set[str] = field(default_factory=set)

    @property
    def source(self) -> Path:
        return self.folder / "samples.jsonl"

    @property
    def destination(self) -> Path:
        return self.folder / "samples.edited.jsonl"

    @classmethod
    def open(cls, folder) -> "Session":
        session = cls(Path(folder))
        if not session.source.exists():
            raise ValueError(f"no samples.jsonl under {session.folder}")
        session.rows = read_jsonl(session.source)
        return session

    def sample(self, sample_id: str | None) -> dict | None:
        return next((s for s in self.rows if s["id"] == sample_id), None)

    def review(
        self, sample: dict, texts: list[str], decision: str | None, note: str
    ) -> set[str]:
        sample["turns"] = updated_turns(sample, texts)
        sample["review_decision"] = decision
        sample["review_note"] = note or None

        touched = {sample["id"]}
        if decision == "regenerate":
            sample["regenerate"] = 1
            twin = find_partner(self.rows, sample)
            if twin is not None:
                twin["regenerate"] = 1
                twin["review_decision"] = "regenerate"
                touched.add(twin["id"])
        elif decision in ("valid", "fixed"):
            sample["regenerate"] = 0

        self.dirty |= touched
        return touched

    def save(self) -> str:
        write_jsonl(self.destination, self.rows)
        self.dirty = set()
        return f"wrote {len(self.rows)} samples -> {self.destination}"
