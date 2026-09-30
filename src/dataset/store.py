import datetime
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SCENARIOS_PATH = Path(__file__).with_name("scenarios.json")

KINDS = ("single_turn", "multi_turn")


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", "utf-8")


def load_scenarios() -> list[dict]:
    return read_json(SCENARIOS_PATH)


@dataclass(frozen=True)
class Run:
    id: str

    @property
    def dir(self) -> Path:
        return DATA_DIR / self.id

    @property
    def plan(self) -> Path:
        return self.dir / "plan.jsonl"

    @property
    def facts(self) -> Path:
        return self.dir / "facts.jsonl"

    def kind_dir(self, kind: str) -> Path:
        _check_kind(kind)
        return self.dir / kind

    def samples(self, kind: str) -> Path:
        return self.kind_dir(kind) / "samples.jsonl"

    def audit(self, kind: str) -> Path:
        return self.kind_dir(kind) / "audit.jsonl"

    def reports(self, kind: str) -> Path:
        return self.kind_dir(kind) / "reports"

    def llm_dir(self, kind_or_stage: str) -> Path:
        """Raw provider traffic. `facts` gets its own folder, kinds get theirs."""
        if kind_or_stage in KINDS:
            return self.kind_dir(kind_or_stage) / "llm"
        return self.dir / kind_or_stage / "llm"

    def exists(self) -> bool:
        return self.plan.exists()


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.datetime.now():%Y%m%d-%H%M%S}"


def latest_run() -> Run | None:
    if not DATA_DIR.exists():
        return None
    runs = [p for p in DATA_DIR.iterdir() if (p / "plan.jsonl").exists()]
    if not runs:
        return None
    return Run(max(runs, key=lambda p: p.stat().st_mtime).name)


def open_run(run_id: str | None) -> Run:
    if run_id:
        run = Run(run_id)
        if not run.exists():
            raise ValueError(f"run {run_id!r} has no plan.jsonl at {run.dir}")
        return run
    run = latest_run()
    if run is None:
        raise ValueError("no runs under data/; create one with `ds plan`")
    return run
