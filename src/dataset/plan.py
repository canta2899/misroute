import random

from . import config

_SHORT = {"plain": "pln", "natural": "nat", "adversarial": "adv"}
_PREFIX = {"single_turn": "st", "multi_turn": "mt"}


def targets(scale: float = 1.0) -> dict[str, int]:
    out = {}
    for construction in config.CONSTRUCTION_IDS:
        n = config.per_expert(construction) * scale
        if config.mode(construction) == "pair":
            out[construction] = max(4, 4 * round(n / 4))
        else:
            out[construction] = max(1, round(n))
    return out


def _spread(values, n: int, rng) -> list:
    values = list(values)
    seq = (values * (n // len(values) + 1))[:n]
    rng.shuffle(seq)
    return seq


def build(scenarios: list[dict], kinds=("single_turn", "multi_turn"), scale=1.0):
    rng = random.Random(config.SEED)
    scenario_ids = [a["id"] for a in scenarios]
    per_construction = targets(scale)

    rows = []
    for kind in kinds:
        counter = {c: 0 for c in config.CONSTRUCTION_IDS}
        for construction in config.CONSTRUCTION_IDS:
            n = per_construction[construction]
            if config.mode(construction) == "single":
                rows += _single_rows(kind, construction, n, scenario_ids, rng, counter)
            else:
                rows += _pair_rows(kind, construction, n, scenario_ids, rng, counter)
    return rows


def _row_id(kind: str, construction: str, n: int, variant: str | None) -> str:
    return f"{_PREFIX[kind]}-{_SHORT[construction]}-{n:05d}{variant or ''}"


def _base(kind, construction, gold, decoy, scenario, persona, pattern, rid, variant):
    return {
        "id": _row_id(kind, construction, rid, variant),
        "request_id": _row_id(kind, construction, rid, None),
        "kind": kind,
        "construction": construction,
        "gold": gold,
        "decoy": decoy,
        "scenario_id": scenario,
        "persona": persona,
        "pattern": pattern,
        "variant": variant,
    }


def _single_rows(kind, construction, n, scenario_ids, rng, counter) -> list[dict]:
    rows = []
    for gold in config.EXPERT_IDS:
        scenarios = _spread(scenario_ids, n, rng)
        personas = _spread(config.PERSONA_IDS, n, rng)
        decoys = _spread(config.partners(gold), n, rng)
        for i in range(n):
            counter[construction] += 1
            rows.append(
                _base(
                    kind,
                    construction,
                    gold,
                    decoys[i] if construction != "plain" else None,
                    scenarios[i],
                    personas[i],
                    None,
                    counter[construction],
                    None,
                )
            )
    return rows


def _pair_rows(kind, construction, n, scenario_ids, rng, counter) -> list[dict]:
    k = n // 4
    rows = []
    for collision in config.COLLISIONS:
        a, b = collision["pair"]
        scenarios = _spread(scenario_ids, k, rng)
        personas = _spread(config.PERSONA_IDS, k, rng)
        patterns = _spread(config.PATTERN_IDS, k, rng)
        for i in range(k):
            counter[construction] += 1
            rid = counter[construction]
            for variant, gold, decoy in (("a", a, b), ("b", b, a)):
                rows.append(
                    _base(
                        kind,
                        construction,
                        gold,
                        decoy,
                        scenarios[i],
                        personas[i],
                        patterns[i],
                        rid,
                        variant,
                    )
                )
    return rows


def by_kind(rows: list[dict], kind: str) -> list[dict]:
    return [r for r in rows if r["kind"] == kind]


def requests(rows: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["request_id"], []).append(row)
    return [sorted(g, key=lambda r: r["id"]) for g in groups.values()]


def summarize(rows: list[dict]) -> dict:
    from collections import Counter

    out = {"total": len(rows)}
    for kind in sorted({r["kind"] for r in rows}):
        subset = by_kind(rows, kind)
        out[kind] = {  # type: ignore
            "total": len(subset),
            "by_construction": dict(Counter(r["construction"] for r in subset)),
            "by_expert": dict(Counter(r["gold"] for r in subset)),
        }
    return out
