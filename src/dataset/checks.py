"""Structural checks on generated samples"""

import re

from . import config


def user_text(sample: dict) -> str:
    return "\n".join(t["content"] for t in sample["turns"] if t["role"] == "user")


def transcript(sample: dict) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in sample["turns"])


_WORD = re.compile(r"[a-z0-9']+")


def overlap(a: str, b: str) -> float:
    """Jaccard over word sets"""
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def problems(sample: dict) -> list[str]:
    out = []
    turns = sample.get("turns") or []
    roles = [t.get("role") for t in turns]

    if not turns or any(
        r not in ("user", "assistant") or not str(t.get("content", "")).strip()
        for r, t in zip(roles, turns)
    ):
        return ["malformed turns"]

    if sample["kind"] == "single_turn":
        if roles != ["user"]:
            out.append("single_turn must be exactly one user turn")
    else:
        if len(turns) != config.MT_MESSAGES:
            out.append(f"expected {config.MT_MESSAGES} turns, got {len(turns)}")
        if roles[:1] != ["user"] or roles[-1:] != ["user"]:
            out.append("conversation must start and end with the developer")
        if any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)):
            out.append("roles must alternate")

    text = user_text(sample)
    for banned in config.BANNED_SUBSTRINGS:
        if banned in text:
            out.append(f"contains banned markup: {banned!r}")

    n_user = sum(1 for r in roles if r == "user")
    if len(text) < config.MIN_CHARS * n_user:
        out.append(f"too short ({len(text)} chars over {n_user} developer turns)")
    if len(text) > config.MAX_CHARS * n_user:
        out.append(f"too long ({len(text)} chars over {n_user} developer turns)")

    for eid in config.EXPERT_IDS:
        if eid in text or config.EXPERTS[eid]["name"].lower() in text.lower():
            out.append(f"names the expert {eid}")

    return out


def validate(samples: list[dict]) -> tuple[list[dict], dict]:
    """Split samples into usable and rejected"""
    good, rejected = [], []
    seen = set()
    for s in samples:
        errors = problems(s)
        if s["id"] in seen:
            errors.append("duplicate id")
        seen.add(s["id"])
        if errors:
            rejected.append(
                {"id": s["id"], "request_id": s["request_id"], "errors": errors}
            )
        else:
            good.append(s)
    return good, {
        "n_input": len(samples),
        "n_valid": len(good),
        "n_rejected": len(rejected),
        "rejected": rejected,
    }


def contrast_text(sample: dict) -> str:
    if sample["kind"] == "multi_turn":
        return sample["turns"][-1]["content"]
    return user_text(sample)


def min_pair_overlap(kind: str) -> float:
    return config.MIN_PAIR_OVERLAP[kind]


def pair_overlaps(samples: list[dict]) -> tuple[list[dict], dict]:
    """Check that each adversarial pair really is a minimal pair."""
    pairs: dict[str, list[dict]] = {}
    for s in samples:
        if s["construction"] == "adversarial":
            pairs.setdefault(s["request_id"], []).append(s)

    thresholds = {min_pair_overlap(s["kind"]) for s in samples} or {None}
    scores, weak = [], []
    for request_id, members in sorted(pairs.items()):
        if len(members) != 2:
            weak.append(
                {"request_id": request_id, "overlap": None, "reason": "incomplete pair"}
            )
            continue
        a, b = contrast_text(members[0]), contrast_text(members[1])
        score = overlap(a, b)
        scores.append(score)
        if a.strip() == b.strip():
            weak.append(
                {
                    "request_id": request_id,
                    "overlap": 1.0,
                    "reason": "identical members",
                }
            )
        elif score < min_pair_overlap(members[0]["kind"]):
            weak.append({"request_id": request_id, "overlap": round(score, 3)})

    mean = round(sum(scores) / len(scores), 4) if scores else None
    return weak, {
        "n_pairs": len(pairs),
        "mean_overlap": mean,
        "min_overlap_threshold": min(thresholds),  # pyright: ignore[reportArgumentType]
        "n_weak": len(weak),
        "weak": weak,
    }
