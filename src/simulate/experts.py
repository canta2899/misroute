from dataclasses import dataclass

from dataset.config import COLLISIONS
from dataset.config import EXPERTS as _EXPERT_SPECS


@dataclass(frozen=True)
class Expert:
    key: str  # routing label, also used as the tool name
    name: str  # human-readable name
    handles: str  # what this expert handles
    example: str  # a one-shot developer request that routes here


_EXAMPLES = {
    "time_to_market": "we're demoing the ingest flow to the buyer on thursday and the deal is contingent on it running end to end. i'd rather wire the mapping properly but that can wait.",
    "scale_arch": "product has committed us to onboarding the three biggest tenants next quarter, which is roughly twenty times the events we take now. our worker pulls from one queue and holds a db connection open the whole time.",
    "resource_opt": "finance capped our per-node spend and we can't move off the 2gb instances until next fiscal year. the parser copies every frame into a fresh buffer before decoding it.",
    "legacy_compat": "two external customers still call the v1 export endpoint and their contracts say we can't change its response shape for another year. i want to rework what happens behind it.",
    "strict_compliance": "our dpo signed off on a retention policy that says raw uploads never leave the audited bucket. right now the fine-tuning job reads straight from it by filename prefix.",
}

EXPERTS = [
    Expert(key, spec["name"], spec["handles"], _EXAMPLES[key])
    for key, spec in _EXPERT_SPECS.items()
]

EXPERT_KEYS = [e.key for e in EXPERTS]


def _collision_notes() -> str:
    return "\n".join(
        f"- {a} vs {b}: {c['tension']}" for c in COLLISIONS for a, b in [c["pair"]]
    )


SYSTEM_PROMPT = (
    "You are a coordinator that routes an incoming developer request to exactly "
    "one specialist expert. Read the request, decide which single expert should "
    "handle it, and route it by calling that expert's tool exactly once. The "
    "experts are not topic areas: they look at the same system and optimise for "
    "different constraints, so route on which constraint the developer is "
    "actually held to, not on which one they talk about most.\n\n"
    "Pairs that pull against each other:\n"
    f"{_collision_notes()}"
)
