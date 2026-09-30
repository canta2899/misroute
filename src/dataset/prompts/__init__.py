from .. import config
from .loader import render
from .tools import (
    turn_keys,
    FACTS as FACTS_TOOL,
    MULTI as MULTI_TOOL,
    MULTI_PAIR as MULTI_PAIR_TOOL,
    ORACLE as ORACLE_TOOL,
    SINGLE as SINGLE_TOOL,
    SINGLE_PAIR as SINGLE_PAIR_TOOL,
)

__all__ = [
    "turn_keys",
    "FACTS_TOOL",
    "SINGLE_TOOL",
    "SINGLE_PAIR_TOOL",
    "MULTI_TOOL",
    "MULTI_PAIR_TOOL",
    "ORACLE_TOOL",
    "GENERATION_SYSTEM",
    "FACTS_SYSTEM",
    "ORACLE_SYSTEM",
    "facts_user",
    "single_user",
    "single_pair_user",
    "multi_user",
    "multi_pair_user",
    "audit_user",
]


EXPERTS = "\n".join(
    f"- {eid} ({spec['name']}): {spec['handles']}" for eid, spec in config.EXPERTS.items()
)

COLLISIONS = "\n".join(
    f"- {a} vs {b}: {c['tension']}" for c in config.COLLISIONS for a, b in [c["pair"]]
)

_TAXONOMY = {
    "experts": EXPERTS,
    "collisions": COLLISIONS,
    "resolution_rule": config.RESOLUTION_RULE,
}

GENERATION_SYSTEM = render("system/generation", **_TAXONOMY)
FACTS_SYSTEM = render("system/facts", experts=EXPERTS)
ORACLE_SYSTEM = render("system/oracle", **_TAXONOMY)


def facts_user(scenario: dict) -> str:
    return render("user/facts", scenario=scenario["text"], experts=EXPERTS)

def _standalone(row: dict, scenario: dict, facts: dict) -> dict:
    values = {
        "scenario": scenario["text"],
        "persona": config.PERSONAS[row["persona"]],
        "fact_gold": facts[row["gold"]],
    }
    if row["construction"] != "plain":
        values["fact_decoy"] = facts[row["decoy"]]
        values["tension"] = config.tension(row["gold"], row["decoy"])
    return values


def _pair(rows: list[dict], scenario: dict, facts: dict) -> dict:
    north, south = rows[0]["gold"], rows[1]["gold"]
    return {
        "scenario": scenario["text"],
        "persona": config.PERSONAS[rows[0]["persona"]],
        "pattern": config.PATTERNS[rows[0]["pattern"]],
        "fact_north": facts[north],
        "fact_south": facts[south],
        "tension": config.tension(north, south),
    }


def single_user(row: dict, scenario: dict, facts: dict) -> str:
    """Prompt for one `plain` or `natural` single-turn query."""
    return render(f"user/single_{row['construction']}", **_standalone(row, scenario, facts))


def single_pair_user(rows: list[dict], scenario: dict, facts: dict) -> str:
    """Prompt for one adversarial contrastive pair (two rows, one call)."""
    return render("user/single_pair", **_pair(rows, scenario, facts))


def multi_user(row: dict, scenario: dict, facts: dict) -> str:
    """Prompt for one `plain` or `natural` multi-turn conversation."""
    return render(f"user/multi_{row['construction']}", **_standalone(row, scenario, facts))


def multi_pair_user(rows: list[dict], scenario: dict, facts: dict) -> str:
    """Prompt for one adversarial pair: a shared opening and two final turns."""
    return render("user/multi_pair", **_pair(rows, scenario, facts))


def audit_user(text: str) -> str:
    return render("user/audit", text=text)
