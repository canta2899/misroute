"""Static configuration: taxonomy"""

# --- models ------------------------------------------------------------------


def generator_model(kind: str = "", construction: str = "") -> str:
    if kind == "multi_turn" and construction == "adversarial":
        return "anthropic/claude-sonnet-5"
    return "anthropic/claude-haiku-4-5-20251001"


ORACLE_MODEL = "anthropic/claude-sonnet-5"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

MAX_TOKENS = {
    "facts": 1024,
    "single": 1024,
    "multi": 3072,
    "audit": 512,
}

SEED = 20260724


EXPERTS = {
    "time_to_market": {
        "name": "Time-to-Market",
        "handles": (
            "Absolute time-to-market. Accepts hardcoding, skipped abstractions "
            "and technical debt to get something demonstrable now."
        ),
    },
    "scale_arch": {
        "name": "Scalable Architecture",
        "handles": (
            "Future concurrency and horizontal growth. Decouples components, "
            "removes shared state, designs for many times the current load."
        ),
    },
    "resource_opt": {
        "name": "Resource Optimization",
        "handles": (
            "Hard CPU, memory or cost ceilings. Strips abstractions and takes "
            "low-level control to fit inside a fixed budget."
        ),
    },
    "legacy_compat": {
        "name": "Legacy Compatibility",
        "handles": (
            "Strict backward compatibility. Bound by published API contracts, "
            "existing clients and outdated dependencies that cannot move."
        ),
    },
    "strict_compliance": {
        "name": "Strict Compliance",
        "handles": (
            "Security, privacy and regulatory standards. Zero-trust data "
            "handling, auditability, and rules that cannot be traded away."
        ),
    },
}

EXPERT_IDS = tuple(EXPERTS)
CHANCE = 1 / len(EXPERT_IDS)


# The rule that makes "which constraint wins" decidable
RESOLUTION_RULE = """\
Exactly one requirement in the text is BINDING: someone with authority over this
team - their org, a customer, a regulator, a budget owner - is holding them to it
right now, and it is not open for discussion. The correct expert is the one that
owns that requirement.

Every other requirement has been WAIVED: whoever owns it has explicitly relieved
this team of it for the moment - signed off on the current behaviour, granted an
exception, accepted the tradeoff, moved it to a later milestone. A waived
requirement can still be serious, can still be what the developer cares about
most, and can still take up most of the message."""


COLLISIONS = [
    {
        "pair": ("time_to_market", "scale_arch"),
        "tension": "get it working today vs. build it for many times this load",
    },
    {
        "pair": ("time_to_market", "resource_opt"),
        "tension": "the fastest path to build vs. the cheapest path to run",
    },
    {
        "pair": ("time_to_market", "legacy_compat"),
        "tension": "hack it in now vs. never break an existing contract",
    },
    {
        "pair": ("time_to_market", "strict_compliance"),
        "tension": "bypass the boundary to ship vs. the boundary is not optional",
    },
    {
        "pair": ("scale_arch", "resource_opt"),
        "tension": "spread the work across more machines vs. fit it on the machines you pay for",
    },
    {
        "pair": ("scale_arch", "legacy_compat"),
        "tension": "reshape the topology vs. keep the published interface intact",
    },
    {
        "pair": ("scale_arch", "strict_compliance"),
        "tension": "move and replicate data freely vs. data may not leave its boundary",
    },
    {
        "pair": ("resource_opt", "legacy_compat"),
        "tension": "strip the abstraction out vs. the signature must not change",
    },
    {
        "pair": ("resource_opt", "strict_compliance"),
        "tension": "cut the overhead vs. the overhead is a required control",
    },
    {
        "pair": ("legacy_compat", "strict_compliance"),
        "tension": "preserve the old behaviour vs. the old behaviour is now disallowed",
    },
]


def partners(expert: str) -> list[str]:
    """The four experts `expert` collides with."""
    out = []
    for c in COLLISIONS:
        a, b = c["pair"]
        if a == expert:
            out.append(b)
        elif b == expert:
            out.append(a)
    return out


def tension(a: str, b: str) -> str:
    for c in COLLISIONS:
        if set(c["pair"]) == {a, b}:
            return c["tension"]
    raise KeyError(f"no collision entry for {a!r} / {b!r}")


CONSTRUCTIONS = {
    # one constraint stated cleanly
    "plain": {"mode": "single", "per_expert": 30},
    # realist query where several constraints are in play but the mandate is stated plainly
    "natural": {"mode": "single", "per_expert": 120},
    # contrastive minimal pairs, two queries differing only in the pivot that names the mandate
    "adversarial": {"mode": "pair", "per_expert": 180},
}

CONSTRUCTION_IDS = tuple(CONSTRUCTIONS)


def mode(construction: str) -> str:
    return CONSTRUCTIONS[construction]["mode"]


def per_expert(construction: str) -> int:
    return CONSTRUCTIONS[construction]["per_expert"]


# how to treat the decoy in the contrastive pair
PATTERNS = {
    "buried_rule": (
        "The user asks how to do a specific thing (their X). What actually "
        "decides the routing is a rule they mention in passing as the reason "
        "they are stuck with X in the first place."
    ),
    "wrong_cause": (
        "The user confidently asserts the wrong constraint is the problem and "
        "spends most of the message on it. The real mandate appears as a "
        "condition they state matter-of-factly, as if it were obvious."
    ),
    "late_mandate": (
        "The whole message is about the decoy concern. The mandate arrives at "
        "the very end, tacked on ('oh and we can't ...', 'forgot to say ...'), "
        "in a flat tone that makes it sound minor."
    ),
    "bad_fix": (
        "The user proposes a concrete fix that serves the decoy constraint and "
        "asks for help executing it. Their own description of the situation "
        "contains the mandate that rules the fix out."
    ),
}

PATTERN_IDS = tuple(PATTERNS)

PERSONAS = {
    "normal_writing": "Ordinary professional developer prose. Complete sentences, correct punctuation.",
    "frustrated": "Tired and irritated. Blunt, some venting, occasional sentence fragments. No profanity.",
    "casual": "Relaxed chat register. Contractions, lowercase starts, a little sloppy.",
    "extremely_casual": "Rushed instant message. Mostly lowercase, minimal punctuation, abbreviations, no greeting.",
}

PERSONA_IDS = tuple(PERSONAS)


MT_MESSAGES = 5  # user, assistant, user, assistant, user: 2 turns + 1 final request


BANNED_SUBSTRINGS = ("```", "[log", "[attach", "![", "<img", "$$")
MIN_CHARS = 120
MAX_CHARS = 1600

MIN_PAIR_OVERLAP = {"single_turn": 0.60, "multi_turn": 0.60}
DEDUP_COSINE_THRESHOLD = 0.95
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}
