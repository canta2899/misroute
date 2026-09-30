"""Tool schemas"""

from .. import config

_STR = {"type": "string"}

_EXPERT_ENUM = {"type": "string", "enum": list(config.EXPERT_IDS)}


def _tool(name: str, description: str, properties: dict) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


def turn_keys(count: int, prefix: str = "") -> list[tuple[str, str]]:
    """(field name, internal role) for a conversation of `count` turns."""
    out = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        speaker = "developer" if role == "user" else "assistant"
        out.append((f"{prefix}turn{i + 1}_{speaker}", role))
    return out


def _turn_fields(count: int, prefix: str = "", what: str = "conversation") -> dict:
    return {
        key: {
            **_STR,
            "description": f"Turn {i + 1} of the {what}, spoken by the "
            f"{'developer' if role == 'user' else 'assistant'}.",
        }
        for i, (key, role) in enumerate(turn_keys(count, prefix))
    }


FACTS = _tool(
    "emit_facts",
    "One operational constraint per specialist for the given system.",
    {
        eid: {**_STR, "description": f"Constraint for {eid}"}
        for eid in config.EXPERT_IDS
    },
)

SINGLE = _tool(
    "emit_query",
    "One developer message.",
    {"query": {**_STR, "description": "The developer's message, plain text."}},
)

SINGLE_PAIR = _tool(
    "emit_query_pair",
    "Two near-identical developer messages that route differently.",
    {
        "query_north_binding": {
            **_STR,
            "description": "Version where requirement NORTH is binding and SOUTH is waived.",
        },
        "query_south_binding": {
            **_STR,
            "description": "Version where requirement SOUTH is binding and NORTH is waived.",
        },
        "pivot_north_binding": {
            **_STR,
            "description": "The sentence in query_north_binding that waives SOUTH.",
        },
        "pivot_south_binding": {
            **_STR,
            "description": "The sentence in query_south_binding that waives NORTH.",
        },
    },
)

MULTI = _tool(
    "emit_conversation",
    f"A developer/assistant conversation of exactly {config.MT_MESSAGES} turns, "
    "one per field, starting and ending with the developer.",
    _turn_fields(config.MT_MESSAGES),
)

MULTI_PAIR = _tool(
    "emit_conversation_pair",
    "A shared conversation opening plus two alternative final developer turns.",
    {
        **_turn_fields(config.MT_MESSAGES - 1, "opening_", "shared opening"),
        "final_north_binding": {
            **_STR,
            "description": (
                "Required. The final developer turn where requirement SOUTH "
                "has been let go and NORTH is what they are held to."
            ),
        },
        "final_south_binding": {
            **_STR,
            "description": (
                "Required. The final developer turn where requirement NORTH "
                "has been let go and SOUTH is what they are held to."
            ),
        },
    },
)

ORACLE = _tool(
    "route_with_evidence",
    "Route the request and show the evidence.",
    {
        "expert": _EXPERT_ENUM,
        "quote": {
            **_STR,
            "description": "The exact words that state the binding requirement.",
        },
        "constraints_present": {
            "type": "array",
            "items": _EXPERT_ENUM,
            "description": "Every specialist's concern that is visibly in play in the text.",
        },
        "decidable": {
            "type": "boolean",
            "description": "False if no single requirement is stated as binding, or if two are.",
        },
    },
)
