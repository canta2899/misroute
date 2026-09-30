import datetime
import json

from . import config, llm, plan, prompts
from .providers import Spec
from .store import Run, append_jsonl, load_scenarios, read_jsonl


def generate_facts(run: Run, *, mode="batch", limit=None) -> dict:
    model = config.generator_model()
    scenarios = load_scenarios()
    done = {row["scenario_id"] for row in read_jsonl(run.facts)}
    todo = [a for a in scenarios if a["id"] not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"requested": 0, "written": 0, "failures": [], "already": len(done)}

    specs = {
        a["id"]: Spec(
            system=prompts.FACTS_SYSTEM,
            user=prompts.facts_user(a),
            tool=prompts.FACTS_TOOL,
            max_tokens=config.MAX_TOKENS["facts"],
        )
        for a in todo
    }
    results = llm.run(
        model, specs, mode=mode, log_dir=run.llm_dir("facts"), tag="facts"
    )
    outputs, failures = llm.split_results(results)

    rows = []
    for scenario_id, output in outputs.items():
        missing = [e for e in config.EXPERT_IDS if not str(output.get(e, "")).strip()]
        if missing:
            failures.append({"id": scenario_id, "error": f"missing facts: {missing}"})
            continue
        rows.append(
            {
                "scenario_id": scenario_id,
                "facts": {e: output[e].strip() for e in config.EXPERT_IDS},
                "model": model,
                "generated_at": _now(),
            }
        )
    append_jsonl(run.facts, rows)
    return {
        "requested": len(todo),
        "written": len(rows),
        "failures": failures,
        "already": len(done),
    }


def generate_samples(
    run: Run, kind: str, *, mode="batch", limit=None, ids=None
) -> dict:
    rows = plan.by_kind(read_jsonl(run.plan), kind)
    facts = {r["scenario_id"]: r["facts"] for r in read_jsonl(run.facts)}
    scenarios = {a["id"]: a for a in load_scenarios()}
    have = {s["id"] for s in read_jsonl(run.samples(kind))}

    pending, skipped_no_facts, partial = [], [], []
    for group in plan.requests(rows):
        wanted = [r for r in group if r["id"] not in have]
        if not wanted:
            continue
        if len(wanted) != len(group):
            partial.append(group[0]["request_id"])
            continue
        if ids and not any(r["id"] in ids for r in group):
            continue
        if group[0]["scenario_id"] not in facts:
            skipped_no_facts.append(group[0]["request_id"])
            continue
        pending.append(group)

    if limit:
        pending = pending[:limit]
    if not pending:
        return {
            "requests": 0,
            "written": 0,
            "failures": [],
            "missing_facts": skipped_no_facts,
            "partial_requests": partial,
            "already": len(have),
        }

    # Requests are grouped by the model that serves them, so one call site
    # covers both the uniform case and a per-construction escalation.
    by_model: dict[str, dict] = {}
    for group in pending:
        chosen = config.generator_model(kind, group[0]["construction"])
        by_model.setdefault(chosen, {})[group[0]["request_id"]] = _spec(
            kind, group, scenarios, facts
        )

    by_request = {g[0]["request_id"]: g for g in pending}
    samples, failures = [], []
    for chosen, specs in by_model.items():
        results = llm.run(
            chosen, specs, mode=mode, log_dir=run.llm_dir(kind), tag=f"{kind}_generate"
        )
        outputs, model_failures = llm.split_results(results)
        failures.extend(model_failures)
        for request_id, output in outputs.items():
            try:
                samples.extend(_expand(kind, by_request[request_id], output, chosen))
            except (KeyError, ValueError, TypeError) as exc:
                failures.append({"id": request_id, "error": f"bad output: {exc}"})

    append_jsonl(run.samples(kind), samples)
    return {
        "requests": len(pending),
        "written": len(samples),
        "failures": failures,
        "missing_facts": skipped_no_facts,
        "partial_requests": partial,
        "already": len(have),
    }


def _spec(kind: str, group: list[dict], scenarios: dict, facts: dict) -> Spec:
    row = group[0]
    scenario = scenarios[row["scenario_id"]]
    fact = facts[row["scenario_id"]]
    is_pair = len(group) == 2
    if kind == "single_turn":
        user = (
            prompts.single_pair_user(group, scenario, fact)
            if is_pair
            else prompts.single_user(row, scenario, fact)
        )
        tool = prompts.SINGLE_PAIR_TOOL if is_pair else prompts.SINGLE_TOOL
        max_tokens = config.MAX_TOKENS["single"]
    else:
        user = (
            prompts.multi_pair_user(group, scenario, fact)
            if is_pair
            else prompts.multi_user(row, scenario, fact)
        )
        tool = prompts.MULTI_PAIR_TOOL if is_pair else prompts.MULTI_TOOL
        max_tokens = config.MAX_TOKENS["multi"]
    return Spec(
        system=prompts.GENERATION_SYSTEM, user=user, tool=tool, max_tokens=max_tokens
    )


def _expand(kind: str, group: list[dict], output: dict, model: str) -> list[dict]:
    if kind == "single_turn":
        if len(group) == 1:
            texts = {None: _text(output["query"])}
            pivots = {None: None}
        else:
            texts = {
                "a": _text(output["query_north_binding"]),
                "b": _text(output["query_south_binding"]),
            }
            pivots = {
                "a": output.get("pivot_north_binding"),
                "b": output.get("pivot_south_binding"),
            }
        return [
            _sample(
                row,
                [_turn("user", texts[row["variant"]])],
                model,
                pivot=pivots[row["variant"]],
            )
            for row in group
        ]

    if len(group) == 1:
        return [_sample(group[0], _turns(output, config.MT_MESSAGES), model)]

    shared = _turns(output, config.MT_MESSAGES - 1, "opening_")
    finals = {
        "a": _text(output["final_north_binding"]),
        "b": _text(output["final_south_binding"]),
    }
    return [
        _sample(
            row,
            shared + [_turn("user", finals[row["variant"]])],
            model,
            pivot=finals[row["variant"]],
        )
        for row in group
    ]


def _turn(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _decode(value, key: str):
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    if isinstance(value, dict) and key in value:
        return _decode(value[key], key)
    return value


def _turns(output: dict, count: int, prefix: str = "") -> list[dict]:
    return [
        _turn(role, _text(output[key]))
        for key, role in prompts.turn_keys(count, prefix)
    ]


def _text(value) -> str:
    value = _decode(value, "content")
    # a final turn sometimes arrives as the turn object, or a list holding it
    if isinstance(value, list) and value:
        value = value[-1]
    if isinstance(value, dict):
        value = value.get("content")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty text field")
    return value.strip()


def _sample(row: dict, turns: list[dict], model: str, pivot=None) -> dict:
    return {
        **row,
        "turns": turns,
        "pivot": pivot,
        "generator_model": model,
        "generated_at": _now(),
    }


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")
