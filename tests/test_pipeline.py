import pytest

from dataset import config, generate, plan, prompts, repair, store
from dataset.store import load_scenarios

SCENARIOS = load_scenarios()
FACTS = {e: f"constraint text for {e}" for e in config.EXPERT_IDS}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    return store.Run("test-run")


# --- turning model output into samples ---------------------------------------


def _rows(construction, kind="single_turn"):
    rows = [
        r
        for r in plan.build(SCENARIOS, kinds=(kind,))
        if r["construction"] == construction
    ]
    return plan.requests(rows)[0]


def test_single_pair_expands_into_two_samples_sharing_a_request():
    group = _rows("adversarial")
    out = generate._expand(
        "single_turn",
        group,
        {
            "query_north_binding": "text A",
            "query_south_binding": "text B",
            "pivot_north_binding": "pa",
            "pivot_south_binding": "pb",
        },
        "m",
    )
    assert [s["id"] for s in out] == [group[0]["id"], group[1]["id"]]
    assert [s["turns"][0]["content"] for s in out] == ["text A", "text B"]
    assert {s["request_id"] for s in out} == {group[0]["request_id"]}
    assert out[0]["gold"] != out[1]["gold"]


def _opening(prefix="opening_"):
    return {key: f"t{i}" for i, (key, _) in enumerate(prompts.turn_keys(config.MT_MESSAGES - 1, prefix))}


def test_multi_pair_shares_the_opening_and_differs_only_in_the_last_turn():
    group = _rows("adversarial", kind="multi_turn")
    a, b = generate._expand(
        "multi_turn",
        group,
        {**_opening(), "final_north_binding": "fa", "final_south_binding": "fb"},
        "m",
    )
    assert len(a["turns"]) == len(b["turns"]) == config.MT_MESSAGES
    assert a["turns"][:-1] == b["turns"][:-1]
    assert [t["role"] for t in a["turns"]] == ["user", "assistant"] * 2 + ["user"]
    assert (a["turns"][-1]["content"], b["turns"][-1]["content"]) == ("fa", "fb")


def test_multi_standalone_takes_one_turn_per_field():
    (row,) = _rows("plain", kind="multi_turn")
    output = {key: f"t{i}" for i, (key, _) in enumerate(prompts.turn_keys(config.MT_MESSAGES))}
    (s,) = generate._expand("multi_turn", [row], output, "m")
    assert [t["role"] for t in s["turns"]] == ["user", "assistant"] * 2 + ["user"]
    assert [t["content"] for t in s["turns"]] == [f"t{i}" for i in range(config.MT_MESSAGES)]


def test_a_missing_turn_field_is_a_failure_not_a_short_conversation():
    group = _rows("adversarial", kind="multi_turn")
    output = {**_opening(), "final_north_binding": "fa", "final_south_binding": "fb"}
    output.pop("opening_turn3_developer")
    with pytest.raises(KeyError):
        generate._expand("multi_turn", group, output, "m")


def test_generator_model_escalates_only_for_multi_turn_pairs():
    assert config.generator_model("single_turn", "adversarial") == "anthropic/claude-haiku-4-5-20251001"
    assert config.generator_model("multi_turn", "natural") == "anthropic/claude-haiku-4-5-20251001"
    assert config.generator_model("multi_turn", "adversarial") == "anthropic/claude-sonnet-5"


def test_empty_model_output_is_an_error_not_a_sample():
    group = _rows("natural")
    with pytest.raises(ValueError):
        generate._expand("single_turn", group, {"query": "   "}, "m")


# --- prompts -----------------------------------------------------------------


def test_prompts_render_without_leaking_the_answer():
    group = _rows("adversarial")
    scenario = {"id": "scn_01", "text": "a system"}
    text = prompts.single_pair_user(group, scenario, FACTS)
    assert FACTS[group[0]["gold"]] in text and FACTS[group[1]["gold"]] in text
    # the prompt must not tell the model which version is which by label name
    assert group[0]["gold"] not in text.replace(FACTS[group[0]["gold"]], "")


def test_generation_system_prompt_carries_the_resolution_rule():
    assert config.RESOLUTION_RULE in prompts.GENERATION_SYSTEM
    assert config.RESOLUTION_RULE in prompts.ORACLE_SYSTEM


def test_tool_schemas_are_strict_objects():
    for tool in (
        prompts.FACTS_TOOL,
        prompts.SINGLE_TOOL,
        prompts.SINGLE_PAIR_TOOL,
        prompts.MULTI_TOOL,
        prompts.MULTI_PAIR_TOOL,
        prompts.ORACLE_TOOL,
    ):
        schema = tool["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


# --- repair ------------------------------------------------------------------


def _sample(row):
    return {**row, "turns": [{"role": "user", "content": "x"}], "pivot": None}


def test_repair_removes_whole_requests_and_backs_up(run):
    group = _rows("adversarial")
    other = _rows("natural")
    samples = [_sample(r) for r in group + other]
    store.write_jsonl(run.samples("single_turn"), samples)
    result = repair.repair(run, "single_turn", sample_ids=[group[0]["id"]])

    assert result["removed"] == 2  # the sibling goes with it
    kept = store.read_jsonl(run.samples("single_turn"))
    assert [s["id"] for s in kept] == [other[0]["id"]]
    assert result["backup"].exists()


def test_repair_leaves_the_plan_alone_so_generate_refills_the_gap(run):
    group = _rows("adversarial")
    store.write_jsonl(run.plan, group)
    store.write_jsonl(run.samples("single_turn"), [_sample(r) for r in group])

    repair.repair(run, "single_turn", sample_ids=[group[0]["id"]])

    assert len(store.read_jsonl(run.plan)) == len(group)


def test_drop_plan_removes_the_rows_so_nothing_recreates_them(run):
    group = _rows("adversarial")
    other = _rows("natural")
    store.write_jsonl(run.plan, group + other)
    store.write_jsonl(run.samples("single_turn"), [_sample(r) for r in group + other])

    result = repair.repair(
        run, "single_turn", sample_ids=[group[0]["id"]], drop_plan=True
    )

    assert result["plan_rows_dropped"] == len(group)  # the pair goes as a unit
    assert [r["id"] for r in store.read_jsonl(run.plan)] == [r["id"] for r in other]


def test_repair_before_analyze_is_an_error(run):
    store.write_jsonl(run.samples("single_turn"), [_sample(_rows("natural")[0])])
    with pytest.raises(ValueError, match="regenerate flag"):
        repair.repair(run, "single_turn")


def test_repair_targets_the_samples_marked_for_regeneration(run):
    natural = [r for r in plan.build(SCENARIOS, kinds=("single_turn",))
               if r["construction"] == "natural"]
    keep, drop = natural[0], natural[1]
    samples = [_sample(keep) | {"regenerate": 0}, _sample(drop) | {"regenerate": 1}]
    store.write_jsonl(run.samples("single_turn"), samples)

    result = repair.repair(run, "single_turn")

    assert result["removed"] == 1
    assert [s["id"] for s in store.read_jsonl(run.samples("single_turn"))] == [keep["id"]]
