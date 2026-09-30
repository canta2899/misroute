from collections import Counter

from dataset import config, plan
from dataset.store import load_scenarios

SCENARIOS = load_scenarios()


def test_plan_is_deterministic():
    assert plan.build(SCENARIOS) == plan.build(SCENARIOS)


def test_every_expert_gets_its_target_in_every_construction():
    rows = plan.build(SCENARIOS, kinds=("single_turn",))
    targets = plan.targets()
    counts = Counter((r["gold"], r["construction"]) for r in rows)
    for construction, target in targets.items():
        for expert in config.EXPERT_IDS:
            assert counts[(expert, construction)] == target


def test_adversarial_rows_come_in_contrastive_pairs():
    rows = [
        r
        for r in plan.build(SCENARIOS, kinds=("single_turn",))
        if r["construction"] == "adversarial"
    ]
    groups = plan.requests(rows)
    assert groups
    for group in groups:
        a, b = group
        assert (a["variant"], b["variant"]) == ("a", "b")
        assert {a["gold"], b["gold"]} == {a["decoy"], b["decoy"]}
        assert a["gold"] == b["decoy"] and b["gold"] == a["decoy"]
        # a pair shares everything except which side is binding
        for field in ("scenario_id", "pattern", "persona", "construction"):
            assert a[field] == b[field]
        assert {a["gold"], b["gold"]} in [set(c["pair"]) for c in config.COLLISIONS]


def test_non_adversarial_rows_are_single_requests():
    rows = [
        r
        for r in plan.build(SCENARIOS, kinds=("single_turn",))
        if r["construction"] != "adversarial"
    ]
    assert all(len(g) == 1 for g in plan.requests(rows))


def test_ids_are_unique_across_kinds():
    rows = plan.build(SCENARIOS)
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)


def test_scale_shrinks_the_plan_but_keeps_it_balanced():
    rows = plan.build(SCENARIOS, kinds=("single_turn",), scale=0.05)
    counts = Counter(r["gold"] for r in rows)
    assert len(set(counts.values())) == 1
    assert len(rows) < len(plan.build(SCENARIOS, kinds=("single_turn",)))


def test_collision_map_covers_every_pair_once():
    pairs = [frozenset(c["pair"]) for c in config.COLLISIONS]
    n = len(config.EXPERT_IDS)
    assert len(set(pairs)) == len(pairs) == n * (n - 1) // 2
    for expert in config.EXPERT_IDS:
        assert len(config.partners(expert)) == n - 1
