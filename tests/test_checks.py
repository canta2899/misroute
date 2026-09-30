import pytest

from dataset import checks, config

GOOD_TEXT = (
    "we are wiring the nightly export up for the customer dashboard and legal has "
    "signed us up to a gdpr retention window that we cannot move, every personal "
    "data field has to stay inside the audited store. i would personally rather "
    "just hardcode the whole thing before the demo on friday since it is only a "
    "prototype, but that is my own call not something anyone is holding me to. "
    "what is the sanest way to wire this up."
)


def sample(**overrides):
    base = {
        "id": "st-adv-00001a",
        "request_id": "st-adv-00001",
        "kind": "single_turn",
        "construction": "adversarial",
        "gold": "strict_compliance",
        "decoy": "time_to_market",
        "scenario_id": "scn_01",
        "variant": "a",
        "turns": [{"role": "user", "content": GOOD_TEXT}],
    }
    return {**base, **overrides}


def test_clean_sample_has_no_problems():
    assert checks.problems(sample()) == []


@pytest.mark.parametrize("banned", ["```", "[logs.txt]", "![](x)"])
def test_markup_is_rejected(banned):
    bad = sample(turns=[{"role": "user", "content": GOOD_TEXT + banned}])
    assert any("banned markup" in p for p in checks.problems(bad))


def test_naming_an_expert_is_rejected():
    bad = sample(turns=[{"role": "user", "content": GOOD_TEXT + " route to strict_compliance"}])
    assert any("names the expert" in p for p in checks.problems(bad))


def test_multi_turn_shape_is_enforced():
    turns = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": GOOD_TEXT}
        for i in range(config.MT_MESSAGES)
    ]
    good = sample(kind="multi_turn", turns=turns)
    assert checks.problems(good) == []

    short = sample(kind="multi_turn", turns=turns[:3])
    assert any("expected" in p for p in checks.problems(short))

    ends_wrong = sample(kind="multi_turn", turns=turns[:-1] + [{"role": "assistant", "content": GOOD_TEXT}])
    assert checks.problems(ends_wrong)


def test_pair_overlap_flags_non_minimal_pairs():
    # a real minimal pair: same words, the waiver moved from one requirement to
    # the other, so the word sets match and the pair is not weak
    a = sample()
    b = sample(
        id="st-adv-00001b",
        variant="b",
        gold="time_to_market",
        decoy="strict_compliance",
        turns=[{"role": "user", "content": GOOD_TEXT.replace(
            "that is my own call not something anyone is holding me to",
            "that is not my own call, something anyone is holding me to",
        )}],
    )
    weak, report = checks.pair_overlaps([a, b])
    assert weak == [] and report["mean_overlap"] == 1.0

    b_far = sample(
        id="st-adv-00001b",
        variant="b",
        turns=[{"role": "user", "content": "totally different words about a queue and shards everywhere"}],
    )
    weak, report = checks.pair_overlaps([a, b_far])
    assert len(weak) == 1 and report["n_weak"] == 1


def test_identical_pair_members_are_flagged():
    """Word overlap cannot see this: a perfect pair also scores 1.0.

    Two members with the same text carry opposite labels on one document, so
    one of them is mislabelled no matter what the oracle says.
    """
    a = sample()
    b = sample(id="st-adv-00001b", variant="b", gold="time_to_market", decoy="strict_compliance")
    weak, report = checks.pair_overlaps([a, b])
    assert [w["reason"] for w in weak] == ["identical members"]
    assert report["n_weak"] == 1


def test_multi_turn_pairs_are_compared_on_the_final_turn_only():
    """The shared opening is identical by construction, so it must not count."""
    opening = [
        {"role": "user", "content": GOOD_TEXT},
        {"role": "assistant", "content": "which store is the export reading from"},
        {"role": "user", "content": GOOD_TEXT},
        {"role": "assistant", "content": "and who signed off on the retention window"},
    ]
    def final(text):
        return opening + [{"role": "user", "content": text}]

    a = sample(kind="multi_turn", id="mt-adv-00001a", request_id="mt-adv-00001",
               turns=final("legal will not move the retention window this quarter"))
    b = sample(kind="multi_turn", id="mt-adv-00001b", request_id="mt-adv-00001",
               variant="b", turns=final("the demo on friday is the only thing i am held to"))
    weak, report = checks.pair_overlaps([a, b])
    # identical openings would score near 1.0 on the whole developer text; these
    # two finals share almost nothing, so the pair is correctly called weak
    assert report["mean_overlap"] < 0.2
    assert report["min_overlap_threshold"] == config.MIN_PAIR_OVERLAP["multi_turn"]
    assert [w["request_id"] for w in weak] == ["mt-adv-00001"]


def test_validate_separates_and_reports():
    good = sample()
    bad = sample(id="st-adv-00002a", turns=[{"role": "user", "content": "too short"}])
    kept, report = checks.validate([good, bad])
    assert [s["id"] for s in kept] == [good["id"]]
    assert report["n_rejected"] == 1 and report["rejected"][0]["id"] == bad["id"]
