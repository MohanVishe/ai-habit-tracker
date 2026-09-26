"""The evaluation's ground truth and scorer. No model is called here."""
import json
from datetime import date
from pathlib import Path

import pytest

import seed
from eval import run, scoring
from habitloop import analytics, db

MW, RP, DW, NS = "Morning walk", "Read 20 pages", "Deep work block", "No screens after 10pm"
HABITS = [MW, RP, DW, NS]
RESULTS = Path(scoring.__file__).with_name("results")


@pytest.fixture(scope="module")
def summaries():
    return {d: scoring.summary_for(d) for d in scoring.AS_OF_DATES}


def q(kind, answer_type, **kw):
    return {"id": "t", "category": "t", "kind": kind, "answer_type": answer_type,
            "question": "?", **kw}


# --- the question set -----------------------------------------------------


def test_question_set_is_well_formed(summaries):
    questions = scoring.load_questions()
    assert 40 <= len(questions) <= 60
    assert len({x["id"] for x in questions}) == len(questions)
    assert {x["answer_type"] for x in questions} <= set(scoring.SPEC)
    for summary in summaries.values():
        for question in questions:
            truth = scoring.ground_truth(question, summary)
            assert (truth is None) == (question["kind"] == "unknown"), question["id"]
            if isinstance(truth, list):
                assert truth, question["id"]


def test_answers_differ_between_the_seeded_logs(summaries):
    questions = [x for x in scoring.load_questions() if x["kind"] != "unknown"]
    truths = {x["id"]: {json.dumps(scoring.ground_truth(x, s)) for s in summaries.values()}
              for x in questions}
    changed = sum(len(v) > 1 for v in truths.values())
    assert changed >= len(questions) // 2


# --- the held-out set -----------------------------------------------------


@pytest.fixture(scope="module")
def heldout_summaries():
    return {d: scoring.summary_for(d) for d in scoring.HELDOUT_AS_OF_DATES}


def _stats(summary):
    return {k: v for k, v in summary.items() if k != "generated_on"} | {
        "completion": {h: {k: v for k, v in c.items() if k != "tracked_since"}
                       for h, c in summary["completion"].items()}}


def test_heldout_set_is_well_formed_and_separate(heldout_summaries):
    questions = scoring.load_questions(scoring.HELDOUT)
    dev = scoring.load_questions()
    assert 30 <= len(questions) <= 45
    assert len({x["id"] for x in questions}) == len(questions)
    assert not {x["id"] for x in questions} & {x["id"] for x in dev}
    assert not {x["question"] for x in questions} & {x["question"] for x in dev}
    assert {x["answer_type"] for x in questions} <= set(scoring.SPEC)
    comparisons = [x for x in questions if x["category"] in
                   ("best_worst", "compare_two", "which_day", "ranking")]
    assert len(comparisons) >= len(questions) / 2
    assert not set(scoring.HELDOUT_AS_OF_DATES) & set(scoring.AS_OF_DATES)
    for summary in heldout_summaries.values():
        for question in questions:
            truth = scoring.ground_truth(question, summary)
            assert (truth is None) == (question["kind"] == "unknown"), question["id"]
            if isinstance(truth, list):
                assert truth, question["id"]


def test_heldout_logs_are_new_logs(summaries, heldout_summaries):
    # seed.py repeats a log for an end date on the same weekday; these must not.
    dev = [_stats(s) for s in summaries.values()]
    held = [_stats(s) for s in heldout_summaries.values()]
    assert all(h not in dev for h in held)
    assert len({json.dumps(h, sort_keys=True) for h in held}) == 3


def test_same_weekday_gives_the_same_log():
    # why the held-out dates are on other weekdays: 2026-08-02 and 2026-08-09 are both Sundays
    assert _stats(scoring.summary_for("2026-08-02")) == _stats(scoring.summary_for("2026-08-09"))


def test_heldout_ground_truth_kinds(heldout_summaries):
    s = heldout_summaries["2026-09-19"]
    # completion: Morning walk 0.933, Read 20 pages 0.619, Deep work 0.571, No screens 0.333
    assert scoring.ground_truth(q("second_best_habit", "habit", scope="overall"), s) == [RP]
    assert scoring.ground_truth(q("second_worst_habit", "habit", scope="overall"), s) == [DW]
    assert scoring.ground_truth(q("rank_habits", "ranking", scope="overall"), s)         == [[MW], [RP], [DW], [NS]]
    # weekends: Deep work block and Read 20 pages tie at 0.0
    assert scoring.ground_truth(q("rank_habits", "ranking", scope="sat_sun"), s)         == [[MW], [NS], [DW, RP]]
    assert scoring.ground_truth(q("worst_habit", "habit", scope="sat_sun"), s) == [DW, RP]
    assert scoring.ground_truth(q("worst_habit", "habit", scope="Monday"), s) == [NS]
    assert scoring.ground_truth(
        q("lower_of_two", "habit", scope="sat_sun", habits=[RP, NS]), s) == [RP]
    assert scoring.ground_truth(
        q("higher_of_two", "habit", scope="current_streak", habits=[MW, DW]), s) == [MW]
    # Mon-Fri minus Sat+Sun: Read 20 pages 0.619, the largest drop
    assert scoring.ground_truth(q("biggest_weekend_drop", "habit"), s) == [RP]
    pattern = s["weekday_pattern"]
    drops = {h: p["mon_to_fri"]["rate"] - p["sat_sun"]["rate"] for h, p in pattern.items()}
    assert max(drops, key=drops.get) == RP


@pytest.mark.parametrize("value, ok", [
    (f"{MW} > {NS} > {DW} > {RP}", True),
    (f"{MW} > {NS} > {RP} > {DW}", True),       # the tied pair in either order
    (f"{MW} > {RP} > {NS} > {DW}", False),      # a tied habit placed above a better one
    (f"{MW} > {NS} > {DW}", False),             # one habit left out
    (f"{MW} > {NS} > {DW} > {RP} > {MW}", False),  # named twice
])
def test_ranking_rule(value, ok):
    truth = [[MW], [NS], [DW, RP]]
    assert scoring.compare("ranking", value, truth, HABITS) == (ok, ok)


# --- ground truth comes from the analytics functions ----------------------


@pytest.mark.parametrize("as_of", scoring.AS_OF_DATES)
def test_ground_truth_matches_the_analytics_functions(tmp_path, summaries, as_of):
    day = date.fromisoformat(as_of)
    seed.seed(day, tmp_path / "h.db")
    entries = db.recent(scoring.WINDOW, day, tmp_path / "h.db")
    summary = summaries[as_of]
    completion = analytics.completion_by_habit(entries, scoring.WINDOW, day)

    for habit in HABITS:
        assert scoring.ground_truth(q("current_streak", "number", habit=habit), summary) \
            == analytics.current_streak(entries, habit, day)
        assert scoring.ground_truth(q("longest_streak", "number", habit=habit), summary) \
            == analytics.longest_streak(entries, habit)
        assert scoring.ground_truth(q("completion_rate", "rate", habit=habit), summary) \
            == completion[habit]["rate"]
        pattern = analytics.weekday_pattern(
            entries, habit, date.fromisoformat(completion[habit]["tracked_since"]), day)
        assert scoring.ground_truth(q("weekend_rate", "rate", habit=habit), summary) \
            == pattern["sat_sun"]["rate"]

    # best/worst agree with the rankings the app computes
    for scope, key in (("overall", "completion_rate_worst_first"),
                       ("sat_sun", "sat_sun_rate_worst_first"),
                       ("mon_to_fri", "mon_to_fri_rate_worst_first")):
        ranking = summary["rankings"][key]
        assert ranking[0][0] in scoring.ground_truth(q("worst_habit", "habit", scope=scope), summary)
        assert ranking[-1][0] in scoring.ground_truth(q("best_habit", "habit", scope=scope), summary)


def test_known_answers_on_the_readme_log(summaries):
    s = summaries["2026-09-15"]  # the README's seed block: Morning walk streak 1/28
    assert scoring.ground_truth(q("current_streak", "number", habit=MW), s) == 1
    assert scoring.ground_truth(q("longest_streak", "number", habit=MW), s) == 28
    assert scoring.ground_truth(q("worst_habit", "habit", scope="sat_sun"), s) == [DW]
    assert scoring.ground_truth(q("more_reliable", "choice", habit=MW), s) == "weekends"
    assert scoring.ground_truth(q("worst_day", "weekday", habit=MW), s) == ["Monday"]
    assert scoring.ground_truth(q("top_current_streak", "habit"), s) == [DW, NS]
    assert scoring.ground_truth(q("habits_tracked", "number"), s) == 4
    # a different end date: Deep work block and Read 20 pages tie at 0.0 on weekends
    tie = scoring.ground_truth(q("worst_habit", "habit", scope="sat_sun"), summaries["2026-08-02"])
    assert tie == [DW, RP]


# --- parsing --------------------------------------------------------------


@pytest.mark.parametrize("text, value", [
    ("ANSWER: 1\nYour current streak is 1 day.", "1"),
    ("**ANSWER:** Deep work block\nIt has 0.0 at weekends.", "Deep work block"),
    ("Answer: `0.967`.\nMorning walk is at 0.967.", "0.967"),
    ("ANSWER:\nUNKNOWN\nThe log has no sleep data.", "UNKNOWN"),
    ("ANSWER: <UNKNOWN>", "UNKNOWN"),
])
def test_parse_answer_line(text, value):
    assert scoring.parse_response(text)[0] == value


def test_parse_separates_the_explanation():
    value, explanation = scoring.parse_response("ANSWER: 2\nDeep work block's current streak is 2.")
    assert (value, explanation) == ("2", "Deep work block's current streak is 2.")
    assert scoring.parse_response("Your streak is 2 days.") == (None, "Your streak is 2 days.")


# --- comparison rules -----------------------------------------------------


@pytest.mark.parametrize("value, ok", [
    ("28", True), ("28 days", True), ("Read 20 pages: 28", True), ("27", False), ("none", False),
])
def test_number_rule(value, ok):
    assert scoring.compare("number", value, 28, HABITS)[0] is ok


@pytest.mark.parametrize("value, ok", [
    ("0.967", True), ("96.7%", True), ("96.7", True), ("97%", True), ("0.962", True),
    ("0.96", False), ("0.5", False), ("No screens after 10pm: 0.967", True),
])
def test_rate_rule(value, ok):
    assert scoring.compare("rate", value, 0.967, HABITS)[0] is ok


@pytest.mark.parametrize("value, correct, complete", [
    ("Deep work block | Read 20 pages", True, True),
    ("deep work block", True, False),          # one of two tied habits: right, not complete
    ("Morning walk", False, False),
    ("Deep work block | Morning walk", False, False),
    ("Meditate", False, False),
])
def test_habit_rule_with_a_tie(value, correct, complete):
    assert scoring.compare("habit", value, [DW, RP], HABITS) == (correct, complete)


@pytest.mark.parametrize("value, ok", [
    ("Saturday", True), ("Saturdays", True), ("Sat", True), ("Sunday", False),
    ("Saturday | Monday", False), ("weekends", False),
])
def test_weekday_rule(value, ok):
    assert scoring.compare("weekday", value, ["Saturday"], HABITS)[0] is ok


@pytest.mark.parametrize("value, truth, ok", [
    ("weekdays", "weekdays", True), ("Weekends.", "weekends", True), ("weekend", "weekends", True),
    ("the same", "equal", True), ("weekdays and weekends", "weekdays", False),
    ("weekends", "weekdays", False),
])
def test_choice_rule(value, truth, ok):
    assert scoring.compare("choice", value, truth, HABITS)[0] is ok


def test_first_named_is_by_position_in_the_answer():
    assert scoring.first_named("habit", "No screens after 10pm | Deep work block", HABITS) == NS
    assert scoring.first_named("weekday", "Sun | Saturday", HABITS) == "Sunday"
    assert scoring.first_named("habit", "Meditate", HABITS) is None


def test_sensitivity_counts_an_over_long_list_by_its_first_name(summaries):
    s = summaries["2026-09-15"]  # worst overall: No screens after 10pm only
    question = q("worst_habit", "habit", scope="overall")
    listed = scoring.evaluate(question, "2026-09-15",
                              f"ANSWER: {NS} | {DW}\n{NS} is at 0.367.", s)
    wrong_first = scoring.evaluate(question, "2026-09-15",
                                   f"ANSWER: {DW} | {NS}\n{DW} is at 0.762.", s)
    assert listed["outcome"] == scoring.WRONG and listed["first_named_correct"] is True
    assert wrong_first["first_named_correct"] is False


# --- outcomes -------------------------------------------------------------


@pytest.mark.parametrize("truth, response, outcome", [
    (1, "ANSWER: 1\nstreak 1", scoring.CORRECT),
    (1, "ANSWER: 28\nstreak 28", scoring.WRONG),
    (1, "ANSWER: UNKNOWN\nnot in the log", scoring.DECLINED),
    (None, "ANSWER: UNKNOWN\nnot in the log", scoring.REFUSED_OK),
    (None, "ANSWER: 5\nMeditate is at 5", scoring.ANSWERED_UNANSWERABLE),
    (1, "Your streak is 1 day.", scoring.NO_ANSWER_LINE),
    (None, "The log doesn't cover sleep.", scoring.NO_ANSWER_LINE),
])
def test_outcomes(truth, response, outcome):
    assert scoring.score(q("x", "number"), truth, response, HABITS)["outcome"] == outcome


def test_validator_runs_on_the_explanation(summaries):
    s = summaries["2026-09-15"]
    question = q("weekend_rate", "rate", habit=DW)
    right = scoring.evaluate(question, "2026-09-15",
                             "ANSWER: 0.0\nDeep work block has a weekend rate of 0.0.", s)
    wrong = scoring.evaluate(question, "2026-09-15",
                             "ANSWER: 0.5\nDeep work block has a weekend rate of 0.5.", s)
    assert (right["outcome"], right["validator_flags"]) == (scoring.CORRECT, [])
    assert wrong["outcome"] == scoring.WRONG and wrong["validator_flags"]


# --- statistics -----------------------------------------------------------


def test_wilson_interval():
    lo, hi = scoring.wilson(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)
    lo, hi = scoring.wilson(0, 10)
    assert lo == 0.0 and round(hi, 4) == 0.2775


def test_cluster_bootstrap():
    records = [{"id": f"q{i % 5}", "ok": i % 5 != 0} for i in range(15)]
    lo, hi = scoring.cluster_bootstrap(records, lambda r: r["ok"], reps=2000)
    assert 0 <= lo <= 0.8 <= hi <= 1
    assert scoring.cluster_bootstrap(records, lambda r: True, reps=200) == (1.0, 1.0)
    assert scoring.cluster_bootstrap(records, lambda r: r["ok"], reps=500, seed=3) \
        == scoring.cluster_bootstrap(records, lambda r: r["ok"], reps=500, seed=3)


# --- the committed run ----------------------------------------------------


@pytest.mark.parametrize("jsonl", sorted(RESULTS.glob("*.jsonl")), ids=lambda p: p.stem)
def test_committed_summary_is_reproduced_by_rescoring(jsonl):
    raw = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    committed = json.loads(jsonl.with_suffix(".summary.json").read_text(encoding="utf-8"))
    records = run.score_all(raw, run.ROOT / committed["run"]["questions_file"])
    assert [(r["id"], r["as_of"], r["outcome"], r["truth"]) for r in records] \
        == [(r["id"], r["as_of"], r["outcome"], r["truth"]) for r in raw]
    committed.pop("run")
    assert json.loads(json.dumps(scoring.summarize(records))) == committed


def test_flag_review_covers_exactly_the_flagged_records():
    rows = [json.loads(line) for line in
            (RESULTS / "qwen2.5-7b-instruct.jsonl").read_text(encoding="utf-8").splitlines()]
    review = json.loads((RESULTS / "qwen2.5-7b-instruct.flag-review.json")
                        .read_text(encoding="utf-8"))
    for key, outcome in (("flags_on_wrong_answers", scoring.WRONG),
                         ("flags_on_correct_answers", scoring.CORRECT),
                         ("flags_on_correct_refusals", scoring.REFUSED_OK)):
        flagged = {(r["as_of"], r["id"]) for r in rows
                   if r["outcome"] == outcome and r["validator_flags"]}
        assert {(e["as_of"], e["id"]) for e in review[key]} == flagged, key
