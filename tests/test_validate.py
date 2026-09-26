"""Tests for the claim checker that runs over every model answer.

The fixture is a 14-day log with two habits whose numbers are easy to confuse:
Morning walk is done every day but one Sunday; No screens after 10pm is done
every weekday and never at weekends.
"""
from datetime import date, timedelta

import pytest

from habitloop import analytics, validate

TODAY = date(2026, 6, 15)  # a Monday


@pytest.fixture(scope="module")
def summary():
    entries = []
    for i in range(14):
        day = TODAY - timedelta(days=i)
        entries.append({"name": "Morning walk", "on_date": day.isoformat(),
                        "done": int(day != date(2026, 6, 14)), "target_per_week": 7})
        entries.append({"name": "No screens after 10pm", "on_date": day.isoformat(),
                        "done": int(day.weekday() < 5), "target_per_week": 7})
    s = analytics.build_summary(entries, 14, TODAY)
    # Pin the facts the tests rely on, so a change in analytics fails loudly here.
    assert s["completion"]["Morning walk"]["rate"] == 0.929
    assert s["completion"]["No screens after 10pm"]["completed"] == 10
    assert s["streaks"]["Morning walk"]["longest_in_window"] == 12
    assert s["weekday_pattern"]["No screens after 10pm"]["sat_sun"]["missed"] == 4
    assert s["weekday_pattern"]["Morning walk"]["sat_sun"]["rate"] == 0.75
    return s


def flagged(text, summary):
    return [issue.value for issue in validate.check(text, summary).issues]


class TestSupportedClaims:
    def test_exact_numbers_pass(self, summary):
        text = "Morning walk is at 92.9% (13 of 14 days) with a current streak of 1 day."
        report = validate.check(text, summary)
        assert report.ok
        assert report.checked == 4

    def test_rounded_percentages_pass(self, summary):
        assert flagged("Morning walk is at 93%.", summary) == []

    def test_correct_weekend_claim_passes(self, summary):
        assert flagged("No screens after 10pm was missed on 4 weekend days, 0% of them done.",
                       summary) == []

    def test_overall_then_weekend_in_one_sentence(self, summary):
        assert flagged("Morning walk is at 92.9% overall, but 75% on weekends.", summary) == []

    def test_weekend_context_carries_to_the_next_clause(self, summary):
        assert flagged("On weekends, No screens after 10pm drops to 0%.", summary) == []

    def test_window_level_numbers_pass_anywhere(self, summary):
        assert flagged("Across the 14-day window you logged 28 entries.", summary) == []

    def test_digits_in_habit_names_are_not_claims(self, summary):
        assert flagged("No screens after 10pm is at 71.4%.", summary) == []

    def test_list_markers_and_dates_are_not_claims(self, summary):
        text = "1. **What's working** — as of 2026-06-15, Morning walk is at 92.9%."
        assert flagged(text, summary) == []

    def test_times_and_durations_in_advice_are_not_claims(self, summary):
        """From a live weekly review: 'before 10pm' is advice, not a statistic."""
        text = "For No screens after 10pm, try a 30-minute wind-down before 10pm or 9:30 pm."
        assert flagged(text, summary) == []

    def test_short_names_are_recognised(self, summary):
        """'your walk' refers to Morning walk, so its numbers are checked against it."""
        assert flagged("Your walk streak peaked at 12 days.", summary) == []
        assert flagged("Your screens habit peaked at 12 days.", summary) == ["12"]

    def test_decimal_rates_and_named_streaks_pass(self, summary):
        text = ("Morning walk has a rate of 0.929, a weekend rate of 0.75, "
                "a current streak of 1 day and a longest streak of 12 days.")
        assert flagged(text, summary) == []

    def test_correct_superlatives_pass(self, summary):
        assert flagged("No screens after 10pm is your worst habit at weekends.", summary) == []
        assert flagged("Morning walk is your best habit overall.", summary) == []
        assert flagged("Your weakest habit is No screens after 10pm, at 71.4%.", summary) == []

    def test_no_numbers_nothing_to_check(self, summary):
        report = validate.check("The log doesn't cover how you felt.", summary)
        assert report.ok and report.checked == 0


class TestUnsupportedClaims:
    def test_invented_number_is_flagged(self, summary):
        assert flagged("Morning walk is at 95%.", summary) == ["95%"]

    def test_number_from_another_habit_is_flagged(self, summary):
        """12 is Morning walk's longest streak, not No screens'."""
        assert flagged("No screens after 10pm has a longest streak of 12 days.", summary) == ["12"]

    def test_total_reported_as_weekend_figure_is_flagged(self, summary):
        """The audit's failure: a habit's overall count quoted as its weekend count."""
        assert flagged("No screens after 10pm was done on 10 weekend days.", summary) == ["10"]

    def test_weekend_rate_quoted_as_overall_is_flagged(self, summary):
        assert flagged("Morning walk is at 75%.", summary) == ["75%"]

    def test_habit_carries_into_the_next_sentence(self, summary):
        text = "No screens after 10pm is your weakest habit. It was done on 10 Saturdays."
        report = validate.check(text, summary)
        assert [i.value for i in report.issues] == ["10"]
        assert report.issues[0].habits == ["No screens after 10pm"]

    def test_unnamed_habit_must_match_some_habit(self, summary):
        assert flagged("Your best habit is at 92.9%.", summary) == []
        assert flagged("Your best habit is at 97%.", summary) == ["97%"]

    def test_longest_streak_quoted_as_current_is_flagged(self, summary):
        """From a live weekly review: the longest streak reported as the current one."""
        text = "Morning walk has a current streak of 12 days and a longest streak of 12 days."
        assert flagged(text, summary) == ["12"]

    def test_single_day_rate_quoted_as_weekend_rate_is_flagged(self, summary):
        """From a live weekly review: 'weekends' means Sat+Sun, not Sunday alone (0.5)."""
        text = "On weekends (Saturday and Sunday), Morning walk has a completion rate of 0.5."
        assert flagged(text, summary) == ["0.5"]

    def test_wrong_worst_at_weekends_is_flagged(self, summary):
        """Right numbers, wrong ranking: the live-model failure on the UI's own question."""
        text = "Morning walk is your weakest habit on weekends, done 3 of 4 weekend days."
        assert flagged(text, summary) == ["weakest"]

    def test_wrong_best_overall_is_flagged(self, summary):
        assert flagged("No screens after 10pm is your strongest habit.", summary) == ["strongest"]

    def test_issue_text_names_the_context(self, summary):
        issue = validate.check("Morning walk is at 95%.", summary).issues[0]
        assert "Morning walk" in str(issue) and "95%" in str(issue)


# --- ranking claims, on the seeded logs the evaluation uses ---------------
#
# The texts below are explanations the model wrote in the committed
# evaluation run (eval/results/qwen2.5-7b-instruct.jsonl, dev set), quoted
# verbatim apart from the ANSWER line. Most wrong comparisons there named the
# right habit or day first and then added another.


@pytest.fixture(scope="module")
def seeded():
    from eval import scoring
    return {d: scoring.summary_for(d) for d in scoring.AS_OF_DATES}


def ranking_flags(text, summary):
    return [(i.value, i.habits[0]) for i in validate.check(text, summary).issues if i.claim]


class TestRankingClaims:
    def test_extra_habit_in_a_best_claim_is_flagged(self, seeded):
        """q31, 2026-08-02: Read 20 pages (0.619) is not tied with Morning walk (0.933)."""
        text = ("The Morning walk and Read 20 pages habits have the highest overall completion "
                "rates at 0.619 and 0.933, respectively.")
        assert ranking_flags(text, seeded["2026-08-02"]) == [("highest", "Read 20 pages")]

    def test_them_covers_every_habit_in_the_sentence(self, seeded):
        """q34, 2026-08-02: only No screens after 10pm is worst on weekdays."""
        text = ('On weekdays, "No screens after 10pm" has a rate of 0.35 and "Deep work block" '
                'has a rate of 0.55, making them the worst habits on weekdays.')
        assert ranking_flags(text, seeded["2026-08-02"]) == [("worst", "Deep work block")]

    def test_these_habits_can_point_at_the_next_sentence(self, seeded):
        """q32, 2026-09-15: the habits are named only in the sentence after the claim."""
        text = ('These habits have the lowest completion rates on weekends. For "Deep work '
                'block", the rate is 0.0, and for "No screens after 10pm", the rate is 0.125.')
        assert ranking_flags(text, seeded["2026-09-15"]) == [("lowest", "No screens after 10pm")]

    def test_both_in_a_later_sentence(self, seeded):
        """q31, 2026-09-25: 'Both are tied for the best' after naming two habits."""
        text = ("Morning walk has the highest completion rate overall at 0.9, while Read 20 pages "
                "has a rate of 0.667. Both are tied for the best overall performance.")
        assert ranking_flags(text, seeded["2026-09-25"]) == [("best", "Read 20 pages")]

    def test_a_tie_is_not_flagged(self, seeded):
        """2026-08-02: Deep work block and Read 20 pages are both 0.0 at weekends."""
        text = "Deep work block and Read 20 pages have the lowest weekend rates, both 0.0."
        assert validate.check(text, seeded["2026-08-02"]).ok

    def test_a_contrast_clause_is_not_part_of_the_claim(self, seeded):
        """q30, 2026-09-15: the explanation is right (the ANSWER line was not)."""
        text = ("No screens after 10pm has the lowest overall completion rate of 0.367, while "
                "Deep work block has a rate of 0.762, which is still lower than the other habits.")
        assert validate.check(text, seeded["2026-09-15"]).ok

    def test_second_lowest_is_not_a_top_claim(self, seeded):
        """q34, 2026-09-15: a correct sentence the old rule flagged."""
        text = ("No screens after 10pm has a weekday rate of 0.455, which is the lowest among the "
                "habits, and Deep work block has a weekday rate of 0.727, the second lowest, on "
                "weekdays (mon_to_fri).")
        assert validate.check(text, seeded["2026-09-15"]).ok

    def test_wrong_days_are_flagged(self, seeded):
        """q36, 2026-09-15: Morning walk's lowest day is Monday (0.8); weekends are 1.0."""
        text = ("The completion rate for Morning walk is lowest on Saturdays and Sundays, with a "
                "rate of 0.0, while on weekdays (Monday to Friday), the rate is 0.955.")
        assert ranking_flags(text, seeded["2026-09-15"]) == [
            ("lowest", "Saturday"), ("lowest", "Sunday")]

    def test_extra_days_in_a_list_are_flagged(self, seeded):
        """q40, 2026-09-15: Tuesday is 0.8; Wednesday and Thursday are 0.75."""
        text = ("The completion rate for Read 20 pages is highest on Tuesday, Wednesday, and "
                "Thursday, with a rate of 0.8 for each of these days.")
        assert ranking_flags(text, seeded["2026-09-15"]) == [
            ("highest", "Wednesday"), ("highest", "Thursday")]

    def test_right_days_pass(self, seeded):
        """q38 and q41: correct answers the old rule flagged as claims about habits."""
        assert validate.check(
            "The completion rate for Deep work block is 0.0 on both Saturdays and Sundays, "
            "which is the lowest rate among the weekdays.", seeded["2026-09-15"]).ok
        assert validate.check(
            "Thursday has the highest completion rate for Deep work block at 0.8, with 4 out "
            "of 5 days completed.", seeded["2026-09-25"]).ok

    def test_a_habit_on_one_day(self, seeded):
        """'worst habit on Mondays' compares habits on that day, not days."""
        s = seeded["2026-09-15"]
        rates = {h: p["by_day"]["Monday"]["rate"] for h, p in s["weekday_pattern"].items()}
        worst = min(rates, key=rates.get)
        best = max(rates, key=rates.get)
        assert validate.check(f"{worst} is your worst habit on Mondays.", s).ok
        assert ranking_flags(f"{best} is your worst habit on Mondays.", s) == [("worst", best)]

    def test_longest_current_streak_claim(self, seeded):
        """q09, 2026-09-15: Morning walk's current streak is 1; two habits are on 2."""
        text = ("The Morning walk habit has the longest current streak of 1 day, which is also "
                "its longest streak in the window.")
        assert ranking_flags(text, seeded["2026-09-15"]) == [
            ("the longest current streak", "Morning walk")]

    def test_the_longest_streak_for_a_habit_is_a_number_not_a_ranking(self, seeded):
        text = "The longest streak for Read 20 pages in the last 30 days is 5 days."
        assert validate.check(text, seeded["2026-09-15"]).ok


class TestJointNumbers:
    def test_a_number_for_both_habits_must_hold_for_each(self, seeded):
        """q31, 2026-09-15: 0.81 is Read 20 pages' rate; Morning walk is 0.967."""
        text = ("The Morning walk and Read 20 pages habits both have an overall completion rate "
                "of 0.81, which is the highest among all habits.")
        issues = validate.check(text, seeded["2026-09-15"]).issues
        assert ("0.81", ["Morning walk"]) in [(i.value, i.habits) for i in issues]

    def test_respectively_pairs_the_numbers(self, seeded):
        text = ("On weekdays, Morning walk and Read 20 pages have rates of 0.909 and 0.636 "
                "respectively.")
        assert validate.check(text, seeded["2026-09-25"]).ok


def test_written_out_dates_are_not_claims(seeded):
    """q12, 2026-09-25: a correct answer whose date was flagged as two numbers."""
    text = ("The completion rate for Read 20 pages is 0.667, based on completing 14 out of the "
            "expected 21 days tracked since August 27, 2026.")
    assert validate.check(text, seeded["2026-09-25"]).ok
