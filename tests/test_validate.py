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
