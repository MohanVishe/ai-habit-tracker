"""Tests for the statistics the model is grounded on.

These are the numbers the LLM reports back to the user as fact, so they are the
part worth testing. A wrong streak here becomes a confidently wrong statement
in the coach.
"""
from datetime import date, timedelta

import pytest

from habitloop import analytics


def entry(name, day, done=True, target=7):
    return {
        "name": name,
        "on_date": day.isoformat(),
        "done": int(done),
        "target_per_week": target,
        "category": "test",
    }


TODAY = date(2026, 6, 15)  # a Monday


class TestCurrentStreak:
    def test_no_entries(self):
        assert analytics.current_streak([], "Walk", TODAY) == 0

    def test_counts_consecutive_days_back_from_today(self):
        entries = [entry("Walk", TODAY - timedelta(days=i)) for i in range(5)]
        assert analytics.current_streak(entries, "Walk", TODAY) == 5

    def test_a_gap_breaks_the_streak(self):
        entries = [entry("Walk", TODAY), entry("Walk", TODAY - timedelta(days=2))]
        assert analytics.current_streak(entries, "Walk", TODAY) == 1

    def test_unlogged_today_does_not_break_the_streak(self):
        """The day isn't over — yesterday's streak should still stand."""
        entries = [entry("Walk", TODAY - timedelta(days=i)) for i in range(1, 4)]
        assert analytics.current_streak(entries, "Walk", TODAY) == 3

    def test_a_missed_day_is_not_a_completed_day(self):
        entries = [entry("Walk", TODAY, done=False), entry("Walk", TODAY - timedelta(days=1))]
        assert analytics.current_streak(entries, "Walk", TODAY) == 1

    def test_other_habits_are_ignored(self):
        entries = [entry("Walk", TODAY), entry("Read", TODAY - timedelta(days=1))]
        assert analytics.current_streak(entries, "Walk", TODAY) == 1


class TestLongestStreak:
    def test_finds_the_longest_run_not_the_latest(self):
        days = [0, 1, 2, 3, 7, 8]  # a run of 4, then a run of 2
        entries = [entry("Walk", TODAY - timedelta(days=d)) for d in days]
        assert analytics.longest_streak(entries, "Walk") == 4

    def test_single_day(self):
        assert analytics.longest_streak([entry("Walk", TODAY)], "Walk") == 1


class TestCompletion:
    def test_denominator_is_the_window_not_the_row_count(self):
        """Two completions in 30 days is 2/30-ish, never 100%."""
        entries = [entry("Walk", TODAY), entry("Walk", TODAY - timedelta(days=1))]
        result = analytics.completion_by_habit(entries, window_days=30)
        assert result["Walk"]["rate"] < 0.1

    def test_rate_is_capped_at_one(self):
        entries = [entry("Walk", TODAY - timedelta(days=i), target=1) for i in range(14)]
        assert analytics.completion_by_habit(entries, 14)["Walk"]["rate"] == 1.0

    def test_sorted_best_first(self):
        entries = [entry("Bad", TODAY)] + [
            entry("Good", TODAY - timedelta(days=i)) for i in range(10)
        ]
        assert list(analytics.completion_by_habit(entries, 14))[0] == "Good"


class TestWeekdayPattern:
    def test_buckets_by_weekday(self):
        entries = [entry("Walk", TODAY)]  # Monday
        table = analytics.by_weekday(entries)
        assert table["Monday"]["done"] == 1
        assert table["Sunday"]["done"] == 0

    def test_separates_done_from_missed(self):
        entries = [entry("Walk", TODAY), entry("Read", TODAY, done=False)]
        assert analytics.by_weekday(entries)["Monday"] == {"done": 1, "missed": 1}


class TestSummary:
    def test_shape(self):
        entries = [entry("Walk", TODAY - timedelta(days=i)) for i in range(3)]
        summary = analytics.build_summary(entries, 30, TODAY)

        assert summary["habits_tracked"] == 1
        assert summary["total_entries"] == 3
        assert summary["streaks"]["Walk"]["current"] == 3
        assert "Walk" in summary["completion"]

    def test_empty_log_does_not_crash(self):
        summary = analytics.build_summary([], 30, TODAY)
        assert summary["habits_tracked"] == 0
        assert summary["completion"] == {}
