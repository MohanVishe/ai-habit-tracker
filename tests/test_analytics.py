"""Tests for the statistics the model is grounded on.

These are the numbers the LLM reports back to the user as fact, so they are the
part worth testing. A wrong streak here becomes a confidently wrong statement
in the coach.
"""
from datetime import date, timedelta

from habitloop import analytics


def entry(name, day, done=True, target=7, created=None):
    row = {
        "name": name,
        "on_date": day.isoformat(),
        "done": int(done),
        "target_per_week": target,
        "category": "test",
    }
    if created:
        row["created_on"] = created.isoformat()
    return row


TODAY = date(2026, 6, 15)  # a Monday


def days_ago(n):
    return TODAY - timedelta(days=n)


class TestCurrentStreak:
    def test_no_entries(self):
        assert analytics.current_streak([], "Walk", TODAY) == 0

    def test_counts_consecutive_days_back_from_today(self):
        entries = [entry("Walk", days_ago(i)) for i in range(5)]
        assert analytics.current_streak(entries, "Walk", TODAY) == 5

    def test_a_gap_breaks_the_streak(self):
        entries = [entry("Walk", TODAY), entry("Walk", days_ago(2))]
        assert analytics.current_streak(entries, "Walk", TODAY) == 1

    def test_unlogged_today_does_not_break_the_streak(self):
        """The day isn't over — yesterday's streak should still stand."""
        entries = [entry("Walk", days_ago(i)) for i in range(1, 4)]
        assert analytics.current_streak(entries, "Walk", TODAY) == 3

    def test_a_miss_logged_today_breaks_the_streak(self):
        """An explicit miss today is a broken streak, not an unfinished day."""
        entries = [entry("Walk", TODAY, done=False), entry("Walk", days_ago(1))]
        assert analytics.current_streak(entries, "Walk", TODAY) == 0

    def test_miss_today_after_five_done_days_is_zero(self):
        entries = [entry("Walk", TODAY, done=False)]
        entries += [entry("Walk", days_ago(i)) for i in range(1, 6)]
        assert analytics.current_streak(entries, "Walk", TODAY) == 0

    def test_miss_yesterday_with_today_unlogged_is_zero(self):
        entries = [entry("Walk", days_ago(1), done=False)]
        entries += [entry("Walk", days_ago(i)) for i in range(2, 6)]
        assert analytics.current_streak(entries, "Walk", TODAY) == 0

    def test_a_miss_in_the_middle_ends_the_count(self):
        entries = [entry("Walk", TODAY), entry("Walk", days_ago(1)),
                   entry("Walk", days_ago(2), done=False), entry("Walk", days_ago(3))]
        assert analytics.current_streak(entries, "Walk", TODAY) == 2

    def test_only_misses_is_zero(self):
        entries = [entry("Walk", days_ago(i), done=False) for i in range(3)]
        assert analytics.current_streak(entries, "Walk", TODAY) == 0

    def test_other_habits_are_ignored(self):
        entries = [entry("Walk", TODAY), entry("Read", days_ago(1))]
        assert analytics.current_streak(entries, "Walk", TODAY) == 1

    def test_another_habits_miss_today_does_not_reset_this_one(self):
        entries = [entry("Read", TODAY, done=False)]
        entries += [entry("Walk", days_ago(i)) for i in range(1, 4)]
        assert analytics.current_streak(entries, "Walk", TODAY) == 3


class TestLongestStreak:
    def test_finds_the_longest_run_not_the_latest(self):
        days = [0, 1, 2, 3, 7, 8]  # a run of 4, then a run of 2
        entries = [entry("Walk", days_ago(d)) for d in days]
        assert analytics.longest_streak(entries, "Walk") == 4

    def test_single_day(self):
        assert analytics.longest_streak([entry("Walk", TODAY)], "Walk") == 1

    def test_misses_do_not_extend_a_run(self):
        entries = [entry("Walk", TODAY), entry("Walk", days_ago(1), done=False),
                   entry("Walk", days_ago(2))]
        assert analytics.longest_streak(entries, "Walk") == 1


class TestCompletion:
    def test_denominator_is_the_window_not_the_row_count(self):
        """Two completions in 30 days is 2/30-ish, never 100%."""
        entries = [entry("Walk", TODAY), entry("Walk", days_ago(1))]
        result = analytics.completion_by_habit(entries, window_days=30, today=TODAY)
        assert result["Walk"]["rate"] < 0.1

    def test_old_habit_logged_twice_is_still_measured_over_the_window(self):
        created = days_ago(90)
        entries = [entry("Walk", TODAY, created=created),
                   entry("Walk", days_ago(1), created=created)]
        walk = analytics.completion_by_habit(entries, 30, TODAY)["Walk"]
        assert (walk["days_tracked"], walk["expected"]) == (30, 30)
        assert walk["rate"] == round(2 / 30, 3)

    def test_new_habit_is_measured_from_its_start_date(self):
        """Created three days ago and done all three days is 100%, not 10%."""
        created = days_ago(2)
        entries = [entry("New", days_ago(i), created=created) for i in range(3)]
        new = analytics.completion_by_habit(entries, 30, TODAY)["New"]
        assert new["days_tracked"] == 3
        assert new["tracked_since"] == created.isoformat()
        assert new["rate"] == 1.0

    def test_new_habit_with_one_miss(self):
        created = days_ago(3)
        entries = [entry("New", days_ago(i), created=created) for i in range(3)]
        entries.append(entry("New", created, done=False, created=created))
        new = analytics.completion_by_habit(entries, 30, TODAY)["New"]
        assert (new["completed"], new["expected"], new["rate"]) == (3, 4, 0.75)

    def test_new_habit_with_unlogged_days_is_not_full_marks(self):
        created = days_ago(4)  # 5 days, 2 logged
        entries = [entry("New", TODAY, created=created), entry("New", days_ago(1), created=created)]
        assert analytics.completion_by_habit(entries, 30, TODAY)["New"]["rate"] == 0.4

    def test_back_dated_logs_before_created_on_count(self):
        """Logging last week on a habit added today measures from the first log."""
        entries = [entry("New", days_ago(i), created=TODAY) for i in range(7)]
        new = analytics.completion_by_habit(entries, 30, TODAY)["New"]
        assert new["days_tracked"] == 7
        assert new["rate"] == 1.0

    def test_start_is_bounded_by_the_window(self):
        entries = [entry("Old", TODAY, created=days_ago(400))]
        assert analytics.completion_by_habit(entries, 30, TODAY)["Old"]["days_tracked"] == 30

    def test_weekly_target_scales_with_days_tracked(self):
        created = days_ago(13)  # 14 days
        entries = [entry("Read", days_ago(i), target=5, created=created) for i in range(5)]
        read = analytics.completion_by_habit(entries, 30, TODAY)["Read"]
        assert read["expected"] == 10
        assert read["rate"] == 0.5

    def test_rate_is_capped_at_one(self):
        entries = [entry("Walk", days_ago(i), target=1) for i in range(14)]
        assert analytics.completion_by_habit(entries, 14, TODAY)["Walk"]["rate"] == 1.0

    def test_sorted_best_first(self):
        entries = [entry("Bad", TODAY)] + [entry("Good", days_ago(i)) for i in range(10)]
        assert list(analytics.completion_by_habit(entries, 14, TODAY))[0] == "Good"


class TestWeekdayPattern:
    START = days_ago(13)  # two full weeks, Tuesday..Monday

    def pattern(self, entries, name):
        return analytics.weekday_pattern(entries, name, self.START, TODAY)

    def test_buckets_by_weekday(self):
        table = self.pattern([entry("Walk", TODAY)], "Walk")["by_day"]  # Monday
        assert table["Monday"]["done"] == 1
        assert table["Sunday"]["done"] == 0

    def test_is_per_habit(self):
        """Another habit's miss on the same day does not leak into this one."""
        entries = [entry("Walk", TODAY), entry("Read", TODAY, done=False)]
        assert self.pattern(entries, "Walk")["by_day"]["Monday"] == {
            "done": 1, "missed": 0, "days": 2, "rate": 0.5}
        assert self.pattern(entries, "Read")["by_day"]["Monday"] == {
            "done": 0, "missed": 1, "days": 2, "rate": 0.0}

    def test_weekend_and_weekday_totals(self):
        """Done every weekday, never at weekends: 10/10 vs 0/4."""
        entries = [entry("Deep work", days_ago(i), done=days_ago(i).weekday() < 5, target=5)
                   for i in range(14)]
        pattern = self.pattern(entries, "Deep work")
        assert pattern["mon_to_fri"] == {"done": 10, "missed": 0, "days": 10, "rate": 1.0}
        assert pattern["sat_sun"] == {"done": 0, "missed": 4, "days": 4, "rate": 0.0}

    def test_unlogged_days_count_in_days_not_missed(self):
        assert self.pattern([], "Walk")["sat_sun"] == {
            "done": 0, "missed": 0, "days": 4, "rate": 0.0}

    def test_best_and_worst_days_and_stronger_part_of_the_week(self):
        """Done every weekday, never at weekends: every weekday ties for best."""
        entries = [entry("Deep work", days_ago(i), done=days_ago(i).weekday() < 5, target=5)
                   for i in range(14)]
        pattern = self.pattern(entries, "Deep work")
        assert pattern["best_days"] == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        assert pattern["worst_days"] == ["Saturday", "Sunday"]
        assert pattern["stronger_on"] == "weekdays"

    def test_single_worst_day(self):
        entries = [entry("Walk", days_ago(i), done=days_ago(i).weekday() != 2) for i in range(14)]
        pattern = self.pattern(entries, "Walk")
        assert pattern["worst_days"] == ["Wednesday"]
        assert pattern["stronger_on"] == "weekends"

    def test_entries_outside_the_period_are_ignored(self):
        pattern = self.pattern([entry("Walk", self.START - timedelta(days=1))], "Walk")
        assert pattern["mon_to_fri"]["done"] + pattern["sat_sun"]["done"] == 0


class TestSummary:
    def test_shape(self):
        entries = [entry("Walk", days_ago(i)) for i in range(3)]
        summary = analytics.build_summary(entries, 30, TODAY)

        assert summary["habits_tracked"] == 1
        assert summary["total_entries"] == 3
        assert summary["streaks"]["Walk"] == {"current": 3, "longest_in_window": 3}
        assert "Walk" in summary["completion"]
        assert set(summary["weekday_pattern"]["Walk"]) == {
            "by_day", "mon_to_fri", "sat_sun", "best_days", "worst_days", "stronger_on"}

    def test_weekday_pattern_uses_the_habit_start(self):
        created = days_ago(2)  # Saturday, Sunday, Monday
        entries = [entry("New", days_ago(i), created=created) for i in range(3)]
        summary = analytics.build_summary(entries, 30, TODAY)
        assert summary["weekday_pattern"]["New"]["sat_sun"] == {
            "done": 2, "missed": 0, "days": 2, "rate": 1.0}

    def test_rankings_are_worst_first_per_scope(self):
        entries = []
        for i in range(14):
            day = days_ago(i)
            weekend = day.weekday() >= 5
            entries.append(entry("Walk", day, done=True))
            entries.append(entry("Work", day, done=not weekend))
            entries.append(entry("Read", day, done=weekend))
        ranks = analytics.build_summary(entries, 14, TODAY)["rankings"]
        assert [n for n, _ in ranks["sat_sun_rate_worst_first"]] == ["Work", "Read", "Walk"]
        assert ranks["sat_sun_rate_worst_first"][0] == ["Work", 0.0]
        assert [n for n, _ in ranks["mon_to_fri_rate_worst_first"]] == ["Read", "Walk", "Work"]
        assert [n for n, _ in ranks["completion_rate_worst_first"]] == ["Read", "Work", "Walk"]

    def test_best_and_worst_are_named_with_ties(self):
        entries = []
        for i in range(14):
            day = days_ago(i)
            weekend = day.weekday() >= 5
            entries.append(entry("Walk", day, done=True))
            entries.append(entry("Work", day, done=not weekend))
            entries.append(entry("Read", day, done=weekend))
            entries.append(entry("Gym", day, done=not weekend))
        ranks = analytics.build_summary(entries, 14, TODAY)["rankings"]
        assert ranks["best"]["sat_sun"] == ["Read", "Walk"]        # a tie at 1.0
        assert ranks["worst"]["sat_sun"] == ["Gym", "Work"]        # a tie at 0.0
        assert ranks["best"]["overall"] == ["Walk"]
        assert ranks["worst"]["overall"] == ["Read"]
        assert ranks["best"]["mon_to_fri"] == ["Gym", "Walk", "Work"]
        assert ranks["best"]["current_streak"] == ["Walk"]
        assert ranks["best"]["longest_streak_in_window"] == ["Walk"]

    def test_empty_log_does_not_crash(self):
        summary = analytics.build_summary([], 30, TODAY)
        assert summary["habits_tracked"] == 0
        assert summary["completion"] == {}
        assert summary["weekday_pattern"] == {}
