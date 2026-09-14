"""Statistics over habit entries.

Everything here is a pure function over plain dicts — no database, no LLM.
That is deliberate: these numbers are what the model gets fed, so they are the
part that has to be right, and pure functions are the part that can be tested.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta


def _to_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def completion_by_habit(entries: list[dict], window_days: int) -> dict[str, dict]:
    """Per-habit completion over the window.

    `window_days` is the denominator, not the number of rows — a habit logged
    twice in thirty days is 2/30, not 2/2. Using row count as the denominator
    is the easy mistake here and it reports 100% for total neglect.
    """
    done = defaultdict(int)
    missed = defaultdict(int)
    targets: dict[str, int] = {}

    for entry in entries:
        name = entry["name"]
        targets[name] = entry.get("target_per_week", 7)
        if entry["done"]:
            done[name] += 1
        else:
            missed[name] += 1

    weeks = max(window_days / 7, 1)
    summary = {}

    for name, target in targets.items():
        completed = done[name]
        expected = max(round(target * weeks), 1)
        summary[name] = {
            "completed": completed,
            "explicitly_missed": missed[name],
            "expected": expected,
            "rate": round(min(completed / expected, 1.0), 3),
            "target_per_week": target,
        }

    return dict(sorted(summary.items(), key=lambda kv: kv[1]["rate"], reverse=True))


def current_streak(entries: list[dict], habit_name: str, today: date | None = None) -> int:
    """Consecutive days completed, counting back from today.

    A day with no entry breaks the streak — absence of a log is a miss, not a
    gap to be skipped over.
    """
    today = today or date.today()
    completed = {
        _to_date(e["on_date"])
        for e in entries
        if e["name"] == habit_name and e["done"]
    }

    if not completed:
        return 0

    # Today not yet logged is not a broken streak — the day isn't over.
    cursor = today if today in completed else today - timedelta(days=1)

    streak = 0
    while cursor in completed:
        streak += 1
        cursor -= timedelta(days=1)

    return streak


def longest_streak(entries: list[dict], habit_name: str) -> int:
    completed = sorted(
        _to_date(e["on_date"]) for e in entries if e["name"] == habit_name and e["done"]
    )
    if not completed:
        return 0

    best = run = 1
    for previous, current in zip(completed, completed[1:]):
        run = run + 1 if current - previous == timedelta(days=1) else 1
        best = max(best, run)
    return best


def by_weekday(entries: list[dict]) -> dict[str, dict[str, int]]:
    """Completions per weekday — surfaces the 'weekends always collapse' pattern."""
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    table = {n: {"done": 0, "missed": 0} for n in names}

    for entry in entries:
        weekday = names[_to_date(entry["on_date"]).weekday()]
        table[weekday]["done" if entry["done"] else "missed"] += 1

    return table


def build_summary(entries: list[dict], window_days: int = 30,
                  today: date | None = None) -> dict:
    """The single object handed to the LLM.

    The model never sees raw rows. It sees this — computed in Python, where the
    arithmetic is deterministic and testable. Asking an LLM to count days is
    asking it to do the one thing it is worst at.
    """
    today = today or date.today()
    habits = sorted({e["name"] for e in entries})

    return {
        "window_days": window_days,
        "generated_on": today.isoformat(),
        "total_entries": len(entries),
        "habits_tracked": len(habits),
        "completion": completion_by_habit(entries, window_days),
        "streaks": {
            name: {
                "current": current_streak(entries, name, today),
                "longest": longest_streak(entries, name),
            }
            for name in habits
        },
        "weekday_pattern": by_weekday(entries),
    }
