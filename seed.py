"""Populate the database with 60 days of plausible sample data.

Deliberately uneven: one habit that is nearly perfect, one that collapses at
weekends, one that was abandoned three weeks in. A seed where everything is
80% complete makes the analysis look good and proves nothing.

    python seed.py                         # 60 days ending today
    python seed.py --as-of 2026-09-15      # 60 days ending on a fixed date

The draws are seeded, so the same --as-of date always produces the same log
and the same printed summary, on any machine and any day. Without --as-of the
numbers depend on which weekday today is (the weekends move). Re-running
replaces the sample habits rather than adding to them.
"""
from __future__ import annotations

import argparse
import random
from datetime import date, timedelta

from habitloop import analytics, db

SEED = 7
DAYS = 60
WINDOW = 30

HABITS = [
    # name, category, target, weekday probability, weekend probability, abandoned after
    ("Morning walk", "health", 7, 0.92, 0.85, None),
    ("Read 20 pages", "learning", 5, 0.74, 0.20, None),
    ("Deep work block", "work", 5, 0.66, 0.05, None),
    ("Meditate", "mind", 7, 0.55, 0.45, 21),
    ("No screens after 10pm", "health", 7, 0.38, 0.12, None),
]

NOTES = [
    None, None, None,
    "felt good", "rushed it", "almost skipped", "early start",
    "tired", "travelling", "long day",
]


def seed(as_of: date, db_path=None) -> None:
    rng = random.Random(SEED)
    db.init(db_path)
    first_day = as_of - timedelta(days=DAYS - 1)

    for name, category, target, weekday_p, weekend_p, abandon_after in HABITS:
        db.delete_habit(name, db_path=db_path)
        habit_id = db.add_habit(name, category, target, created_on=first_day, db_path=db_path)

        for offset in range(DAYS - 1, -1, -1):
            day = as_of - timedelta(days=offset)
            age = DAYS - offset

            if abandon_after and age > abandon_after:
                continue

            probability = weekend_p if day.weekday() >= 5 else weekday_p
            done = rng.random() < probability

            db.log(habit_id, day, done, rng.choice(NOTES) if done else None, db_path=db_path)


def report(summary: dict) -> str:
    """The block quoted in the README: rate, current/longest streak, weekends."""
    lines = []
    for name, stats in summary["completion"].items():
        streak = summary["streaks"][name]
        weekend = summary["weekday_pattern"][name]["sat_sun"]
        lines.append(
            f"{name:<22} {stats['rate'] * 100:5.1f}%  "
            f"streak {streak['current']}/{streak['longest_in_window']:<3} "
            f"weekends {weekend['done']:>2}/{weekend['days']} done"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today(),
                        help="last day of sample data (YYYY-MM-DD, default today)")
    parser.add_argument("--db", default=None,
                        help="database file (default $HABITLOOP_DB or data/habits.db)")
    args = parser.parse_args(argv)

    seed(args.as_of, args.db)
    summary = analytics.build_summary(
        db.recent(WINDOW, today=args.as_of, db_path=args.db), WINDOW, args.as_of
    )
    print(f"Seeded {len(HABITS)} habits, {DAYS} days ending {args.as_of} "
          f"({args.as_of:%A}). Last {WINDOW} days:\n")
    print(report(summary))
    print("\nRun: streamlit run app.py")


if __name__ == "__main__":
    main()
