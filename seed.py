"""Populate the database with 60 days of plausible sample data.

Deliberately uneven: one habit that is nearly perfect, one that collapses at
weekends, one that was abandoned three weeks in. A seed where everything is
80% complete makes the analysis look good and proves nothing.

    python seed.py
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from habitloop import db

random.seed(7)

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


def main() -> None:
    db.init()
    today = date.today()

    for name, category, target, weekday_p, weekend_p, abandon_after in HABITS:
        habit_id = db.add_habit(name, category, target)

        for offset in range(59, -1, -1):
            day = today - timedelta(days=offset)
            age = 60 - offset

            if abandon_after and age > abandon_after:
                continue

            probability = weekend_p if day.weekday() >= 5 else weekday_p
            done = random.random() < probability

            db.log(habit_id, day, done, random.choice(NOTES) if done else None)

        print(f"seeded {name}")

    print("\nDone. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
