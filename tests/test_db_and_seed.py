"""Storage behaviour the summary depends on, and the README's seed block."""
import re
from datetime import date, timedelta
from pathlib import Path

import seed
from habitloop import analytics, db

README = Path(__file__).resolve().parents[1] / "README.md"
AS_OF = date(2026, 9, 15)


def test_archived_habits_are_left_out_of_entries(tmp_path):
    path = tmp_path / "h.db"
    db.init(path)
    keep = db.add_habit("Walk", db_path=path)
    drop = db.add_habit("Read", db_path=path)
    for habit in (keep, drop):
        db.log(habit, AS_OF, True, db_path=path)
    db.archive_habit(drop, db_path=path)

    names = {e["name"] for e in db.entries_between(AS_OF, AS_OF, db_path=path)}
    assert names == {"Walk"}
    everything = db.entries_between(AS_OF, AS_OF, include_archived=True, db_path=path)
    assert {e["name"] for e in everything} == {"Walk", "Read"}
    assert set(analytics.build_summary(db.recent(30, AS_OF, path), 30, AS_OF)["completion"]) == {"Walk"}


def test_created_on_reaches_the_denominator(tmp_path):
    path = tmp_path / "h.db"
    db.init(path)
    habit = db.add_habit("New", created_on=AS_OF - timedelta(days=2), db_path=path)
    for i in range(3):
        db.log(habit, AS_OF - timedelta(days=i), True, db_path=path)

    stats = analytics.completion_by_habit(db.recent(30, AS_OF, path), 30, AS_OF)["New"]
    assert (stats["days_tracked"], stats["rate"]) == (3, 1.0)


def test_relogging_a_day_overwrites(tmp_path):
    path = tmp_path / "h.db"
    db.init(path)
    habit = db.add_habit("Walk", db_path=path)
    db.log(habit, AS_OF, True, db_path=path)
    db.log(habit, AS_OF, False, db_path=path)
    rows = db.entries_between(AS_OF, AS_OF, db_path=path)
    assert [(r["name"], r["done"]) for r in rows] == [("Walk", 0)]


def _seeded_report(path):
    seed.seed(AS_OF, path)
    summary = analytics.build_summary(db.recent(seed.WINDOW, AS_OF, path), seed.WINDOW, AS_OF)
    return seed.report(summary)


def test_seed_is_deterministic_and_rerunnable(tmp_path):
    first = _seeded_report(tmp_path / "a.db")
    again = _seeded_report(tmp_path / "a.db")  # re-seed the same file
    other = _seeded_report(tmp_path / "b.db")
    assert first == again == other


def test_readme_seed_block_matches_a_real_run(tmp_path):
    """The block in the README is exactly what `seed.py --as-of 2026-09-15` prints."""
    text = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    block = re.search(r"--as-of 2026-09-15.*?```\n.*?Last 30 days:\n\n(.*?)\n\nRun:", text, re.S)
    assert block, "README seed block not found"
    assert block.group(1) == _seeded_report(tmp_path / "r.db")
