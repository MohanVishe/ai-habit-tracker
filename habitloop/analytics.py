"""Statistics over habit entries.

Everything here is a pure function over plain dicts — no database, no LLM.
That is deliberate: these numbers are what the model gets fed, so they are the
part that has to be right, and pure functions are the part that can be tested.

An entry is a dict with at least `name`, `on_date` (date or ISO string) and
`done` (truthy/falsy). Optional: `target_per_week` (default 7) and
`created_on` (the day the habit was added; used to size the denominator).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _to_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def tracked_since(entries: list[dict], habit_name: str, window_days: int,
                  today: date) -> date:
    """First day of the window that counts for this habit.

    The window start, unless the habit is younger than the window: a habit
    added three days ago and done all three days is 3/3, not 3/30. Uses the
    earlier of `created_on` and the first logged day (the UI allows back-dated
    logging). With no `created_on` on the entries, the whole window counts —
    the conservative choice, since it can only under-report.
    """
    window_start = today - timedelta(days=window_days - 1)
    rows = [e for e in entries if e["name"] == habit_name]
    created = [_to_date(e["created_on"]) for e in rows if e.get("created_on")]
    if not created:
        return window_start
    first_logged = min(_to_date(e["on_date"]) for e in rows)
    return min(max(window_start, min(min(created), first_logged)), today)


def completion_by_habit(entries: list[dict], window_days: int,
                        today: date | None = None) -> dict[str, dict]:
    """Per-habit completion over the days the habit existed inside the window.

    The denominator is days, not rows — a habit logged twice in thirty days is
    2/30, not 2/2. Using row count as the denominator is the easy mistake here
    and it reports 100% for total neglect. The days counted run from
    `tracked_since` to today, so a new habit is not penalised for the part of
    the window before it existed.
    """
    today = today or date.today()
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

    summary = {}
    for name, target in targets.items():
        start = tracked_since(entries, name, window_days, today)
        days = (today - start).days + 1
        expected = max(round(target * days / 7), 1)
        completed = done[name]
        summary[name] = {
            "completed": completed,
            "explicitly_missed": missed[name],
            "days_tracked": days,
            "tracked_since": start.isoformat(),
            "expected": expected,
            "rate": round(min(completed / expected, 1.0), 3),
            "target_per_week": target,
        }

    return dict(sorted(summary.items(), key=lambda kv: kv[1]["rate"], reverse=True))


def current_streak(entries: list[dict], habit_name: str, today: date | None = None) -> int:
    """Consecutive days completed, counting back from today.

    A day with no entry breaks the streak — absence of a log is a miss, not a
    gap to be skipped over. The one exception is today: if today has no entry
    yet the day isn't over, so counting starts from yesterday. If today is
    logged as *missed*, the streak is 0.
    """
    today = today or date.today()
    log = {
        _to_date(e["on_date"]): bool(e["done"])
        for e in entries
        if e["name"] == habit_name
    }

    if today in log:
        if not log[today]:
            return 0
        cursor = today
    else:
        cursor = today - timedelta(days=1)

    streak = 0
    while log.get(cursor):
        streak += 1
        cursor -= timedelta(days=1)

    return streak


def longest_streak(entries: list[dict], habit_name: str) -> int:
    """Longest run of consecutive completed days in the entries given."""
    completed = sorted({
        _to_date(e["on_date"]) for e in entries if e["name"] == habit_name and e["done"]
    })
    if not completed:
        return 0

    best = run = 1
    for previous, current in zip(completed, completed[1:]):
        run = run + 1 if current - previous == timedelta(days=1) else 1
        best = max(best, run)
    return best


def _bucket(done: int, missed: int, days: int) -> dict:
    return {
        "done": done,
        "missed": missed,
        "days": days,
        "rate": round(done / days, 3) if days else None,
    }


def tied_best(values: dict[str, float | None], pick=max) -> list[str]:
    """Every key sharing the highest (pick=min: lowest) value, in input order.
    More than one name means an exact tie; None values are skipped."""
    known = {k: v for k, v in values.items() if v is not None}
    if not known:
        return []
    target = pick(known.values())
    return [k for k, v in known.items() if v == target]


def weekday_pattern(entries: list[dict], habit_name: str, start: date,
                    today: date) -> dict:
    """One habit's completions per weekday, from `start` to `today` inclusive.

    `days` is how many of that weekday fell in the period, so `rate` is the
    share of, say, Saturdays on which the habit was done. `sat_sun` and
    `mon_to_fri` aggregate the same counts — they answer "worst at weekends?"
    directly instead of leaving the model to add up rows.

    `best_days` / `worst_days` (every day tied at the top or bottom rate) and
    `stronger_on` answer "which day?" and "weekdays or weekends?" outright. On
    the evaluation set, reading seven rates and picking the extreme was where
    the model went wrong most: it added a second day, or quoted a rate that
    isn't there.
    """
    counts = {n: {"done": 0, "missed": 0} for n in WEEKDAYS}
    for entry in entries:
        if entry["name"] != habit_name:
            continue
        day = _to_date(entry["on_date"])
        if start <= day <= today:
            counts[WEEKDAYS[day.weekday()]]["done" if entry["done"] else "missed"] += 1

    days = {n: 0 for n in WEEKDAYS}
    cursor = start
    while cursor <= today:
        days[WEEKDAYS[cursor.weekday()]] += 1
        cursor += timedelta(days=1)

    def total(names):
        return _bucket(
            sum(counts[n]["done"] for n in names),
            sum(counts[n]["missed"] for n in names),
            sum(days[n] for n in names),
        )

    by_day = {n: _bucket(counts[n]["done"], counts[n]["missed"], days[n]) for n in WEEKDAYS}
    mon_to_fri, sat_sun = total(WEEKDAYS[:5]), total(WEEKDAYS[5:])
    rates = {n: b["rate"] for n, b in by_day.items()}
    if mon_to_fri["rate"] is None or sat_sun["rate"] is None:
        stronger = None
    elif mon_to_fri["rate"] == sat_sun["rate"]:
        stronger = "equal"
    else:
        stronger = "weekdays" if mon_to_fri["rate"] > sat_sun["rate"] else "weekends"
    return {
        "by_day": by_day,
        "mon_to_fri": mon_to_fri,
        "sat_sun": sat_sun,
        "best_days": tied_best(rates, max),
        "worst_days": tied_best(rates, min),
        "stronger_on": stronger,
    }


def build_summary(entries: list[dict], window_days: int = 30,
                  today: date | None = None) -> dict:
    """The single object handed to the LLM.

    The model never sees raw rows. It sees this — computed in Python, where the
    arithmetic is deterministic and testable. Asking an LLM to count days is
    asking it to do the one thing it is worst at.
    """
    today = today or date.today()
    habits = sorted({e["name"] for e in entries})
    completion = completion_by_habit(entries, window_days, today)
    streaks = {
        name: {
            "current": current_streak(entries, name, today),
            "longest_in_window": longest_streak(entries, name),
        }
        for name in habits
    }
    weekdays = {
        name: weekday_pattern(
            entries, name, date.fromisoformat(completion[name]["tracked_since"]), today
        )
        for name in habits
    }

    return {
        "window_days": window_days,
        "generated_on": today.isoformat(),
        "total_entries": len(entries),
        "habits_tracked": len(habits),
        "completion": completion,
        "streaks": streaks,
        "weekday_pattern": weekdays,
        "rankings": rankings(completion, weekdays, streaks),
    }


def rankings(completion: dict, weekdays: dict, streaks: dict | None = None) -> dict:
    """Habits ordered worst-first by each rate, as [name, rate] pairs, plus the
    answer to "which is best / worst?" for each rate and for streaks.

    "Which habit am I worst at on weekends?" is a comparison across habits.
    Left to the model, it is one more place to get the order wrong while every
    individual number is right, so the order is computed here too. The
    ordered lists alone were not enough: on the evaluation set the model read
    the right first entry and then added the next one. So `best` and `worst`
    hold only the habit(s) at the top or bottom; two names there means an
    exact tie.
    """
    def order(rates: dict[str, float | None]) -> list:
        known = [(n, r) for n, r in rates.items() if r is not None]
        return [[n, r] for n, r in sorted(known, key=lambda kv: (kv[1], kv[0]))]

    rates = {
        "overall": {n: c["rate"] for n, c in completion.items()},
        "sat_sun": {n: w["sat_sun"]["rate"] for n, w in weekdays.items()},
        "mon_to_fri": {n: w["mon_to_fri"]["rate"] for n, w in weekdays.items()},
    }
    best = {scope: sorted(tied_best(r, max)) for scope, r in rates.items()}
    worst = {scope: sorted(tied_best(r, min)) for scope, r in rates.items()}
    if streaks:
        best["current_streak"] = sorted(tied_best({n: s["current"] for n, s in streaks.items()}))
        best["longest_streak_in_window"] = sorted(
            tied_best({n: s["longest_in_window"] for n, s in streaks.items()}))
    return {
        "completion_rate_worst_first": order(rates["overall"]),
        "sat_sun_rate_worst_first": order(rates["sat_sun"]),
        "mon_to_fri_rate_worst_first": order(rates["mon_to_fri"]),
        "best": best,
        "worst": worst,
    }
