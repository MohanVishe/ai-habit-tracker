"""Ground truth, answer parsing and scoring for the coach evaluation.

Nothing here calls a model, so all of it runs offline and is unit-tested
(tests/test_eval_scoring.py). eval/run.py asks the model; this module decides
whether each answer was right.

Ground truth
    Every expected answer is read from the summary that habitloop.analytics
    computes from the seeded log (seed.py with a fixed --as-of date), the same
    summary the coach is given. No model is involved in producing it.

Answer format
    Free text is hard to score without a judge, so the eval appends a format
    instruction to each question (FORMAT below): a first line
    "ANSWER: <value>" and one sentence of explanation, or "ANSWER: UNKNOWN"
    when the summary cannot answer. Only the ANSWER line is scored.

Scoring rule (by the question's answer_type)
    number   first number in the value must equal the truth exactly.
    rate     first number in the value; a % sign, or a value above 1, is read
             as a percentage. Correct if within 0.005 of the truth (half a
             percentage point, the validator's own tolerance).
    habit    habit names (case-insensitive) found in the value. Correct if at
             least one is named and every name is in the truth set; with a tie,
             naming any of the tied habits is correct (the validator allows
             ties the same way). "complete" records whether all were named.
    weekday  same rule over day names (Saturday, Saturdays or Sat).
    choice   weekdays / weekends / equal ("same" counts as equal); a value
             naming both weekdays and weekends is wrong.
    ranking  every habit named once, in an order consistent with the rates
             (best first); tied habits may come in either order.

    Sensitivity (reported, not the headline): for habit and weekday answers,
    score only the first name in the value, so "A | B" when only A is right
    counts as right. This separates over-long lists from wrong picks.

Outcomes
    correct                answerable, value right
    wrong                  answerable, value wrong
    declined               answerable, but ANSWER: UNKNOWN
    refused_correctly      unanswerable, ANSWER: UNKNOWN
    answered_unanswerable  unanswerable, but a value was given
    no_answer_line         no ANSWER line found (counted as an error)

Validator
    habitloop.validate.check runs on the explanation (the response minus the
    ANSWER line). The bare value on that line has no habit or part of the
    week next to it, so checking it would flag correct weekend rates as
    unsupported overall rates: an artifact of the format, not of the app.
"""
from __future__ import annotations

import json
import math
import random
import re
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from habitloop import analytics, validate
from habitloop.analytics import WEEKDAYS

QUESTIONS = Path(__file__).with_name("questions.jsonl")
# Three end dates for the seeded log: a Sunday, a Tuesday (the README's date)
# and a Friday. The weekends fall on different draws, so the answers differ.
AS_OF_DATES = ("2026-08-02", "2026-09-15", "2026-09-25")

# The held-out set: different questions, asked of logs ending on three other
# weekdays (a Monday, a Wednesday, a Saturday). seed.py draws the same random
# sequence for every end date and only the weekday decides which draws land on
# weekends, so an end date on a weekday already used would give the same log
# again; these three give three new ones (a test checks it).
HELDOUT = Path(__file__).with_name("questions-heldout.jsonl")
HELDOUT_AS_OF_DATES = ("2026-08-24", "2026-09-09", "2026-09-19")

SETS = {
    "dev": (QUESTIONS, AS_OF_DATES),
    "heldout": (HELDOUT, HELDOUT_AS_OF_DATES),
}
WINDOW = 30
RATE_TOLERANCE = 0.005

CORRECT = "correct"
WRONG = "wrong"
DECLINED = "declined"
REFUSED_OK = "refused_correctly"
ANSWERED_UNANSWERABLE = "answered_unanswerable"
NO_ANSWER_LINE = "no_answer_line"
OUTCOMES = (CORRECT, WRONG, DECLINED, REFUSED_OK, ANSWERED_UNANSWERABLE, NO_ANSWER_LINE)
SUCCESS = {CORRECT, REFUSED_OK}

SPEC = {
    "number": "a single whole number",
    "rate": "the rate as a decimal between 0 and 1, exactly as it appears in the summary",
    "habit": ("the habit name exactly as it appears in the summary; if several habits are "
              "tied, list them all separated by \" | \""),
    "weekday": "a day of the week; if several days are tied, list them all separated by \" | \"",
    "choice": "one word: weekdays, weekends or equal",
    "ranking": ("every habit in the summary, best first, exactly as named in the summary, "
                "separated by \" > \""),
}

FORMAT = (
    "\n\nReply in exactly this format. First line: ANSWER: <value>, where <value> is {spec}. "
    "If the summary cannot answer the question, the first line is ANSWER: UNKNOWN. "
    "Second line: one sentence explaining the answer, naming the habit (and weekdays or "
    "weekends, if relevant) and giving the numbers from the summary it rests on."
)


# --- inputs ---------------------------------------------------------------


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_for(question: dict) -> str:
    """The text sent as the user's message: the question plus the format rule."""
    return question["question"] + FORMAT.format(spec=SPEC[question["answer_type"]])


def summary_for(as_of: date | str) -> dict:
    """The summary the app would build for the seeded log ending on `as_of`."""
    import seed  # repo root; imported here so the pure functions need no database
    from habitloop import db

    as_of = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "habits.db"
        seed.seed(as_of, path)
        entries = db.recent(WINDOW, as_of, path)
    return analytics.build_summary(entries, WINDOW, as_of)


# --- ground truth ---------------------------------------------------------


def _all_best(values: dict[str, float | None], pick) -> list[str]:
    known = {k: v for k, v in values.items() if v is not None}
    if not known:
        return []
    target = pick(known.values())
    return sorted(k for k, v in known.items() if v == target)


def _metric(summary: dict, scope: str) -> dict[str, float | None]:
    """Per-habit value that a comparison in `scope` is about: overall rate,
    Sat+Sun or Mon-Fri rate, one weekday's rate, or a streak."""
    if scope == "overall":
        return {h: c["rate"] for h, c in summary["completion"].items()}
    if scope in ("sat_sun", "mon_to_fri"):
        return {h: p[scope]["rate"] for h, p in summary["weekday_pattern"].items()}
    if scope in WEEKDAYS:
        return {h: p["by_day"][scope]["rate"] for h, p in summary["weekday_pattern"].items()}
    if scope == "current_streak":
        return {h: s["current"] for h, s in summary["streaks"].items()}
    raise ValueError(f"unknown scope {scope!r}")


def _tiers(values: dict[str, float | None]) -> list[list[str]]:
    """Habits grouped by value, best (highest) first; a group of two is a tie."""
    known = {k: v for k, v in values.items() if v is not None}
    return [sorted(k for k, v in known.items() if v == target)
            for target in sorted(set(known.values()), reverse=True)]


def ground_truth(question: dict, summary: dict):
    """The expected answer, from the computed summary; None = unanswerable.

    number/rate kinds return a number, habit/weekday kinds a sorted list (more
    than one entry means a tie), choice kinds a string.
    """
    kind = question["kind"]
    habit = question.get("habit")
    streaks = summary["streaks"]
    completion = summary["completion"]
    pattern = summary["weekday_pattern"]

    if kind == "unknown":
        return None
    if kind == "current_streak":
        return streaks[habit]["current"]
    if kind == "longest_streak":
        return streaks[habit]["longest_in_window"]
    if kind == "top_current_streak":
        return _all_best({h: s["current"] for h, s in streaks.items()}, max)
    if kind == "top_longest_streak":
        return _all_best({h: s["longest_in_window"] for h, s in streaks.items()}, max)
    if kind == "completion_rate":
        return completion[habit]["rate"]
    if kind == "completed":
        return completion[habit]["completed"]
    if kind == "missed":
        return completion[habit]["explicitly_missed"]
    if kind == "habits_tracked":
        return summary["habits_tracked"]
    if kind == "weekend_rate":
        return pattern[habit]["sat_sun"]["rate"]
    if kind == "weekday_rate":
        return pattern[habit]["mon_to_fri"]["rate"]
    if kind == "day_rate":
        return pattern[habit]["by_day"][question["day"]]["rate"]
    if kind == "more_reliable":
        weekend = pattern[habit]["sat_sun"]["rate"]
        weekday = pattern[habit]["mon_to_fri"]["rate"]
        return "equal" if weekend == weekday else ("weekends" if weekend > weekday else "weekdays")
    if kind in ("worst_habit", "best_habit"):
        rates = _metric(summary, question["scope"])
        return _all_best(rates, min if kind == "worst_habit" else max)
    if kind in ("worst_day", "best_day"):
        rates = {d: b["rate"] for d, b in pattern[habit]["by_day"].items()}
        return _all_best(rates, min if kind == "worst_day" else max)
    # --- kinds used only by the held-out set ---
    if kind in ("second_best_habit", "second_worst_habit"):
        # the habit(s) at the second-highest (second-lowest) distinct value
        tiers = _tiers(_metric(summary, question["scope"]))
        if kind == "second_worst_habit":
            tiers = tiers[::-1]
        return tiers[1] if len(tiers) > 1 else []
    if kind in ("higher_of_two", "lower_of_two"):
        values = {h: v for h, v in _metric(summary, question["scope"]).items()
                  if h in question["habits"]}
        return _all_best(values, max if kind == "higher_of_two" else min)
    if kind == "biggest_weekend_drop":
        drops = {h: round(p["mon_to_fri"]["rate"] - p["sat_sun"]["rate"], 3)
                 for h, p in pattern.items()
                 if p["mon_to_fri"]["rate"] is not None and p["sat_sun"]["rate"] is not None}
        return _all_best(drops, max)
    if kind == "rank_habits":
        return _tiers(_metric(summary, question["scope"]))
    raise ValueError(f"unknown question kind {kind!r}")


# --- parsing and scoring --------------------------------------------------


_ANSWER = re.compile(r"^[\s>*_`#-]*answer[\s*_`]*:(.*)$", re.IGNORECASE | re.MULTILINE)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_RATE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%|percent\b)?", re.IGNORECASE)


def _clean(value: str) -> str:
    """Drop markdown, quotes, brackets and a trailing full stop around a value."""
    previous = None
    while value != previous:
        previous = value
        value = value.strip().strip("\"'<>`*_[]").rstrip(".")
    return value


def parse_response(text: str) -> tuple[str | None, str]:
    """(value on the first ANSWER line or None, the rest of the text)."""
    match = _ANSWER.search(text)
    if not match:
        return None, text.strip()
    value = _clean(match.group(1))
    rest = text[:match.start()] + text[match.end():]
    if not value:  # "ANSWER:" with the value on the next line
        lines = rest[match.start():].lstrip("\n").split("\n", 1)
        value = _clean(lines[0])
        rest = text[:match.start()] + (lines[1] if len(lines) > 1 else "")
    return value, rest.strip()


def is_unknown(value: str) -> bool:
    return value.lower().startswith("unknown")


def _days_named(value: str) -> set[str]:
    return {
        d for d in WEEKDAYS
        if re.search(rf"\b{d[:3]}(?:{d[3:]})?s?\b", value, re.IGNORECASE)
    }


def _in_order(value: str, habits: list[str]) -> list[str]:
    """Habits named in the value, in the order they appear."""
    found = []
    for habit in habits:
        found += [(m.start(), habit) for m in re.finditer(re.escape(habit), value, re.IGNORECASE)]
    return [habit for _, habit in sorted(found)]


def compare(answer_type: str, value: str, truth, habits: list[str]) -> tuple[bool, bool]:
    """(correct, complete) for an answerable question. `complete` differs from
    `correct` only for ties, where naming a subset of the tied items is correct."""
    if answer_type == "ranking":
        # truth: tiers best first. Every habit once, and no habit placed above
        # one from a better tier.
        named = _in_order(value, habits)
        tier = {h: i for i, group in enumerate(truth) for h in group}
        ok = (sorted(named) == sorted(tier)
              and all(tier[a] <= tier[b] for a, b in zip(named, named[1:])))
        return ok, ok
    if answer_type in ("number", "rate"):
        # Digits inside a habit name ("Read 20 pages") are not the answer.
        for habit in habits:
            value = re.sub(re.escape(habit), " ", value, flags=re.IGNORECASE)
    if answer_type == "number":
        found = _NUM.search(value)
        ok = bool(found) and float(found.group(0)) == float(truth)
        return ok, ok
    if answer_type == "rate":
        found = _RATE.search(value)
        if not found:
            return False, False
        x = float(found.group(1))
        if found.group(2) or x > 1:
            x /= 100
        ok = abs(x - float(truth)) <= RATE_TOLERANCE + 1e-9
        return ok, ok
    if answer_type in ("habit", "weekday"):
        if answer_type == "habit":
            named = {h for h in habits if h.lower() in value.lower()}
        else:
            named = _days_named(value)
        ok = bool(named) and named <= set(truth)
        return ok, ok and named == set(truth)
    if answer_type == "choice":
        v = value.lower()
        if re.search(r"\b(equal|same)\b", v):
            pick = "equal"
        elif "weekend" in v and "weekday" not in v:
            pick = "weekends"
        elif "weekday" in v and "weekend" not in v:
            pick = "weekdays"
        else:
            pick = None
        ok = pick == truth
        return ok, ok
    raise ValueError(f"unknown answer_type {answer_type!r}")


def first_named(answer_type: str, value: str, habits: list[str]) -> str | None:
    """The habit or day named first in the value (for the sensitivity check)."""
    names = habits if answer_type == "habit" else WEEKDAYS
    found = []
    for name in names:
        pattern = (re.escape(name) if answer_type == "habit"
                   else rf"\b{name[:3]}(?:{name[3:]})?s?\b")
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            found.append((match.start(), name))
    return min(found)[1] if found else None


def score(question: dict, truth, response: str, habits: list[str]) -> dict:
    value, explanation = parse_response(response)
    complete = None
    if value is None:
        outcome = NO_ANSWER_LINE
    elif truth is None:
        outcome = REFUSED_OK if is_unknown(value) else ANSWERED_UNANSWERABLE
    elif is_unknown(value):
        outcome = DECLINED
    else:
        correct, complete = compare(question["answer_type"], value, truth, habits)
        outcome = CORRECT if correct else WRONG
    return {"value": value, "explanation": explanation, "outcome": outcome, "complete": complete}


def evaluate(question: dict, as_of: str, response: str, summary: dict) -> dict:
    """One scored record: ground truth, outcome and the validator's verdict."""
    truth = ground_truth(question, summary)
    scored = score(question, truth, response, list(summary["completion"]))
    report = validate.check(scored["explanation"], summary)
    # Sensitivity: habit/weekday answers scored on the first name alone, so an
    # over-long list ("A | B" when only A is right) counts as right.
    first_ok = scored["outcome"] in SUCCESS
    if scored["outcome"] == WRONG and question["answer_type"] in ("habit", "weekday"):
        first = first_named(question["answer_type"], scored["value"], list(summary["completion"]))
        first_ok = first in truth
    return {
        "id": question["id"],
        "as_of": as_of,
        "category": question["category"],
        "kind": question["kind"],
        "answer_type": question["answer_type"],
        "question": question["question"],
        "truth": truth,
        "response": response,
        "value": scored["value"],
        "outcome": scored["outcome"],
        "complete": scored["complete"],
        "first_named_correct": first_ok,
        "validator_checked": report.checked,
        "validator_flags": [str(issue) for issue in report.issues],
    }


# --- statistics -----------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap(records: list[dict], hit, reps: int = 10000,
                      seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for the share of records where hit(record) is
    true, resampling whole questions: the same question asked of three logs is
    not three independent trials."""
    by_q: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_q[r["id"]].append(bool(hit(r)))
    groups = list(by_q.values())
    if not groups:
        return (0.0, 1.0)
    rng = random.Random(seed)
    shares = []
    for _ in range(reps):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        n = sum(len(g) for g in sample)
        shares.append(sum(sum(g) for g in sample) / n)
    shares.sort()
    return (shares[int(0.025 * reps)], shares[int(0.975 * reps) - 1])


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)]}


def summarize(records: list[dict]) -> dict:
    """Error rate and validator performance over scored records."""
    n = len(records)
    errors = [r for r in records if r["outcome"] not in SUCCESS]
    lo, hi = cluster_bootstrap(records, lambda r: r["outcome"] not in SUCCESS)

    def flagged(rows):
        return sum(1 for r in rows if r["validator_flags"])

    def rows(outcome):
        return [r for r in records if r["outcome"] == outcome]

    by_category = {}
    for category in dict.fromkeys(r["category"] for r in records):
        subset = [r for r in records if r["category"] == category]
        by_category[category] = {
            **_rate(sum(r["outcome"] in SUCCESS for r in subset), len(subset)),
            "outcomes": dict(Counter(r["outcome"] for r in subset)),
        }

    return {
        "n_items": n,
        "n_questions": len({r["id"] for r in records}),
        "as_of_dates": sorted({r["as_of"] for r in records}),
        "outcomes": {o: sum(r["outcome"] == o for r in records) for o in OUTCOMES},
        "accuracy": _rate(n - len(errors), n),
        "error_rate": {**_rate(len(errors), n),
                       "question_bootstrap95": [round(lo, 4), round(hi, 4)]},
        "accuracy_if_first_named_counts": _rate(
            sum(bool(r["first_named_correct"]) for r in records), n),
        "ties_named_partially": sum(1 for r in records if r["complete"] is False
                                    and r["outcome"] == CORRECT),
        "by_category": by_category,
        "by_as_of": {
            d: _rate(sum(r["outcome"] in SUCCESS for r in records if r["as_of"] == d),
                     sum(1 for r in records if r["as_of"] == d))
            for d in sorted({r["as_of"] for r in records})
        },
        "validator": {
            "flagged_wrong_answers": _rate(flagged(rows(WRONG)), len(rows(WRONG))),
            "flagged_answered_unanswerable": _rate(flagged(rows(ANSWERED_UNANSWERABLE)),
                                                   len(rows(ANSWERED_UNANSWERABLE))),
            "flagged_any_error": _rate(flagged(errors), len(errors)),
            "flagged_correct_answers": _rate(flagged(rows(CORRECT)), len(rows(CORRECT))),
            "flagged_correct_refusals": _rate(flagged(rows(REFUSED_OK)), len(rows(REFUSED_OK))),
        },
    }
