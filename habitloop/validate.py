"""Check the numbers in a model answer against the summary it was given.

The prompt asks the model to stay inside the summary; this checks, after the
fact, whether it did — at the level of *claims*, not bare numbers. Checking
only that "every number appears somewhere in the summary" passes most real
hallucinations, because a 30-day summary contains almost every small integer.
The failure that matters is a real number attached to the wrong thing: a
habit's 20 total misses reported as its weekend misses, or one habit's rate
quoted for another.

So each number is checked against the facts for the habit(s) and the part of
the week the sentence is talking about:

- Habit context: habits named in the clause, else the previous clause or
  sentence (so "It dropped to 20%" still refers to the habit just named).
  No habit named anywhere yet means any habit's facts are allowed.
- Day context: a clause that mentions weekends, weekdays or a day name is
  checked against that habit's weekday facts ("weekends" means the Sat+Sun
  totals, not either day alone); otherwise against its overall
  facts (completion counts, rate, target, streaks). A day context carries
  forward to later clauses of the same sentence until "overall"/"total"
  resets it.
- Window-level numbers (window length, entry count, habit count) are allowed
  anywhere.

Percentages and decimals such as 0.667 match rates within half a point
(66.7% may be written 67%); whole numbers match counts exactly, or a rate
written without a % sign. A number right after "current streak" or "longest
streak" must be that streak, not the other one.

Best/worst claims about a single habit ("Deep work is your worst habit at
weekends") are checked against the rates in the same context: the habit must
have the lowest (or highest) rate there, ties allowed. This catches the case
where every number quoted is right but the ranking drawn from them is wrong.

Not checked (by design, and stated in the README): numbers written as words,
other claims without numbers ("you met your target"), comparisons between two
named habits, invented habit names, times of day and durations, and
anything in the 7-day plan, whose day numbers and dates are not statistics.
The result flags; it does not rewrite the answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .analytics import WEEKDAYS

GENERIC_WORDS = {
    "after", "before", "block", "daily", "days", "every", "from", "habit", "into",
    "minutes", "night", "session", "than", "that", "this", "time", "week", "with",
    "work", "your",
}

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s*(?:%|percent\b))?")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Times of day and durations ("before 10pm", "a 30-minute wind-down") are advice, not statistics.
_TIME_OR_DURATION = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d+(?:\.\d+)?[\s-]*(?:min(?:ute)?s?|hours?|hrs?)\b",
    re.IGNORECASE)
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|#+)\s+", re.MULTILINE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE = re.compile(r"[,;:()]|\s(?:but|while|whereas|compared (?:with|to)|versus|vs\.?)\s",
                     re.IGNORECASE)
_WORST = re.compile(r"\b(worst|weakest|lowest|poorest|least consistent)\b", re.IGNORECASE)
_BEST = re.compile(r"\b(best|strongest|highest|most consistent)\b", re.IGNORECASE)
# Streaks exist only over the whole window, so a streak clause is never day-scoped.
_RESET = re.compile(r"\b(overall|total|in all|across the (?:window|month)|streaks?)\b",
                    re.IGNORECASE)


@dataclass
class Issue:
    value: str
    sentence: str
    habits: list[str]
    scope: list[str]

    def __str__(self) -> str:
        who = ", ".join(self.habits) if self.habits else "any habit"
        where = ", ".join(self.scope)
        return f"{self.value} — not in the summary for {who} ({where}): \"{self.sentence}\""


@dataclass
class Report:
    checked: int = 0
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


# --- facts ----------------------------------------------------------------


def _facts(summary: dict) -> tuple[dict, set[float]]:
    """facts[habit][scope] -> {"counts": set, "rates": set (as percentages)}."""
    facts: dict[str, dict[str, dict[str, set[float]]]] = {}

    def add(habit, scope, counts=(), rates=()):
        slot = facts.setdefault(habit, {}).setdefault(scope, {"counts": set(), "rates": set()})
        slot["counts"].update(float(c) for c in counts if c is not None)
        slot["rates"].update(float(r) * 100 for r in rates if r is not None)

    for habit, c in summary.get("completion", {}).items():
        add(habit, "overall",
            counts=[c["completed"], c["explicitly_missed"], c["expected"],
                    c["days_tracked"], c["target_per_week"]],
            rates=[c["rate"]])
    for habit, s in summary.get("streaks", {}).items():
        add(habit, "overall", counts=[s["current"], s["longest_in_window"]])
        add(habit, "streak_current", counts=[s["current"]])
        add(habit, "streak_longest", counts=[s["longest_in_window"]])
    for habit, pattern in summary.get("weekday_pattern", {}).items():
        buckets = dict(pattern["by_day"])
        buckets["sat_sun"] = pattern["sat_sun"]
        buckets["mon_to_fri"] = pattern["mon_to_fri"]
        for scope, b in buckets.items():
            add(habit, scope, counts=[b["done"], b["missed"], b["days"]], rates=[b["rate"]])

    global_counts = {float(summary.get(k)) for k in ("window_days", "total_entries",
                                                    "habits_tracked") if k in summary}
    return facts, global_counts


def _aliases(habits: list[str]) -> dict[str, re.Pattern]:
    """Full name, plus any distinctive word ('walk', 'screens') unique to one habit."""
    words: dict[str, set[str]] = {}
    for habit in habits:
        for word in re.findall(r"[a-z]+", habit.lower()):
            if len(word) >= 4 and word not in GENERIC_WORDS:
                words.setdefault(word, set()).add(habit)

    patterns = {}
    for habit in habits:
        options = [re.escape(habit.lower())]
        for word, owners in words.items():
            if owners == {habit}:
                stem = word[:-1] if word.endswith(("s", "e")) and len(word) > 4 else word
                options.append(r"\b" + re.escape(stem) + r"\w*")
        patterns[habit] = re.compile("|".join(options), re.IGNORECASE)
    return patterns


_WEEKEND = re.compile(r"\bweekends?\b|\bsat(?:urday)?s?\s*(?:and|&|\+)\s*sun(?:day)?s?\b")
_WEEKDAYS = re.compile(r"\bweekdays?\b|\bworkdays?\b"
                       r"|\bmon(?:day)?s?\s*(?:-|–|to|through)\s*fri(?:day)?s?\b")


def _scopes(clause: str) -> set[str]:
    """Which part of the week a clause is about.

    "weekends" / "Saturday and Sunday" means the Sat+Sun aggregate, not either
    day on its own; a single day name means that day (or its aggregate).
    """
    text = clause.lower()
    found = set()
    if _WEEKEND.search(text):
        found.add("sat_sun")
        text = _WEEKEND.sub(" ", text)
    if _WEEKDAYS.search(text):
        found.add("mon_to_fri")
        text = _WEEKDAYS.sub(" ", text)
    for day in WEEKDAYS:
        if re.search(r"\b" + day.lower() + r"s?\b", text):
            found.add(day)
            found.add("sat_sun" if day in WEEKDAYS[5:] else "mon_to_fri")
    return found


def _rates_in_scope(summary: dict, scope: set[str]) -> tuple[str, dict[str, float]]:
    """The per-habit rate a best/worst claim in this scope is about."""
    days = [d for d in WEEKDAYS if d in scope]
    pattern = summary.get("weekday_pattern", {})
    if len(days) == 1:
        key = days[0]
        rates = {h: p["by_day"][key]["rate"] for h, p in pattern.items()}
    elif "sat_sun" in scope:
        key = "sat_sun"
        rates = {h: p["sat_sun"]["rate"] for h, p in pattern.items()}
    elif "mon_to_fri" in scope:
        key = "mon_to_fri"
        rates = {h: p["mon_to_fri"]["rate"] for h, p in pattern.items()}
    else:
        key = "overall"
        rates = {h: c["rate"] for h, c in summary.get("completion", {}).items()}
    return key, {h: r for h, r in rates.items() if r is not None}


# --- check ----------------------------------------------------------------


def _matches(value: float, kind: str, slot: dict) -> bool:
    """kind: 'percent' (66.7%), 'fraction' (0.667 — a rate) or 'number'."""
    if kind == "percent":
        return any(abs(value - r) <= 0.51 for r in slot["rates"])
    if kind == "fraction":
        return any(abs(value * 100 - r) <= 0.51 for r in slot["rates"])
    return value in slot["counts"] or any(abs(value - r) <= 0.51 for r in slot["rates"])


_CURRENT_STREAK = re.compile(
    r"\bcurrent(?:ly)?\b.*?\b(?:streak|run)\b|\b(?:streak|run)\b.*?\bcurrent", re.IGNORECASE)
_LONGEST_STREAK = re.compile(r"\b(?:longest|best)\b.*?\b(?:streak|run)\b", re.IGNORECASE)


def check(text: str, summary: dict) -> Report:
    """Return every number, and every best/worst claim about one habit, that
    the summary does not support in context."""
    facts, global_counts = _facts(summary)
    habits = list(facts)
    aliases = _aliases(habits)
    report = Report()

    body = _LIST_MARKER.sub("", _TIME_OR_DURATION.sub(" ", _ISO_DATE.sub(" ", text)))
    last_habits: list[str] = []

    for sentence in (s.strip() for s in _SENTENCE.split(body)):
        if not sentence:
            continue
        scope: set[str] = set()
        sentence_habits: list[str] = []
        for clause in _CLAUSE.split(sentence):
            if not clause or not clause.strip():
                continue
            named = [h for h in habits if aliases[h].search(clause)]
            if named:
                last_habits = sentence_habits = named
            if _RESET.search(clause):
                scope = set()
            scope = _scopes(clause) or scope

            # "X is the worst at weekends": is X really the lowest rate there?
            superlative = _WORST.search(clause) or _BEST.search(clause)
            subject = named or sentence_habits
            if superlative and len(subject) == 1 and len(habits) > 1:
                report.checked += 1
                key, rates = _rates_in_scope(summary, scope)
                if rates and subject[0] in rates:
                    target = (min if _WORST.search(clause) else max)(rates.values())
                    if rates[subject[0]] != target:
                        report.issues.append(Issue(
                            value=superlative.group(0),
                            sentence=sentence,
                            habits=subject,
                            scope=[key],
                        ))

            # Digits inside habit names ("Read 20 pages", "10pm") are not claims.
            stripped = clause
            for habit in habits:
                stripped = re.sub(re.escape(habit), " ", stripped, flags=re.IGNORECASE)

            previous_end = 0
            for match in _NUMBER.finditer(stripped):
                value = float(match.group(1))
                if match.group(2):
                    kind = "percent"
                elif "." in match.group(1) and value <= 1:
                    kind = "fraction"
                else:
                    kind = "number"
                report.checked += 1
                # The words just before a number say which streak it is.
                lead = stripped[previous_end:match.start()]
                previous_end = match.end()
                if kind == "number" and value in global_counts:
                    continue
                pool = last_habits or habits
                scopes = scope or {"overall"}
                if not scope and kind == "number":
                    if _CURRENT_STREAK.search(lead):
                        scopes = {"streak_current"}
                    elif _LONGEST_STREAK.search(lead):
                        scopes = {"streak_longest"}
                if any(
                    _matches(value, kind, facts[h][s])
                    for h in pool for s in scopes if s in facts[h]
                ):
                    continue
                report.issues.append(Issue(
                    value=match.group(0).strip(),
                    sentence=sentence,
                    habits=list(last_habits),
                    scope=sorted(scopes),
                ))
    return report
