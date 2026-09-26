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

Best/worst claims are checked against the computed ranking, and every name
in the claim must be at the top (or bottom), ties allowed. On the evaluation
set most wrong comparisons named the right habit first and then added one,
so a claim about two names is checked name by name:

- Habits: "Morning walk and Read 20 pages are your best habits" is checked
  against the rates in the same context (overall, weekends, weekdays or one
  day), and Read 20 pages is flagged if it isn't tied for the top. "The
  longest current streak" is checked against the current streaks.
- Days: "Morning walk is lowest on Saturdays and Sundays" is a claim about
  that habit's days, checked against its by-day rates.
- The names claimed are those in the same clause, with lists that run over
  commas joined up ("Tuesday, Wednesday and Thursday"). A clause after
  "while", "but" or "compared with" is a separate claim. "Which is the
  lowest" points at the clause before; "both", "these" or "them" point at every
  habit in the sentence (or the one before, or after, if it names none).
  "Second lowest" and "not the best" are not top claims.
- A number given for "both" or "each" of two habits must hold for every one,
  not just one of them.

This catches the case where every number quoted is right but the ranking drawn
from them is wrong.

Not checked (by design, and stated in the README): numbers written as words,
other claims without numbers ("you met your target"), comparative claims
without a superlative ("X is lower than Y"), invented habit names, times of day
and durations, and anything in the 7-day plan, whose day numbers and dates are
not statistics. Dates are skipped, both ISO and written out ("August 27, 2026").
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
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
          r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
# "August 27, 2026", "Aug 27", "27 August 2026": a date, not a count.
_WRITTEN_DATE = re.compile(
    rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}}\b)?"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\b(?:,?\s+\d{{4}}\b)?",
    re.IGNORECASE)
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
# "X has the longest current streak (of 24 days)" ranks habits; "a longest
# streak of 5", "the longest streak for X is 5" and "the longest run of
# consecutive days" describe one habit's figure.
_TOP_STREAK = re.compile(
    r"\bthe\s+longest\s+(current\s+)?(?:streak|run)\b(?!\s+(?:for\b|of\s+(?!\d)))",
    re.IGNORECASE)
_SUPERLATIVE = re.compile(rf"{_TOP_STREAK.pattern}|{_WORST.pattern}|{_BEST.pattern}",
                          re.IGNORECASE)
# "the second lowest", "not the best": not a claim about the top.
_NOT_TOP = re.compile(r"\b(?:second|third|next|2nd|3rd|not)\b[\s-]+(?:the\s+|your\s+)?$",
                      re.IGNORECASE)
_PLURAL_REF = re.compile(r"\b(?:both|they|them|these|those|each)\b", re.IGNORECASE)
_HABIT_WORD = re.compile(r"\bhabits?\b", re.IGNORECASE)
_JOINT = re.compile(r"\b(?:both|each)\b", re.IGNORECASE)
_DAY = {d: re.compile(rf"\b{d.lower()}s?\b", re.IGNORECASE) for d in WEEKDAYS}
_FILLER = re.compile(r"\b(?:and|or|the|on|both|also|as well)\b|[&\"'*`.]", re.IGNORECASE)
# Streaks exist only over the whole window, so a streak clause is never day-scoped.
_RESET = re.compile(r"\b(overall|total|in all|across the (?:window|month)|streaks?)\b",
                    re.IGNORECASE)


@dataclass
class Issue:
    value: str
    sentence: str
    habits: list[str]
    scope: list[str]
    claim: str | None = None  # set for ranking claims: what the summary says instead

    def __str__(self) -> str:
        if self.claim:
            return f"{self.value} — {self.claim}: \"{self.sentence}\""
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


_SCOPE_WORDS = {"overall": "overall", "sat_sun": "at weekends", "mon_to_fri": "on weekdays"}


def _segments(sentence: str, habits: list[str], aliases: dict) -> list[dict]:
    """Clauses of a sentence with the habits and days each names. A clause that
    only continues a list ("Wednesday", "and Thursday") joins the one before."""
    segments: list[dict] = []
    for clause in _CLAUSE.split(sentence):
        if not clause or not clause.strip():
            continue
        named = [h for h in habits if aliases[h].search(clause)]
        days = [d for d in WEEKDAYS if _DAY[d].search(clause)]
        rest = clause
        for pattern in [aliases[h] for h in named] + [_DAY[d] for d in days]:
            rest = pattern.sub(" ", rest)
        if segments and (named or days) and not re.search(r"\w", _FILLER.sub(" ", rest)):
            segments[-1]["text"] += ", " + clause
            segments[-1]["habits"] += [h for h in named if h not in segments[-1]["habits"]]
            segments[-1]["days"] += [d for d in days if d not in segments[-1]["days"]]
        else:
            segments.append({"text": clause, "habits": named, "days": days})
    return segments


def _ranking_issues(sentences: list[str], summary: dict, habits: list[str],
                    aliases: dict, report: Report) -> None:
    """Check every habit or day named in a best/worst claim against the ranking."""
    pattern = summary.get("weekday_pattern", {})
    streaks = summary.get("streaks", {})
    seen: set[tuple] = set()
    names_by_sentence = [[h for h in habits if aliases[h].search(s)] for s in sentences]
    last_habits: list[str] = []

    def flag(word, name, sentence, scope, claim):
        if (word.lower(), name, sentence) not in seen:
            seen.add((word.lower(), name, sentence))
            report.issues.append(Issue(value=word, sentence=sentence, habits=[name],
                                       scope=[scope], claim=claim))

    for index, sentence in enumerate(sentences):
        scope: set[str] = set()
        sentence_habits: list[str] = []
        segments = _segments(sentence, habits, aliases)
        for position, seg in enumerate(segments):
            text = seg["text"]
            if _RESET.search(text):
                scope = set()
            scope = _scopes(text) or scope
            sentence_habits += [h for h in seg["habits"] if h not in sentence_habits]
            if seg["habits"]:
                last_habits = seg["habits"]
            match = _SUPERLATIVE.search(text)
            if not match or _NOT_TOP.search(text[:match.start()]):
                continue
            word = match.group(0)
            lowest = bool(_WORST.fullmatch(word))

            # Which names does the claim cover?
            named, days = seg["habits"], seg["days"]
            if not named and not days:
                if _PLURAL_REF.search(text):
                    named = (sentence_habits
                             or (names_by_sentence[index - 1] if index else [])
                             or (names_by_sentence[index + 1]
                                 if index + 1 < len(sentences) else []))
                else:  # "which is the lowest": the clause before
                    prior = next((s for s in reversed(segments[:position])
                                  if s["habits"] or s["days"]), None)
                    if prior:
                        named, days = prior["habits"], prior["days"]
                    elif len(last_habits) == 1:
                        named = last_habits

            if _TOP_STREAK.fullmatch(word):
                key = "current" if match.group(1) else "longest_in_window"
                values = {h: s[key] for h, s in streaks.items()}
                if len(values) < 2 or not named:
                    continue
                report.checked += 1
                top = max(values.values())
                leaders = [h for h, v in values.items() if v == top]
                label = "current streak" if key == "current" else "longest streak"
                for h in named:
                    if h in values and values[h] != top:
                        flag(word, h, sentence, key,
                             f"{h}'s {label} is {values[h]}; the longest is "
                             f"{' and '.join(leaders)} ({top})")
                continue

            if days and len(named) <= 1 and not _HABIT_WORD.search(text):
                # A claim about one habit's days.
                subject = named or (sentence_habits if len(sentence_habits) == 1 else
                                    last_habits if len(last_habits) == 1 else [])
                if not subject or subject[0] not in pattern:
                    continue
                by_day = {d: b["rate"] for d, b in pattern[subject[0]]["by_day"].items()
                          if b["rate"] is not None}
                lowered = sentence.lower()
                if re.search(r"\bweekdays?\b", lowered) and all(d in WEEKDAYS[:5] for d in days):
                    by_day = {d: r for d, r in by_day.items() if d in WEEKDAYS[:5]}
                elif re.search(r"\bweekends?\b", lowered) and all(d in WEEKDAYS[5:] for d in days):
                    by_day = {d: r for d, r in by_day.items() if d in WEEKDAYS[5:]}
                if not by_day:
                    continue
                report.checked += 1
                target = (min if lowest else max)(by_day.values())
                extremes = [d for d, r in by_day.items() if r == target]
                for d in days:
                    if d in by_day and by_day[d] != target:
                        flag(word, d, sentence, d,
                             f"{subject[0]} is at {by_day[d]} on {d}s; its {word.lower()} "
                             f"day is {' and '.join(extremes)} ({target})")
                continue

            # A claim about habits, in the part of the week the sentence is about.
            if not named or len(habits) < 2:
                continue
            report.checked += 1
            key, rates = _rates_in_scope(summary, scope)
            if not rates:
                continue
            target = (min if lowest else max)(rates.values())
            extremes = sorted(h for h, r in rates.items() if r == target)
            where = _SCOPE_WORDS.get(key, f"on {key}s")
            for h in named:
                if h in rates and rates[h] != target:
                    flag(word, h, sentence, key,
                         f"{h} is at {rates[h]} {where}; the {word.lower()} is "
                         f"{' and '.join(extremes)} ({target})")


def check(text: str, summary: dict) -> Report:
    """Return every number, and every best/worst claim, that the summary does
    not support in context."""
    facts, global_counts = _facts(summary)
    habits = list(facts)
    aliases = _aliases(habits)
    report = Report()

    body = _WRITTEN_DATE.sub(" ", _ISO_DATE.sub(" ", text))
    body = _LIST_MARKER.sub("", _TIME_OR_DURATION.sub(" ", body))
    sentences = [s.strip() for s in _SENTENCE.split(body) if s.strip()]
    _ranking_issues(sentences, summary, habits, aliases, report)

    last_habits: list[str] = []
    joint = False  # last_habits were named together with "both" or "each"
    for sentence in sentences:
        scope: set[str] = set()
        paired = "respectively" in sentence.lower()
        for clause in _CLAUSE.split(sentence):
            if not clause or not clause.strip():
                continue
            named = [h for h in habits if aliases[h].search(clause)]
            if named:
                last_habits = named
                joint = bool(_JOINT.search(clause))
            if _RESET.search(clause):
                scope = set()
            scope = _scopes(clause) or scope

            # Digits inside habit names ("Read 20 pages", "10pm") are not claims.
            stripped = clause
            for habit in habits:
                stripped = re.sub(re.escape(habit), " ", stripped, flags=re.IGNORECASE)

            # "Both A and B are at 0.81": the number must hold for each of them.
            every = (len(last_habits) > 1 and not paired
                     and (joint or bool(_JOINT.search(clause))))

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

                def holds(h, value=value, kind=kind, scopes=scopes):
                    return any(_matches(value, kind, facts[h][s])
                               for s in scopes if s in facts[h])

                if (all if every else any)(holds(h) for h in pool):
                    continue
                report.issues.append(Issue(
                    value=match.group(0).strip(),
                    sentence=sentence,
                    habits=[h for h in pool if not holds(h)] if every else list(last_habits),
                    scope=sorted(scopes),
                ))
    return report
