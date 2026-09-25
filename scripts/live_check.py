"""Run the coach and the weekly review against the fixed sample log, and check
every answer with the validator.

    python scripts/live_check.py                        # uses LLM_PROVIDER etc.
    LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b-instruct python scripts/live_check.py

A spot check, not an evaluation: a handful of questions on one log, one run.
The output shows the answer, then how many numbers and best/worst claims the
validator checked and which it flagged.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

import seed  # noqa: E402
from habitloop import analytics, db, validate  # noqa: E402
from habitloop.coach import answer  # noqa: E402
from habitloop.insights import weekly_analysis  # noqa: E402
from habitloop.llm import describe_provider  # noqa: E402

AS_OF = date(2026, 9, 15)
QUESTIONS = [
    "Which habit am I worst at on weekends?",        # the UI's own placeholder
    "How is my meditation going?",                   # abandoned before the window
    "What is my completion rate for Morning walk?",  # answerable exactly
    "How did I feel on the days I skipped my walk?", # not in the log
]


def show(label: str, text: str, summary: dict) -> None:
    report = validate.check(text, summary)
    print(f"{label}\n{text.strip()}\n")
    print(f"[validator] checked {report.checked}, flagged {len(report.issues)}")
    for issue in report.issues:
        print(f"  FLAG {issue}")
    print("\n" + "-" * 72 + "\n")


def main() -> None:
    load_dotenv()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "habits.db"
        seed.seed(AS_OF, path)
        summary = analytics.build_summary(db.recent(seed.WINDOW, AS_OF, path), seed.WINDOW, AS_OF)

    print(f"Model: {describe_provider()} · sample log: seed.py --as-of {AS_OF}\n")
    for question in QUESTIONS:
        show(f"Q: {question}", answer(question, summary), summary)
    show("Weekly review:", weekly_analysis(summary), summary)


if __name__ == "__main__":
    main()
