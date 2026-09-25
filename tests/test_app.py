"""Smoke test the Streamlit app end to end, without a model.

Runs app.py headless against a freshly seeded database and checks that every
tab renders and that the no-key path fails with a readable message instead of
a traceback.
"""
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import seed

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = tmp_path / "habits.db"
    seed.seed(date.today(), path)
    monkeypatch.setenv("HABITLOOP_DB", str(path))
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")  # present but empty: .env cannot fill it in
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_renders_all_tabs_without_errors(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["📝 Log", "📊 Progress", "🧠 Weekly review", "💬 Coach"]
    # One metric per habit with entries in the window; Meditate was abandoned
    # before the window, so four.
    assert sorted(m.label for m in app.metric) == [
        "Deep work block", "Morning walk", "No screens after 10pm", "Read 20 pages",
    ]


def test_archived_habit_disappears_from_progress(app):
    app.selectbox(key="arch").set_value("Morning walk")
    next(b for b in app.button if b.label == "Archive").click()
    app.run()
    assert not app.exception
    assert "Morning walk" not in [m.label for m in app.metric]


def test_coach_without_a_key_explains_itself(app):
    app.chat_input[0].set_value("Which habit am I worst at on weekends?").run()
    assert not app.exception
    replies = [m for m in app.chat_message if m.name == "assistant"]
    assert "GROQ_API_KEY is not set" in replies[-1].markdown[0].value


def test_review_without_a_key_explains_itself(app):
    next(b for b in app.button if b.label == "Generate review").click()
    app.run()
    assert not app.exception
    assert any("GROQ_API_KEY is not set" in e.value for e in app.error)
