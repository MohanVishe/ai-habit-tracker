"""HabitLoop — Streamlit UI.

    streamlit run app.py
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from habitloop import analytics, db
from habitloop.llm import describe_provider

load_dotenv()
db.init()

st.set_page_config(page_title="HabitLoop", page_icon="🎯", layout="wide")

WINDOW = 30


@st.cache_data(ttl=30)
def load_summary(window: int = WINDOW) -> tuple[dict, list[dict]]:
    entries = db.recent(window)
    return analytics.build_summary(entries, window), entries


def invalidate():
    load_summary.clear()


# --- sidebar --------------------------------------------------------------

with st.sidebar:
    st.title("🎯 HabitLoop")
    st.caption(f"Model: {describe_provider()}")

    st.subheader("Add a habit")
    with st.form("add_habit", clear_on_submit=True):
        name = st.text_input("Name", placeholder="Morning walk")
        category = st.selectbox("Category", ["health", "learning", "work", "mind", "general"])
        target = st.slider("Target per week", 1, 7, 5)
        if st.form_submit_button("Add") and name.strip():
            db.add_habit(name, category, target)
            invalidate()
            st.success(f"Added {name}")

    habits = db.list_habits()
    if habits:
        st.subheader("Archive")
        to_archive = st.selectbox("Habit", [h["name"] for h in habits], key="arch")
        if st.button("Archive", use_container_width=True):
            db.archive_habit(next(h["id"] for h in habits if h["name"] == to_archive))
            invalidate()
            st.rerun()

habits = db.list_habits()

if not habits:
    st.info("No habits yet. Add one in the sidebar — or run `python seed.py` for sample data.")
    st.stop()

summary, entries = load_summary()

# --- tabs -----------------------------------------------------------------

log_tab, progress_tab, review_tab, coach_tab = st.tabs(
    ["📝 Log", "📊 Progress", "🧠 Weekly review", "💬 Coach"]
)

with log_tab:
    st.subheader("Log a day")
    on_date = st.date_input("Date", value=date.today(), max_value=date.today())

    logged = {
        (e["name"], e["on_date"]): e["done"]
        for e in entries
    }

    for habit in habits:
        left, right = st.columns([3, 1])
        already = logged.get((habit["name"], on_date.isoformat()))

        with left:
            state = "✅" if already == 1 else "❌" if already == 0 else "·"
            st.write(f"{state} **{habit['name']}** · {habit['category']} · {habit['target_per_week']}×/week")

        with right:
            done_col, miss_col = st.columns(2)
            if done_col.button("Done", key=f"d{habit['id']}", use_container_width=True):
                db.log(habit["id"], on_date, True)
                invalidate()
                st.rerun()
            if miss_col.button("Miss", key=f"m{habit['id']}", use_container_width=True):
                db.log(habit["id"], on_date, False)
                invalidate()
                st.rerun()

with progress_tab:
    st.subheader(f"Last {WINDOW} days")

    cols = st.columns(min(len(summary["completion"]), 4) or 1)
    for index, (name, stats) in enumerate(summary["completion"].items()):
        with cols[index % len(cols)]:
            st.metric(
                name,
                f"{stats['rate'] * 100:.0f}%",
                f"{summary['streaks'][name]['current']} day streak",
            )

    if entries:
        frame = pd.DataFrame(entries)
        frame["on_date"] = pd.to_datetime(frame["on_date"])

        st.markdown("**Completions per day**")
        daily = frame[frame["done"] == 1].groupby("on_date").size()
        st.bar_chart(daily)

        st.markdown("**By weekday**")
        weekday = pd.DataFrame(summary["weekday_pattern"]).T
        st.bar_chart(weekday)

        with st.expander("Streaks"):
            st.dataframe(pd.DataFrame(summary["streaks"]).T, use_container_width=True)

with review_tab:
    st.subheader("Weekly review")
    st.caption(
        "Generated from the statistics on the Progress tab — the model interprets "
        "those numbers, it does not compute them."
    )

    left, right = st.columns(2)

    if left.button("Generate review", use_container_width=True):
        from habitloop.insights import weekly_analysis
        with st.spinner("Reading your log…"):
            try:
                st.session_state["review"] = weekly_analysis(summary)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")

    if right.button("Generate 7-day plan", use_container_width=True):
        from habitloop.insights import action_plan
        with st.spinner("Planning…"):
            try:
                st.session_state["plan"] = action_plan(summary, date.today() + timedelta(days=1))
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")

    if st.session_state.get("review"):
        st.markdown("### Review")
        st.markdown(st.session_state["review"])

    if st.session_state.get("plan"):
        st.markdown("### Next 7 days")
        st.markdown(st.session_state["plan"])

    with st.expander("What the model actually receives"):
        st.json(summary)

with coach_tab:
    st.subheader("Coach")
    st.caption("Answers come from your logged history. Questions it can't answer from the log, it declines.")

    st.session_state.setdefault("chat", [])

    for role, text in st.session_state["chat"]:
        with st.chat_message(role):
            st.markdown(text)

    if question := st.chat_input("Which habit am I worst at on weekends?"):
        st.session_state["chat"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"), st.spinner("…"):
            from habitloop.coach import answer
            try:
                reply = answer(question, summary, st.session_state["chat"][:-1])
            except Exception as exc:
                reply = f"Couldn't reach the model — {type(exc).__name__}: {exc}"
            st.markdown(reply)

        st.session_state["chat"].append(("assistant", reply))
