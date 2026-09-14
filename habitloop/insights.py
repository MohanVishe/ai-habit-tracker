"""Weekly analysis and the 7-day plan."""
from __future__ import annotations

import json
from datetime import date, timedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import get_llm
from .prompts import ACTION_PLAN, WEEKLY_ANALYSIS


def _as_json(summary: dict) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)


def weekly_analysis(summary: dict) -> str:
    chain = (
        ChatPromptTemplate.from_template(WEEKLY_ANALYSIS)
        | get_llm(temperature=0.3)
        | StrOutputParser()
    )
    return chain.invoke({"summary": _as_json(summary)})


def action_plan(summary: dict, start: date | None = None) -> str:
    start = start or date.today() + timedelta(days=1)
    chain = (
        ChatPromptTemplate.from_template(ACTION_PLAN)
        | get_llm(temperature=0.4)
        | StrOutputParser()
    )
    return chain.invoke({
        "summary": _as_json(summary),
        "start_date": start.strftime("%A %d %B %Y"),
    })
