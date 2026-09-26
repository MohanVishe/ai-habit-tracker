# 🎯 HabitLoop

**A habit tracker whose AI coach is only given statistics computed from your own log — and whose answers are checked against them.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![Llama 3](https://img.shields.io/badge/Llama_3-0467DF?style=flat-square&logo=meta&logoColor=white)](https://llama.meta.com)
[![tests](https://github.com/MohanVishe/ai-habit-tracker/actions/workflows/tests.yml/badge.svg)](https://github.com/MohanVishe/ai-habit-tracker/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

Log habits → get a weekly review, a 7-day plan, and a coach you can ask questions — all grounded in your own history, running on open-weight models.

> Rebuilt and expanded from a project I originally put together for a community AI hackathon. Same idea, properly engineered: tested statistics, three swappable model backends, a documented grounding design and a claim checker on the output.

---

## The problem with AI habit coaches

Point an LLM at "be my habit coach" and it produces encouragement. It will tell you you're doing great, that consistency is key, that you should try habit stacking. None of it is about *you*, because none of it came from your data — and it reads the same whether you logged thirty days or three.

The failure is worse than useless, it's actively misleading: ask "how am I doing on meditation?" and a model with no grounding will confidently describe a streak you never had.

HabitLoop is built the other way round. **The model never sees your raw log, it never does arithmetic, and what it says back is checked against the numbers it was given.**

---

## The design

```mermaid
flowchart LR
    A["Daily logs<br/>SQLite"] --> B["analytics.py<br/>pure Python"]
    B --> C["Statistics summary<br/>rates · streaks · per-habit weekdays · rankings"]
    C --> D["Grounding prompt"]
    D --> E["LLM<br/>Groq · Ollama · OpenAI"]
    E --> F["Weekly review"]
    E --> G["7-day plan"]
    E --> H["Chat coach"]
    F --> V["validate.py<br/>claim checker"]
    H --> V
    C --> V

    style B fill:#3fb950,color:#fff
    style C fill:#6E56CF,color:#fff
    style V fill:#d29922,color:#fff
```

### Three decisions do the work

**1. Statistics are computed in Python, never by the model.**

Streaks, completion rates, per-habit weekday breakdowns, best/worst rankings and the best and worst weekday per habit come out of [`habitloop/analytics.py`](habitloop/analytics.py) — pure functions over plain dicts, tested in [`tests/test_analytics.py`](tests/test_analytics.py).

Asking a language model to count consecutive days across 300 rows is asking it to do the single thing it's least reliable at. And it won't error — it'll return a plausible wrong number with total confidence, which is far harder to catch than a crash. So the model gets handed finished numbers — including the *order* of habits for "which is worst at weekends?" — and its job is interpretation.

What the numbers mean:

- **Completion** is done days ÷ the weekly target scaled to the days the habit has existed inside the 30-day window. A habit added three days ago and done all three days is 100%, not 10%.
- **Current streak** counts back from today. Today not yet logged doesn't break it (the day isn't over); a miss logged today does — the streak is 0.
- **Weekday pattern** is per habit: for each weekday, done / missed / how many of that weekday fell in the period, plus Sat+Sun and Mon–Fri totals.
- **Archived habits** are left out of Progress and out of the summary.

**2. Grounding is structural, not retrieval-based.**

There's no vector store here, and that's deliberate rather than a shortcut. A habit log is small — a few hundred rows. The *computed summary* fits in the context window many times over, so the coach receives the complete picture every turn and there is no retrieval step that can return the wrong thing.

RAG solves "the corpus doesn't fit." That problem doesn't exist at this scale, and adding retrieval anyway would mean an embedding model, a vector store and a new class of silent failures bought in exchange for nothing.

**3. The output is checked, claim by claim.**

[`habitloop/validate.py`](habitloop/validate.py) runs over every coach answer and every weekly review. The UI shows "Checked N number(s) against the summary: all supported" or a warning listing what isn't. What it enforces is set out exactly below.

### The prompt rule

Every one of the three prompts opens with the same block ([`habitloop/prompts.py`](habitloop/prompts.py)), which also explains what each field in the summary means:

```
- Every claim you make must be traceable to a number in the summary below.
- Never invent a habit name, a date, a streak length or a percentage.
- If the summary does not contain enough data to answer, say so plainly.
  "You have only three days logged, so there isn't a pattern yet" is a correct
  and useful answer. A confident narrative built on three data points is not.
- Attach each number to the habit and the part of the week it belongs to. A
  habit's total misses are not its weekend misses.
```

Temperature runs at 0.2–0.4. This app interprets numbers it was given; creativity here surfaces as fabricated detail about your own life, which is the one failure that matters.

---

## Grounding: what is enforced and what isn't

Checking only that "every number in the answer appears somewhere in the summary" catches very little: a 30-day summary contains almost every small integer, so a habit's 20 total misses reported as its *weekend* misses passes. The checker therefore works on claims — a number, the habit it's attached to, and the part of the week it's about.

**Enforced (flagged in the UI, covered by [`tests/test_validate.py`](tests/test_validate.py)):**

- **Every number is checked in context.** The habit is the one named in the clause, or the last one named (so "It dropped to 20%" still refers to it). A clause about weekends, weekdays or a named day is checked against that habit's weekday figures; otherwise against its overall figures (completion counts, rate, target, streaks). "Weekends" means the Sat+Sun totals, not either day alone.
- **Current vs longest streak.** A number right after "current streak" must be the current streak.
- **Rates.** 66.7%, 67% and 0.667 all match a rate of 0.667 (half a point of rounding); a whole number matches a count exactly.
- **Best/worst claims, name by name.** Every habit or day named in a superlative claim must be at the top (bottom) of the computed ranking in that context, ties allowed, and each extra name is flagged: "Morning walk and Read 20 pages are your best habits" flags Read 20 pages if it isn't tied; "lowest on Saturdays and Sundays" is checked against that habit's by-day rates; "the longest current streak" against the streaks. A clause after "while" or "but" is a separate claim; "which", "both", "these" and "them" are followed back to the names they refer to; "second lowest" is not a top claim.
- **Numbers for "both" habits** must hold for each of them.
- Window-level numbers (window length, entry count, habit count) are allowed anywhere. Dates, list numbering, digits inside habit names ("Read 20 pages"), times of day and durations are not treated as claims.

**Not enforced:**

- Numbers written as words ("once every eight days").
- Claims with no number and no superlative: "you met your target", "a perfect record on weekdays", "even lower than" comparisons between two figures or two named habits.
- Invented habit names, and anything in the 7-day plan (its day numbers and dates aren't statistics).
- Ambiguous wording that the rules can't pin down — "maintained a streak of 28 days" with no "current" or "longest" is accepted if 28 is either.

The checker flags; it doesn't rewrite or block the answer. It is rule-based, so it can also flag a correct sentence whose wording it misreads — the warning names the number, the habit and the context it checked, so you can see why.

### A live run

[`scripts/live_check.py`](scripts/live_check.py) seeds the fixed sample log, asks the coach four questions (including the UI's own placeholder, "Which habit am I worst at on weekends?") and requests a weekly review, then runs the checker on each answer. One run with **qwen2.5:7b-instruct via Ollama**, 2026-09-26, is committed as [`examples/live_check_qwen2.5-7b-instruct.txt`](examples/live_check_qwen2.5-7b-instruct.txt). In that run the coach named Deep work block as worst at weekends (0.0, correct per the summary's rankings), declined the meditation and feelings questions, and the checker flagged nothing. The review's "maintained a streak of 28 consecutive days" (28 is the longest, the current streak is 1) was not flagged: it is the ambiguous-wording case listed above.

This is one run of one model on one log, a spot check; the measured error rate is in the next section. Earlier runs while building this surfaced three kinds of error that are now unit tests: a wrong "worst at weekends" with every number correct, a longest streak reported as the current one, and a single day's rate quoted as the weekend rate.

---

## Measured accuracy

[`eval/`](eval/) asks the coach fixed questions about seeded logs and scores every answer against the statistics computed in Python. No model grades anything. There are two question sets:

- **Dev set** ([`questions.jsonl`](eval/questions.jsonl)): 51 questions x logs ending 2026-08-02, 09-15 and 09-25 = 153 answers. The comparison fixes (precomputed best/worst answers, a one-name rule in the coach prompt, the validator's ranking rule) were written from its failures, so its "after" figure is a **dev-set result** and flatters the fixes.
- **Held-out set** ([`questions-heldout.jsonl`](eval/questions-heldout.jsonl)): 43 new questions, 27 of them comparisons (best/worst worded differently, a habit on one weekday, second-best, two named habits, full rankings, the biggest weekend drop), x logs ending 2026-08-24, 09-09 and 09-19 = 129 answers. It was committed (`821e555`) before any of the fixes and run **once**, after them. This is the figure to quote.

**qwen2.5:7b-instruct via Ollama, 2026-09-26:**

| | Dev, before fixes (`fc886c6`) | Dev, after fixes (`1af6dc9`) | **Held-out, after fixes** |
|---|---|---|---|
| Correct, including correct refusals | 119 / 153 = 77.8% (Wilson 95% 70.6–83.6%) | 135 / 153 = 88.2% (82.2–92.4%) | **65 / 129 = 50.4% (41.9–58.9%)** |
| Resampling whole questions, error rate | 12.4–33.3% | 4.6–20.3% | 35.7–62.8% |

| Question type | Dev before | Dev after | Held-out |
|---|---|---|---|
| Single per-habit figures | 27 / 27 | 27 / 27 | 9 / 9 |
| Streaks (incl. which habit leads) | 27 / 30 | 30 / 30 | 13 / 15 |
| Weekday vs weekend | 26 / 30 | 30 / 30 | 13 / 15 |
| Which day of the week | 12 / 24 | 18 / 24 | 16 / 21 |
| Best / worst habit | 3 / 18 | 6 / 18 | 4 / 30 |
| Two named habits | — | — | 5 / 12 |
| Full ranking / biggest drop | — | — | 2 / 12 |
| Unanswerable, declined | 24 / 24 | 24 / 24 | 3 / 15 |

**What this says.** On the dev set the fixes moved accuracy from 77.8% to 88.2%. On new questions and new logs the coach answers 50.4%, and that is the honest number. Reading a figure out of the summary holds up (35 of 39 held-out figure, streak and weekday/weekend answers are right). Comparisons still fail: held-out best/worst 4/30 — including the dev set's own "worst overall" worded as "least consistently" — second-best, rankings (the second place is often wrong) and the biggest-drop question. 18 of the 52 wrong held-out answers still name the right habit or day first. The held-out set also shows a failure the dev set never did: 12 of the 15 unanswerable questions were answered (a habit never logged in March, pages read per day, a forecast of next month), against 24/24 declined on the dev set.

**What the validator catches** (run on each answer's explanation; Wilson 95% intervals):

| | Wrong answers flagged | Correct answers flagged |
|---|---|---|
| Dev before, old validator (`fc886c6`) | 17 / 33 = 51.5% (35.2–67.5%) | 9 / 95 = 9.5% (5.1–17.0%) |
| Dev before, same answers, new validator | 26 / 33 = 78.8% (62.3–89.3%) | 4 / 95 = 4.2% (1.7–10.3%) |
| Dev after | 13 / 18 = 72.2% (49.1–87.5%) | 16 / 111 = 14.4% (9.1–22.1%) |
| **Held-out** | **21 / 52 = 40.4% (28.2–53.9%)** | **12 / 62 = 19.4% (11.4–30.9%)** |

It also flagged 2 of the 12 answered-unanswerable held-out questions, and 3 of 24 (dev before), 4 of 24 (dev after) and 0 of 3 (held-out) correct refusals. The dev rows are the rule's own development data. Read by hand ([`flag-review.json`](eval/results/qwen2.5-7b-instruct.flag-review.json)), 25 of the 26 new flags on the dev-before wrong answers point at the actual error. Of the 4 on correct answers, 2 are arguable and 2 are real false side claims. The held-out flags have not been read by hand. What it cannot see: a wrong `ANSWER:` line with a correct explanation ("A has the lowest rate, while B has 0.52"), comparative claims without a superlative, and, on held-out, answers to unanswerable questions that quote real numbers.

**How it's scored.** Each question gets a format instruction appended: first line `ANSWER: <value>` (or `ANSWER: UNKNOWN`), then one sentence of explanation. The value is scored by a fixed rule per answer type: counts and streaks must match exactly; rates within 0.005 (0.667, 66.7% and 67% all match 0.667); every habit or day named must be in the true set (naming one of several tied is fine); "weekdays", "weekends" or "equal" for the choice questions; a ranking must name every habit once in an order consistent with the rates. The full rule is at the top of [`eval/scoring.py`](eval/scoring.py), with unit tests for each. The expected answers are read from `analytics.build_summary`, and tests cross-check them against the individual analytics functions and check that the held-out logs differ from the dev logs (`seed.py` repeats a log for an end date on the same weekday, so the held-out dates fall on other weekdays).

**Setup.** `qwen2.5:7b-instruct` (Q4_K_M, digest `845dbda0ea48`) on Ollama 0.34.2, through the app's own chat path (`habitloop.coach.answer` with `LLM_PROVIDER=ollama`) at the coach's temperature of 0.2, with `OLLAMA_SEED=42`, context 4,096 tokens. The largest prompt plus answer was 2,936 tokens before the fixes, and 3,424 (dev) and 3,459 (held-out) after them. A repeat of the dev-before run ([`.repeat`](eval/results/qwen2.5-7b-instruct.repeat.summary.json)) gave the same text for 141 of 153 answers. The other 12 changed wording only, with the same outcome and validator verdict on all 153.

**Limits.**

- The logs are synthetic: one seeded generator, four habits in the window. Real logs are messier.
- One model, one seed, one run of each set after the fixes. The held-out set was not run on the code before the fixes, so it shows where the coach stands, not how much the fixes moved it on new questions.
- The questions, the scoring rules and the validator were all written by the same author, so they may share blind spots.
- The format instruction makes answers scoreable, but it is not how the chat tab is used. Free-form answers may do better or worse.
- Each question is asked of three logs, so answers are not independent. That is why the question-level intervals are wider.

```bash
python -m eval.run              # ask the model (Ollama with the model pulled), then score
python -m eval.run --rescore    # re-score the committed responses offline; a test does the same
python -m eval.run --set heldout   # the held-out questions and logs
```

Raw answers with their scores, in [`eval/results/`](eval/results/): `qwen2.5-7b-instruct` (dev, before), `qwen2.5-7b-instruct.after` (dev, after), `heldout.qwen2.5-7b-instruct` (held-out). Each has a `.jsonl` and a `.summary.json`.

---

## What it does

| | |
|---|---|
| 📝 **Log** | Mark habits done or missed, any date. Re-logging a day overwrites rather than duplicating. |
| 📊 **Progress** | Completion rates against your weekly target, current and longest streaks, the share of each weekday done per habit. |
| 🧠 **Weekly review** | What's working, what's slipping, one pattern you probably haven't spotted, one thing to change. Under 250 words, no motivational filler. Checked by the validator. |
| 📅 **7-day plan** | A day-by-day plan sized to your actual completion rates — a habit at 20% doesn't get a daily commitment. Weak weekdays get the lightest version. |
| 💬 **Coach** | Ask about your history. The model is told to decline what the summary doesn't cover; each answer is checked. |

The Weekly review tab has a **"What the model actually receives"** expander showing the exact JSON. If the coach says something surprising, you can check whether the data supports it.

---

## Run it

Python 3.11–3.13 (tested in CI).

```bash
git clone https://github.com/MohanVishe/ai-habit-tracker.git
cd ai-habit-tracker

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # runtime: exact pins for every package, exported from uv.lock

cp .env.example .env              # add GROQ_API_KEY — free at console.groq.com
python seed.py                    # 60 days of sample data ending today (optional)
streamlit run app.py
```

Or with [uv](https://docs.astral.sh/uv/): `uv sync --locked`, then `uv run streamlit run app.py`.

Both requirements files are generated from `uv.lock`, and CI fails if either drifts from it:

```bash
uv export --frozen --no-hashes --no-emit-project --no-dev -o requirements.txt           # runtime (what the Docker image installs)
uv export --frozen --no-hashes --no-emit-project --all-groups -o requirements-dev.txt   # runtime + pytest
```

→ **http://localhost:8501**

### Choosing a backend

```bash
LLM_PROVIDER=groq     GROQ_MODEL=llama-3.3-70b-versatile    # default — open weights, free tier, fast
LLM_PROVIDER=ollama   OLLAMA_MODEL=llama3.1                 # fully local, nothing leaves the machine
LLM_PROVIDER=openai   OPENAI_MODEL=gpt-4o-mini              # for comparison
```

One env var, no code change; every model id is an env var too. `llama-3.3-70b-versatile` is listed as a production model on [Groq's models page](https://console.groq.com/docs/models) and is not on its [deprecations page](https://console.groq.com/docs/deprecations) (both checked 2026-09-26). Ollama is the one to use if you'd rather your habit log never touch a third-party API — which is a reasonable thing to want from this particular category of data. Any chat model Ollama serves works; the live run above used `qwen2.5:7b-instruct`.

### Tests

```bash
pip install -r requirements-dev.txt   # the runtime pins plus pytest (or: uv sync --locked)
pytest -q                             # 153 passing, no model or network needed
```

- `test_analytics.py` — the statistics: streak edge cases (an unlogged today doesn't break a streak, a miss logged today does, a gap does), start-date-aware completion (3/3 on a new habit is 100%; two completions in thirty days is not), per-habit weekday totals, rankings.
- `test_validate.py` — the claim checker: what passes, what is flagged, including the errors seen from a live model.
- `test_db_and_seed.py` — archived habits leave the summary, `created_on` reaches the denominator, the seed is deterministic, and the README's sample block below is exactly what `seed.py` prints.
- `test_eval_scoring.py` — the evaluation: expected answers cross-checked against the analytics functions, the answer parser, every scoring rule, the interval code, and that re-scoring the committed responses reproduces the committed summary.
- `test_app.py` — runs `app.py` headless with Streamlit's AppTest: all four tabs render, archiving removes a habit from Progress, and a missing API key gives a readable message rather than a traceback.

CI runs these on Python 3.11, 3.12 and 3.13 after `pip install -r requirements-dev.txt`, again from `uv sync --locked`, and builds the Docker image, checks pytest is not in it, and waits for its healthcheck.

### Docker

```bash
docker build -t habitloop .
docker run -p 8501:8501 --env-file .env habitloop
```

The image installs `requirements.txt` only, the runtime dependencies without the test tooling. The healthcheck uses Python's standard library (the slim base image has no curl).

---

## The sample data

`seed.py` generates 60 days that are deliberately uneven — a near-perfect habit, two that collapse at weekends, one abandoned after three weeks, one that barely happens. A seed where everything sits at 80% makes the analysis look good and demonstrates nothing.

The draws are seeded, so a fixed end date gives the same log on any machine and any day:

```bash
python seed.py --as-of 2026-09-15 --db data/example.db
```

prints

```
Seeded 5 habits, 60 days ending 2026-09-15 (Tuesday). Last 30 days:

Morning walk            96.7%  streak 1/28  weekends  8/8 done
Read 20 pages           81.0%  streak 1/5   weekends  1/8 done
Deep work block         76.2%  streak 2/3   weekends  0/8 done
No screens after 10pm   36.7%  streak 2/2   weekends  1/8 done

Run: streamlit run app.py
```

(streak is current/longest in the window; a test keeps this block in step with the code.) Plain `python seed.py` ends the log today, so its numbers shift with the weekday. Meditate doesn't appear: it was abandoned before the 30-day window — see Limitations.

The weekend collapse in Read 20 pages and Deep work block is the kind of thing the review is supposed to surface — and it's in the per-habit data, so it can. Part of it is by design: those two habits target 5×/week, and the seed logs their weekends as misses.

---

## Layout

```
├── app.py                      # Streamlit UI — four tabs
├── habitloop/
│   ├── db.py                   # SQLite: schema and CRUD ($HABITLOOP_DB overrides the path)
│   ├── analytics.py            # pure statistics — the tested core
│   ├── validate.py             # claim checker for model output
│   ├── llm.py                  # provider factory: Groq / Ollama / OpenAI
│   ├── prompts.py              # the grounding rule + three prompts
│   ├── insights.py             # weekly review, 7-day plan
│   └── coach.py                # grounded chat
├── tests/                      # analytics, validator, storage + seed, app smoke test
├── eval/                       # question set, scorer, runner and committed results
├── scripts/live_check.py       # coach + review against the sample log, checked
├── examples/                   # committed output of a live_check run
├── seed.py                     # sample data
├── pyproject.toml, uv.lock     # dependencies; requirements*.txt are exported from the lock
└── Dockerfile
```

---

## Limitations

- **The claim checker is rule-based.** It covers numbers and best/worst claims in context; the list of what it doesn't cover is above. It flags rather than blocks. On the held-out set it flagged 21 of 52 wrong answers and 12 of 62 correct ones.
- **The coach is weak at comparisons.** On the held-out set, best/worst habit answers were right 4 times in 30 and rankings 2 in 12, against 35 of 39 for single figures, streaks and weekday/weekend.
- **It answers some questions the log can't.** 12 of 15 held-out unanswerable questions got an answer.
- **The prompt is about 3,400 tokens** with the seeded log. Ollama's default 4,096-token context leaves little room for chat history.
- **Streaks are daily.** A 5×/week habit's streak resets at a skipped weekend even when the weekly target is met.
- **A habit with no entries in the window disappears** rather than being reported as dropped (Meditate in the sample data) — arguably the more useful signal.
- **30-day window is fixed** in the UI. The analytics functions take any window; the UI doesn't expose it yet.
- **Single user, local file.** No accounts, no sync. `data/` is gitignored.
- **The coach has no memory across sessions.** Chat history lives in Streamlit session state and resets on reload.
- **No reminders or notifications.** It tracks; it doesn't nag.
- **Not a clinical tool.** The prompts explicitly refuse medical and psychiatric territory, but this is a habit log and nothing more.

## Next

1. **Comparisons that generalise**: answer "which is best/worst/second/rank" questions from the computed rankings in code (route the question, let the model phrase the answer) rather than by prompt, since the prompt rule helped on the dev wording and not on new wording (held-out best/worst 4/30).
2. **Refusals on held-out wording**: 12 of 15 unanswerable held-out questions were answered. Add a check for claims about periods, units or habits the summary doesn't hold.
3. **A fresh held-out set** written after these results, the held-out set run on the code before the fixes, and the default Groq Llama 3.3 70B (not yet run); checker coverage for numbers written as words and a compacter summary format for small contexts.
4. **Weekly-target streaks** for habits below 7×/week, and reporting abandoned habits as dropped.
5. **Configurable window** and habit-level history charts.
6. **Correlation between habits** — does the walk happening predict the deep-work block happening? The data supports asking; the analytics don't compute it yet.

## Credits

- Default model: Meta **Llama 3.3 70B** (served by Groq), under the [Llama 3.3 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE).
- Live-run example and evaluation: **Qwen2.5-7B-Instruct** (via Ollama), Apache 2.0.

## License

MIT — see [LICENSE](LICENSE).
