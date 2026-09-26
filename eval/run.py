"""Ask the coach every question in eval/questions.jsonl about three seeded logs,
score the answers against the computed statistics, and check each with the
validator.

    python -m eval.run                          # qwen2.5:7b-instruct via local Ollama
    python -m eval.run --model llama3.1         # any model Ollama serves
    python -m eval.run --rescore                # re-score the committed responses, no model

The answers come from habitloop.coach.answer, the app's own chat path, with
LLM_PROVIDER=ollama; only the sampling seed is added (OLLAMA_SEED). Writes
eval/results/<model>.jsonl (one scored record per question and log, with the
raw response) and eval/results/<model>.summary.json. See eval/scoring.py for
the scoring rule.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from eval import scoring

ROOT = Path(__file__).resolve().parents[1]

RESULTS = Path(__file__).with_name("results")
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SEED = 42


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def _ollama_meta(host: str, model: str) -> dict:
    tags = _get(f"{host}/api/tags")["models"]
    match = next((m for m in tags if m["name"] == model), None)
    if match is None:
        raise SystemExit(f"{model} is not pulled on {host}: ollama pull {model}")
    return {
        "ollama_version": _get(f"{host}/api/version")["version"],
        "model_digest": match["digest"],
        "model_details": match.get("details", {}),
    }


def _token_counts(runs) -> dict:
    """prompt/eval token counts reported by Ollama for the one model call."""
    stack = list(runs)
    while stack:
        run = stack.pop()
        if run.run_type in ("llm", "chat_model"):
            for gen in (run.outputs or {}).get("generations", [[]])[0]:
                info = gen.get("generation_info") or {}
                if "prompt_eval_count" in info:
                    return {"prompt_tokens": info["prompt_eval_count"],
                            "output_tokens": info.get("eval_count")}
        stack.extend(run.child_runs or [])
    return {}


def ask(model: str, seed: int, host: str, limit: int | None) -> tuple[list[dict], dict]:
    os.environ.update({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": model,
                       "OLLAMA_SEED": str(seed), "OLLAMA_HOST": host})
    from langchain_core.tracers.context import collect_runs

    from habitloop import coach

    meta = {
        "model": model,
        "provider_path": "habitloop.coach.answer -> habitloop.llm.get_llm (LLM_PROVIDER=ollama)",
        "temperature": coach.TEMPERATURE,
        "seed": seed,
        **_ollama_meta(host, model),
    }
    questions = scoring.load_questions()[:limit]
    raw = []
    for as_of in scoring.AS_OF_DATES:
        summary = scoring.summary_for(as_of)
        for q in questions:
            start = time.perf_counter()
            with collect_runs() as cb:
                text = coach.answer(scoring.prompt_for(q), summary)
            raw.append({"id": q["id"], "as_of": as_of, "response": text,
                        "seconds": round(time.perf_counter() - start, 2),
                        **_token_counts(cb.traced_runs)})
            print(f"{as_of} {q['id']}: {text.splitlines()[0][:70] if text else ''}", flush=True)

    # The prompt must fit the context Ollama loaded the model with, or it is
    # silently truncated.
    loaded = next((m for m in _get(f"{host}/api/ps")["models"] if m["name"] == model), {})
    meta["ollama_context_length"] = loaded.get("context_length")
    longest = max(r.get("prompt_tokens", 0) + (r.get("output_tokens") or 0) for r in raw)
    meta["max_prompt_plus_output_tokens"] = longest
    if meta["ollama_context_length"] and longest >= meta["ollama_context_length"]:
        raise SystemExit(f"a prompt filled the {meta['ollama_context_length']}-token context")
    return raw, meta


def score_all(raw: list[dict]) -> list[dict]:
    questions = {q["id"]: q for q in scoring.load_questions()}
    summaries = {d: scoring.summary_for(d) for d in {r["as_of"] for r in raw}}
    records = []
    for r in raw:
        record = scoring.evaluate(questions[r["id"]], r["as_of"], r["response"],
                                  summaries[r["as_of"]])
        record.update({k: r[k] for k in ("seconds", "prompt_tokens", "output_tokens") if k in r})
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--limit", type=int, default=None, help="first N questions (smoke test)")
    parser.add_argument("--rescore", action="store_true",
                        help="re-score the committed responses without calling a model")
    args = parser.parse_args(argv)

    stem = re.sub(r"[^\w.-]+", "-", args.model)
    jsonl = RESULTS / f"{stem}.jsonl"
    summary_path = RESULTS / f"{stem}.summary.json"

    if args.rescore:
        raw = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
        meta = json.loads(summary_path.read_text(encoding="utf-8"))["run"]
    else:
        raw, meta = ask(args.model, args.seed, args.host, args.limit)
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=False).stdout.strip()
        meta.update({
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "code_commit": commit,
            "python": platform.python_version(),
            "questions_file": "eval/questions.jsonl",
            "format_instruction": scoring.FORMAT,
        })

    records = score_all(raw)
    RESULTS.mkdir(exist_ok=True)
    with open(jsonl, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    result = {"run": meta, **scoring.summarize(records)}
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8", newline="\n")

    acc, err, val = result["accuracy"], result["error_rate"], result["validator"]
    print(f"\n{meta['model']}: {acc['k']}/{acc['n']} correct, error rate {err['rate']:.1%} "
          f"(Wilson 95% {err['wilson95'][0]:.1%}-{err['wilson95'][1]:.1%})")
    print(f"outcomes: {result['outcomes']}")
    for key, value in val.items():
        print(f"validator {key}: {value['k']}/{value['n']}")
    print(f"wrote {jsonl.relative_to(ROOT)} and {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
