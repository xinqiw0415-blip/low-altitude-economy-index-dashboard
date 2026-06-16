from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "data" / "interim" / "deepseek_company_results.jsonl",
    ROOT / "data" / "interim" / "deepseek_policy_results.jsonl",
]
PRICE_PER_MILLION_USD = {"cached_input": 0.0028, "uncached_input": 0.14, "output": 0.28}


def main() -> None:
    prompt = completion = cached = calls = 0
    models = set()
    for path in FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("status") != "ok":
                continue
            calls += 1
            models.add(record.get("model", ""))
            usage = record.get("usage") or {}
            prompt += int(usage.get("prompt_tokens", 0))
            completion += int(usage.get("completion_tokens", 0))
            cached += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    uncached = max(0, prompt - cached)
    cost = (
        cached / 1_000_000 * PRICE_PER_MILLION_USD["cached_input"]
        + uncached / 1_000_000 * PRICE_PER_MILLION_USD["uncached_input"]
        + completion / 1_000_000 * PRICE_PER_MILLION_USD["output"]
    )
    report = {
        "successful_calls": calls,
        "models": sorted(models),
        "prompt_tokens": prompt,
        "cached_input_tokens": cached,
        "uncached_input_tokens": uncached,
        "completion_tokens": completion,
        "estimated_cost_usd": round(cost, 6),
        "pricing_assumption_usd_per_million": PRICE_PER_MILLION_USD,
    }
    output = ROOT / "reports" / "deepseek_usage.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
