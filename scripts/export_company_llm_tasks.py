from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    candidates = pd.read_csv(ROOT / "data" / "interim" / "company_event_candidates.csv")
    documents = pd.read_csv(ROOT / "data" / "interim" / "company_announcement_documents.csv")
    candidates = candidates[candidates["eligible_for_labeling"]].merge(
        documents[["art_code", "full_text"]], on="art_code", how="left"
    )
    template = (ROOT / "prompts" / "company_event_extraction_prompt.md").read_text(encoding="utf-8")
    output = ROOT / "data" / "interim" / "company_llm_event_tasks.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in candidates.itertuples(index=False):
            metadata = {
                "event_id": row.event_id,
                "art_code": row.art_code,
                "ts_code": row.ts_code,
                "name": row.name,
                "event_date": row.event_date,
                "title": row.title,
            }
            prompt = template.replace("{{metadata}}", json.dumps(metadata, ensure_ascii=False, indent=2))
            prompt = prompt.replace("{{document_text}}", str(row.full_text))
            handle.write(json.dumps({"task_id": row.event_id, "prompt": prompt}, ensure_ascii=False) + "\n")
    print(f"导出公司公告LLM任务 {len(candidates)} 条")


if __name__ == "__main__":
    main()
