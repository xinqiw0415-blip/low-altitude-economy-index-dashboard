from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    documents = pd.read_csv(ROOT / "data" / "interim" / "policy_documents.csv")
    documents = documents[documents["fetch_status"] == "ok"]
    prompt_template = (ROOT / "prompts" / "event_extraction_prompt.md").read_text(encoding="utf-8")
    output = ROOT / "data" / "interim" / "llm_event_tasks.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in documents.itertuples(index=False):
            metadata = {
                "document_id": row.document_id,
                "title": row.title,
                "publisher": row.publisher,
                "publish_date": row.publish_date,
                "region": row.region,
                "level": row.level,
                "url": row.url,
            }
            prompt = prompt_template.replace("{{metadata}}", json.dumps(metadata, ensure_ascii=False, indent=2))
            prompt = prompt.replace("{{document_text}}", str(row.full_text))
            handle.write(json.dumps({"task_id": row.document_id, "prompt": prompt}, ensure_ascii=False) + "\n")
    print(f"导出 {len(documents)} 个LLM任务：{output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
