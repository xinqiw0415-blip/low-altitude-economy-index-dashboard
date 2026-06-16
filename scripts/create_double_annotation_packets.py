from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVENT_TYPES = "order_contract / strategic_cooperation / project_investment / product_technology / government_support / performance / market_risk / legal_safety_risk / other_business"


def clean(value: object) -> object:
    if isinstance(value, str):
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    return value


def main() -> None:
    events = pd.read_csv(ROOT / "data" / "processed" / "deepseek_company_events.csv")
    documents = pd.read_csv(ROOT / "data" / "interim" / "company_announcement_documents.csv")[["art_code", "full_text"]]
    data = events.merge(documents, on="art_code", how="left")
    positive = data[data["event_type"] != "other_business"].sort_values(["event_type", "event_date"])
    negative = data[data["event_type"] == "other_business"].sort_values("event_date")
    sample = pd.concat(
        [
            positive.groupby("event_type", group_keys=False).head(5),
            negative.iloc[::max(1, len(negative) // 12)].head(12),
        ]
    ).drop_duplicates("event_id").head(40).sample(frac=1, random_state=20260615).reset_index(drop=True)
    sample.insert(0, "sample_no", range(1, len(sample) + 1))
    task = sample[["sample_no", "event_id", "ts_code", "name", "event_date", "title", "full_text"]].copy()
    task["human_relevant"] = ""
    task["human_event_type"] = ""
    task["human_direction"] = ""
    task["human_intensity"] = ""
    task["human_uncertainty"] = ""
    task["human_evidence_span"] = ""
    task["human_note"] = ""
    for column in task.select_dtypes(include="object"):
        task[column] = task[column].map(clean)
    instructions = pd.DataFrame(
        {
            "item": ["相关性", "事件类型", "方向", "强度", "不确定性", "证据"],
            "rule": [
                "human_relevant填写1或0，只判断公告事件是否与低空经济产业链实质相关",
                EVENT_TYPES,
                "positive / negative / mixed / neutral",
                "1-5；1为例行或很弱，5为重大且已落实",
                "1-5；1为已完成且明确，5为计划性强或条件不确定",
                "从正文复制支持判断的最短原文，不要改写",
            ],
        }
    )
    for annotator in ["A", "B"]:
        output = ROOT / "data" / "interim" / f"human_annotation_{annotator}.xlsx"
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            task.to_excel(writer, sheet_name="tasks", index=False)
            instructions.to_excel(writer, sheet_name="instructions", index=False)
    key = sample[["sample_no", "event_id", "event_type", "direction", "intensity", "uncertainty", "relevance", "evidence_span"]]
    key.to_csv(ROOT / "data" / "interim" / "llm_annotation_comparison_key.csv", index=False, encoding="utf-8-sig")
    print(f"双人盲标任务：{len(task)} 条，已生成A/B两份工作簿")


if __name__ == "__main__":
    main()
