from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TOOL_KEYWORDS = {
    "fiscal_subsidy": ["补贴", "奖励", "奖补", "贴息", "财政资金"],
    "airspace_reform": ["空域", "飞行审批", "飞行服务"],
    "infrastructure": ["起降点", "通用机场", "基础设施", "通信导航", "低空智联网"],
    "industry_support": ["产业园", "企业培育", "招商", "产业集群"],
    "technology_innovation": ["技术创新", "研发", "实验室", "创新平台"],
    "application_scenario": ["应用场景", "示范应用", "商业化运营"],
    "talent": ["人才", "职业教育", "培训"],
    "finance_insurance": ["基金", "融资", "保险", "信贷"],
    "safety_regulation": ["安全监管", "适航", "实名登记", "监管平台"]
}

TARGET_KEYWORDS = {
    "aircraft_manufacturing": ["航空器", "无人机", "eVTOL", "直升机"],
    "operation_service": ["通用航空", "物流配送", "载人飞行", "飞行服务"],
    "infrastructure": ["起降点", "机场", "空管", "通信导航"],
    "supporting_service": ["维修", "培训", "保险", "金融", "检测"]
}


def classify_event(document_type: str, text: str) -> str:
    if "统计分类" in document_type:
        return "industry_standard"
    if "法规" in document_type or "条例" in document_type:
        return "regulation"
    if "实施意见" in document_type and "保险" in text[:500]:
        return "insurance_finance"
    if "措施" in document_type:
        return "support_measure"
    if "方案" in document_type or "纲要" in document_type:
        return "strategic_plan"
    return "other"


def first_evidence(text: str) -> str:
    parts = re.split(r"(?<=[。！？；])", text)
    for part in parts:
        if "低空" in part and 20 <= len(part) <= 500:
            return part.strip()
    return next((part.strip() for part in parts if len(part.strip()) >= 20), text[:300].strip())


def matched_labels(text: str, mapping: dict[str, list[str]]) -> list[str]:
    return [label for label, words in mapping.items() if any(word.lower() in text.lower() for word in words)]


def main() -> None:
    documents = pd.read_csv(ROOT / "data" / "interim" / "policy_documents.csv")
    documents = documents[documents["fetch_status"] == "ok"].copy()
    level_score = {"central": 5, "province": 4, "city": 3, "district": 2}
    events = []
    for row in documents.itertuples(index=False):
        text = str(row.full_text)
        event_type = classify_event(row.document_type, text)
        intensity = level_score.get(row.level, 2)
        if event_type in {"regulation", "industry_standard"}:
            intensity = min(5, intensity + 1)
        events.append(
            {
                "event_id": f"E-{row.document_id}-01",
                "document_id": row.document_id,
                "event_date": row.publish_date,
                "event_type": event_type,
                "actor": row.publisher,
                "region": row.region,
                "policy_level": row.level,
                "policy_tools": json.dumps(matched_labels(text, TOOL_KEYWORDS), ensure_ascii=False),
                "targets": json.dumps(matched_labels(text, TARGET_KEYWORDS), ensure_ascii=False),
                "direction": "positive" if event_type != "regulation" else "mixed",
                "intensity": intensity,
                "uncertainty": max(1, 6 - intensity),
                "novelty": 3,
                "evidence_span": first_evidence(text),
                "confidence": 0.6,
                "extraction_method": "rule_baseline",
                "review_status": "pending_human_review",
            }
        )
    output = ROOT / "data" / "interim" / "policy_events_baseline.csv"
    pd.DataFrame(events).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"生成 {len(events)} 条弱标签事件：{output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
