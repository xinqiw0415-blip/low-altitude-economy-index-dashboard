from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).replace("“", '"').replace("”", '"')


def evidence_match_score(evidence: str, source: str) -> float:
    needle = compact(evidence)
    haystack = compact(source)
    if not needle:
        return 0.0
    if needle in haystack:
        return 1.0
    if len(needle) >= 40 and needle[:40] in haystack:
        return 0.95
    sentences = [compact(part) for part in re.split(r"(?<=[。！？；])", str(source)) if len(compact(part)) >= 10]
    candidates = sentences + [sentences[i] + sentences[i + 1] for i in range(len(sentences) - 1)]
    if not candidates:
        return 0.0
    return max(SequenceMatcher(None, needle, candidate).ratio() for candidate in candidates)


def normalize_policy_type(raw_type: str, evidence: str, tools: object) -> str:
    text = " ".join([str(raw_type), str(evidence), json.dumps(tools, ensure_ascii=False)]).lower()
    rules = [
        ("fiscal_subsidy", ["financial", "fund", "subsid", "reward", "财政", "补贴", "奖励", "资金", "保险", "融资"]),
        ("airspace_reform", ["airspace", "route_approval", "空域", "航线", "飞行审批"]),
        ("infrastructure", ["infrastructure", "机场", "起降", "导航", "低空智联网", "基础设施"]),
        ("regulation", ["regulat", "safety", "certification", "条例", "监管", "安全", "适航", "实名"]),
        ("industry_standard", ["standard", "classification", "标准", "统计分类"]),
        ("insurance_finance", ["insurance", "保险"]),
        ("strategic_plan", ["strategic", "industry_development", "industrial_development", "产业发展", "规划"]),
        ("support_measure", ["support", "promotion", "innovation", "talent", "application", "service", "支持", "促进", "创新", "人才", "场景"]),
    ]
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return "other"


def clean_excel(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].map(
            lambda value: re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value) if isinstance(value, str) else value
        )
    return frame


def finalize_company() -> pd.DataFrame:
    raw = read_jsonl(ROOT / "data" / "interim" / "deepseek_company_results.jsonl")
    metadata = pd.read_csv(ROOT / "data" / "interim" / "company_event_candidates.csv")
    documents = pd.read_csv(ROOT / "data" / "interim" / "company_announcement_documents.csv")[["art_code", "full_text"]]
    source = metadata.merge(documents, on="art_code", how="left")
    source_by_id = source.set_index("event_id")
    rows = []
    for record in raw:
        event = record["result"]["event"]
        meta = source_by_id.loc[record["task_id"]]
        match_score = evidence_match_score(event["evidence_span"], meta["full_text"])
        grounded = match_score >= 0.82
        rows.append(
            {
                "event_id": record["task_id"], "art_code": meta["art_code"], "ts_code": meta["ts_code"],
                "name": meta["name"], "event_date": meta["event_date"], "title": meta["title"],
                "event_type": event["event_type"], "direction": event["direction"],
                "intensity": int(event["intensity"]), "uncertainty": int(event["uncertainty"]),
                "relevance": float(event["relevance"]), "is_initial_event": bool(event["is_initial_event"]),
                "event_chain_key": event["event_chain_key"], "evidence_span": event["evidence_span"],
                "confidence": float(event["confidence"]), "evidence_match_score": match_score, "evidence_grounded": grounded,
                "rule_event_type": meta["event_type"], "rule_relevance": meta["relevance_score"],
                "accepted": grounded and event["event_type"] != "other_business" and float(event["relevance"]) >= 0.5,
                "review_status": "pending_human_review",
            }
        )
    return pd.DataFrame(rows)


def finalize_policy() -> pd.DataFrame:
    raw = read_jsonl(ROOT / "data" / "interim" / "deepseek_policy_results.jsonl")
    documents = pd.read_csv(ROOT / "data" / "interim" / "policy_documents.csv").set_index("document_id")
    rows = []
    for record in raw:
        meta = documents.loc[record["task_id"]]
        seen = set()
        for index, event in enumerate(record["result"]["events"], start=1):
            evidence = event["evidence_span"]
            event_type = normalize_policy_type(event["event_type"], evidence, event.get("policy_tools", []))
            signature = (event_type, compact(evidence)[:80])
            if signature in seen:
                continue
            seen.add(signature)
            match_score = evidence_match_score(evidence, meta["full_text"])
            grounded = match_score >= 0.82
            rows.append(
                {
                    "event_id": f"L-{record['task_id']}-{index:02d}", "document_id": record["task_id"],
                    "event_date": meta["publish_date"], "title": meta["title"], "publisher": meta["publisher"],
                    "region": event.get("region", meta["region"]), "policy_level": event.get("policy_level", meta["level"]),
                    "event_type": event_type, "raw_event_type": event["event_type"],
                    "direction": event["direction"], "intensity": int(event["intensity"]),
                    "uncertainty": int(event["uncertainty"]), "novelty": int(event.get("novelty", 3)),
                    "policy_tools": json.dumps(event.get("policy_tools", []), ensure_ascii=False),
                    "targets": json.dumps(event.get("targets", []), ensure_ascii=False),
                    "evidence_span": evidence, "confidence": float(event["confidence"]),
                    "evidence_match_score": match_score, "evidence_grounded": grounded, "accepted": grounded, "review_status": "pending_human_review",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    company = finalize_company()
    policy = finalize_policy()
    company.to_csv(ROOT / "data" / "processed" / "deepseek_company_events.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(ROOT / "data" / "processed" / "deepseek_policy_events.csv", index=False, encoding="utf-8-sig")
    review_path = ROOT / "data" / "interim" / "deepseek_event_review.xlsx"
    with pd.ExcelWriter(review_path, engine="openpyxl") as writer:
        clean_excel(company).to_excel(writer, sheet_name="company_events", index=False)
        clean_excel(policy).to_excel(writer, sheet_name="policy_events", index=False)

    company_types = company[company["accepted"]]["event_type"].value_counts().rename_axis("event_type").reset_index(name="events")
    policy_types = policy[policy["accepted"]]["event_type"].value_counts().rename_axis("event_type").reset_index(name="events")
    report = [
        "# DeepSeek事件抽取质量报告", "",
        f"- 公司任务：{len(company)}，证据可回溯：{int(company['evidence_grounded'].sum())}，自动接纳：{int(company['accepted'].sum())}",
        f"- 公司规则误报/无关：{int((company['event_type'] == 'other_business').sum())}",
        f"- 政策子事件：{len(policy)}，证据可回溯：{int(policy['evidence_grounded'].sum())}，自动接纳：{int(policy['accepted'].sum())}",
        "", "## 公司事件类型", "", company_types.to_markdown(index=False),
        "", "## 政策事件类型（统一映射后）", "", policy_types.to_markdown(index=False),
        "", "所有事件仍标记为待人工复核。`accepted`只表示通过自动证据与相关性门槛，不代表人工金标准。",
    ]
    (ROOT / "reports" / "deepseek_extraction_quality.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("公司", company[["evidence_grounded", "accepted"]].sum().to_dict())
    print("政策", policy[["evidence_grounded", "accepted"]].sum().to_dict())


if __name__ == "__main__":
    main()
