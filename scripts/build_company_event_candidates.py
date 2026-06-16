from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOMAIN_TERMS = ["低空", "无人机", "通用航空", "通航", "航空器", "飞行", "机场", "空管", "eVTOL", "直升机"]
EVENT_TYPES = {
    "order_contract": ["中标", "订单", "重大合同", "购销合同", "销售合同", "供货合同"],
    "strategic_cooperation": ["战略合作", "合作框架", "合作协议"],
    "project_investment": ["项目", "对外投资", "设立", "收购", "重组"],
    "product_technology": ["研发", "新产品", "试飞", "交付", "获批", "资质", "型号装备"],
    "government_support": ["政府补助", "专项资金", "财政补贴", "奖励资金"],
    "performance": ["业绩预告", "业绩快报"],
    "market_risk": ["异常波动", "风险提示"],
    "legal_safety_risk": ["事故", "处罚", "诉讼"]
}
NOISE_TERMS = [
    "投资者调研", "投资者保护", "金融服务协议", "募集资金", "理财", "股东协议转让",
    "会计师事务所", "项目质量复核", "证券投资", "内控制度", "管理办法", "工商变更登记"
]


def classify(title: str) -> str:
    for label, words in EVENT_TYPES.items():
        if any(word in title for word in words):
            return label
    return "other_business"


def normalized_chain_title(title: str) -> str:
    title = title.split(":", 1)[-1]
    title = re.sub(r"关于|公告|提示性|自愿性信息披露|进展|暨关联交易|的", "", title)
    title = re.sub(r"20\d{2}年(?:度)?|第[一二三四五六七八九十]+季度|\d+%|\s+", "", title)
    return title[:100]


def evidence(text: str, title: str) -> str:
    terms = [word for words in EVENT_TYPES.values() for word in words] + DOMAIN_TERMS
    sentences = re.split(r"(?<=[。！？；])", text)
    ranked = sorted(sentences, key=lambda sentence: sum(word in sentence for word in terms), reverse=True)
    return next((sentence.strip() for sentence in ranked if 30 <= len(sentence.strip()) <= 500), title)


def relevance(title: str, text: str, tier: str, event_type: str) -> tuple[float, list[str]]:
    title_hits = [term for term in DOMAIN_TERMS if term.lower() in title.lower()]
    text_hits = [term for term in DOMAIN_TERMS if term.lower() in text.lower()]
    noise = [term for term in NOISE_TERMS if term in title]
    score = 0.0
    if title_hits:
        score += 0.55
    if text_hits:
        score += min(0.3, 0.05 * len(text_hits))
    if tier == "core":
        score += 0.15
    elif tier == "extended":
        score += 0.05
    if noise:
        score -= 0.5
    substantive = {"order_contract", "strategic_cooperation", "project_investment", "product_technology", "government_support", "legal_safety_risk"}
    if event_type in substantive:
        score += 0.3 if tier == "core" else 0.18
    return max(0.0, min(1.0, score)), title_hits + text_hits


def direction(event_type: str, title: str) -> str:
    if event_type in {"market_risk", "legal_safety_risk"}:
        return "negative"
    if "终止" in title or "下修" in title or "亏损" in title:
        return "negative"
    if event_type == "performance":
        return "mixed"
    return "positive"


def main() -> None:
    documents = pd.read_csv(ROOT / "data" / "interim" / "company_announcement_documents.csv")
    universe = pd.read_csv(ROOT / "config" / "stock_universe.csv")[["ts_code", "tier"]]
    documents = documents.merge(universe, on="ts_code", how="left")
    documents = documents[documents["fetch_status"] == "ok"].copy()
    rows = []
    for item in documents.itertuples(index=False):
        title, text = str(item.title), str(item.full_text)
        event_type = classify(title)
        score, hits = relevance(title, text, item.tier, event_type)
        chain_title = normalized_chain_title(title)
        chain_id = hashlib.sha1(f"{item.ts_code}|{chain_title}".encode("utf-8")).hexdigest()[:14]
        rows.append(
            {
                "event_id": f"C-{item.art_code}",
                "art_code": item.art_code,
                "ts_code": item.ts_code,
                "name": item.name,
                "event_date": item.notice_date,
                "event_type": event_type,
                "direction": direction(event_type, title),
                "relevance_score": score,
                "domain_hits": json.dumps(sorted(set(hits)), ensure_ascii=False),
                "title": title,
                "evidence_span": evidence(text, title),
                "event_chain_id": chain_id,
                "is_progress_update": bool("进展" in title or "更正" in title),
                "confidence": 0.55,
                "extraction_method": "rule_baseline",
                "review_status": "pending_human_review",
            }
        )
    events = pd.DataFrame(rows).sort_values(["event_date", "ts_code"])
    events["eligible_for_labeling"] = (events["relevance_score"] >= 0.45) & (events["event_type"] != "other_business")
    output = ROOT / "data" / "interim" / "company_event_candidates.csv"
    events.to_csv(output, index=False, encoding="utf-8-sig")

    eligible = events[events["eligible_for_labeling"]].copy()
    sample_parts = []
    for _, group in eligible.groupby("event_type"):
        sample_parts.append(group.sort_values("relevance_score", ascending=False).head(40))
    sample = pd.concat(sample_parts, ignore_index=True).drop_duplicates("event_id")
    for column in sample.select_dtypes(include="object").columns:
        sample[column] = sample[column].map(
            lambda value: re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value) if isinstance(value, str) else value
        )
    sample.to_excel(ROOT / "data" / "interim" / "company_event_annotation_sample.xlsx", index=False)
    print(f"候选事件 {len(events)}；进入标注池 {len(eligible)}；分层标注样本 {len(sample)}")
    print(eligible["event_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
