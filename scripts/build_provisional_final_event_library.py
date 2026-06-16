from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    llm = pd.read_csv(ROOT / "data" / "processed" / "deepseek_company_events.csv")
    gold = pd.read_csv(ROOT / "data" / "processed" / "human_gold_adjudicated.csv").set_index("event_id")
    rows = []
    for item in llm.itertuples(index=False):
        record = item._asdict()
        if item.event_id in gold.index:
            label = gold.loc[item.event_id]
            record.update({
                "event_type": label["human_event_type"],
                "direction": label["human_direction"],
                "intensity": int(label["human_intensity"]),
                "uncertainty": int(label["human_uncertainty"]),
                "evidence_span": label["human_evidence_span"],
                "accepted": bool(int(label["human_relevant"])),
                "label_source": label["gold_source"],
                "review_status": "adjudicated_sample",
            })
        else:
            record["label_source"] = "deepseek_evidence_filtered"
            record["review_status"] = "not_human_reviewed"
        rows.append(record)
    company = pd.DataFrame(rows)
    policy = pd.read_csv(ROOT / "data" / "processed" / "deepseek_policy_events.csv")
    policy["label_source"] = "deepseek_evidence_filtered"
    policy["review_status"] = "not_human_reviewed"

    company.to_csv(ROOT / "data" / "processed" / "final_company_events_hybrid.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(ROOT / "data" / "processed" / "final_policy_events_hybrid.csv", index=False, encoding="utf-8-sig")
    # Compatibility paths used by existing downstream scripts.
    company.to_csv(ROOT / "data" / "processed" / "final_company_events_provisional.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(ROOT / "data" / "processed" / "final_policy_events_provisional.csv", index=False, encoding="utf-8-sig")

    reviewed = int(company["review_status"].eq("adjudicated_sample").sum())
    report = [
        "# 裁决后混合事件库", "",
        f"- 公司候选事件：{len(company)}条，接纳{int(company['accepted'].sum())}条。",
        f"- 双人标注并完成裁决：{reviewed}条。",
        f"- 未经人工复核、采用DeepSeek证据过滤：{len(company) - reviewed}条。",
        f"- 政策子事件：{len(policy)}条，接纳{int(policy['accepted'].sum())}条，当前均为DeepSeek证据过滤。", "",
        "因此该库应称为“人工裁决样本增强的混合事件库”，不应表述为全量人工金标准。",
    ]
    (ROOT / "reports" / "final_event_library.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Accepted company events: {int(company['accepted'].sum())}/{len(company)}")


if __name__ == "__main__":
    main()
