from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


# Decisions are based only on the source announcement and the two annotators'
# evidence spans. DeepSeek output is deliberately excluded from adjudication.
DECISIONS = {
    1: (1, "project_investment", "positive", 4, 4, "收购航空器制造资产属于重大产业投资，仍处于筹划阶段。"),
    3: (1, "legal_risk", "mixed", 3, 4, "无人机专利诉讼仅剩一案，风险收敛但尚未完全消除。"),
    4: (1, "order_contract", "positive", 4, 2, "无人机销售合同金额重大且已签署，履约仍有一般不确定性。"),
    5: (1, "government_support", "positive", 3, 1, "通航专项资金已经获得，金额具有实质性。"),
    6: (1, "strategic_cooperation", "positive", 3, 5, "无人机及基础设施合作仅为框架协议，落地不确定性高。"),
    8: (1, "strategic_cooperation", "positive", 3, 5, "无人机通航产业化框架方向明确，但投资和实施尚未确定。"),
    9: (1, "government_support", "positive", 3, 1, "通航政府补助已确认，直接形成正向支持。"),
    11: (1, "project_investment", "positive", 4, 3, "年产220架飞机项目及投资额明确，但建设投产仍需推进。"),
    12: (0, "other_business", "neutral", 1, 1, "主要是同一控制体系内的股权收购披露，未形成新增低空经营事件。"),
    13: (0, "other_business", "neutral", 1, 1, "仅说明停牌前股价波动，不包含新增低空产业风险或经营信息。"),
    14: (0, "other_business", "neutral", 1, 3, "证监会反馈回复修订属于融资审核程序，原文未给出直接低空经营事件。"),
    15: (1, "legal_risk", "positive", 3, 1, "相关诉讼已被法院裁定驳回，风险得到实质解除。"),
    16: (0, "other_business", "neutral", 1, 1, "集团内部收购事项完成，未体现新增直升机业务活动或产业增量。"),
    18: (0, "other_business", "neutral", 1, 1, "会计师专项核查属于重组财务说明，不是独立低空产业事件。"),
    20: (1, "strategic_cooperation", "positive", 3, 4, "协议直接涉及通航和低空航线，但具体项目仍待落地。"),
    21: (0, "other_business", "neutral", 1, 3, "审计评估仍在进行，属于无新增实质信息的例行重组进展。"),
    22: (1, "legal_risk", "positive", 3, 1, "无人机专利诉讼全部撤回，法律风险已经实质解除。"),
    24: (1, "project_investment", "positive", 4, 3, "航空资产重组获得国资委批复，是关键审批里程碑。"),
    25: (1, "government_support", "positive", 3, 1, "通航专项资金已经获得，金额具有实质性。"),
    27: (1, "project_investment", "positive", 4, 2, "航空资产重组通过上交所审核，是重要且较确定的审批进展。"),
    28: (1, "order_contract", "positive", 4, 4, "低空数字化系统项目已中标，但正式合同尚未签署。"),
    30: (1, "government_support", "positive", 3, 1, "通航专项资金已经获得，金额具有实质性。"),
    32: (1, "order_contract", "positive", 4, 3, "22架直升机购销合同规模重大，后续履约存在正常不确定性。"),
    33: (1, "legal_risk", "negative", 3, 5, "无人机相关诉讼已受理但尚未开庭，结果高度不确定。"),
    34: (0, "other_business", "neutral", 1, 3, "尽调和预审仍在推进，未披露新的审批、交易或经营结果。"),
    35: (1, "project_investment", "mixed", 3, 4, "首次披露航空资产重组预案具有实质信息，同时明确提示较高风险。"),
    36: (1, "project_investment", "positive", 3, 2, "董事会批准设立通航子公司，项目明确但仍需完成设立。"),
    37: (1, "order_contract", "positive", 3, 2, "航空消防服务项目已经中标，金额明确且与通航运营直接相关。"),
    38: (1, "strategic_cooperation", "positive", 3, 5, "无人机合作框架方向明确，但没有约束性订单和确定金额。"),
    39: (1, "government_support", "positive", 3, 1, "通航专项资金已经获得，金额具有实质性。"),
}


def choose_evidence(row: pd.Series, relevant: int) -> str:
    a = str(row.get("human_evidence_span_a", "") or "").strip()
    b = str(row.get("human_evidence_span_b", "") or "").strip()
    if relevant:
        return a if len(a) >= len(b) else b
    return b if b and b.lower() != "nan" else a


def main() -> None:
    workbook = INTERIM / "human_annotation_adjudication.xlsx"
    task = pd.read_excel(workbook, sheet_name="adjudication")
    instructions = pd.read_excel(workbook, sheet_name="instructions")
    if set(task["sample_no"].astype(int)) != set(DECISIONS):
        raise ValueError("Adjudication rows do not match the scripted decisions.")

    text_columns = ["final_event_type", "final_direction", "final_evidence_span", "adjudication_reason"]
    task[text_columns] = task[text_columns].astype("object")
    for idx, row in task.iterrows():
        relevant, event_type, direction, intensity, uncertainty, reason = DECISIONS[int(row["sample_no"])]
        task.loc[idx, [
            "final_relevant", "final_event_type", "final_direction",
            "final_intensity", "final_uncertainty", "final_evidence_span",
            "adjudication_reason",
        ]] = [
            relevant, event_type, direction, intensity, uncertainty,
            choose_evidence(row, relevant), f"Codex辅助裁决：{reason}",
        ]

    metadata = pd.DataFrame({
        "item": ["adjudicator", "method", "model_leakage_control", "scope"],
        "value": [
            "Codex辅助裁决（非第三位独立人工标注）",
            "依据公告原文、A/B标签和证据片段逐条裁决",
            "裁决过程不读取DeepSeek预测标签",
            f"{len(task)}条A/B争议样本",
        ],
    })
    filled_workbook = INTERIM / "human_annotation_adjudication_filled.xlsx"
    with pd.ExcelWriter(filled_workbook, engine="openpyxl") as writer:
        task.to_excel(writer, sheet_name="adjudication", index=False)
        instructions.to_excel(writer, sheet_name="instructions", index=False)
        metadata.to_excel(writer, sheet_name="metadata", index=False)

    a = pd.read_excel(INTERIM / "human_annotation_A.xlsx", sheet_name="tasks")
    b = pd.read_excel(INTERIM / "human_annotation_B_independent.xlsx", sheet_name="tasks")
    gold = a.merge(
        b[["event_id", "human_relevant", "human_event_type", "human_direction", "human_intensity", "human_uncertainty", "human_evidence_span"]],
        on="event_id", suffixes=("_a", "_b"), validate="one_to_one",
    )
    adjudicated = task.set_index("event_id")
    rows = []
    for row in gold.itertuples(index=False):
        record = row._asdict()
        event_id = record["event_id"]
        if event_id in adjudicated.index:
            final = adjudicated.loc[event_id]
            source = "codex_assisted_adjudication"
            values = {
                "human_relevant": int(final["final_relevant"]),
                "human_event_type": final["final_event_type"],
                "human_direction": final["final_direction"],
                "human_intensity": int(final["final_intensity"]),
                "human_uncertainty": int(final["final_uncertainty"]),
                "human_evidence_span": final["final_evidence_span"],
                "adjudication_reason": final["adjudication_reason"],
            }
        else:
            source = "independent_annotator_consensus"
            values = {
                "human_relevant": int(record["human_relevant_a"]),
                "human_event_type": record["human_event_type_a"],
                "human_direction": record["human_direction_a"],
                "human_intensity": int(record["human_intensity_a"]),
                "human_uncertainty": int(record["human_uncertainty_a"]),
                "human_evidence_span": record["human_evidence_span_a"],
                "adjudication_reason": "A/B标签一致，无需裁决。",
            }
        base = {key: value for key, value in record.items() if not key.endswith("_a") and not key.endswith("_b")}
        # Restore source fields that acquired an _a suffix during the merge.
        for key, value in record.items():
            if key.endswith("_a") and not key.startswith("human_"):
                base[key[:-2]] = value
        base.update(values)
        base["gold_source"] = source
        rows.append(base)
    final_gold = pd.DataFrame(rows)
    final_gold.to_csv(PROCESSED / "human_gold_adjudicated.csv", index=False, encoding="utf-8-sig")
    print(f"Filled {len(task)} adjudications in {filled_workbook.name}; built {len(final_gold)} gold labels.")


if __name__ == "__main__":
    main()
