from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["human_relevant", "human_event_type", "human_direction", "human_intensity", "human_uncertainty"]


def clean(value: object) -> object:
    if isinstance(value, str):
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    return value


def main() -> None:
    a = pd.read_excel(ROOT / "data" / "interim" / "human_annotation_A.xlsx", sheet_name="tasks")
    b = pd.read_excel(ROOT / "data" / "interim" / "human_annotation_B_independent.xlsx", sheet_name="tasks")
    merged = a.merge(b, on=["sample_no", "event_id"], suffixes=("_a", "_b"))
    mask = False
    for field in FIELDS:
        mask = mask | (merged[f"{field}_a"] != merged[f"{field}_b"])
    disagreements = merged[mask].copy()
    output_columns = ["sample_no", "event_id", "ts_code_a", "name_a", "event_date_a", "title_a", "full_text_a"]
    for field in FIELDS + ["human_evidence_span"]:
        output_columns.extend([f"{field}_a", f"{field}_b"])
    task = disagreements[output_columns].copy()
    task = task.rename(columns={"ts_code_a": "ts_code", "name_a": "name", "event_date_a": "event_date", "title_a": "title", "full_text_a": "full_text"})
    task["final_relevant"] = ""
    task["final_event_type"] = ""
    task["final_direction"] = ""
    task["final_intensity"] = ""
    task["final_uncertainty"] = ""
    task["final_evidence_span"] = ""
    task["adjudication_reason"] = ""
    for column in task.select_dtypes(include="object"):
        task[column] = task[column].map(clean)
    instructions = pd.DataFrame(
        {
            "principle": [
                "相关性边界",
                "进展公告",
                "风险文件",
                "数值评分",
                "裁决独立性",
            ],
            "rule": [
                "必须有原文直接关联低空、通航、无人机、航空器、直升机、空管或低空基础设施；仅因公司属于概念股不算相关。",
                "同一事项的关键审批、完成、中止或实质变化可以作为事件；例行进展且无新增信息可判other_business。",
                "风险提示或价格波动说明只有在披露新增实质风险时纳入；纯模板说明通常判other_business。",
                "强度与不确定性必须重新独立判断，不要机械取A/B平均。",
                "裁决人可查看A/B意见，但必须依据正文填写final字段，并复制最终证据原文。",
            ],
        }
    )
    output = ROOT / "data" / "interim" / "human_annotation_adjudication.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        task.to_excel(writer, sheet_name="adjudication", index=False)
        instructions.to_excel(writer, sheet_name="instructions", index=False)
    print(f"待裁决 {len(task)} 条：{output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
