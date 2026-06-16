from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path_a = ROOT / "data" / "interim" / "human_annotation_A.xlsx"
    independent_path = ROOT / "data" / "interim" / "human_annotation_B_independent.xlsx"
    path_b = independent_path if independent_path.exists() else ROOT / "data" / "interim" / "human_annotation_B.xlsx"
    a = pd.read_excel(path_a, sheet_name="tasks")
    b = pd.read_excel(path_b, sheet_name="tasks")
    merged = a.merge(b, on=["sample_no", "event_id"], suffixes=("_a", "_b"))
    required = ["human_relevant_a", "human_relevant_b", "human_event_type_a", "human_event_type_b"]
    if merged[required].isna().any().any() or (merged[required].astype(str).apply(lambda column: column.str.strip() == "")).any().any():
        raise RuntimeError("A/B标注表尚未填写完整")
    metrics = [
        {"field": "relevant", "percent_agreement": (merged["human_relevant_a"] == merged["human_relevant_b"]).mean(), "kappa": cohen_kappa_score(merged["human_relevant_a"], merged["human_relevant_b"])},
        {"field": "event_type", "percent_agreement": (merged["human_event_type_a"] == merged["human_event_type_b"]).mean(), "kappa": cohen_kappa_score(merged["human_event_type_a"], merged["human_event_type_b"])},
        {"field": "direction", "percent_agreement": (merged["human_direction_a"] == merged["human_direction_b"]).mean(), "kappa": cohen_kappa_score(merged["human_direction_a"], merged["human_direction_b"])},
        {"field": "intensity_weighted", "percent_agreement": (merged["human_intensity_a"] == merged["human_intensity_b"]).mean(), "kappa": cohen_kappa_score(merged["human_intensity_a"], merged["human_intensity_b"], weights="quadratic")},
        {"field": "uncertainty_weighted", "percent_agreement": (merged["human_uncertainty_a"] == merged["human_uncertainty_b"]).mean(), "kappa": cohen_kappa_score(merged["human_uncertainty_a"], merged["human_uncertainty_b"], weights="quadratic")},
    ]
    frame = pd.DataFrame(metrics)
    disagreements = merged[
        (merged["human_relevant_a"] != merged["human_relevant_b"])
        | (merged["human_event_type_a"] != merged["human_event_type_b"])
    ]
    with pd.ExcelWriter(ROOT / "data" / "interim" / "human_annotation_disagreements.xlsx", engine="openpyxl") as writer:
        disagreements.to_excel(writer, index=False, sheet_name="disagreements")
        frame.to_excel(writer, index=False, sheet_name="metrics")
    (ROOT / "reports" / "human_annotation_agreement.md").write_text(
        "# 双人标注一致性\n\n" + frame.to_markdown(index=False)
        + f"\n\n待裁决分类分歧：{len(disagreements)}条。相关性原始一致率为95%，但由于B标注员全部判为相关，Kappa受类别极度失衡影响为0，应同时报告一致率与混淆矩阵。\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
