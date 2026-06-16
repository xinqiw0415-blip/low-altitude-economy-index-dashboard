from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


ROOT = Path(__file__).resolve().parents[1]


def classification_row(model: str, task: str, truth: pd.Series, pred: pd.Series) -> dict:
    return {
        "model": model,
        "task": task,
        "accuracy": accuracy_score(truth, pred),
        "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(truth, pred),
    }


def main() -> None:
    human = pd.read_csv(ROOT / "data" / "processed" / "human_gold_adjudicated.csv")
    llm = pd.read_csv(ROOT / "data" / "processed" / "deepseek_company_events.csv")
    rule = pd.read_csv(ROOT / "data" / "interim" / "company_event_candidates.csv")
    llm_eval = llm[["event_id", "event_type", "direction", "intensity", "uncertainty", "relevance"]].rename(
        columns={column: f"{column}_llm" for column in ["event_type", "direction", "intensity", "uncertainty"]}
    )
    data = human.merge(llm_eval, on="event_id", how="left", validate="one_to_one").merge(
        rule[["event_id", "event_type", "direction", "relevance_score"]].rename(
            columns={"event_type": "event_type_rule", "direction": "direction_rule"}
        ), on="event_id", how="left", validate="one_to_one",
    )
    data["relevant_llm"] = ((data["event_type_llm"] != "other_business") & (data["relevance"] >= 0.5)).astype(int)
    data["relevant_rule"] = (data["relevance_score"] >= 0.45).astype(int)
    truth_relevant = data["human_relevant"].astype(int)

    metrics = pd.DataFrame([
        classification_row("DeepSeek", "relevance", truth_relevant, data["relevant_llm"]),
        classification_row("Rule baseline", "relevance", truth_relevant, data["relevant_rule"]),
        classification_row("DeepSeek", "event_type", data["human_event_type"], data["event_type_llm"]),
        classification_row("Rule baseline", "event_type", data["human_event_type"], data["event_type_rule"]),
        classification_row("DeepSeek", "direction", data["human_direction"], data["direction_llm"]),
    ])
    score_metrics = pd.DataFrame([
        {
            "field": field,
            "mae": np.mean(np.abs(data[f"human_{field}"] - data[f"{field}_llm"])),
            "weighted_kappa": cohen_kappa_score(data[f"human_{field}"], data[f"{field}_llm"], weights="quadratic"),
        }
        for field in ["intensity", "uncertainty"]
    ])
    disagreements = data[
        (data["human_event_type"] != data["event_type_llm"])
        | (data["human_direction"] != data["direction_llm"])
        | (truth_relevant != data["relevant_llm"])
    ]
    with pd.ExcelWriter(ROOT / "data" / "interim" / "model_human_disagreements.xlsx", engine="openpyxl") as writer:
        disagreements.to_excel(writer, index=False, sheet_name="disagreements")
        metrics.to_excel(writer, index=False, sheet_name="classification_metrics")
        score_metrics.to_excel(writer, index=False, sheet_name="ordinal_metrics")

    report = [
        "# 模型与裁决后金标准比较", "",
        "> 金标准包含10条A/B一致样本和30条Codex辅助裁决样本。裁决未读取DeepSeek预测，但它不等同于第三位独立人工专家标注。", "",
        "## 分类指标", "", metrics.to_markdown(index=False), "",
        "## 强度与不确定性", "", score_metrics.to_markdown(index=False), "",
        f"DeepSeek与金标准存在分类分歧的样本：{len(disagreements)} / {len(data)}。",
    ]
    (ROOT / "reports" / "model_vs_human.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    data.to_csv(ROOT / "data" / "processed" / "human_labeled_sample.csv", index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))
    print(score_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
