from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RNG = np.random.default_rng(20260616)


def latest_event_key(panel: pd.DataFrame) -> pd.Series:
    event_date = panel["trade_date"].where(panel["company_event_count"].gt(0))
    latest = event_date.groupby(panel["ts_code"]).ffill()
    return panel["ts_code"] + "|" + latest.dt.strftime("%Y-%m-%d").fillna("none")


def cluster_bootstrap(data: pd.DataFrame, probability_columns: list[str], repetitions: int = 2000) -> pd.DataFrame:
    clusters = data["event_key"].dropna().unique()
    rows = []
    estimates = {column: roc_auc_score(data["truth"], data[column]) for column in probability_columns}
    draws = {column: [] for column in probability_columns}
    draws["delta"] = []
    for _ in range(repetitions):
        selected = RNG.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([data[data["event_key"].eq(cluster)] for cluster in selected], ignore_index=True)
        if sample["truth"].nunique() < 2:
            continue
        values = {column: roc_auc_score(sample["truth"], sample[column]) for column in probability_columns}
        for column, value in values.items():
            draws[column].append(value)
        draws["delta"].append(values[probability_columns[1]] - values[probability_columns[0]])
    for column in probability_columns:
        rows.append({
            "metric": column, "estimate": estimates[column],
            "ci_low": np.quantile(draws[column], 0.025), "ci_high": np.quantile(draws[column], 0.975),
            "clusters": len(clusters), "observations": len(data),
        })
    delta = estimates[probability_columns[1]] - estimates[probability_columns[0]]
    rows.append({
        "metric": f"delta_{probability_columns[1]}_minus_{probability_columns[0]}", "estimate": delta,
        "ci_low": np.quantile(draws["delta"], 0.025), "ci_high": np.quantile(draws["delta"], 0.975),
        "clusters": len(clusters), "observations": len(data),
    })
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_csv(PROCESSED / "stock_panel_daily.csv", parse_dates=["trade_date"])
    panel["event_key"] = latest_event_key(panel)
    predictions = pd.read_csv(PROCESSED / "stock_panel_prediction_oos.csv", parse_dates=["trade_date"])
    selected = predictions[
        predictions["target"].eq("extreme_relative_5d")
        & predictions["model"].eq("logistic")
        & predictions["feature_set"].isin(["technical", "technical_events", "technical_global_events"])
        & predictions["event_window_20d"].eq(1)
    ]
    wide = selected.pivot_table(
        index=["trade_date", "ts_code", "name", "truth", "fold"],
        columns="feature_set", values="probability"
    ).reset_index()
    wide = wide.merge(panel[["trade_date", "ts_code", "event_key"]], on=["trade_date", "ts_code"], how="left")
    fold_rows = []
    for feature in ["technical", "technical_events", "technical_global_events"]:
        for fold, group in wide.groupby("fold"):
            if group["truth"].nunique() == 2:
                fold_rows.append({"feature_set": feature, "fold": fold, "observations": len(group), "auc": roc_auc_score(group["truth"], group[feature])})
    fold_metrics = pd.DataFrame(fold_rows)
    bootstrap_event = cluster_bootstrap(wide, ["technical", "technical_events"])
    bootstrap_global = cluster_bootstrap(wide, ["technical", "technical_global_events"])
    bootstrap = pd.concat([bootstrap_event.assign(comparison="events_without_global"), bootstrap_global.assign(comparison="events_with_global")], ignore_index=True)
    fold_metrics.to_csv(PROCESSED / "stock_panel_event_fold_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(PROCESSED / "stock_panel_event_uplift_bootstrap.csv", index=False, encoding="utf-8-sig")
    report = [
        "# 公司事件特征增量检验", "",
        "针对事件后20日的极端相对强弱目标，比较同一批样本上的技术基线与事件增强模型。置信区间按最近一次公司事件聚类，自助抽样2000次。", "",
        "## 分时间折结果", "", fold_metrics.to_markdown(index=False), "",
        "## 事件簇自助法", "", bootstrap.to_markdown(index=False), "",
        "若AUC增量置信区间包含0，则不能认为事件特征带来了稳定提升，即使点估计更高。",
    ]
    (ROOT / "reports" / "stock_panel_event_uplift.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(fold_metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
