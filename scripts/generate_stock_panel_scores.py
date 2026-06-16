from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from run_stock_panel_prediction import CATEGORICAL, EVENT, GLOBAL, TECHNICAL, pipelines


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def ranking_metrics(predictions: pd.DataFrame, fraction: float = 0.20) -> pd.DataFrame:
    rows = []
    for (target, feature_set, ensemble), group in predictions.groupby(["target", "feature_set", "ensemble"]):
        selected_parts = []
        for _, day in group.groupby("trade_date"):
            count = max(1, int(np.ceil(len(day) * fraction)))
            selected_parts.append(day.nlargest(count, "ensemble_probability"))
        selected = pd.concat(selected_parts, ignore_index=True)
        baseline = group["truth"].mean()
        precision = selected["truth"].mean()
        rows.append({
            "target": target, "feature_set": feature_set, "ensemble": ensemble,
            "selection_fraction": fraction, "selected_observations": len(selected),
            "precision_at_top": precision, "baseline_positive_share": baseline,
            "lift": precision / baseline if baseline > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def fit_ensemble(panel: pd.DataFrame, target: str, numeric: list[str], model_names: list[str]) -> pd.Series:
    features = numeric + CATEGORICAL
    train = panel.dropna(subset=[target]).copy()
    latest_date = panel["trade_date"].max()
    latest = panel[panel["trade_date"].eq(latest_date)].copy()
    probabilities = []
    templates = pipelines(numeric)
    for model_name in model_names:
        estimator = clone(templates[model_name])
        estimator.fit(train[features], train[target].astype(int))
        probabilities.append(estimator.predict_proba(latest[features])[:, 1])
    return latest.assign(score=np.mean(probabilities, axis=0)).set_index("ts_code")["score"]


def main() -> None:
    oos = pd.read_csv(PROCESSED / "stock_panel_ensemble_oos.csv", parse_dates=["trade_date"])
    ranking = ranking_metrics(oos)
    ranking.to_csv(PROCESSED / "stock_panel_ranking_metrics.csv", index=False, encoding="utf-8-sig")

    panel = pd.read_csv(PROCESSED / "stock_panel_daily.csv", parse_dates=["trade_date"])
    panel = panel[panel["realtime_sentiment_lag1"].notna()].copy()
    extreme = fit_ensemble(
        panel, "future_extreme_relative_5d", TECHNICAL + GLOBAL + EVENT,
        ["logistic", "hist_gradient_boosting", "extra_trees"],
    )
    volatility = fit_ensemble(
        panel, "future_volatility_jump_5d", TECHNICAL + EVENT,
        ["logistic", "extra_trees"],
    )
    latest = panel[panel["trade_date"].eq(panel["trade_date"].max())][[
        "trade_date", "ts_code", "name", "industry_role", "tier", "event_window_20d", "days_since_company_event"
    ]].copy()
    latest["relative_strength_probability"] = latest["ts_code"].map(extreme)
    latest["volatility_jump_probability"] = latest["ts_code"].map(volatility)
    latest["relative_strength_rank"] = latest["relative_strength_probability"].rank(ascending=False, method="min").astype(int)
    latest["volatility_risk_rank"] = latest["volatility_jump_probability"].rank(ascending=False, method="min").astype(int)
    latest.to_csv(PROCESSED / "stock_panel_latest_scores.csv", index=False, encoding="utf-8-sig")
    report = [
        "# 个股面板最新评分", "",
        f"评分日期：{latest['trade_date'].iloc[0].date()}。模型使用此前所有具有可观测目标的样本重新训练。", "",
        "## 前20%排序能力", "", ranking.to_markdown(index=False), "",
        "## 最新评分", "", latest.sort_values("relative_strength_rank").to_markdown(index=False), "",
        "评分用于研究展示，不是投资建议。极端强弱概率来自仅使用上下30%标签训练的模型，应理解为相对排序分数，而非无条件上涨概率。",
    ]
    (ROOT / "reports" / "stock_panel_latest_scores.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))
    print(latest.sort_values("relative_strength_rank").to_string(index=False))


if __name__ == "__main__":
    main()
