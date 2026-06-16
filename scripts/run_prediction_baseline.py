from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TECHNICAL = [
    "return_lag_1", "return_lag_2", "return_lag_5", "return_lag_10", "return_lag_20",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "momentum_5d", "momentum_10d", "momentum_20d", "HS300", "SSE", "CYB"
]
POLICY = [
    "policy_event_count", "policy_intensity_sum", "policy_intensity_max",
    "central_event_count", "positive_event_count", "policy_climate_factor", "policy_climate_index",
    "company_event_count", "company_event_relevance_sum", "positive_company_events",
    "negative_company_events", "unique_event_chains"
]
LLM_EVENTS = [
    "llm_company_event_count", "llm_company_intensity", "llm_company_relevance",
    "llm_company_positive", "llm_company_negative", "llm_policy_event_count",
    "llm_policy_intensity", "llm_policy_uncertainty", "llm_policy_novelty",
    "llm_company_intensity_ewm20", "llm_company_relevance_ewm20", "llm_policy_intensity_ewm20"
]
FINAL_EVENTS = [
    "final_company_event_count", "final_company_intensity", "final_company_positive",
    "final_company_negative", "final_human_event_count", "final_policy_event_count",
    "final_policy_intensity", "final_policy_uncertainty",
    "final_company_intensity_ewm20", "final_policy_intensity_ewm20"
]


def evaluate(data: pd.DataFrame, features: list[str], model_name: str, feature_set: str) -> dict:
    target = "future_up_5d"
    usable = data.dropna(subset=[target]).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=5, gap=20)
    probabilities = pd.Series(index=usable.index, dtype=float)
    predictions = pd.Series(index=usable.index, dtype=float)

    for train, test in splitter.split(usable):
        if model_name == "logistic":
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            )
        else:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3),
            )
        model.fit(usable.loc[train, features], usable.loc[train, target].astype(int))
        probabilities.loc[test] = model.predict_proba(usable.loc[test, features])[:, 1]
        predictions.loc[test] = (probabilities.loc[test] >= 0.5).astype(int)

    valid = probabilities.notna()
    truth = usable.loc[valid, target].astype(int)
    return {
        "model": model_name,
        "feature_set": feature_set,
        "observations": int(valid.sum()),
        "auc": roc_auc_score(truth, probabilities[valid]),
        "accuracy": accuracy_score(truth, predictions[valid]),
        "f1": f1_score(truth, predictions[valid]),
        "brier": brier_score_loss(truth, probabilities[valid]),
    }


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "processed" / "model_daily_dataset.csv")
    results = []
    for model in ["logistic", "hist_gradient_boosting"]:
        results.append(evaluate(data, TECHNICAL, model, "technical_only"))
        results.append(evaluate(data, TECHNICAL + POLICY, model, "technical_plus_rule_events"))
        results.append(evaluate(data, TECHNICAL + LLM_EVENTS, model, "technical_plus_llm_events"))
        results.append(evaluate(data, TECHNICAL + FINAL_EVENTS, model, "technical_plus_adjudicated_hybrid"))
    frame = pd.DataFrame(results)
    frame.to_csv(ROOT / "data" / "processed" / "prediction_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    (ROOT / "reports" / "prediction_baseline.md").write_text(
        "# 严格时序预测基线\n\n"
        + frame.to_markdown(index=False)
        + "\n\n使用5折时间序列切分并设置20个交易日间隔。裁决后混合事件特征由40条双人标注样本及其裁决结果增强，其余公司事件和政策事件仍采用DeepSeek证据过滤。近期新闻仅覆盖2026年4月至6月，未进入长期预测。结果用于检验增量信息，不据此宣称可交易性。\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
