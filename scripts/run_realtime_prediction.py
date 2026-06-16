from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
TARGET = "future_up_5d"
TECHNICAL = [
    "return_lag_1", "return_lag_2", "return_lag_5", "return_lag_10", "return_lag_20",
    "volatility_5d", "volatility_10d", "volatility_20d", "momentum_5d", "momentum_10d",
    "momentum_20d", "HS300", "SSE", "CYB",
]
REALTIME = ["realtime_sentiment_lag1", "realtime_sentiment_change_lag1"]
CHANNEL_SOURCE = ["policy_support", "company_sentiment", "confidence_balance", "signed_breadth", "risk_impulse"]


def models() -> dict:
    return {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_depth=3, l2_regularization=1.0),
        ),
    }


def evaluate(data: pd.DataFrame, feature_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, prediction_rows, importance_rows = [], [], []
    splitter = TimeSeriesSplit(n_splits=5, gap=20)
    for feature_set, features in feature_sets.items():
        sample = data.dropna(subset=[TARGET, *REALTIME]).reset_index(drop=True)
        for model_name, template in models().items():
            probabilities = pd.Series(np.nan, index=sample.index)
            fold_ids = pd.Series(pd.NA, index=sample.index, dtype="Int64")
            for fold, (train, test) in enumerate(splitter.split(sample), start=1):
                estimator = clone(template)
                estimator.fit(sample.loc[train, features], sample.loc[train, TARGET].astype(int))
                probabilities.loc[test] = estimator.predict_proba(sample.loc[test, features])[:, 1]
                fold_ids.loc[test] = fold
                permutation = permutation_importance(
                    estimator, sample.loc[test, features], sample.loc[test, TARGET].astype(int),
                    scoring="roc_auc", n_repeats=10, random_state=20260616 + fold,
                )
                for feature, mean, std in zip(features, permutation.importances_mean, permutation.importances_std):
                    importance_rows.append({
                        "model": model_name, "feature_set": feature_set, "fold": fold,
                        "feature": feature, "importance_mean": mean, "importance_std": std,
                    })
            valid = probabilities.notna()
            truth = sample.loc[valid, TARGET].astype(int)
            pred = (probabilities.loc[valid] >= 0.5).astype(int)
            metric_rows.append({
                "model": model_name, "feature_set": feature_set, "observations": int(valid.sum()),
                "auc": roc_auc_score(truth, probabilities.loc[valid]),
                "accuracy": accuracy_score(truth, pred), "f1": f1_score(truth, pred),
                "brier": brier_score_loss(truth, probabilities.loc[valid]),
                "log_loss": log_loss(truth, probabilities.loc[valid]),
            })
            for idx in sample.index[valid]:
                prediction_rows.append({
                    "trade_date": sample.loc[idx, "trade_date"], "model": model_name,
                    "feature_set": feature_set, "fold": int(fold_ids.loc[idx]),
                    "truth": int(sample.loc[idx, TARGET]), "probability_up": probabilities.loc[idx],
                })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), pd.DataFrame(importance_rows)


def shap_diagnostic(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    sample = data.dropna(subset=[TARGET, *REALTIME]).reset_index(drop=True)
    train = sample.iloc[:-250]
    explain = sample.iloc[-250:]
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_explain = imputer.transform(explain[features])
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_depth=3, l2_regularization=1.0)
    model.fit(x_train, train[TARGET].astype(int))
    explainer = shap.Explainer(model)
    values = explainer(x_explain).values
    if values.ndim == 3:
        values = values[:, :, -1]
    rows = []
    for feature, mean_abs, signed_mean in zip(features, np.abs(values).mean(axis=0), values.mean(axis=0)):
        rows.append({"feature": feature, "mean_abs_shap": mean_abs, "mean_signed_shap": signed_mean})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def main() -> None:
    data = pd.read_csv(PROCESSED / "model_daily_dataset.csv", parse_dates=["trade_date"])
    for source in CHANNEL_SOURCE:
        data[f"{source}_lag1"] = data[source].shift(1)
    channel_features = [f"{source}_lag1" for source in CHANNEL_SOURCE]
    feature_sets = {
        "technical_only": TECHNICAL,
        "technical_plus_realtime_index": TECHNICAL + REALTIME,
        "technical_plus_realtime_channels": TECHNICAL + REALTIME + channel_features,
    }
    metrics, predictions, importance = evaluate(data, feature_sets)
    shap_values = shap_diagnostic(data, feature_sets["technical_plus_realtime_channels"])
    metrics.to_csv(PROCESSED / "realtime_prediction_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(PROCESSED / "realtime_prediction_oos.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(PROCESSED / "realtime_prediction_permutation_importance.csv", index=False, encoding="utf-8-sig")
    shap_values.to_csv(PROCESSED / "realtime_prediction_shap.csv", index=False, encoding="utf-8-sig")

    aggregate_importance = importance.groupby(["model", "feature_set", "feature"], as_index=False)["importance_mean"].mean()
    top = aggregate_importance[
        (aggregate_importance["model"] == "hist_gradient_boosting")
        & (aggregate_importance["feature_set"] == "technical_plus_realtime_channels")
    ].nlargest(12, "importance_mean")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    top.sort_values("importance_mean").plot.barh(x="feature", y="importance_mean", ax=axes[0], legend=False)
    axes[0].set_title("Out-of-sample permutation importance")
    shap_values.head(12).sort_values("mean_abs_shap").plot.barh(x="feature", y="mean_abs_shap", ax=axes[1], legend=False, color="#d95f02")
    axes[1].set_title("SHAP importance (final holdout diagnostic)")
    fig.tight_layout()
    fig.savefig(REPORTS / "realtime_prediction_explainability.png", dpi=180)
    plt.close(fig)

    best = metrics.sort_values("auc", ascending=False).iloc[0]
    report = [
        "# 实时指数严格时序预测", "",
        "使用5折扩展时间序列切分和20个交易日间隔。实时情绪指数按月仅使用历史数据估计，进入预测时再滞后1个交易日。未知未来收益保持为缺失值，不参与评估。", "",
        "## 样本外指标", "", metrics.to_markdown(index=False), "",
        f"最高AUC为{best['auc']:.4f}，对应{best['model']} / {best['feature_set']}。", "",
        "## SHAP全局重要性", "", shap_values.head(15).to_markdown(index=False), "",
        "SHAP使用最后250个可用观测作为诊断集；更严格的主要解释依据是各时间折测试集上的置换重要性。任何接近0.5的AUC都应解释为预测能力有限，而不是可交易策略。", "",
        "![预测解释](realtime_prediction_explainability.png)",
    ]
    (REPORTS / "realtime_prediction.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(shap_values.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
