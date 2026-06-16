from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CATEGORICAL = ["ts_code", "industry_role", "tier"]
TECHNICAL = [
    "stock_return_lag_1", "stock_return_lag_2", "stock_return_lag_5", "stock_return_lag_10", "stock_return_lag_20",
    "excess_return_lag_1", "excess_return_lag_2", "excess_return_lag_5", "excess_return_lag_10", "excess_return_lag_20",
    "stock_momentum_5d", "stock_momentum_10d", "stock_momentum_20d", "stock_momentum_60d",
    "stock_volatility_5d", "stock_volatility_10d", "stock_volatility_20d", "stock_volatility_60d",
    "volume_z_5d", "volume_z_10d", "volume_z_20d", "volume_z_60d",
    "intraday_range", "overnight_gap", "hs300_return_lag1", "hs300_volatility_20d",
    "rank_stock_return_lag_1", "rank_excess_return_lag_5", "rank_stock_momentum_20d",
    "rank_stock_volatility_20d", "rank_volume_z_20d",
]
GLOBAL = [
    "realtime_sentiment_lag1", "realtime_sentiment_change_lag1",
    "policy_support_lag1", "risk_impulse_lag1", "final_policy_event_count_lag1",
]
EVENT = [
    "company_event_count", "company_event_signed_impulse", "company_event_risk_impulse",
    "company_event_amount_log", "company_event_initial_count", "company_event_intensity",
    "company_event_count_ewm20", "company_event_signed_impulse_ewm20", "company_event_risk_impulse_ewm20",
    "company_event_amount_log_ewm20", "company_event_initial_count_ewm20", "company_event_intensity_ewm20",
    "days_since_company_event", "event_window_20d",
]
TARGETS = {
    "outperform_5d": "future_outperform_5d",
    "extreme_relative_5d": "future_extreme_relative_5d",
    "volatility_jump_5d": "future_volatility_jump_5d",
}


def date_folds(dates: pd.Series, splits: int = 5, gap: int = 20) -> list[tuple[np.ndarray, np.ndarray, int]]:
    unique = np.array(sorted(pd.to_datetime(dates.unique())))
    test_size = len(unique) // (splits + 1)
    folds = []
    for fold in range(1, splits + 1):
        test_start = len(unique) - (splits - fold + 1) * test_size
        test_end = min(test_start + test_size, len(unique))
        train_end = max(0, test_start - gap)
        train_dates, test_dates = unique[:train_end], unique[test_start:test_end]
        train_idx = dates[dates.isin(train_dates)].index.to_numpy()
        test_idx = dates[dates.isin(test_dates)].index.to_numpy()
        folds.append((train_idx, test_idx, fold))
    return folds


def pipelines(numeric: list[str]) -> dict:
    logistic_pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    dense_pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return {
        "logistic": make_pipeline(logistic_pre, LogisticRegression(max_iter=3000, class_weight="balanced", C=0.2)),
        "hist_gradient_boosting": make_pipeline(
            dense_pre, HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_depth=4, l2_regularization=2.0)
        ),
        "extra_trees": make_pipeline(
            dense_pre,
            ExtraTreesClassifier(
                n_estimators=400, max_depth=10, min_samples_leaf=20, max_features=0.7,
                class_weight="balanced", n_jobs=-1, random_state=20260616,
            ),
        ),
    }


def metrics(truth: pd.Series, probability: pd.Series) -> dict:
    prediction = probability.ge(0.5).astype(int)
    return {
        "observations": len(truth), "positive_share": truth.mean(),
        "auc": roc_auc_score(truth, probability), "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction), "f1": f1_score(truth, prediction),
        "brier": brier_score_loss(truth, probability),
    }


def main() -> None:
    panel = pd.read_csv(PROCESSED / "stock_panel_daily.csv", parse_dates=["trade_date"])
    panel = panel[panel["realtime_sentiment_lag1"].notna()].reset_index(drop=True)
    feature_sets = {
        "technical": TECHNICAL,
        "technical_global": TECHNICAL + GLOBAL,
        "technical_events": TECHNICAL + EVENT,
        "technical_global_events": TECHNICAL + GLOBAL + EVENT,
    }
    metric_rows, prediction_rows = [], []
    for target_name, target in TARGETS.items():
        sample = panel.dropna(subset=[target]).reset_index(drop=True)
        folds = date_folds(sample["trade_date"])
        for feature_set, numeric in feature_sets.items():
            features = numeric + CATEGORICAL
            for model_name, template in pipelines(numeric).items():
                probabilities = pd.Series(np.nan, index=sample.index)
                fold_ids = pd.Series(pd.NA, index=sample.index, dtype="Int64")
                for train, test, fold in folds:
                    estimator = clone(template)
                    estimator.fit(sample.loc[train, features], sample.loc[train, target].astype(int))
                    probabilities.loc[test] = estimator.predict_proba(sample.loc[test, features])[:, 1]
                    fold_ids.loc[test] = fold
                valid = probabilities.notna()
                truth = sample.loc[valid, target].astype(int)
                overall = metrics(truth, probabilities.loc[valid])
                metric_rows.append({"target": target_name, "subset": "all", "model": model_name, "feature_set": feature_set, **overall})
                event_mask = valid & sample["event_window_20d"].eq(1)
                if event_mask.sum() >= 30 and sample.loc[event_mask, target].nunique() == 2:
                    event_result = metrics(sample.loc[event_mask, target].astype(int), probabilities.loc[event_mask])
                    metric_rows.append({"target": target_name, "subset": "event_window_20d", "model": model_name, "feature_set": feature_set, **event_result})
                for idx in sample.index[valid]:
                    prediction_rows.append({
                        "trade_date": sample.loc[idx, "trade_date"], "ts_code": sample.loc[idx, "ts_code"],
                        "name": sample.loc[idx, "name"], "target": target_name, "model": model_name,
                        "feature_set": feature_set, "fold": int(fold_ids.loc[idx]),
                        "truth": int(sample.loc[idx, target]), "probability": probabilities.loc[idx],
                        "event_window_20d": int(sample.loc[idx, "event_window_20d"]),
                    })
    result = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    result.to_csv(PROCESSED / "stock_panel_prediction_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(PROCESSED / "stock_panel_prediction_oos.csv", index=False, encoding="utf-8-sig")

    all_results = result[result["subset"].eq("all")].copy()
    best = all_results.sort_values(["target", "auc"], ascending=[True, False]).groupby("target").head(1)
    fig = plt.figure(figsize=(11, 5))
    chart = all_results[all_results["target"].eq("extreme_relative_5d")].copy()
    chart["label"] = chart["model"] + "\n" + chart["feature_set"]
    plt.barh(chart["label"], chart["auc"], color=["#4C78A8" if "events" not in value else "#F58518" for value in chart["feature_set"]])
    plt.axvline(0.5, color="black", linewidth=0.8)
    plt.title("Panel model: extreme relative return classification")
    plt.xlabel("Out-of-sample AUC")
    plt.tight_layout()
    fig.savefig(REPORTS / "stock_panel_prediction.png", dpi=180)
    plt.close(fig)

    report = [
        "# 个股面板严格时序预测", "",
        "所有股票按交易日期整体进入训练集或测试集，使用5折扩展窗口和20个交易日间隔。公司代码、产业角色和样本层级以独热编码进入模型。", "",
        "## 各目标最佳结果", "", best.to_markdown(index=False), "",
        "## 全部结果", "", result.to_markdown(index=False), "",
        "横截面极端强弱目标只保留每个交易日未来异常收益排名处于上下30%的股票，减少接近零收益的标签噪声。事件窗口结果样本较少，必须与全样本指标共同解释。", "",
        "![面板预测](stock_panel_prediction.png)",
    ]
    (REPORTS / "stock_panel_prediction.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
