from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from statsmodels.regression.quantile_regression import QuantReg


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
RNG = np.random.default_rng(20260616)


def robust_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    median = frame.median()
    scale = ((frame.quantile(0.75) - frame.quantile(0.25)) / 1.349).replace(0, np.nan)
    scale = scale.fillna(frame.std()).replace(0, 1).fillna(1)
    return ((frame - median) / scale).clip(-8, 8)


def orient(values: pd.Series, channels: pd.DataFrame) -> pd.Series:
    return -values if values.corr(channels.mean(axis=1)) < 0 else values


def alternative_index(daily: pd.DataFrame, company_half_life: int, policy_half_life: int) -> pd.Series:
    channels = pd.DataFrame(index=daily.index)
    channels["policy_support"] = daily["policy_support_impulse"].ewm(halflife=policy_half_life, adjust=False).mean()
    channels["company_sentiment"] = daily["company_sentiment_impulse"].ewm(halflife=company_half_life, adjust=False).mean()
    channels["confidence_balance"] = daily["confidence_impulse"].ewm(halflife=company_half_life, adjust=False).mean()
    channels["signed_breadth"] = daily["signed_breadth_impulse"].ewm(halflife=max(5, company_half_life // 2), adjust=False).mean()
    standardized = robust_zscore(channels)
    values = pd.Series(PCA(n_components=1).fit_transform(standardized).ravel(), index=daily.index)
    values = orient(values, standardized)
    return (values - values.mean()) / values.std()


def quantile_coefficient(data: pd.DataFrame, index: pd.Series, quantile: float) -> tuple[float, float]:
    work = data.copy()
    work["alternative_index"] = index
    columns = ["future_return_5d", "alternative_index", "return_lag_1", "HS300", "volatility_20d"]
    sample = work[columns].dropna()
    result = QuantReg(sample["future_return_5d"], sm.add_constant(sample[columns[1:]])).fit(q=quantile, max_iter=10000)
    return float(result.params["alternative_index"]), float(result.pvalues["alternative_index"])


def placebo_event_study(data: pd.DataFrame, simulations: int = 1000) -> pd.DataFrame:
    threshold = data["dynamic_sentiment_change"].quantile(0.95)
    candidates = data.index[data["dynamic_sentiment_change"] >= threshold].tolist()
    selected = []
    for idx in candidates:
        if not selected or idx - selected[-1] >= 20:
            selected.append(idx)
    eligible = np.arange(0, len(data) - 21)
    rows = []
    for horizon in [1, 5, 10, 20]:
        abnormal = []
        for idx in range(len(data) - horizon):
            index_return = data.loc[idx + horizon, "index_level"] / data.loc[idx, "index_level"] - 1
            benchmark_return = np.prod(1 + data.loc[idx + 1:idx + horizon, "HS300"].fillna(0)) - 1
            abnormal.append(index_return - benchmark_return)
        abnormal = np.asarray(abnormal)
        actual = float(np.mean([abnormal[idx] for idx in selected if idx < len(abnormal)]))
        placebo_means = np.asarray([np.mean(abnormal[RNG.choice(eligible[eligible < len(abnormal)], len(selected), replace=False)]) for _ in range(simulations)])
        two_sided = (np.sum(np.abs(placebo_means) >= abs(actual)) + 1) / (simulations + 1)
        rows.append({
            "horizon": horizon, "events": len(selected), "actual_mean_abnormal_return": actual,
            "placebo_mean": placebo_means.mean(), "placebo_std": placebo_means.std(ddof=1),
            "placebo_p_value_two_sided": two_sided,
        })
    return pd.DataFrame(rows)


def main() -> None:
    daily = pd.read_csv(PROCESSED / "dynamic_sentiment_index_daily.csv", parse_dates=["trade_date"])
    model = pd.read_csv(PROCESSED / "model_daily_dataset.csv", parse_dates=["trade_date"])
    active_mask = daily["trade_date"] >= "2021-01-06"
    daily = daily.loc[active_mask].reset_index(drop=True)
    model = model[model["trade_date"] >= "2021-01-06"].reset_index(drop=True)
    base = (daily["dynamic_sentiment_filtered"] - daily["dynamic_sentiment_filtered"].mean()) / daily["dynamic_sentiment_filtered"].std()

    specifications = [(10, 30), (10, 60), (20, 30), (20, 60), (20, 90), (40, 60), (40, 90)]
    rows = []
    for company_half_life, policy_half_life in specifications:
        alternative = alternative_index(daily, company_half_life, policy_half_life)
        q50, p50 = quantile_coefficient(model, alternative, 0.5)
        q90, p90 = quantile_coefficient(model, alternative, 0.9)
        rows.append({
            "company_half_life": company_half_life, "policy_half_life": policy_half_life,
            "correlation_with_dynamic_index": alternative.corr(base),
            "q50_coefficient": q50, "q50_p_value": p50,
            "q90_coefficient": q90, "q90_p_value": p90,
        })
    sensitivity = pd.DataFrame(rows)

    z_columns = ["policy_support_z", "company_sentiment_z", "confidence_balance_z", "signed_breadth_z"]
    equal_weight = daily[z_columns].mean(axis=1)
    alternatives = pd.DataFrame({
        "pair": ["dynamic_vs_pca", "dynamic_vs_equal_weight", "pca_vs_equal_weight"],
        "correlation": [
            daily["dynamic_sentiment_filtered"].corr(daily["pca_sentiment_index"]),
            base.corr(equal_weight),
            daily["pca_sentiment_index"].corr(equal_weight),
        ],
    })
    placebo = placebo_event_study(model)
    sensitivity.to_csv(PROCESSED / "dynamic_index_half_life_sensitivity.csv", index=False, encoding="utf-8-sig")
    alternatives.to_csv(PROCESSED / "dynamic_index_alternative_correlations.csv", index=False, encoding="utf-8-sig")
    placebo.to_csv(PROCESSED / "dynamic_index_placebo_event_study.csv", index=False, encoding="utf-8-sig")

    report = [
        "# 动态情绪指数稳健性检验", "",
        "## 替代半衰期", "", sensitivity.to_markdown(index=False), "",
        "## 替代合成方法", "", alternatives.to_markdown(index=False), "",
        "## 随机事件日安慰剂", "", placebo.to_markdown(index=False), "",
        "半衰期检验使用PCA替代指数，以降低重复估计动态因子造成的数值噪声。安慰剂检验随机抽取与真实高冲击事件相同数量的日期，报告双侧经验p值。",
    ]
    (REPORTS / "dynamic_index_robustness.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(sensitivity.to_string(index=False))
    print(alternatives.to_string(index=False))
    print(placebo.to_string(index=False))


if __name__ == "__main__":
    main()
