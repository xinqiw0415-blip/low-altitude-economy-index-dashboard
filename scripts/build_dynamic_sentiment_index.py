from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def next_trading_date(values: pd.Series, calendar: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    return dates.map(lambda value: calendar[calendar > value].iloc[0] if (calendar > value).any() else pd.NaT)


def signed_direction(value: str) -> float:
    return {"positive": 1.0, "negative": -1.0, "mixed": -0.25, "neutral": 0.0}.get(str(value), 0.0)


def robust_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    median = frame.median()
    scale = (frame.quantile(0.75) - frame.quantile(0.25)) / 1.349
    scale = scale.mask(scale < 1e-8, frame.std()).replace(0, 1).fillna(1)
    return (frame - median) / scale


def scale_index(values: pd.Series) -> pd.Series:
    std = values.std()
    standardized = (values - values.mean()) / (std if std > 1e-8 else 1.0)
    return (50 + 10 * standardized).clip(0, 100)


def main() -> None:
    market = pd.read_csv(ROOT / "data" / "interim" / "market_daily.csv", parse_dates=["trade_date"])
    calendar = pd.Series(pd.to_datetime(sorted(market["trade_date"].unique())))
    company = pd.read_csv(PROCESSED / "final_company_events_hybrid.csv")
    policy = pd.read_csv(PROCESSED / "final_policy_events_hybrid.csv")
    company = company[company["accepted"].astype(bool)].copy()
    policy = policy[policy["accepted"].astype(bool)].copy()

    company["trade_date"] = next_trading_date(company["event_date"], calendar)
    policy["trade_date"] = next_trading_date(policy["event_date"], calendar)
    company["certainty"] = 1 - (company["uncertainty"].clip(1, 5) - 1) / 5
    policy["certainty"] = 1 - (policy["uncertainty"].clip(1, 5) - 1) / 5
    company["signed_score"] = company["direction"].map(signed_direction) * company["intensity"] * company["certainty"]
    company["signed_count"] = company["direction"].map(signed_direction)
    policy["support_score"] = policy["intensity"] * policy["certainty"] * (0.8 + 0.1 * policy["novelty"].clip(1, 5))
    policy["signed_count"] = policy["direction"].map(signed_direction)
    company["confidence_score"] = company["direction"].map(signed_direction) * company["intensity"] * company["certainty"] ** 2
    policy["confidence_score"] = policy["intensity"] * policy["certainty"] ** 2
    company["risk_score"] = company["intensity"] * (company["uncertainty"] / 5) * company["direction"].map(
        {"negative": 1.0, "mixed": 0.7, "neutral": 0.2, "positive": 0.1}
    ).fillna(0.2)
    policy["risk_score"] = policy["intensity"] * (policy["uncertainty"] / 5) * 0.25

    company_daily = company.groupby("trade_date").agg(
        company_sentiment_impulse=("signed_score", "sum"),
        company_risk_impulse=("risk_score", "sum"),
        company_attention=("event_id", "count"),
        company_signed_count=("signed_count", "sum"),
        company_confidence_impulse=("confidence_score", "sum"),
    ).reset_index()
    policy_daily = policy.groupby("trade_date").agg(
        policy_support_impulse=("support_score", "sum"),
        policy_risk_impulse=("risk_score", "sum"),
        policy_attention=("event_id", "count"),
        policy_signed_count=("signed_count", "sum"),
        policy_confidence_impulse=("confidence_score", "sum"),
    ).reset_index()
    daily = pd.DataFrame({"trade_date": calendar}).merge(company_daily, how="left").merge(policy_daily, how="left")
    impulse_columns = [column for column in daily if column != "trade_date"]
    daily[impulse_columns] = daily[impulse_columns].fillna(0)
    daily["risk_impulse"] = daily["company_risk_impulse"] + daily["policy_risk_impulse"]
    daily["attention_impulse"] = np.log1p(daily["company_attention"] + daily["policy_attention"])
    daily["confidence_impulse"] = daily["company_confidence_impulse"] + daily["policy_confidence_impulse"]
    daily["signed_breadth_impulse"] = daily["company_signed_count"] + daily["policy_signed_count"]

    channels = pd.DataFrame(index=daily.index)
    channels["policy_support"] = daily["policy_support_impulse"].ewm(halflife=60, adjust=False).mean()
    channels["company_sentiment"] = daily["company_sentiment_impulse"].ewm(halflife=20, adjust=False).mean()
    channels["confidence_balance"] = daily["confidence_impulse"].ewm(halflife=20, adjust=False).mean()
    channels["signed_breadth"] = daily["signed_breadth_impulse"].ewm(halflife=10, adjust=False).mean()

    first_active = daily.index[(daily["company_attention"] + daily["policy_attention"]) > 0].min()
    active = channels.loc[first_active:].copy()
    standardized = robust_zscore(active).clip(-8, 8)
    pca = PCA(n_components=1).fit(standardized)
    pca_factor = pd.Series(pca.transform(standardized).ravel(), index=standardized.index)
    if pca_factor.corr(standardized.mean(axis=1)) < 0:
        pca_factor *= -1
        pca.components_ *= -1

    model = DynamicFactor(standardized, k_factors=1, factor_order=1, error_order=1)
    result = model.fit(method="powell", maxiter=1000, disp=False)
    filtered = pd.Series(result.factors.filtered[0], index=standardized.index)
    smoothed = pd.Series(result.factors.smoothed[0], index=standardized.index)
    orientation = standardized.mean(axis=1)
    if smoothed.corr(orientation) < 0:
        filtered *= -1
        smoothed *= -1

    factor_se = pd.Series(
        np.sqrt(np.maximum(result.filter_results.smoothed_state_cov[0, 0, :], 0)),
        index=standardized.index,
    )
    output = daily.copy()
    for column in channels:
        output[column] = channels[column]
        output[f"{column}_z"] = np.nan
        output.loc[standardized.index, f"{column}_z"] = standardized[column]
    output["pca_sentiment_index"] = 50.0
    output["dynamic_sentiment_filtered"] = 50.0
    output["dynamic_sentiment_smoothed"] = 50.0
    output["dynamic_factor_se"] = np.nan
    output.loc[standardized.index, "pca_sentiment_index"] = scale_index(pca_factor)
    output.loc[standardized.index, "dynamic_sentiment_filtered"] = scale_index(filtered)
    output.loc[standardized.index, "dynamic_sentiment_smoothed"] = scale_index(smoothed)
    output.loc[standardized.index, "dynamic_factor_se"] = factor_se
    output["dynamic_sentiment_change"] = output["dynamic_sentiment_filtered"].diff().fillna(0)
    output.to_csv(PROCESSED / "dynamic_sentiment_index_daily.csv", index=False, encoding="utf-8-sig")

    loadings = pd.DataFrame({
        "channel": standardized.columns,
        "pca_loading": pca.components_[0],
        "dynamic_factor_loading": [result.params.get(f"loading.f1.{column}", np.nan) for column in standardized.columns],
    })
    loadings.to_csv(PROCESSED / "dynamic_sentiment_loadings.csv", index=False, encoding="utf-8-sig")
    diagnostics = pd.DataFrame([{
        "active_start": output.loc[first_active, "trade_date"],
        "observations": len(standardized),
        "company_events": len(company),
        "policy_events": len(policy),
        "pca_explained_variance": pca.explained_variance_ratio_[0],
        "dynamic_factor_converged": bool(result.mle_retvals.get("converged", False)),
        "aic": result.aic,
        "bic": result.bic,
        "pca_dynamic_correlation": output.loc[standardized.index, "pca_sentiment_index"].corr(
            output.loc[standardized.index, "dynamic_sentiment_smoothed"]
        ),
    }])
    diagnostics.to_csv(PROCESSED / "dynamic_sentiment_diagnostics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 5))
    chart = output.loc[first_active:]
    ax.plot(chart["trade_date"], chart["dynamic_sentiment_smoothed"], label="Dynamic factor (smoothed)", linewidth=1.8)
    ax.plot(chart["trade_date"], chart["pca_sentiment_index"], label="PCA benchmark", linewidth=1, alpha=0.65)
    ax.axhline(50, color="black", linewidth=0.8, alpha=0.5)
    ax.set(title="Low-altitude Economy Dynamic Sentiment Index", ylabel="Index (mean=50)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "dynamic_sentiment_index.png", dpi=180)
    plt.close(fig)

    report = [
        "# 低空经济动态情绪指数", "",
        "## 构造方法", "",
        "指数由政策支持、企业景气、置信度余额和有方向的事件广度四个通道构成。公司事件采用方向、强度和不确定性加权；政策事件进一步加入新颖度。事件冲击通过指数衰减转化为日频状态，再使用单因子动态因子模型提取共同变化。风险压力保留在日频明细中，但不作为无方向的活跃度信号进入因子。", "",
        "- 企业事件影响半衰期：20个交易日。",
        "- 政策支持影响半衰期：60个交易日。",
        "- 置信度余额半衰期：20个交易日。",
        "- 有方向事件广度半衰期：10个交易日。", "",
        "## 载荷", "", loadings.to_markdown(index=False), "",
        "## 诊断", "", diagnostics.to_markdown(index=False), "",
        "`dynamic_sentiment_smoothed`适合历史解释和可视化；`dynamic_sentiment_filtered`适合时序统计分析。模型参数仍由全样本估计，严格预测时需要滚动重估。指数在首个有效事件之前设为中性值50。", "",
        "![动态情绪指数](dynamic_sentiment_index.png)",
    ]
    (REPORTS / "dynamic_sentiment_index.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(diagnostics.to_string(index=False))
    print(loadings.to_string(index=False))


if __name__ == "__main__":
    main()
