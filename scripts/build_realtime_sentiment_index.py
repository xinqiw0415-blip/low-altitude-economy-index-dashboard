from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHANNELS = ["policy_support", "company_sentiment", "confidence_balance", "signed_breadth"]
MIN_TRAIN = 252


def fit_snapshot(history: pd.DataFrame) -> dict:
    median = history.median()
    scale = ((history.quantile(0.75) - history.quantile(0.25)) / 1.349).replace(0, np.nan)
    scale = scale.fillna(history.std()).replace(0, 1).fillna(1)
    standardized = ((history - median) / scale).clip(-8, 8)
    pca = PCA(n_components=1).fit(standardized)
    factor = pd.Series(pca.transform(standardized).ravel(), index=history.index)
    loadings = pca.components_[0].copy()
    if factor.corr(standardized.mean(axis=1)) < 0:
        factor *= -1
        loadings *= -1
    factor_std = factor.std()
    return {
        "median": median,
        "scale": scale,
        "pca": pca,
        "sign": 1 if np.allclose(loadings, pca.components_[0]) else -1,
        "factor_mean": factor.mean(),
        "factor_std": factor_std if factor_std > 1e-8 else 1.0,
        "loadings": loadings,
        "explained_variance": pca.explained_variance_ratio_[0],
    }


def main() -> None:
    data = pd.read_csv(PROCESSED / "dynamic_sentiment_index_daily.csv", parse_dates=["trade_date"])
    active = data.index[data[CHANNELS].abs().sum(axis=1) > 0]
    first_active = int(active.min())
    output = data[["trade_date"]].copy()
    output["realtime_sentiment_index"] = np.nan
    output["realtime_sentiment_z"] = np.nan
    output["realtime_model_train_end"] = pd.NaT
    output["realtime_explained_variance"] = np.nan
    output["realtime_available"] = False
    loading_rows = []
    snapshot = None
    snapshot_month = None

    for idx in range(first_active + MIN_TRAIN, len(data)):
        date = data.loc[idx, "trade_date"]
        month = date.to_period("M")
        if snapshot is None or month != snapshot_month:
            history = data.loc[first_active:idx - 1, CHANNELS]
            snapshot = fit_snapshot(history)
            snapshot_month = month
            loading_rows.append({
                "effective_month": str(month),
                "train_start": data.loc[first_active, "trade_date"],
                "train_end": data.loc[idx - 1, "trade_date"],
                "observations": len(history),
                "explained_variance": snapshot["explained_variance"],
                **{f"loading_{channel}": value for channel, value in zip(CHANNELS, snapshot["loadings"])},
            })
        row = data.loc[[idx], CHANNELS]
        standardized = ((row - snapshot["median"]) / snapshot["scale"]).clip(-8, 8)
        raw = float(snapshot["pca"].transform(standardized)[0, 0]) * snapshot["sign"]
        value_z = (raw - snapshot["factor_mean"]) / snapshot["factor_std"]
        output.loc[idx, "realtime_sentiment_z"] = value_z
        output.loc[idx, "realtime_sentiment_index"] = np.clip(50 + 10 * value_z, 0, 100)
        output.loc[idx, "realtime_model_train_end"] = data.loc[idx - 1, "trade_date"]
        output.loc[idx, "realtime_explained_variance"] = snapshot["explained_variance"]
        output.loc[idx, "realtime_available"] = True

    output["realtime_sentiment_change"] = output["realtime_sentiment_index"].diff()
    output["realtime_sentiment_lag1"] = output["realtime_sentiment_index"].shift(1)
    output["realtime_sentiment_change_lag1"] = output["realtime_sentiment_change"].shift(1)
    output.to_csv(PROCESSED / "realtime_sentiment_index_daily.csv", index=False, encoding="utf-8-sig")
    loadings = pd.DataFrame(loading_rows)
    loadings.to_csv(PROCESSED / "realtime_sentiment_loadings.csv", index=False, encoding="utf-8-sig")

    valid = output[output["realtime_available"]]
    correlation = valid["realtime_sentiment_index"].corr(
        data.loc[valid.index, "dynamic_sentiment_filtered"]
    )
    report = [
        "# 实时扩展窗口情绪指数", "",
        "该指数用于严格时序预测。四个事件通道本身只依赖当日及历史事件；PCA载荷和稳健标准化参数在每月首个交易日使用上一交易日以前的数据重新估计。预测模型使用指数的一日滞后值。", "",
        f"- 最小训练窗口：{MIN_TRAIN}个交易日。",
        f"- 首个可用日期：{valid['trade_date'].min().date() if len(valid) else 'NA'}。",
        f"- 可用观测：{len(valid)}。",
        f"- 与全样本过滤指数相关系数：{correlation:.4f}。",
        f"- 月度重估次数：{len(loadings)}。", "",
        "全样本动态因子指数继续用于历史解释；本实时PCA指数用于预测和可复现的无参数前视对照。",
    ]
    (ROOT / "reports" / "realtime_sentiment_index.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Realtime observations: {len(valid)}, correlation with full-sample index: {correlation:.4f}")


if __name__ == "__main__":
    main()
