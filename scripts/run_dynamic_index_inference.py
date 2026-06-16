from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def stationarity(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["dynamic_sentiment_filtered", "dynamic_sentiment_change", "return_ew", "excess_return_hs300"]:
        values = data[column].dropna()
        statistic, p_value, used_lag, nobs, *_ = adfuller(values, autolag="AIC")
        rows.append({"variable": column, "adf_statistic": statistic, "p_value": p_value, "used_lag": used_lag, "nobs": nobs})
    return pd.DataFrame(rows)


def local_projections(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    work["sentiment_shock_z"] = zscore(work["dynamic_sentiment_change"])
    rows = []
    for horizon in range(1, 21):
        work["outcome"] = work["index_level"].shift(-horizon).div(work["index_level"]).sub(1)
        x_columns = ["sentiment_shock_z", "return_lag_1", "HS300", "volatility_20d"]
        sample = work[["outcome", *x_columns]].dropna()
        result = sm.OLS(sample["outcome"], sm.add_constant(sample[x_columns])).fit(
            cov_type="HAC", cov_kwds={"maxlags": horizon + 1}
        )
        coefficient = result.params["sentiment_shock_z"]
        se = result.bse["sentiment_shock_z"]
        rows.append({
            "horizon": horizon, "coefficient": coefficient, "std_error": se,
            "ci_low": coefficient - 1.96 * se, "ci_high": coefficient + 1.96 * se,
            "p_value": result.pvalues["sentiment_shock_z"], "nobs": int(result.nobs),
        })
    return pd.DataFrame(rows)


def quantile_regression(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    work["sentiment_level_z"] = zscore(work["dynamic_sentiment_filtered"])
    features = ["sentiment_level_z", "return_lag_1", "HS300", "volatility_20d"]
    sample = work[["future_return_5d", *features]].dropna()
    x = sm.add_constant(sample[features])
    rows = []
    for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
        result = QuantReg(sample["future_return_5d"], x).fit(q=quantile, max_iter=10000)
        coefficient = result.params["sentiment_level_z"]
        se = result.bse["sentiment_level_z"]
        rows.append({
            "quantile": quantile, "coefficient": coefficient, "std_error": se,
            "ci_low": coefficient - 1.96 * se, "ci_high": coefficient + 1.96 * se,
            "p_value": result.pvalues["sentiment_level_z"], "nobs": int(result.nobs),
        })
    output = pd.DataFrame(rows)
    output["p_value_fdr"] = multipletests(output["p_value"], method="fdr_bh")[1]
    return output


def var_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sample = data[["dynamic_sentiment_change", "excess_return_hs300", "volatility_20d"]].dropna().copy()
    sample.columns = ["sentiment_shock", "excess_return", "volatility"]
    sample = (sample - sample.mean()) / sample.std()
    result = VAR(sample).fit(maxlags=10, ic="aic", trend="c")
    irf_values = result.irf(20).orth_irfs[:, 1, 0]
    irf = pd.DataFrame({
        "horizon": range(len(irf_values)),
        "excess_return_response_to_sentiment_shock": irf_values,
        "selected_lag": result.k_ar,
    })
    tests = []
    for caused, causing in [("excess_return", ["sentiment_shock"]), ("sentiment_shock", ["excess_return"])]:
        test = result.test_causality(caused, causing, kind="wald")
        tests.append({
            "caused": caused, "causing": causing[0], "test_statistic": float(test.test_statistic),
            "p_value": float(test.pvalue), "df": str(test.df),
        })
    diagnostics = {"selected_lag": result.k_ar, "stable": bool(result.is_stable()), "nobs": int(result.nobs), "aic": result.aic, "bic": result.bic}
    return irf, pd.DataFrame(tests), diagnostics


def shock_event_study(data: pd.DataFrame) -> pd.DataFrame:
    active = data[data["dynamic_sentiment_filtered"].ne(50)].copy()
    threshold = active["dynamic_sentiment_change"].quantile(0.95)
    candidates = active.index[active["dynamic_sentiment_change"] >= threshold].tolist()
    selected = []
    for idx in candidates:
        if not selected or idx - selected[-1] >= 20:
            selected.append(idx)
    rows = []
    for horizon in [0, 1, 5, 10, 20]:
        values = []
        for idx in selected:
            if idx + horizon >= len(data):
                continue
            if horizon == 0:
                values.append(float(data.loc[idx, "excess_return_hs300"]))
            else:
                index_return = data.loc[idx + horizon, "index_level"] / data.loc[idx, "index_level"] - 1
                benchmark_return = np.prod(1 + data.loc[idx + 1:idx + horizon, "HS300"].fillna(0)) - 1
                values.append(float(index_return - benchmark_return))
        array = np.asarray(values)
        se = array.std(ddof=1) / np.sqrt(len(array)) if len(array) > 1 else np.nan
        rows.append({
            "horizon": horizon, "events": len(array), "mean_abnormal_return": array.mean() if len(array) else np.nan,
            "std_error": se, "ci_low": array.mean() - 1.96 * se if len(array) > 1 else np.nan,
            "ci_high": array.mean() + 1.96 * se if len(array) > 1 else np.nan,
            "shock_threshold": threshold,
        })
    return pd.DataFrame(rows)


def save_plots(lp: pd.DataFrame, qr: pd.DataFrame, irf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(lp["horizon"], lp["coefficient"], color="#1f77b4")
    axes[0].fill_between(lp["horizon"], lp["ci_low"], lp["ci_high"], alpha=0.2)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Local projection")
    axes[1].plot(qr["quantile"], qr["coefficient"], marker="o", color="#d62728")
    axes[1].fill_between(qr["quantile"], qr["ci_low"], qr["ci_high"], alpha=0.2)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Quantile regression")
    axes[2].plot(irf["horizon"], irf["excess_return_response_to_sentiment_shock"], color="#2ca02c")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("VAR impulse response")
    fig.tight_layout()
    fig.savefig(REPORTS / "dynamic_index_inference.png", dpi=180)
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(PROCESSED / "model_daily_dataset.csv", parse_dates=["trade_date"])
    active = data[data["trade_date"] >= "2021-01-06"].reset_index(drop=True)
    adf = stationarity(active)
    lp = local_projections(active)
    qr = quantile_regression(active)
    irf, granger, var_diagnostics = var_analysis(active)
    events = shock_event_study(active)
    outputs = {
        "dynamic_index_stationarity.csv": adf,
        "dynamic_index_local_projections.csv": lp,
        "dynamic_index_quantile_regression.csv": qr,
        "dynamic_index_var_irf.csv": irf,
        "dynamic_index_granger.csv": granger,
        "dynamic_index_shock_event_study.csv": events,
    }
    for name, frame in outputs.items():
        frame.to_csv(PROCESSED / name, index=False, encoding="utf-8-sig")
    save_plots(lp, qr, irf)
    diagnostics = pd.DataFrame([var_diagnostics])
    report = [
        "# 动态情绪指数统计检验", "",
        "> 以下结果是探索性关联检验，不构成严格因果识别。指数参数使用全样本估计，局部投影和VAR使用过滤状态，但仍存在参数估计层面的前视信息。", "",
        "## 平稳性", "", adf.to_markdown(index=False), "",
        "## 局部投影", "", lp.to_markdown(index=False), "",
        "## 五日收益分位数回归", "", qr.to_markdown(index=False), "",
        "## VAR诊断", "", diagnostics.to_markdown(index=False), "", granger.to_markdown(index=False), "",
        "## 高情绪冲击事件研究", "", events.to_markdown(index=False), "",
        "![统计检验图](dynamic_index_inference.png)", "",
        "显著性结果应结合样本量、事件聚集、公告选择偏差和多重检验谨慎解释。下一步正式论文版本应加入滚动估计、安慰剂日期和替代半衰期稳健性检验。",
    ]
    (REPORTS / "dynamic_index_inference.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(adf.to_string(index=False))
    print(qr.to_string(index=False))
    print(diagnostics.to_string(index=False))
    print(granger.to_string(index=False))


if __name__ == "__main__":
    main()
