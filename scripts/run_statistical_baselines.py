from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.api import VAR


ROOT = Path(__file__).resolve().parents[1]


def local_projections(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in range(0, 21):
        work = data.copy()
        work["outcome"] = work["index_level"].shift(-horizon).div(work["index_level"]).sub(1)
        x = work[["policy_intensity_sum", "return_lag_1", "HS300", "volatility_20d"]]
        sample = pd.concat([work["outcome"], x], axis=1).dropna()
        model = sm.OLS(sample["outcome"], sm.add_constant(sample[x.columns])).fit(
            cov_type="HAC", cov_kwds={"maxlags": max(1, horizon + 1)}
        )
        rows.append(
            {
                "horizon": horizon,
                "coefficient": model.params["policy_intensity_sum"],
                "std_error": model.bse["policy_intensity_sum"],
                "p_value": model.pvalues["policy_intensity_sum"],
                "nobs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def quantile_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = ["policy_climate_factor", "return_lag_1", "HS300", "volatility_20d"]
    sample = data[["future_return_5d", *features]].dropna()
    x = sm.add_constant(sample[features])
    for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
        model = QuantReg(sample["future_return_5d"], x).fit(q=quantile, max_iter=5000)
        rows.append(
            {
                "quantile": quantile,
                "coefficient": model.params["policy_climate_factor"],
                "std_error": model.bse["policy_climate_factor"],
                "p_value": model.pvalues["policy_climate_factor"],
                "nobs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def var_irf(data: pd.DataFrame) -> pd.DataFrame:
    sample = data[["policy_climate_factor", "return_ew", "HS300", "volatility_20d"]].dropna().copy()
    sample = sample.loc[sample["policy_climate_factor"].ne(0).idxmax():]
    sample = (sample - sample.mean()) / sample.std()
    result = VAR(sample).fit(maxlags=5, ic="aic", trend="c")
    response = result.irf(20).orth_irfs[:, 1, 0]
    return pd.DataFrame({"horizon": range(len(response)), "return_response_to_policy_shock": response, "selected_lag": result.k_ar})


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "processed" / "model_daily_dataset.csv")
    lp = local_projections(data)
    qr = quantile_models(data)
    irf = var_irf(data)
    lp.to_csv(ROOT / "data" / "processed" / "local_projection_baseline.csv", index=False, encoding="utf-8-sig")
    qr.to_csv(ROOT / "data" / "processed" / "quantile_regression_baseline.csv", index=False, encoding="utf-8-sig")
    irf.to_csv(ROOT / "data" / "processed" / "var_irf_baseline.csv", index=False, encoding="utf-8-sig")
    report = [
        "# 统计模型基线",
        "",
        "## 局部投影",
        "",
        lp.to_markdown(index=False),
        "",
        "## 五日收益分位数回归",
        "",
        qr.to_markdown(index=False),
        "",
        "## 解释限制",
        "",
        "当前只有17条规则弱标签政策事件，估计功效很低；这些结果只用于验证模型代码和确定未来数据需求，不构成因果证据。",
    ]
    (ROOT / "reports" / "statistical_baselines.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("局部投影、分位数回归和VAR基线已完成")


if __name__ == "__main__":
    main()
