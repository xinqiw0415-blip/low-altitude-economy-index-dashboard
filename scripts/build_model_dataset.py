from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    index_data = pd.read_csv(ROOT / "data" / "processed" / "low_altitude_index_daily.csv", parse_dates=["trade_date"])
    index_data = index_data[index_data["index_code"] == "LAE_CORE_EW"].copy()
    benchmark = pd.read_csv(ROOT / "data" / "interim" / "benchmark_daily.csv", parse_dates=["trade_date"])
    benchmark = benchmark.pivot(index="trade_date", columns="index_code", values="return").reset_index()
    events = pd.read_csv(ROOT / "data" / "processed" / "policy_event_daily_features.csv", parse_dates=["trade_date"])
    climate = pd.read_csv(ROOT / "data" / "processed" / "policy_climate_daily.csv", parse_dates=["trade_date"])
    company_events = pd.read_csv(ROOT / "data" / "processed" / "company_event_daily_features.csv", parse_dates=["trade_date"])
    news = pd.read_csv(ROOT / "data" / "processed" / "news_daily_features.csv", parse_dates=["trade_date"])
    llm_events = pd.read_csv(ROOT / "data" / "processed" / "deepseek_event_daily_features.csv", parse_dates=["trade_date"])
    final_events = pd.read_csv(ROOT / "data" / "processed" / "provisional_final_event_daily_features.csv", parse_dates=["trade_date"])
    sentiment = pd.read_csv(ROOT / "data" / "processed" / "dynamic_sentiment_index_daily.csv", parse_dates=["trade_date"])
    realtime = pd.read_csv(ROOT / "data" / "processed" / "realtime_sentiment_index_daily.csv", parse_dates=["trade_date", "realtime_model_train_end"])
    data = index_data.merge(benchmark, on="trade_date", how="left").merge(events, on="trade_date", how="left").merge(climate, on="trade_date", how="left").merge(company_events, on="trade_date", how="left").merge(news, on="trade_date", how="left").merge(llm_events, on="trade_date", how="left").merge(final_events, on="trade_date", how="left").merge(sentiment, on="trade_date", how="left").merge(realtime, on="trade_date", how="left")
    data["excess_return_hs300"] = data["return_ew"] - data["HS300"]
    for lag in [1, 2, 5, 10, 20]:
        data[f"return_lag_{lag}"] = data["return_ew"].shift(lag)
    for window in [5, 10, 20]:
        data[f"volatility_{window}d"] = data["return_ew"].shift(1).rolling(window).std()
        data[f"momentum_{window}d"] = data["index_level"].shift(1).pct_change(window)
    for horizon in [1, 5, 20]:
        data[f"future_return_{horizon}d"] = (
            data["index_level"].shift(-horizon).div(data["index_level"]).sub(1)
        )
        data[f"future_up_{horizon}d"] = data[f"future_return_{horizon}d"].gt(0).astype("Int64")
        data.loc[data[f"future_return_{horizon}d"].isna(), f"future_up_{horizon}d"] = pd.NA
    output = ROOT / "data" / "processed" / "model_daily_dataset.csv"
    data.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"建模日频表：{len(data)} 行，{len(data.columns)} 列")


if __name__ == "__main__":
    main()
