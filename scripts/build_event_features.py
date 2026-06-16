from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    events = pd.read_csv(ROOT / "data" / "interim" / "policy_events_baseline.csv", parse_dates=["event_date"])
    index_data = pd.read_csv(ROOT / "data" / "processed" / "low_altitude_index_daily.csv", parse_dates=["trade_date"])
    calendar = pd.Series(sorted(index_data["trade_date"].unique()))

    def next_trade_day(event_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
        future = calendar[calendar > event_date]
        return future.iloc[0] if len(future) else pd.NaT

    events["information_trade_date"] = events["event_date"].map(next_trade_day)
    events.to_csv(ROOT / "data" / "processed" / "policy_events.csv", index=False, encoding="utf-8-sig")

    daily = pd.DataFrame({"trade_date": calendar})
    event_daily = events.dropna(subset=["information_trade_date"]).groupby("information_trade_date").agg(
        policy_event_count=("event_id", "count"),
        policy_intensity_sum=("intensity", "sum"),
        policy_intensity_max=("intensity", "max"),
        central_event_count=("policy_level", lambda values: int((values == "central").sum())),
        positive_event_count=("direction", lambda values: int((values == "positive").sum())),
    ).reset_index().rename(columns={"information_trade_date": "trade_date"})
    daily = daily.merge(event_daily, on="trade_date", how="left")
    feature_columns = [column for column in daily.columns if column != "trade_date"]
    daily[feature_columns] = daily[feature_columns].fillna(0).astype(int)
    daily.to_csv(ROOT / "data" / "processed" / "policy_event_daily_features.csv", index=False, encoding="utf-8-sig")
    print(f"事件数：{len(events)}；有可对齐交易日：{events['information_trade_date'].notna().sum()}")


if __name__ == "__main__":
    main()
