from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    events = pd.read_csv(ROOT / "data" / "interim" / "company_event_candidates.csv", parse_dates=["event_date"])
    events = events[events["eligible_for_labeling"]].copy()
    market = pd.read_csv(ROOT / "data" / "interim" / "market_daily.csv", parse_dates=["trade_date"])

    calendars = {code: pd.Series(sorted(group["trade_date"].unique())) for code, group in market.groupby("ts_code")}
    def next_day(row: pd.Series) -> pd.Timestamp | pd.NaT:
        future = calendars[row["ts_code"]]
        future = future[future > row["event_date"]]
        return future.iloc[0] if len(future) else pd.NaT
    events["information_trade_date"] = events.apply(next_day, axis=1)
    events.to_csv(ROOT / "data" / "processed" / "company_events.csv", index=False, encoding="utf-8-sig")

    daily = events.dropna(subset=["information_trade_date"]).groupby("information_trade_date").agg(
        company_event_count=("event_id", "count"),
        company_event_relevance_sum=("relevance_score", "sum"),
        positive_company_events=("direction", lambda values: int((values == "positive").sum())),
        negative_company_events=("direction", lambda values: int((values == "negative").sum())),
        unique_event_chains=("event_chain_id", "nunique"),
    ).reset_index().rename(columns={"information_trade_date": "trade_date"})
    calendar = pd.DataFrame({"trade_date": sorted(market["trade_date"].unique())})
    daily = calendar.merge(daily, on="trade_date", how="left")
    columns = [column for column in daily if column != "trade_date"]
    daily[columns] = daily[columns].fillna(0)
    daily.to_csv(ROOT / "data" / "processed" / "company_event_daily_features.csv", index=False, encoding="utf-8-sig")
    print(f"公司事件 {len(events)} 条，成功对齐 {events['information_trade_date'].notna().sum()} 条")


if __name__ == "__main__":
    main()
