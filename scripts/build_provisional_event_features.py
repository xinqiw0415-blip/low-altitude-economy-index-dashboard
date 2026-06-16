from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def next_dates(values: pd.Series, calendar: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    return dates.map(lambda value: calendar[calendar > value].iloc[0] if (calendar > value).any() else pd.NaT)


def main() -> None:
    market = pd.read_csv(ROOT / "data" / "interim" / "market_daily.csv", parse_dates=["trade_date"])
    calendar = pd.Series(sorted(market["trade_date"].unique()))
    company = pd.read_csv(ROOT / "data" / "processed" / "final_company_events_provisional.csv")
    policy = pd.read_csv(ROOT / "data" / "processed" / "final_policy_events_provisional.csv")
    company = company[company["accepted"]].copy()
    policy = policy[policy["accepted"]].copy()
    company["trade_date"] = next_dates(company["event_date"], calendar)
    policy["trade_date"] = next_dates(policy["event_date"], calendar)
    company_daily = company.groupby("trade_date").agg(
        final_company_event_count=("event_id", "count"),
        final_company_intensity=("intensity", "sum"),
        final_company_positive=("direction", lambda values: int((values == "positive").sum())),
        final_company_negative=("direction", lambda values: int((values == "negative").sum())),
        final_human_event_count=("review_status", lambda values: int((values == "adjudicated_sample").sum())),
    ).reset_index()
    policy_daily = policy.groupby("trade_date").agg(
        final_policy_event_count=("event_id", "count"),
        final_policy_intensity=("intensity", "sum"),
        final_policy_uncertainty=("uncertainty", "mean"),
    ).reset_index()
    daily = pd.DataFrame({"trade_date": calendar}).merge(company_daily, on="trade_date", how="left").merge(policy_daily, on="trade_date", how="left")
    columns = [column for column in daily if column != "trade_date"]
    daily[columns] = daily[columns].fillna(0)
    for column in ["final_company_intensity", "final_policy_intensity"]:
        daily[f"{column}_ewm20"] = daily[column].ewm(halflife=20, adjust=False).mean()
    daily.to_csv(ROOT / "data" / "processed" / "provisional_final_event_daily_features.csv", index=False, encoding="utf-8-sig")
    print(f"裁决后混合事件特征：公司{len(company)}，政策{len(policy)}")


if __name__ == "__main__":
    main()
