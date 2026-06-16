from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def align_next_trade_day(frame: pd.DataFrame, date_column: str, calendar: pd.Series) -> pd.Series:
    dates = pd.to_datetime(frame[date_column])
    return dates.map(lambda value: calendar[calendar > value].iloc[0] if (calendar > value).any() else pd.NaT)


def main() -> None:
    market = pd.read_csv(ROOT / "data" / "interim" / "market_daily.csv", parse_dates=["trade_date"])
    calendar = pd.Series(sorted(market["trade_date"].unique()))
    company = pd.read_csv(ROOT / "data" / "processed" / "deepseek_company_events.csv")
    policy = pd.read_csv(ROOT / "data" / "processed" / "deepseek_policy_events.csv")
    company = company[company["accepted"]].copy()
    policy = policy[policy["accepted"]].copy()
    company["trade_date"] = align_next_trade_day(company, "event_date", calendar)
    policy["trade_date"] = align_next_trade_day(policy, "event_date", calendar)

    company_daily = company.dropna(subset=["trade_date"]).groupby("trade_date").agg(
        llm_company_event_count=("event_id", "count"),
        llm_company_intensity=("intensity", "sum"),
        llm_company_relevance=("relevance", "sum"),
        llm_company_positive=("direction", lambda values: int((values == "positive").sum())),
        llm_company_negative=("direction", lambda values: int((values == "negative").sum())),
    ).reset_index()
    policy_daily = policy.dropna(subset=["trade_date"]).groupby("trade_date").agg(
        llm_policy_event_count=("event_id", "count"),
        llm_policy_intensity=("intensity", "sum"),
        llm_policy_uncertainty=("uncertainty", "mean"),
        llm_policy_novelty=("novelty", "mean"),
    ).reset_index()
    type_counts = policy.pivot_table(index="trade_date", columns="event_type", values="event_id", aggfunc="count", fill_value=0)
    type_counts.columns = [f"llm_policy_type_{column}" for column in type_counts.columns]
    type_counts = type_counts.reset_index()

    daily = pd.DataFrame({"trade_date": calendar})
    daily = daily.merge(company_daily, on="trade_date", how="left").merge(policy_daily, on="trade_date", how="left").merge(type_counts, on="trade_date", how="left")
    columns = [column for column in daily if column != "trade_date"]
    daily[columns] = daily[columns].fillna(0)
    for column in ["llm_company_intensity", "llm_company_relevance", "llm_policy_intensity"]:
        daily[f"{column}_ewm20"] = daily[column].ewm(halflife=20, adjust=False).mean()
    output = ROOT / "data" / "processed" / "deepseek_event_daily_features.csv"
    daily.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"DeepSeek日频特征：{len(daily)} 行，{len(daily.columns)} 列")


if __name__ == "__main__":
    main()
