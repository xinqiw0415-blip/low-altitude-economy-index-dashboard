from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = [(-1, 1), (-3, 3), (0, 5), (0, 10)]


def main() -> None:
    events = pd.read_csv(ROOT / "data" / "processed" / "company_events.csv", parse_dates=["information_trade_date"])
    market = pd.read_csv(ROOT / "data" / "interim" / "market_daily.csv", parse_dates=["trade_date"])
    benchmark = pd.read_csv(ROOT / "data" / "interim" / "benchmark_daily.csv", parse_dates=["trade_date"])
    hs300 = benchmark[benchmark["index_code"] == "HS300"][["trade_date", "return"]].rename(columns={"return": "market_return"})
    market = market.merge(hs300, on="trade_date", how="left").sort_values(["ts_code", "trade_date"])
    market["stock_return"] = market.groupby("ts_code")["close"].pct_change(fill_method=None)
    market["abnormal_return"] = market["stock_return"] - market["market_return"]
    by_stock = {code: group.reset_index(drop=True) for code, group in market.groupby("ts_code")}
    positions = {code: {day: index for index, day in enumerate(group["trade_date"])} for code, group in by_stock.items()}
    records = []
    for event in events.dropna(subset=["information_trade_date"]).itertuples(index=False):
        center = positions[event.ts_code].get(event.information_trade_date)
        if center is None:
            continue
        series = by_stock[event.ts_code]
        for left, right in WINDOWS:
            start, end = center + left, center + right
            if start < 0 or end >= len(series):
                continue
            window = series.iloc[start:end + 1]
            records.append(
                {
                    "event_id": event.event_id,
                    "event_chain_id": event.event_chain_id,
                    "ts_code": event.ts_code,
                    "event_type": event.event_type,
                    "direction": event.direction,
                    "window": f"[{left},{right}]",
                    "car_market_adjusted": window["abnormal_return"].sum(),
                }
            )
    result = pd.DataFrame(records)
    result.to_csv(ROOT / "data" / "processed" / "company_event_study.csv", index=False, encoding="utf-8-sig")
    first_events = events.sort_values("event_date").drop_duplicates("event_chain_id")["event_id"]
    summary_source = result[result["event_id"].isin(first_events)]
    summary = summary_source.groupby(["event_type", "window"]).agg(
        events=("event_id", "nunique"),
        mean_car=("car_market_adjusted", "mean"),
        median_car=("car_market_adjusted", "median"),
        positive_share=("car_market_adjusted", lambda values: float((values > 0).mean())),
    ).reset_index()
    report = ROOT / "reports" / "company_event_study.md"
    report.write_text(
        "# 公司公告事件研究基线\n\n"
        + summary.to_markdown(index=False)
        + "\n\n仅使用每个事件链的首次公告，收益为个股相对沪深300的市场调整收益。事件仍为规则候选，正式结论须完成人工或LLM复核，并处理同期事件与横截面相关。\n",
        encoding="utf-8",
    )
    print(f"事件研究记录 {len(result)}，首次事件链 {len(first_events)}")


if __name__ == "__main__":
    main()
