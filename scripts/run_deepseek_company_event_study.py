from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = [(-1, 1), (-3, 3), (0, 5), (0, 10)]


def main() -> None:
    events = pd.read_csv(ROOT / "data" / "processed" / "deepseek_company_events.csv", parse_dates=["event_date"])
    events = events[events["accepted"]].copy()
    market = pd.read_csv(ROOT / "data" / "interim" / "market_daily.csv", parse_dates=["trade_date"])
    benchmark = pd.read_csv(ROOT / "data" / "interim" / "benchmark_daily.csv", parse_dates=["trade_date"])
    hs300 = benchmark[benchmark["index_code"] == "HS300"][["trade_date", "return"]].rename(columns={"return": "market_return"})
    market = market.merge(hs300, on="trade_date", how="left").sort_values(["ts_code", "trade_date"])
    market["stock_return"] = market.groupby("ts_code")["close"].pct_change(fill_method=None)
    market["abnormal_return"] = market["stock_return"] - market["market_return"]
    groups = {code: group.reset_index(drop=True) for code, group in market.groupby("ts_code")}
    positions = {code: {day: index for index, day in enumerate(group["trade_date"])} for code, group in groups.items()}
    events["information_trade_date"] = events.apply(
        lambda row: groups[row["ts_code"]].loc[groups[row["ts_code"]]["trade_date"] > row["event_date"], "trade_date"].iloc[0]
        if (groups[row["ts_code"]]["trade_date"] > row["event_date"]).any() else pd.NaT,
        axis=1,
    )
    rows = []
    for event in events.dropna(subset=["information_trade_date"]).itertuples(index=False):
        center = positions[event.ts_code][event.information_trade_date]
        series = groups[event.ts_code]
        for left, right in WINDOWS:
            if center + left < 0 or center + right >= len(series):
                continue
            window = series.iloc[center + left:center + right + 1]
            rows.append({"event_id": event.event_id, "event_type": event.event_type, "window": f"[{left},{right}]", "car": window["abnormal_return"].sum()})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "data" / "processed" / "deepseek_company_event_study.csv", index=False, encoding="utf-8-sig")
    summary = result.groupby(["event_type", "window"]).agg(
        events=("event_id", "nunique"), mean_car=("car", "mean"), median_car=("car", "median"),
        positive_share=("car", lambda values: float((values > 0).mean())),
    ).reset_index()
    (ROOT / "reports" / "deepseek_company_event_study.md").write_text(
        "# DeepSeek复核后的公司事件研究\n\n" + summary.to_markdown(index=False)
        + "\n\n仅使用证据回溯通过且相关性不低于0.5的事件。样本仍小，未完成人工金标准复核。\n",
        encoding="utf-8",
    )
    print(f"DeepSeek事件研究：{len(events)} 个事件")


if __name__ == "__main__":
    main()
