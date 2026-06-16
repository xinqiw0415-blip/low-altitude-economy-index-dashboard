from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = [(-1, 1), (-3, 3), (0, 5), (0, 10)]


def main() -> None:
    events = pd.read_csv(ROOT / "data" / "processed" / "policy_events.csv", parse_dates=["information_trade_date"])
    low = pd.read_csv(ROOT / "data" / "processed" / "low_altitude_index_daily.csv", parse_dates=["trade_date"])
    low = low[low["index_code"] == "LAE_CORE_EW"][["trade_date", "return_ew"]]
    benchmark = pd.read_csv(ROOT / "data" / "interim" / "benchmark_daily.csv", parse_dates=["trade_date"])
    benchmark = benchmark[benchmark["index_code"] == "HS300"][["trade_date", "return"]]
    market = low.merge(benchmark, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    market["abnormal_return"] = market["return_ew"] - market["return"]
    position = {day: index for index, day in enumerate(market["trade_date"])}
    records = []

    for event in events.dropna(subset=["information_trade_date"]).itertuples(index=False):
        center = position.get(event.information_trade_date)
        if center is None:
            continue
        for left, right in WINDOWS:
            start, end = center + left, center + right
            if start < 0 or end >= len(market):
                continue
            window = market.iloc[start : end + 1]
            records.append(
                {
                    "event_id": event.event_id,
                    "event_date": event.event_date,
                    "information_trade_date": event.information_trade_date,
                    "event_type": event.event_type,
                    "policy_level": event.policy_level,
                    "window": f"[{left},{right}]",
                    "car_market_adjusted": window["abnormal_return"].sum(),
                    "raw_return": window["return_ew"].sum(),
                    "benchmark_return": window["return"].sum(),
                }
            )

    result = pd.DataFrame(records)
    output = ROOT / "data" / "processed" / "baseline_event_study.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    summary = result.groupby("window").agg(
        events=("event_id", "nunique"),
        mean_car=("car_market_adjusted", "mean"),
        median_car=("car_market_adjusted", "median"),
        positive_share=("car_market_adjusted", lambda values: (values > 0).mean()),
    ).reset_index()
    report = ROOT / "reports" / "baseline_event_study.md"
    report.write_text(
        "# 基线政策事件研究\n\n"
        + summary.to_markdown(index=False)
        + "\n\n该结果基于规则弱标签和市场调整收益，仅用于管线验证，不能作为最终因果结论。正式分析需人工复核事件、处理事件重叠并使用聚类或随机化推断。\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
