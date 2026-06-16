from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def amount_rmb_proxy(text: object) -> float:
    content = str(text or "")
    values = []
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(亿|万)?\s*(美元|元)")
    for number, unit, currency in pattern.findall(content):
        value = float(number)
        if unit == "亿":
            value *= 1e8
        elif unit == "万":
            value *= 1e4
        if currency == "美元":
            value *= 7.0
        values.append(value)
    return max(values, default=0.0)


def next_trade_date(events: pd.DataFrame, market: pd.DataFrame) -> pd.Series:
    calendars = {code: pd.Series(sorted(group["trade_date"].unique())) for code, group in market.groupby("ts_code")}
    def align(row: pd.Series) -> pd.Timestamp | pd.NaT:
        calendar = calendars.get(row["ts_code"])
        if calendar is None:
            return pd.NaT
        future = calendar[calendar > row["event_date"]]
        return future.iloc[0] if len(future) else pd.NaT
    return events.apply(align, axis=1)


def main() -> None:
    market = pd.read_csv(ROOT / "data" / "interim" / "market_daily.csv", parse_dates=["trade_date"])
    universe = pd.read_csv(ROOT / "config" / "stock_universe.csv")
    universe = universe[universe["include"].eq(1)][["ts_code", "industry_role", "tier"]]
    benchmark = pd.read_csv(ROOT / "data" / "interim" / "benchmark_daily.csv", parse_dates=["trade_date"])
    hs300 = benchmark[benchmark["index_code"].eq("HS300")][["trade_date", "close", "return"]].rename(
        columns={"close": "hs300_close", "return": "hs300_return"}
    )
    global_features = pd.read_csv(PROCESSED / "model_daily_dataset.csv", parse_dates=["trade_date"])[[
        "trade_date", "realtime_sentiment_lag1", "realtime_sentiment_change_lag1",
        "policy_support", "risk_impulse", "final_policy_event_count",
    ]]
    panel = market.merge(universe, on="ts_code", how="inner").merge(hs300, on="trade_date", how="left")
    panel = panel.merge(global_features, on="trade_date", how="left").sort_values(["ts_code", "trade_date"])
    grouped = panel.groupby("ts_code", group_keys=False)
    panel["stock_return"] = grouped["close"].pct_change(fill_method=None)
    panel["excess_return_1d"] = panel["stock_return"] - panel["hs300_return"]
    panel["hs300_return_lag1"] = panel["trade_date"].map(hs300.set_index("trade_date")["hs300_return"].shift(1))
    panel["hs300_volatility_20d"] = panel["trade_date"].map(
        hs300.set_index("trade_date")["hs300_return"].shift(1).rolling(20).std()
    )
    panel["log_volume"] = np.log1p(panel["volume"])
    for lag in [1, 2, 5, 10, 20]:
        panel[f"stock_return_lag_{lag}"] = grouped["stock_return"].shift(lag)
        panel[f"excess_return_lag_{lag}"] = grouped["excess_return_1d"].shift(lag)
    for window in [5, 10, 20, 60]:
        panel[f"stock_momentum_{window}d"] = grouped["close"].transform(lambda s: s.shift(1).pct_change(window))
        panel[f"stock_volatility_{window}d"] = grouped["stock_return"].transform(lambda s: s.shift(1).rolling(window).std())
        panel[f"volume_z_{window}d"] = grouped["log_volume"].transform(
            lambda s: (s.shift(1) - s.shift(1).rolling(window).mean()) / s.shift(1).rolling(window).std()
        )
    panel["intraday_range"] = (panel["high"] - panel["low"]) / panel["open"].replace(0, np.nan)
    panel["overnight_gap"] = grouped.apply(
        lambda g: g["open"].div(g["close"].shift(1)).sub(1), include_groups=False
    ).reset_index(level=0, drop=True).sort_index()
    for column in ["stock_return_lag_1", "excess_return_lag_5", "stock_momentum_20d", "stock_volatility_20d", "volume_z_20d"]:
        panel[f"rank_{column}"] = panel.groupby("trade_date")[column].rank(pct=True)

    for horizon in [1, 5, 20]:
        stock_future = grouped["close"].transform(lambda s: s.shift(-horizon).div(s).sub(1))
        benchmark_future = panel["hs300_close"].shift(-horizon).div(panel["hs300_close"]).sub(1)
        # Benchmark is repeated by stock, so compute by date before merging back.
        benchmark_by_date = hs300.set_index("trade_date")["hs300_close"]
        benchmark_future_by_date = benchmark_by_date.shift(-horizon).div(benchmark_by_date).sub(1)
        panel[f"future_stock_return_{horizon}d"] = stock_future
        panel[f"future_benchmark_return_{horizon}d"] = panel["trade_date"].map(benchmark_future_by_date)
        panel[f"future_abnormal_return_{horizon}d"] = stock_future - panel[f"future_benchmark_return_{horizon}d"]
        panel[f"future_outperform_{horizon}d"] = panel[f"future_abnormal_return_{horizon}d"].gt(0).astype("Int64")
        panel.loc[panel[f"future_abnormal_return_{horizon}d"].isna(), f"future_outperform_{horizon}d"] = pd.NA

    events = pd.read_csv(PROCESSED / "final_company_events_hybrid.csv", parse_dates=["event_date"])
    events = events[events["accepted"].astype(bool)].copy()
    events["information_trade_date"] = next_trade_date(events, market)
    direction = events["direction"].map({"positive": 1.0, "negative": -1.0, "mixed": -0.25, "neutral": 0.0}).fillna(0)
    certainty = 1 - (events["uncertainty"].clip(1, 5) - 1) / 5
    events["event_signed_score"] = direction * events["intensity"] * certainty
    events["event_risk_score"] = events["intensity"] * (events["uncertainty"] / 5) * direction.map(
        lambda value: 1.0 if value < 0 else 0.15
    )
    events["event_amount_rmb"] = (events["title"].fillna("") + " " + events["evidence_span"].fillna("")).map(amount_rmb_proxy)
    events["event_amount_log"] = np.log1p(events["event_amount_rmb"])
    events["event_initial"] = events["is_initial_event"].astype(int)
    event_daily = events.groupby(["ts_code", "information_trade_date"]).agg(
        company_event_count=("event_id", "count"),
        company_event_signed_impulse=("event_signed_score", "sum"),
        company_event_risk_impulse=("event_risk_score", "sum"),
        company_event_amount_log=("event_amount_log", "max"),
        company_event_initial_count=("event_initial", "sum"),
        company_event_intensity=("intensity", "sum"),
    ).reset_index().rename(columns={"information_trade_date": "trade_date"})
    panel = panel.merge(event_daily, on=["ts_code", "trade_date"], how="left")
    event_columns = [column for column in event_daily if column not in ["ts_code", "trade_date"]]
    panel[event_columns] = panel[event_columns].fillna(0)
    for column in event_columns:
        panel[f"{column}_ewm20"] = panel.groupby("ts_code")[column].transform(
            lambda s: s.ewm(halflife=20, adjust=False).mean()
        )
    panel["days_since_company_event"] = panel.groupby("ts_code")["company_event_count"].transform(
        lambda s: s.groupby(s.gt(0).cumsum()).cumcount().where(s.gt(0).cumsum().gt(0), 999)
    ).clip(upper=999)
    panel["event_window_20d"] = panel["days_since_company_event"].le(20).astype(int)
    for column in ["policy_support", "risk_impulse", "final_policy_event_count"]:
        panel[f"{column}_lag1"] = panel.groupby("ts_code")[column].shift(1)

    panel["abnormal_return_rank_5d"] = panel.groupby("trade_date")["future_abnormal_return_5d"].rank(pct=True)
    panel["future_extreme_relative_5d"] = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    panel.loc[panel["abnormal_return_rank_5d"].le(0.30), "future_extreme_relative_5d"] = 0
    panel.loc[panel["abnormal_return_rank_5d"].ge(0.70), "future_extreme_relative_5d"] = 1
    current_vol = panel["stock_volatility_20d"]
    future_vol = grouped["stock_return"].transform(lambda s: s.shift(-1).rolling(5).std().shift(-4))
    panel["future_volatility_5d"] = future_vol
    panel["future_volatility_jump_5d"] = future_vol.gt(current_vol * 1.25).astype("Int64")
    panel.loc[future_vol.isna() | current_vol.isna(), "future_volatility_jump_5d"] = pd.NA

    panel.to_csv(PROCESSED / "stock_panel_daily.csv", index=False, encoding="utf-8-sig")
    event_daily.to_csv(PROCESSED / "stock_panel_event_daily.csv", index=False, encoding="utf-8-sig")
    report = [
        "# 个股日频面板数据", "",
        f"- 面板记录：{len(panel)}。",
        f"- 股票数量：{panel['ts_code'].nunique()}。",
        f"- 接纳公司事件：{len(events)}。",
        f"- 具有金额代理的事件：{int(events['event_amount_rmb'].gt(0).sum())}。",
        f"- 事件后20日窗口记录：{int(panel['event_window_20d'].sum())}。", "",
        "主要目标包括未来5日是否跑赢沪深300、横截面上下30%强弱分类和未来5日波动率跳升。所有未来不可观测日期均保留为缺失值。",
    ]
    (ROOT / "reports" / "stock_panel_dataset.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(panel.shape, panel["ts_code"].nunique(), int(panel["event_window_20d"].sum()))


if __name__ == "__main__":
    main()
