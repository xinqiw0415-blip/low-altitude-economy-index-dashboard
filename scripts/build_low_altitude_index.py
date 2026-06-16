from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def build_index(frame: pd.DataFrame, members: set[str], code: str) -> pd.DataFrame:
    selected = frame[frame["ts_code"].isin(members)].copy()
    selected["stock_return"] = selected.groupby("ts_code")["close"].pct_change(fill_method=None)
    daily = selected.groupby("trade_date", as_index=False).agg(
        return_ew=("stock_return", "mean"),
        constituent_count=("close", "count"),
        total_volume=("volume", "sum"),
    )
    daily["index_code"] = code
    daily["index_level"] = 1000 * np.exp(np.log1p(daily["return_ew"].fillna(0)).cumsum())
    return daily[["index_code", "trade_date", "index_level", "return_ew", "constituent_count", "total_volume"]]


def main() -> None:
    market = pd.read_csv(ROOT / "data" / "interim" / "market_daily.csv")
    universe = pd.read_csv(ROOT / "config" / "stock_universe.csv")
    included = universe[universe["include"] == 1]
    core = set(included.loc[included["tier"] == "core", "ts_code"])
    extended = set(included["ts_code"])
    result = pd.concat(
        [
            build_index(market, core, "LAE_CORE_EW"),
            build_index(market, extended, "LAE_EXTENDED_EW"),
        ],
        ignore_index=True,
    )
    output = ROOT / "data" / "processed" / "low_altitude_index_daily.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    report = result.groupby("index_code").agg(
        first_date=("trade_date", "min"),
        last_date=("trade_date", "max"),
        observations=("trade_date", "size"),
        ending_level=("index_level", "last"),
        min_members=("constituent_count", "min"),
        max_members=("constituent_count", "max"),
    )
    print(report.to_string())
    print(f"输出：{output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
