from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "interim" / "market_daily.csv"
REPORT_PATH = ROOT / "reports" / "market_data_quality.md"


def main() -> None:
    frame = pd.read_csv(INPUT_PATH, parse_dates=["trade_date"])
    duplicate_count = int(frame.duplicated(["ts_code", "trade_date"]).sum())
    invalid_ohlc = frame[
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    ]
    summary = (
        frame.groupby(["ts_code", "name"], as_index=False)
        .agg(
            observations=("trade_date", "size"),
            first_date=("trade_date", "min"),
            last_date=("trade_date", "max"),
            missing_close=("close", lambda values: int(values.isna().sum())),
        )
        .sort_values("ts_code")
    )
    missing = frame.isna().sum().to_dict()

    lines = [
        "# 行情数据质量报告",
        "",
        f"- 总记录数：{len(frame)}",
        f"- 股票数：{frame['ts_code'].nunique()}",
        f"- 日期范围：{frame['trade_date'].min().date()} 至 {frame['trade_date'].max().date()}",
        f"- 证券-日期重复数：{duplicate_count}",
        f"- OHLC 逻辑异常数：{len(invalid_ohlc)}",
        f"- 各字段缺失数：`{missing}`",
        "",
        "## 分证券覆盖",
        "",
        summary.to_markdown(index=False),
        "",
        "## 注意事项",
        "",
        "- 腾讯接口未提供成交额和换手率，因此这两列暂为空，下一轮使用第二数据源补齐。",
        "- 上市日期晚于样本起点的股票属于结构性缺失，构造指数时必须使用当日可交易成分股。",
        "- 当前股票池是候选池，需完成主营业务与低空经济相关性人工审核后冻结版本。",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 {REPORT_PATH.relative_to(ROOT)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
