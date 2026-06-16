from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


def annual_windows(start: str, end: str) -> list[tuple[str, str]]:
    left, right = date.fromisoformat(start), date.fromisoformat(end)
    return [
        (max(left, date(year, 1, 1)).isoformat(), min(right, date(year, 12, 31)).isoformat())
        for year in range(left.year, right.year + 1)
    ]


def fetch(symbol: str, start: str, end: str) -> dict:
    url = f"{API_URL}?param={symbol},day,{start},{end},400,"
    result = subprocess.run(
        ["curl.exe", "--ssl-no-revoke", "-L", "--fail", "--silent", "--show-error", "--retry", "3", url],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout)
    if payload.get("code") != 0 or not payload.get("data", {}).get(symbol):
        raise RuntimeError(f"{symbol} 接口返回异常")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=datetime.now().date().isoformat())
    args = parser.parse_args()

    config = pd.read_csv(ROOT / "config" / "market_benchmarks.csv")
    raw_dir = ROOT / "data" / "raw" / "benchmarks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    records = []

    for item in config.itertuples(index=False):
        print(f"采集 {item.index_code} ...")
        payloads = []
        for start, end in annual_windows(args.start, args.end):
            payload = fetch(item.symbol, start, end)
            payloads.append(payload)
            data = payload["data"][item.symbol]
            source_rows = data.get("day") or data.get("qfqday") or []
            for values in source_rows:
                records.append(
                    {
                        "index_code": item.index_code,
                        "name": item.name,
                        "trade_date": values[0],
                        "open": values[1],
                        "close": values[2],
                        "high": values[3],
                        "low": values[4],
                        "volume": values[5],
                        "source": "tencent_index_kline",
                        "fetched_at": fetched_at,
                    }
                )
        (raw_dir / f"{item.index_code}.json").write_text(
            json.dumps(payloads, ensure_ascii=False), encoding="utf-8"
        )

    frame = pd.DataFrame(records).drop_duplicates(["index_code", "trade_date"])
    for column in ["open", "close", "high", "low", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["index_code", "trade_date"])
    frame["return"] = frame.groupby("index_code")["close"].pct_change(fill_method=None)
    output = ROOT / "data" / "interim" / "benchmark_daily.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"完成：{frame['index_code'].nunique()} 个指数，{len(frame)} 行")


if __name__ == "__main__":
    main()
