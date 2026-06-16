from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TENCENT_API_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
EASTMONEY_API_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
FIELDS = [
    "trade_date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude_pct",
    "pct_change",
    "price_change",
    "turnover_pct",
]


def tencent_symbol(ts_code: str) -> str:
    code, exchange = ts_code.split(".")
    return f"{exchange.lower()}{code}"


def eastmoney_secid(ts_code: str) -> str:
    code, exchange = ts_code.split(".")
    market_id = "1" if exchange == "SH" else "0"
    return f"{market_id}.{code}"


def fetch_tencent_json(ts_code: str, start: str, end: str) -> dict:
    symbol = tencent_symbol(ts_code)
    url = f"{TENCENT_API_URL}?param={symbol},day,{start},{end},400,qfq"
    result = subprocess.run(
        [
            "curl.exe",
            "--ssl-no-revoke",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout)
    if payload.get("code") != 0 or not payload.get("data", {}).get(symbol):
        raise RuntimeError(f"{ts_code} 接口返回异常：{payload}")
    return payload


def fetch_eastmoney_json(ts_code: str, start: str, end: str) -> dict:
    start_compact = start.replace("-", "")
    end_compact = end.replace("-", "")
    url = (
        f"{EASTMONEY_API_URL}?secid={eastmoney_secid(ts_code)}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&klt=101&fqt=1&beg={start_compact}&end={end_compact}"
    )
    result = subprocess.run(
        [
            "curl.exe",
            "--ssl-no-revoke",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout)
    if payload.get("rc") != 0 or not payload.get("data", {}).get("klines"):
        raise RuntimeError(f"{ts_code} 东方财富接口返回异常：{payload}")
    return payload


def fetch_json(ts_code: str, start: str, end: str) -> tuple[str, dict]:
    try:
        return "tencent_qfq_kline", fetch_tencent_json(ts_code, start, end)
    except Exception as exc:
        print(f"  腾讯源失败，切换东方财富：{exc}")
        return "eastmoney_qfq_kline", fetch_eastmoney_json(ts_code, start, end)


def parse_tencent_rows(ts_code: str, name: str, payload: dict, fetched_at: str) -> list[dict]:
    data = payload["data"][tencent_symbol(ts_code)]
    source_rows = data.get("qfqday") or data.get("day") or []
    rows = []
    for values in source_rows:
        trade_date, open_, close, high, low, volume = values[:6]
        record = {
            "trade_date": trade_date,
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "amount": None,
            "amplitude_pct": None,
            "pct_change": None,
            "price_change": None,
            "turnover_pct": None,
        }
        record.update(
            {
                "ts_code": ts_code,
                "name": name,
                "source": "tencent_qfq_kline",
                "fetched_at": fetched_at,
            }
        )
        rows.append(record)
    return rows


def parse_eastmoney_rows(ts_code: str, name: str, payload: dict, fetched_at: str) -> list[dict]:
    rows = []
    for line in payload["data"]["klines"]:
        (
            trade_date,
            open_,
            close,
            high,
            low,
            volume,
            amount,
            amplitude_pct,
            pct_change,
            price_change,
            turnover_pct,
        ) = line.split(",")[:11]
        rows.append(
            {
                "trade_date": trade_date,
                "open": open_,
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
                "amount": amount,
                "amplitude_pct": amplitude_pct,
                "pct_change": pct_change,
                "price_change": price_change,
                "turnover_pct": turnover_pct,
                "ts_code": ts_code,
                "name": name,
                "source": "eastmoney_qfq_kline",
                "fetched_at": fetched_at,
            }
        )
    return rows


def parse_rows(ts_code: str, name: str, source: str, payload: dict, fetched_at: str) -> list[dict]:
    if source == "eastmoney_qfq_kline":
        return parse_eastmoney_rows(ts_code, name, payload, fetched_at)
    return parse_tencent_rows(ts_code, name, payload, fetched_at)


def annual_windows(start: str, end: str) -> list[tuple[str, str]]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    return [
        (
            max(start_date, date(year, 1, 1)).isoformat(),
            min(end_date, date(year, 12, 31)).isoformat(),
        )
        for year in range(start_date.year, end_date.year + 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="采集低空经济股票池日行情")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=datetime.now().date().isoformat())
    args = parser.parse_args()

    universe = pd.read_csv(ROOT / "config" / "stock_universe.csv", dtype=str)
    universe = universe[universe["include"] == "1"]
    raw_dir = ROOT / "data" / "raw" / "market"
    output_path = ROOT / "data" / "interim" / "market_daily.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    for item in universe.itertuples(index=False):
        ts_code = item.ts_code
        print(f"采集 {ts_code} ...")
        payloads = []
        for window_start, window_end in annual_windows(args.start, args.end):
            source, payload = fetch_json(ts_code, window_start, window_end)
            payloads.append({"source": source, "payload": payload})
            records.extend(parse_rows(ts_code, item.name, source, payload, fetched_at))
        (raw_dir / f"{ts_code.replace('.', '_')}.json").write_text(
            json.dumps(payloads, ensure_ascii=False), encoding="utf-8"
        )

    frame = pd.DataFrame(records)
    numeric_columns = [column for column in FIELDS if column != "trade_date"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        ["ts_code", "trade_date"], keep="last"
    )
    previous_close = frame.groupby("ts_code")["close"].shift(1)
    frame["price_change"] = frame["close"] - previous_close
    frame["pct_change"] = frame["price_change"].div(previous_close).mul(100)
    frame["amplitude_pct"] = frame["high"].sub(frame["low"]).div(previous_close).mul(100)
    ordered = ["ts_code", "name", *FIELDS, "source", "fetched_at"]
    frame[ordered].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        f"完成：{frame['ts_code'].nunique()} 只股票，{len(frame)} 行，"
        f"{frame['trade_date'].min()} 至 {frame['trade_date'].max()}"
    )
    print(f"输出：{output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
