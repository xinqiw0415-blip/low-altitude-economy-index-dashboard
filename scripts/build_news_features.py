from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POSITIVE = ["支持", "促进", "突破", "获批", "中标", "签约", "交付", "增长", "创新", "补贴", "利好"]
NEGATIVE = ["事故", "处罚", "亏损", "下滑", "风险", "终止", "延迟", "失败", "调查", "召回"]
UNCERTAIN = ["拟", "计划", "或将", "预计", "有望", "可能", "探索", "尚未"]


def score(text: str, lexicon: list[str]) -> int:
    return sum(text.count(word) for word in lexicon)


def main() -> None:
    news = pd.read_csv(ROOT / "data" / "interim" / "news_metadata.csv", parse_dates=["published_at"])
    calendar = pd.read_csv(ROOT / "data" / "processed" / "low_altitude_index_daily.csv", parse_dates=["trade_date"])
    calendar = pd.Series(sorted(calendar["trade_date"].unique()))
    fundamental = news[news["article_type"] != "market_report"].copy()
    combined = fundamental["title"].fillna("") + " " + fundamental["summary"].fillna("")
    fundamental["positive_terms"] = combined.map(lambda value: score(value, POSITIVE))
    fundamental["negative_terms"] = combined.map(lambda value: score(value, NEGATIVE))
    fundamental["uncertain_terms"] = combined.map(lambda value: score(value, UNCERTAIN))

    def next_trade_day(timestamp: pd.Timestamp) -> pd.Timestamp | pd.NaT:
        future = calendar[calendar > timestamp.normalize()]
        return future.iloc[0] if len(future) else pd.NaT
    fundamental["information_trade_date"] = fundamental["published_at"].map(next_trade_day)
    fundamental.to_csv(ROOT / "data" / "processed" / "news_fundamental_metadata.csv", index=False, encoding="utf-8-sig")
    grouped = fundamental.dropna(subset=["information_trade_date"]).groupby("information_trade_date").agg(
        news_count=("news_id", "count"),
        news_source_count=("media_name", "nunique"),
        news_positive_terms=("positive_terms", "sum"),
        news_negative_terms=("negative_terms", "sum"),
        news_uncertain_terms=("uncertain_terms", "sum"),
    ).reset_index().rename(columns={"information_trade_date": "trade_date"})
    daily = pd.DataFrame({"trade_date": calendar}).merge(grouped, on="trade_date", how="left")
    columns = [column for column in daily if column != "trade_date"]
    daily[columns] = daily[columns].fillna(0)
    daily["news_tone"] = (daily["news_positive_terms"] - daily["news_negative_terms"]) / (1 + daily["news_count"])
    daily.to_csv(ROOT / "data" / "processed" / "news_daily_features.csv", index=False, encoding="utf-8-sig")
    print(f"非行情播报新闻 {len(fundamental)} 条，覆盖 {int((daily['news_count'] > 0).sum())} 个交易日")


if __name__ == "__main__":
    main()
