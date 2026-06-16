from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API = "https://search-api-web.eastmoney.com/search/jsonp"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LowAltitudeResearch/1.0"
MARKET_WORDS = ["板块", "概念", "涨停", "走高", "拉升", "上涨", "下跌", "跌停", "异动", "资金流向", "主力资金"]
FUNDAMENTAL_WORDS = ["政策", "条例", "方案", "规划", "中标", "订单", "合同", "项目", "试飞", "交付", "适航", "获批", "事故", "处罚", "融资", "补贴", "机场", "空域"]


def request_page(keyword: str, page: int, page_size: int) -> dict:
    parameter = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default", "pageIndex": page, "pageSize": page_size}},
    }
    url = API + "?cb=callback&param=" + urllib.parse.quote(json.dumps(parameter, ensure_ascii=False, separators=(",", ":")))
    result = subprocess.run(
        ["curl.exe", "--ssl-no-revoke", "-L", "--fail", "--silent", "--show-error", "--retry", "2", "-A", USER_AGENT, url],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    text = result.stdout.decode("utf-8", errors="replace")
    return json.loads(text[text.index("(") + 1:text.rindex(")")])


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(value or ""))).strip()


def article_type(title: str, content: str) -> str:
    combined = title + " " + content
    market_score = sum(word in combined for word in MARKET_WORDS)
    fundamental_score = sum(word in combined for word in FUNDAMENTAL_WORDS)
    if market_score >= 2 and fundamental_score == 0:
        return "market_report"
    if fundamental_score:
        return "fundamental_event"
    return "industry_commentary"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=3000)
    parser.add_argument("--start-date", default="2024-01-01")
    args = parser.parse_args()
    raw_dir = ROOT / "data" / "raw" / "news" / "search_pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    page_size = 50
    records = []
    page = 1
    total_hits = None
    while len(records) < args.max_records:
        payload = request_page("低空经济", page, page_size)
        if total_hits is None:
            total_hits = int(payload.get("hitsTotal") or 0)
            print(f"检索命中 {total_hits} 条")
        (raw_dir / f"page_{page:04d}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        articles = payload.get("result", {}).get("cmsArticleWebOld") or []
        if not articles:
            break
        for item in articles:
            title, content = clean(item.get("title", "")), clean(item.get("content", ""))
            published_at = item.get("date", "")
            if published_at[:10] < args.start_date:
                continue
            records.append(
                {
                    "news_id": "N-" + str(item.get("code") or hashlib.sha1((title + published_at).encode("utf-8")).hexdigest()[:18]),
                    "published_at": published_at,
                    "title": title,
                    "summary": content,
                    "media_name": item.get("mediaName", ""),
                    "url": item.get("url", ""),
                    "article_type": article_type(title, content),
                    "source": "eastmoney_search",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        print(f"第 {page} 页，累计 {len(records)}")
        if page * page_size >= total_hits:
            break
        page += 1
        time.sleep(0.12)
    frame = pd.DataFrame(records).drop_duplicates("url").sort_values("published_at")
    frame["content_hash"] = (frame["title"] + "|" + frame["summary"]).map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    frame = frame.drop_duplicates("content_hash")
    output = ROOT / "data" / "interim" / "news_metadata.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"新闻元数据去重后 {len(frame)} 条：{frame['article_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
