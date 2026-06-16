from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    news = pd.read_csv(ROOT / "data" / "interim" / "news_metadata.csv", parse_dates=["published_at"])
    monthly = news.groupby(news["published_at"].dt.to_period("M").astype(str)).size().reset_index(name="articles")
    types = news["article_type"].value_counts().rename_axis("article_type").reset_index(name="articles")
    sources = news["media_name"].value_counts().head(20).rename_axis("media_name").reset_index(name="articles")
    report = [
        "# 新闻元数据质量报告", "",
        f"- 去重新闻：{len(news)}",
        f"- 时间范围：{news['published_at'].min()} 至 {news['published_at'].max()}",
        f"- 媒体数量：{news['media_name'].nunique()}",
        "- 限制：公开搜索接口只返回最近约1000条，当前仅覆盖2026年4月至6月，暂不进入正式长期预测特征集。",
        "", "## 内容类型", "", types.to_markdown(index=False),
        "", "## 月度覆盖", "", monthly.to_markdown(index=False),
        "", "## 主要来源", "", sources.to_markdown(index=False),
    ]
    output = ROOT / "reports" / "news_data_quality.md"
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"已生成 {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
