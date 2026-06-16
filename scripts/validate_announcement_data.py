from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metadata = pd.read_csv(ROOT / "data" / "interim" / "company_announcements_metadata.csv")
    documents = pd.read_csv(ROOT / "data" / "interim" / "company_announcement_documents.csv")
    events = pd.read_csv(ROOT / "data" / "interim" / "company_event_candidates.csv")
    status = documents["fetch_status"].value_counts(dropna=False).to_dict()
    selected_by_stock = metadata[metadata["selected"]].groupby(["ts_code", "name"]).size().reset_index(name="selected")
    event_types = events[events["eligible_for_labeling"]]["event_type"].value_counts().rename_axis("event_type").reset_index(name="events")
    failures = documents[documents["fetch_status"] != "ok"][["art_code", "ts_code", "title", "fetch_status", "error"]]
    report = [
        "# 公司公告语料质量报告",
        "",
        f"- 历史公告元数据：{len(metadata)}",
        f"- 标题筛选公告：{int(metadata['selected'].sum())}",
        f"- 正文采集状态：`{status}`",
        f"- 规则事件候选：{len(events)}",
        f"- 进入人工/LLM标注池：{int(events['eligible_for_labeling'].sum())}",
        f"- 独立事件链：{events.loc[events['eligible_for_labeling'], 'event_chain_id'].nunique()}",
        "",
        "## 分公司覆盖",
        "",
        selected_by_stock.to_markdown(index=False),
        "",
        "## 事件类型",
        "",
        event_types.to_markdown(index=False),
        "",
        "## 失败与短文本",
        "",
        failures.to_markdown(index=False),
        "",
        "规则相关性仅用于抽样，不替代人工判断。进展公告通过 `event_chain_id` 与初始事项归组。",
    ]
    output = ROOT / "reports" / "announcement_data_quality.md"
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"已生成 {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
