from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "interim" / "policy_documents.csv")
    frame["text_length"] = frame["full_text"].fillna("").str.len()
    duplicate_hashes = frame.loc[
        frame["content_sha256"].notna() & frame.duplicated("content_sha256", keep=False),
        ["document_id", "content_sha256"],
    ]
    status = frame["fetch_status"].value_counts(dropna=False).to_dict()
    coverage = frame.groupby(["level", "document_type"]).size().reset_index(name="documents")
    failed = frame.loc[frame["fetch_status"] == "failed", ["document_id", "title", "url", "error"]]
    short = frame.loc[frame["text_length"] < 1000, ["document_id", "title", "fetch_status", "text_length"]]
    lines = [
        "# 政策文本质量报告",
        "",
        f"- 清单文档数：{len(frame)}",
        f"- 采集状态：`{status}`",
        f"- 正文长度中位数：{int(frame['text_length'].median())}",
        f"- 重复正文哈希记录数：{len(duplicate_hashes)}",
        "",
        "## 层级与类型覆盖",
        "",
        coverage.to_markdown(index=False),
        "",
        "## 失败记录",
        "",
        failed.to_markdown(index=False) if len(failed) else "无",
        "",
        "## 短文本复核清单",
        "",
        short.to_markdown(index=False) if len(short) else "无",
        "",
        "短文本可能是反爬页面、跳转页或政策解读摘要，必须人工复核后才能进入事件抽取。",
    ]
    output = ROOT / "reports" / "policy_data_quality.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
