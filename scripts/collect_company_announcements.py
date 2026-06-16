from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
API = "https://np-anotice-stock.eastmoney.com/api/security/ann"
PDF_TEMPLATE = "https://pdf.dfcfw.com/pdf/H2_{art_code}_1.pdf"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LowAltitudeResearch/1.0"


def curl_bytes(url: str, timeout: int = 60) -> bytes:
    result = subprocess.run(
        ["curl.exe", "--ssl-no-revoke", "-L", "--fail", "--silent", "--show-error", "--retry", "2", "--max-time", str(timeout), "-A", USER_AGENT, url],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def query_page(stock_code: str, start: str, end: str, page: int, page_size: int) -> dict:
    url = (
        f"{API}?sr=-1&page_size={page_size}&page_index={page}&ann_type=A&client_source=web"
        f"&stock_list={stock_code}&f_node=0&s_node=0&begin_time={start}&end_time={end}"
    )
    return json.loads(curl_bytes(url).decode("utf-8"))


def announcement_rows(payload: dict, ts_code: str, fallback_name: str) -> list[dict]:
    rows = []
    for item in payload.get("data", {}).get("list") or []:
        code_info = next((code for code in item.get("codes", []) if code.get("stock_code") == ts_code[:6]), {})
        rows.append(
            {
                "art_code": item.get("art_code"),
                "ts_code": ts_code,
                "name": code_info.get("short_name") or fallback_name,
                "notice_date": str(item.get("notice_date", ""))[:10],
                "display_time": item.get("display_time", ""),
                "title": item.get("title_ch") or item.get("title") or "",
                "columns": json.dumps([column.get("column_name", "") for column in item.get("columns", [])], ensure_ascii=False),
                "pdf_url": PDF_TEMPLATE.format(art_code=item.get("art_code")),
            }
        )
    return rows


def select_reason(title: str, config: dict) -> tuple[bool, str]:
    exclude = [word for word in config["exclude_keywords"] if word in title]
    if exclude:
        return False, "excluded:" + "|".join(exclude)
    matched = []
    domain_hits = []
    business_hits = []
    risk_hits = []
    for group, words in config["include_keywords"].items():
        hits = [word for word in words if word.lower() in title.lower()]
        if hits:
            matched.append(f"{group}:" + "|".join(hits))
            if group == "domain":
                domain_hits.extend(hits)
            elif group == "business_event":
                business_hits.extend(hits)
            elif group == "risk_event":
                risk_hits.extend(hits)
    direct_domain = {"低空", "通用航空", "通航", "空域", "空管", "eVTOL", "机场"}
    selected = bool(risk_hits or business_hits or direct_domain.intersection(domain_hits))
    return selected, ";".join(matched)


def pdf_to_text(content: bytes) -> str:
    source = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    target = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    source_path, target_path = Path(source.name), Path(target.name)
    source.write(content)
    source.close()
    target.close()
    try:
        subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(source_path), str(target_path)], check=True, capture_output=True)
        return target_path.read_text(encoding="utf-8", errors="replace").strip()
    finally:
        source_path.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config" / "announcement_filter.yaml").read_text(encoding="utf-8"))
    universe = pd.read_csv(ROOT / "config" / "stock_universe.csv", dtype=str)
    universe = universe[universe["include"] == "1"]
    raw_meta = ROOT / "data" / "raw" / "announcements" / "metadata"
    raw_pdf = ROOT / "data" / "raw" / "announcements" / "pdf"
    raw_text = ROOT / "data" / "raw" / "announcements" / "text"
    for directory in [raw_meta, raw_pdf, raw_text]:
        directory.mkdir(parents=True, exist_ok=True)

    all_rows = []
    page_size = 100
    for stock in universe.itertuples(index=False):
        stock_code = stock.ts_code[:6]
        print(f"公告元数据 {stock.ts_code} ...")
        first = query_page(stock_code, config["start_date"], config["end_date"], 1, page_size)
        total = int(first.get("data", {}).get("total_hits") or 0)
        payloads = [first]
        all_rows.extend(announcement_rows(first, stock.ts_code, stock.name))
        for page in range(2, math.ceil(total / page_size) + 1):
            payload = query_page(stock_code, config["start_date"], config["end_date"], page, page_size)
            payloads.append(payload)
            all_rows.extend(announcement_rows(payload, stock.ts_code, stock.name))
            time.sleep(0.15)
        (raw_meta / f"{stock.ts_code.replace('.', '_')}.json").write_text(json.dumps(payloads, ensure_ascii=False), encoding="utf-8")

    frame = pd.DataFrame(all_rows).drop_duplicates("art_code").sort_values(["notice_date", "ts_code"])
    decisions = frame["title"].map(lambda title: select_reason(str(title), config))
    frame["selected"] = decisions.map(lambda value: value[0])
    frame["selection_reason"] = decisions.map(lambda value: value[1])
    frame["fetched_at"] = datetime.now(timezone.utc).isoformat()
    metadata_path = ROOT / "data" / "interim" / "company_announcements_metadata.csv"
    frame.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    print(f"元数据 {len(frame)} 条，标题筛选 {int(frame['selected'].sum())} 条")

    if args.metadata_only:
        return
    selected = frame[frame["selected"]].copy()
    documents = []
    for index, item in enumerate(selected.itertuples(index=False), start=1):
        print(f"正文 {index}/{len(selected)} {item.art_code} {item.title[:35]}")
        status, error, text = "failed", "", ""
        try:
            pdf = curl_bytes(item.pdf_url, timeout=90)
            if not pdf.lstrip().startswith(b"%PDF"):
                raise RuntimeError("响应不是PDF")
            (raw_pdf / f"{item.art_code}.pdf").write_bytes(pdf)
            text = pdf_to_text(pdf)
            (raw_text / f"{item.art_code}.txt").write_text(text + "\n", encoding="utf-8")
            status = "ok" if len(text) >= 200 else "short_text"
        except Exception as exc:
            error = str(exc)[:500]
        documents.append({**item._asdict(), "full_text": text, "text_length": len(text), "fetch_status": status, "error": error})
        time.sleep(0.08)
    document_frame = pd.DataFrame(documents)
    document_frame.to_csv(ROOT / "data" / "interim" / "company_announcement_documents.csv", index=False, encoding="utf-8-sig")
    print(document_frame["fetch_status"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
