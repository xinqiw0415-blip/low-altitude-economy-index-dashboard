from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from urllib.parse import urljoin
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, UnicodeDammit


ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LowAltitudeResearch/1.0"


def fetch_content(url: str) -> tuple[bytes, str]:
    result = subprocess.run(
        [
            "curl.exe",
            "--ssl-no-revoke",
            "-L",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            "--max-time",
            "45",
            "-A",
            USER_AGENT,
            "-w",
            "\n__FINAL_URL__:%{url_effective}",
            url,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    marker = b"\n__FINAL_URL__:"
    content, final_url = result.stdout.rsplit(marker, 1)
    return content, final_url.decode("utf-8", errors="replace")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_document(content: bytes) -> tuple[str, str, str]:
    if content.lstrip().startswith(b"%PDF"):
        source = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        target = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        source_path, target_path = Path(source.name), Path(target.name)
        source.write(content)
        source.close()
        target.close()
        try:
            subprocess.run(
                ["pdftotext", "-layout", "-enc", "UTF-8", str(source_path), str(target_path)],
                check=True,
                capture_output=True,
            )
            body = clean_text(target_path.read_text(encoding="utf-8", errors="replace"))
        finally:
            source_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)
        return "", body, "pdf"
    html = UnicodeDammit(content, is_html=True).unicode_markup or content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        node.decompose()
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    candidates = []
    for selector in ["article", "main", ".article", ".content", ".TRS_Editor", "#zoom", "#content"]:
        candidates.extend(soup.select(selector))
    candidates.append(soup.body or soup)
    texts = [clean_text(node.get_text("\n", strip=True)) for node in candidates]
    body = max(texts, key=len, default="")
    return title, body, "html"


def first_pdf_link(content: bytes, base_url: str) -> str | None:
    html = UnicodeDammit(content, is_html=True).unicode_markup or ""
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        if ".pdf" in href.lower():
            return urljoin(base_url, href)
    return None


def main() -> None:
    sources = pd.read_csv(ROOT / "config" / "policy_sources.csv", dtype=str)
    html_dir = ROOT / "data" / "raw" / "policies" / "html"
    text_dir = ROOT / "data" / "raw" / "policies" / "text"
    html_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    records = []

    for item in sources.itertuples(index=False):
        print(f"采集 {item.document_id} {item.title} ...")
        record = item._asdict()
        record.update({"final_url": "", "page_title": "", "full_text": "", "content_sha256": "", "fetched_at": fetched_at, "fetch_status": "failed", "error": ""})
        try:
            content, final_url = fetch_content(item.url)
            page_title, body, content_type = extract_document(content)
            attachment_url = None
            if content_type == "html" and len(body) < 1000:
                attachment_url = first_pdf_link(content, final_url)
                if attachment_url:
                    attachment, _ = fetch_content(attachment_url)
                    _, attachment_body, attachment_type = extract_document(attachment)
                    if len(attachment_body) > len(body):
                        body = attachment_body
                        content_type = attachment_type
                        content = attachment
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            raw_suffix = ".pdf" if content_type == "pdf" else ".html"
            (html_dir / f"{item.document_id}{raw_suffix}").write_bytes(content)
            (text_dir / f"{item.document_id}.txt").write_text(body + "\n", encoding="utf-8")
            record.update(
                {
                    "final_url": final_url,
                    "page_title": page_title,
                    "full_text": body,
                    "content_sha256": digest,
                    "fetch_status": "ok" if len(body) >= 500 else "short_text",
                }
            )
        except Exception as exc:
            record["error"] = str(exc)[:500]
        records.append(record)

    frame = pd.DataFrame(records)
    output = ROOT / "data" / "interim" / "policy_documents.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(frame[["document_id", "fetch_status", "title"]].to_string(index=False))
    print(f"成功或短文本：{(frame['fetch_status'] != 'failed').sum()}/{len(frame)}")


if __name__ == "__main__":
    main()
