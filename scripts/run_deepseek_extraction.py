from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import tempfile
import time
import winreg
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.deepseek.com/chat/completions"


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as registry:
            value, _ = winreg.QueryValueEx(registry, "DEEPSEEK_API_KEY")
            key = str(value).strip()
    except OSError:
        key = ""
    if not key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。请运行：powershell -ExecutionPolicy Bypass -File .\\setup_deepseek_key.ps1"
        )
    return key


def strip_code_fence(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def normalize_result(task_type: str, content: str) -> dict[str, Any]:
    parsed = json.loads(strip_code_fence(content))
    if task_type == "policy":
        if isinstance(parsed, list):
            parsed = {"events": parsed}
        if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
            raise ValueError("政策输出必须是包含 events 数组的JSON对象")
    else:
        if isinstance(parsed, dict) and isinstance(parsed.get("event"), dict):
            parsed = parsed["event"]
        if isinstance(parsed, list):
            if len(parsed) != 1:
                raise ValueError("公司公告任务必须返回单个事件对象")
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            raise ValueError("公司公告输出必须是JSON对象")
        parsed = {"event": parsed}
    return parsed


def validate_result(task_type: str, result: dict[str, Any]) -> None:
    events = result["events"] if task_type == "policy" else [result["event"]]
    if not events:
        raise ValueError("事件列表为空")
    required = (
        {"event_type", "direction", "intensity", "uncertainty", "evidence_span", "confidence"}
        if task_type == "policy"
        else {"event_type", "direction", "intensity", "uncertainty", "relevance", "evidence_span"}
    )
    for event in events:
        missing = required - set(event)
        if missing:
            raise ValueError(f"缺少字段：{sorted(missing)}")
        if not str(event.get("evidence_span", "")).strip():
            raise ValueError("evidence_span为空")
        for field in ["intensity", "uncertainty"]:
            value = int(event[field])
            if not 1 <= value <= 5:
                raise ValueError(f"{field}超出1-5")


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def add(self, raw: dict[str, Any]) -> None:
        self.prompt_tokens += int(raw.get("prompt_tokens", 0))
        self.completion_tokens += int(raw.get("completion_tokens", 0))
        details = raw.get("prompt_tokens_details") or {}
        self.cached_tokens += int(details.get("cached_tokens", 0))


def call_api(api_key: str, prompt: str, model: str, timeout: int, max_tokens: int) -> tuple[str, dict[str, Any]]:
    system = (
        "你是严谨的中文金融事件抽取器。只能依据输入文本。"
        "输出必须是合法JSON对象，不要Markdown，不要解释，不要臆测。"
    )
    request_body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    request_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    response_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    request_path, response_path = Path(request_file.name), Path(response_file.name)
    request_file.write(json.dumps(request_body, ensure_ascii=False).encode("utf-8"))
    request_file.close()
    response_file.close()
    try:
        result = subprocess.run(
            [
                "curl.exe", "--ssl-no-revoke", "--http1.1", "-L", "--silent", "--show-error",
                "--retry", "2", "--max-time", str(timeout), "-X", "POST", API_URL,
                "-H", f"Authorization: Bearer {api_key}", "-H", "Content-Type: application/json",
                "--data-binary", f"@{request_path}", "-o", str(response_path), "-w", "%{http_code}"
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"curl exit {result.returncode}")
        status = int(result.stdout.strip()[-3:])
        response_text = response_path.read_text(encoding="utf-8", errors="replace")
        if status >= 400:
            raise RuntimeError(f"HTTP {status}: {response_text[:1000]}")
        payload = json.loads(response_text)
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
    content = payload["choices"][0]["message"]["content"]
    return content, payload.get("usage") or {}


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") == "ok":
                completed.add(record["task_id"])
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="调用DeepSeek执行事件抽取，支持断点续跑")
    parser.add_argument("--task-type", choices=["policy", "company"], required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()

    default_input = "llm_event_tasks.jsonl" if args.task_type == "policy" else "company_llm_event_tasks.jsonl"
    default_output = "deepseek_policy_results.jsonl" if args.task_type == "policy" else "deepseek_company_results.jsonl"
    input_path = Path(args.input) if args.input else ROOT / "data" / "interim" / default_input
    output_path = Path(args.output) if args.output else ROOT / "data" / "interim" / default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    api_key = get_api_key()
    max_tokens = args.max_tokens or (4000 if args.task_type == "policy" else 1800)
    tasks = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    completed = load_completed(output_path)
    usage = Usage()
    print(f"任务 {len(tasks)} 条，已完成 {len(completed)} 条，模型 {args.model}")

    with output_path.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(tasks, start=1):
            task_id = task["task_id"]
            if task_id in completed:
                continue
            error = ""
            last_content = ""
            for attempt in range(1, args.max_retries + 1):
                try:
                    content, raw_usage = call_api(api_key, task["prompt"], args.model, args.timeout, max_tokens)
                    last_content = content
                    usage.add(raw_usage)
                    result = normalize_result(args.task_type, content)
                    validate_result(args.task_type, result)
                    record = {
                        "task_id": task_id,
                        "task_type": args.task_type,
                        "model": args.model,
                        "status": "ok",
                        "result": result,
                        "usage": raw_usage,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"[{index}/{len(tasks)}] {task_id} ok")
                    break
                except Exception as exc:
                    error = str(exc)
                    if attempt < args.max_retries:
                        time.sleep(min(20, 2 ** attempt + random.random()))
            else:
                handle.write(
                    json.dumps(
                        {"task_id": task_id, "task_type": args.task_type, "model": args.model, "status": "failed", "error": error[:1500], "raw_content": last_content},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                print(f"[{index}/{len(tasks)}] {task_id} failed: {error[:160]}")

    print(
        f"本次成功调用Token：输入 {usage.prompt_tokens}，缓存命中 {usage.cached_tokens}，"
        f"输出 {usage.completion_tokens}"
    )


if __name__ == "__main__":
    main()
