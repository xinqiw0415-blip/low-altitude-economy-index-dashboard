from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = sorted((ROOT / "data").glob("interim/*")) + sorted((ROOT / "data").glob("processed/*"))
    files = [
        path for path in paths
        if path.is_file() and path.name != ".gitkeep" and not path.name.startswith("~$")
    ]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
    }
    output = ROOT / "data" / "manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据清单：{len(files)} 个文件")


if __name__ == "__main__":
    main()
