from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "human_relevant", "human_event_type", "human_direction", "human_intensity",
    "human_uncertainty", "human_evidence_span", "human_note"
]


def main() -> None:
    source = ROOT / "data" / "interim" / "human_annotation_A.xlsx"
    tasks = pd.read_excel(source, sheet_name="tasks")
    instructions = pd.read_excel(source, sheet_name="instructions")
    tasks[FIELDS] = ""
    output = ROOT / "data" / "interim" / "human_annotation_B_independent.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        tasks.to_excel(writer, sheet_name="tasks", index=False)
        instructions.to_excel(writer, sheet_name="instructions", index=False)
    print(f"已生成独立复标表：{output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
