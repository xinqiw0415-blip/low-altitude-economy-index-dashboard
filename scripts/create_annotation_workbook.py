from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    events = pd.read_csv(ROOT / "data" / "interim" / "policy_events_baseline.csv")
    columns = [
        "event_id", "document_id", "event_date", "event_type", "actor", "region", "policy_level",
        "policy_tools", "targets", "direction", "intensity", "uncertainty", "novelty",
        "evidence_span", "confidence", "annotator", "review_status", "review_note"
    ]
    events["annotator"] = ""
    events["review_note"] = ""
    output = ROOT / "data" / "interim" / "event_annotation_template.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        events[columns].to_excel(writer, sheet_name="events", index=False)
        pd.DataFrame(
            {
                "field": ["event_type", "direction", "intensity", "uncertainty", "novelty"],
                "allowed_values": [
                    "strategic_plan/support_measure/regulation/airspace_reform/infrastructure/fiscal_subsidy/industry_standard/insurance_finance/other",
                    "positive/negative/mixed/neutral", "1-5", "1-5", "1-5"
                ],
            }
        ).to_excel(writer, sheet_name="codebook", index=False)
    print(f"已生成标注模板：{output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
