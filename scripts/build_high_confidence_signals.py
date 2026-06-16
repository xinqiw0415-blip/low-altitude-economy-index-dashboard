from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def selective_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, feature_set, ensemble), group in frame.groupby(["target", "feature_set", "ensemble"]):
        data = group.copy()
        data["confidence"] = (data["ensemble_probability"] - 0.5).abs()
        for coverage in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]:
            n = max(1, int(len(data) * coverage))
            selected = data.nlargest(n, "confidence")
            prediction = selected["ensemble_probability"].ge(0.5).astype(int)
            rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "ensemble": ensemble,
                    "mode": "high_confidence_direction",
                    "coverage": coverage,
                    "observations": len(selected),
                    "accuracy": accuracy_score(selected["truth"], prediction),
                    "balanced_accuracy": balanced_accuracy_score(selected["truth"], prediction),
                    "f1": f1_score(selected["truth"], prediction, zero_division=0),
                    "positive_share": selected["truth"].mean(),
                }
            )
    return pd.DataFrame(rows)


def daily_top_precision(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, feature_set, ensemble), group in frame.groupby(["target", "feature_set", "ensemble"]):
        for fraction in [0.05, 0.10, 0.15, 0.20, 0.30]:
            selected_parts = []
            for _, day in group.groupby("trade_date"):
                k = max(1, int(np.ceil(len(day) * fraction)))
                selected_parts.append(day.nlargest(k, "ensemble_probability"))
            selected = pd.concat(selected_parts, ignore_index=True)
            baseline = group["truth"].mean()
            precision = precision_score(selected["truth"], selected["ensemble_probability"].ge(0.5), zero_division=0)
            hit_rate = selected["truth"].mean()
            rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "ensemble": ensemble,
                    "mode": "daily_top_positive",
                    "daily_top_fraction": fraction,
                    "observations": len(selected),
                    "hit_rate": hit_rate,
                    "baseline_positive_share": baseline,
                    "lift": hit_rate / baseline if baseline > 0 else np.nan,
                    "precision_at_threshold_0_5": precision,
                }
            )
    return pd.DataFrame(rows)


def latest_high_confidence() -> pd.DataFrame:
    scores = pd.read_csv(PROCESSED / "stock_panel_latest_scores.csv", parse_dates=["trade_date"])
    rows = []
    for item in scores.itertuples(index=False):
        strength_confidence = abs(item.relative_strength_probability - 0.5)
        volatility_confidence = abs(item.volatility_jump_probability - 0.5)
        rows.append(
            {
                "trade_date": item.trade_date,
                "ts_code": item.ts_code,
                "name": item.name,
                "industry_role": item.industry_role,
                "relative_strength_probability": item.relative_strength_probability,
                "relative_strength_signal": "强势关注" if item.relative_strength_probability >= 0.60 else ("弱势回避" if item.relative_strength_probability <= 0.40 else "观望"),
                "relative_strength_confidence": strength_confidence,
                "volatility_jump_probability": item.volatility_jump_probability,
                "volatility_signal": "高波动风险" if item.volatility_jump_probability >= 0.55 else ("低波动风险" if item.volatility_jump_probability <= 0.35 else "观望"),
                "volatility_confidence": volatility_confidence,
                "event_window_20d": item.event_window_20d,
                "days_since_company_event": item.days_since_company_event,
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(PROCESSED / "stock_panel_latest_high_confidence_signals.csv", index=False, encoding="utf-8-sig")
    return output


def main() -> None:
    frame = pd.read_csv(PROCESSED / "stock_panel_ensemble_oos.csv", parse_dates=["trade_date"])
    selective = selective_accuracy(frame)
    top = daily_top_precision(frame)
    latest = latest_high_confidence()
    selective.to_csv(PROCESSED / "stock_panel_selective_accuracy.csv", index=False, encoding="utf-8-sig")
    top.to_csv(PROCESSED / "stock_panel_daily_top_precision.csv", index=False, encoding="utf-8-sig")

    best_selective = selective.sort_values("accuracy", ascending=False).groupby("target").head(3)
    best_top = top.sort_values("lift", ascending=False).groupby("target").head(3)
    report = [
        "# 高置信度预测与Top关注策略",
        "",
        "本报告将预测目标从“所有样本都必须给出方向”改为“只在高置信度区域给出信号，其余观望”。这会降低覆盖率，但能显著提高可解释的命中率，适合系统中的风险预警和重点关注名单。",
        "",
        "## 高置信度方向预测",
        "",
        best_selective.to_markdown(index=False),
        "",
        "## 每日Top正类筛选",
        "",
        best_top.to_markdown(index=False),
        "",
        "## 最新信号",
        "",
        latest.sort_values(["relative_strength_signal", "volatility_jump_probability"], ascending=[True, False]).to_markdown(index=False),
        "",
        "解释：高置信度准确率不是全样本准确率，它回答的是“当系统决定出手时有多准”。因此必须同时报告coverage，避免只报高准确率造成误导。",
    ]
    (REPORTS / "stock_panel_high_confidence_signals.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(best_selective.to_string(index=False))
    print(best_top.to_string(index=False))


if __name__ == "__main__":
    main()
