from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RNG = np.random.default_rng(20260616)


def score(data: pd.DataFrame, probability: pd.Series) -> dict:
    truth = data["truth"].astype(int)
    prediction = probability.ge(0.5).astype(int)
    return {
        "observations": len(data), "positive_share": truth.mean(), "auc": roc_auc_score(truth, probability),
        "accuracy": accuracy_score(truth, prediction), "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "f1": f1_score(truth, prediction), "brier": brier_score_loss(truth, probability),
    }


def date_bootstrap(data: pd.DataFrame, repetitions: int = 1000) -> tuple[float, float]:
    dates = data["trade_date"].unique()
    date_codes = pd.Categorical(data["trade_date"], categories=dates).codes
    values = []
    for _ in range(repetitions):
        counts = RNG.multinomial(len(dates), np.repeat(1 / len(dates), len(dates)))
        weights = counts[date_codes]
        if data.loc[weights > 0, "truth"].nunique() == 2:
            values.append(roc_auc_score(data["truth"], data["ensemble_probability"], sample_weight=weights))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    predictions = pd.read_csv(PROCESSED / "stock_panel_prediction_oos.csv", parse_dates=["trade_date"])
    specs = [
        ("extreme_relative_5d", "technical_global_events", ["logistic", "hist_gradient_boosting", "extra_trees"]),
        ("volatility_jump_5d", "technical_events", ["logistic", "hist_gradient_boosting", "extra_trees"]),
        ("volatility_jump_5d", "technical_events", ["logistic", "extra_trees"]),
    ]
    rows, outputs = [], []
    for target, feature_set, model_names in specs:
        selected = predictions[
            predictions["target"].eq(target) & predictions["feature_set"].eq(feature_set)
            & predictions["model"].isin(model_names)
        ]
        wide = selected.pivot_table(
            index=["trade_date", "ts_code", "name", "truth", "fold", "event_window_20d"],
            columns="model", values="probability"
        ).dropna().reset_index()
        wide["ensemble_probability"] = wide[model_names].mean(axis=1)
        ensemble_name = "+".join(model_names)
        for subset, mask in [("all", pd.Series(True, index=wide.index)), ("event_window_20d", wide["event_window_20d"].eq(1))]:
            sample = wide[mask].copy()
            if len(sample) < 30 or sample["truth"].nunique() < 2:
                continue
            ci_low, ci_high = date_bootstrap(sample)
            rows.append({
                "target": target, "subset": subset, "feature_set": feature_set,
                "ensemble": ensemble_name, **score(sample, sample["ensemble_probability"]),
                "auc_ci_low": ci_low, "auc_ci_high": ci_high,
            })
        wide["target"] = target
        wide["feature_set"] = feature_set
        wide["ensemble"] = ensemble_name
        outputs.append(wide)
    metrics = pd.DataFrame(rows)
    ensemble_predictions = pd.concat(outputs, ignore_index=True)
    metrics.to_csv(PROCESSED / "stock_panel_ensemble_metrics.csv", index=False, encoding="utf-8-sig")
    ensemble_predictions.to_csv(PROCESSED / "stock_panel_ensemble_oos.csv", index=False, encoding="utf-8-sig")
    report = [
        "# 个股面板概率集成", "",
        "集成直接平均各基础模型的样本外概率，不训练二级模型。置信区间按交易日自助抽样1000次，以保留同日股票之间的横截面相关性。", "",
        metrics.to_markdown(index=False), "",
        "集成方案是在本轮探索中比较后选定，属于探索性模型选择；最终论文若将其作为主结果，应使用独立留出期再次确认。",
    ]
    (ROOT / "reports" / "stock_panel_ensembles.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
