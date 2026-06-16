from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
LEXICONS = {
    "support": ["支持", "促进", "推动", "鼓励", "加快", "保障", "奖励", "补贴"],
    "implementation": ["责任单位", "完成时限", "实施", "建设", "制定", "落实", "任务"],
    "innovation": ["创新", "研发", "技术", "实验室", "示范", "应用场景"],
    "uncertainty_language": ["探索", "研究", "适时", "逐步", "力争", "有序", "条件成熟"],
    "risk_regulation": ["风险", "安全", "监管", "处罚", "限制", "禁止", "事故"]
}


def density(text: str, words: list[str]) -> float:
    return 1000 * sum(text.count(word) for word in words) / max(len(text), 1)


def main() -> None:
    documents = pd.read_csv(ROOT / "data" / "interim" / "policy_documents.csv")
    events = pd.read_csv(ROOT / "data" / "processed" / "policy_events.csv", parse_dates=["information_trade_date"])
    calendar = pd.read_csv(ROOT / "data" / "processed" / "low_altitude_index_daily.csv", parse_dates=["trade_date"])
    calendar = pd.Series(sorted(calendar["trade_date"].unique()))
    documents = documents[documents["fetch_status"] == "ok"].copy()
    for label, words in LEXICONS.items():
        documents[label] = documents["full_text"].fillna("").map(lambda text: density(text, words))
    document_scores = documents[["document_id", *LEXICONS.keys()]]
    events = events.merge(document_scores, on="document_id", how="left")

    daily = pd.DataFrame({"trade_date": calendar})
    dimensions = list(LEXICONS)
    for dimension in dimensions:
        shocks = events.groupby("information_trade_date").apply(
            lambda group: float((group[dimension] * group["intensity"]).sum()),
            include_groups=False,
        )
        daily[dimension] = daily["trade_date"].map(shocks).fillna(0.0)
        daily[dimension] = daily[dimension].ewm(halflife=20, adjust=False).mean()

    active = daily[dimensions].abs().sum(axis=1) > 0
    standardized = np.zeros((len(daily), len(dimensions)))
    standardized[active] = StandardScaler().fit_transform(daily.loc[active, dimensions])
    factor = np.zeros(len(daily))
    if active.sum() >= 2:
        pca = PCA(n_components=1)
        factor[active] = pca.fit_transform(standardized[active]).ravel()
        support_index = dimensions.index("support")
        if pca.components_[0, support_index] < 0:
            factor *= -1
        explained = float(pca.explained_variance_ratio_[0])
    else:
        explained = 0.0
    daily["policy_climate_factor"] = factor
    active_values = daily.loc[active, "policy_climate_factor"]
    daily["policy_climate_index"] = 50.0
    if len(active_values) and active_values.std() > 0:
        daily.loc[active, "policy_climate_index"] = (
            50 + 10 * (active_values - active_values.mean()) / active_values.std()
        ).clip(0, 100)
    output = ROOT / "data" / "processed" / "policy_climate_daily.csv"
    daily.to_csv(output, index=False, encoding="utf-8-sig")
    metadata = {
        "method": "keyword density -> intensity-weighted daily shocks -> 20-trading-day EWMA -> PCA",
        "explained_variance_ratio": explained,
        "documents": int(len(documents)),
        "warning": "This is a transparent lexical baseline, not the final LLM/dynamic-factor index."
    }
    (ROOT / "reports" / "policy_climate_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
