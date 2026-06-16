from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    low = pd.read_csv(ROOT / "data" / "processed" / "low_altitude_index_daily.csv", parse_dates=["trade_date"])
    bench = pd.read_csv(ROOT / "data" / "interim" / "benchmark_daily.csv", parse_dates=["trade_date"])
    hs300 = bench[bench["index_code"] == "HS300"].copy()
    hs300["normalized"] = 1000 * hs300["close"] / hs300["close"].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 5))
    for code, group in low.groupby("index_code"):
        ax.plot(group["trade_date"], group["index_level"], label=code, linewidth=1.3)
    ax.plot(hs300["trade_date"], hs300["normalized"], label="HS300", linewidth=1.1, alpha=0.8)
    ax.set_title("低空经济自建指数与沪深300（起点=1000）")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "index_comparison.png", dpi=180)
    plt.close(fig)

    climate = pd.read_csv(ROOT / "data" / "processed" / "policy_climate_daily.csv", parse_dates=["trade_date"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(climate["trade_date"], climate["policy_climate_index"], color="#b23a48")
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("政策气候指数：透明词典基线")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "policy_climate_index.png", dpi=180)
    plt.close(fig)

    lp = pd.read_csv(ROOT / "data" / "processed" / "local_projection_baseline.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lp["horizon"], lp["coefficient"], marker="o", markersize=3)
    ax.fill_between(
        lp["horizon"],
        lp["coefficient"] - 1.96 * lp["std_error"],
        lp["coefficient"] + 1.96 * lp["std_error"],
        alpha=0.2,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("政策强度冲击的局部投影基线")
    ax.set_xlabel("交易日响应期")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "local_projection.png", dpi=180)
    plt.close(fig)
    print(f"图表输出：{FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
