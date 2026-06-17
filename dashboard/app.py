from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    "cyan": "#38D5FF",
    "blue": "#2F80ED",
    "deep": "#071A3A",
    "orange": "#FFB86B",
    "green": "#53F2AE",
    "red": "#FF6B8A",
    "muted": "#9FB7D5",
    "grid": "rgba(99, 179, 237, 0.16)",
    "panel": "rgba(10, 30, 66, 0.72)",
}


st.set_page_config(page_title="低空经济动态情绪指数", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 12% 8%, rgba(56, 213, 255, 0.22), transparent 28%),
            radial-gradient(circle at 86% 16%, rgba(47, 128, 237, 0.28), transparent 30%),
            linear-gradient(135deg, #031226 0%, #071a3a 48%, #0a2a5e 100%) !important;
        color: #EAF6FF;
        font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif;
    }
    [data-testid="stHeader"] {
        background: rgba(3, 18, 38, 0.18);
        backdrop-filter: blur(16px);
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.4rem;
        max-width: 1500px;
    }
    h1 {
        color: #F5FBFF;
        letter-spacing: 0.03em;
        text-shadow: 0 0 26px rgba(56, 213, 255, 0.45);
    }
    h2, h3 {
        color: #D9F3FF;
    }
    [data-testid="stCaptionContainer"], .note {
        color: #9FB7D5 !important;
        font-size: 0.94rem;
    }
    [data-testid="stMetric"] {
        background:
            linear-gradient(145deg, rgba(12, 40, 88, 0.92), rgba(8, 24, 56, 0.86));
        border: 1px solid rgba(56, 213, 255, 0.28);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        padding: 16px 18px;
        border-radius: 18px;
    }
    [data-testid="stMetricLabel"] p {
        color: #9FDFFF !important;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
        text-shadow: 0 0 18px rgba(56, 213, 255, 0.36);
    }
    [data-testid="stMetricDelta"] {
        color: #53F2AE !important;
    }
    [data-testid="stTabs"] button {
        color: #AFCBE8;
        background: rgba(9, 34, 74, 0.42);
        border-radius: 999px;
        margin-right: 8px;
        border: 1px solid rgba(56, 213, 255, 0.16);
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #FFFFFF;
        background: linear-gradient(90deg, rgba(47,128,237,0.78), rgba(56,213,255,0.42));
        border-color: rgba(56, 213, 255, 0.55);
        box-shadow: 0 0 22px rgba(56, 213, 255, 0.25);
    }
    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"] {
        background: rgba(5, 20, 48, 0.62);
        border: 1px solid rgba(56, 213, 255, 0.18);
        border-radius: 18px;
        box-shadow: 0 18px 54px rgba(0, 0, 0, 0.24);
        padding: 8px;
    }
    div[data-testid="stDataFrame"] {
        overflow: hidden;
    }
    .stDateInput, .stMultiSelect {
        color: #EAF6FF;
    }
    a { color: #38D5FF; }
    .hero {
        position: relative;
        padding: 26px 30px;
        margin-bottom: 18px;
        border-radius: 24px;
        background:
            linear-gradient(120deg, rgba(8, 32, 78, 0.96), rgba(9, 72, 128, 0.66)),
            radial-gradient(circle at 88% 20%, rgba(56, 213, 255, 0.35), transparent 28%);
        border: 1px solid rgba(56, 213, 255, 0.32);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.10);
        overflow: hidden;
    }
    .hero:before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(56, 213, 255, 0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56, 213, 255, 0.08) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: linear-gradient(90deg, transparent 0%, black 18%, black 76%, transparent 100%);
    }
    .hero > * { position: relative; z-index: 1; }
    .hero-title {
        margin: 0;
        font-size: 2.15rem;
        font-weight: 800;
        color: #FFFFFF;
        text-shadow: 0 0 28px rgba(56, 213, 255, 0.48);
    }
    .hero-subtitle {
        margin-top: 8px;
        color: #BFEAFF;
        font-size: 1.02rem;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 16px;
    }
    .chip {
        padding: 6px 12px;
        border-radius: 999px;
        color: #EAF6FF;
        background: rgba(56, 213, 255, 0.12);
        border: 1px solid rgba(56, 213, 255, 0.28);
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,20,48,0.35)",
        font=dict(color="#DCEEFF", family="Microsoft YaHei, Noto Sans SC, Arial"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(56,213,255,0.18)"),
        margin=dict(l=22, r=22, t=48, b=28),
        hoverlabel=dict(bgcolor="#071A3A", font_color="#F5FBFF", bordercolor="#38D5FF"),
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return fig


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    model = pd.read_csv(DATA / "model_daily_dataset.csv", parse_dates=["trade_date"])
    universe = pd.read_csv(ROOT / "config" / "stock_universe.csv")
    summary_path = ROOT / "config" / "dashboard_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    company = pd.read_csv(DATA / "final_company_events_hybrid.csv", parse_dates=["event_date"])
    policy = pd.read_csv(DATA / "final_policy_events_hybrid.csv", parse_dates=["event_date"])
    metrics = pd.read_csv(DATA / "realtime_prediction_metrics.csv")
    shap = pd.read_csv(DATA / "realtime_prediction_shap.csv")
    quantile = pd.read_csv(DATA / "dynamic_index_quantile_regression.csv")
    lp = pd.read_csv(DATA / "dynamic_index_local_projections.csv")
    irf = pd.read_csv(DATA / "dynamic_index_var_irf.csv")
    placebo = pd.read_csv(DATA / "dynamic_index_placebo_event_study.csv")
    panel_metrics = pd.read_csv(DATA / "stock_panel_ensemble_metrics.csv")
    panel_scores = pd.read_csv(DATA / "stock_panel_latest_scores.csv", parse_dates=["trade_date"])
    ranking_metrics = pd.read_csv(DATA / "stock_panel_ranking_metrics.csv")
    top_precision = pd.read_csv(DATA / "stock_panel_daily_top_precision.csv")
    selective_accuracy = pd.read_csv(DATA / "stock_panel_selective_accuracy.csv")
    latest_signals = pd.read_csv(DATA / "stock_panel_latest_high_confidence_signals.csv", parse_dates=["trade_date"])
    return {
        "universe": universe, "summary": summary,
        "model": model, "company": company, "policy": policy, "metrics": metrics,
        "shap": shap, "quantile": quantile, "lp": lp, "irf": irf, "placebo": placebo,
        "panel_metrics": panel_metrics,
        "panel_scores": panel_scores, "ranking_metrics": ranking_metrics, "top_precision": top_precision,
        "selective_accuracy": selective_accuracy, "latest_signals": latest_signals,
    }


def overview(data: dict[str, pd.DataFrame]) -> None:
    model = data["model"].dropna(subset=["dynamic_sentiment_filtered"])
    latest = model.iloc[-1]
    previous = model.iloc[-2]
    accepted_company = data["company"]["accepted"].astype(bool).sum()
    accepted_policy = data["policy"]["accepted"].astype(bool).sum()
    summary = data["summary"]
    stock_count = int(summary.get("stock_count") or (data["universe"]["include"].astype(str) == "1").sum())
    market_rows = int(summary.get("market_rows") or 0)
    parsed_docs = int(summary.get("parsed_docs") or 0)
    selected_docs = int(summary.get("selected_docs") or 0)
    top5 = data["top_precision"][
        (data["top_precision"]["target"] == "volatility_jump_5d")
        & (data["top_precision"]["daily_top_fraction"].round(2) == 0.05)
    ].sort_values("lift", ascending=False).iloc[0]
    scale_cols = st.columns(5)
    scale_cols[0].metric("股票样本池", f"{stock_count} 只", "核心 + 扩展产业链")
    scale_cols[1].metric("个股日行情", f"{market_rows:,} 条")
    scale_cols[2].metric("有效公告正文", f"{parsed_docs:,} 篇", f"筛选 {selected_docs:,} 篇")
    scale_cols[3].metric("Top 5%风险命中", f"{top5['hit_rate']:.1%}", f"Lift {top5['lift']:.2f}x")
    scale_cols[4].metric("风险基准发生率", f"{top5['baseline_positive_share']:.1%}")

    cols = st.columns(4)
    cols[0].metric("历史解释指数", f"{latest['dynamic_sentiment_filtered']:.1f}", f"{latest['dynamic_sentiment_filtered'] - previous['dynamic_sentiment_filtered']:.2f}")
    realtime = model["realtime_sentiment_index"].dropna()
    cols[1].metric("实时指数", f"{realtime.iloc[-1]:.1f}", f"{realtime.iloc[-1] - realtime.iloc[-2]:.2f}")
    cols[2].metric("接纳公司事件", f"{accepted_company} 条")
    cols[3].metric("接纳政策事件", f"{accepted_policy} 条")

    chart = make_subplots(specs=[[{"secondary_y": True}]])
    chart.add_trace(go.Scatter(x=model["trade_date"], y=model["dynamic_sentiment_filtered"], name="历史解释指数", line=dict(color=COLORS["cyan"], width=2.4)), secondary_y=False)
    chart.add_trace(go.Scatter(x=model["trade_date"], y=model["realtime_sentiment_index"], name="实时扩展窗口指数", line=dict(color=COLORS["orange"], width=1.7)), secondary_y=False)
    chart.add_trace(go.Scatter(x=model["trade_date"], y=model["index_level"], name="低空经济市场指数", line=dict(color=COLORS["muted"], width=1.2)), secondary_y=True)
    chart.add_hline(y=50, line_dash="dot", line_color="rgba(159,183,213,0.65)", secondary_y=False)
    style_figure(chart, height=520)
    chart.update_layout(hovermode="x unified")
    chart.update_yaxes(title_text="情绪指数", secondary_y=False)
    chart.update_yaxes(title_text="市场指数", secondary_y=True)
    st.plotly_chart(chart, width="stretch")
    st.markdown('<p class="note">样本池覆盖60只低空经济核心及扩展产业链A股。历史解释指数使用全样本动态因子参数；实时指数按月仅使用历史数据重估，更适合预测对照。指数不构成投资建议。</p>', unsafe_allow_html=True)


def decomposition(data: dict[str, pd.DataFrame]) -> None:
    model = data["model"]
    start, end = st.date_input(
        "日期范围", value=(model["trade_date"].min().date(), model["trade_date"].max().date()),
        min_value=model["trade_date"].min().date(), max_value=model["trade_date"].max().date(),
    )
    selected = model[(model["trade_date"].dt.date >= start) & (model["trade_date"].dt.date <= end)]
    channels = ["policy_support_z", "company_sentiment_z", "confidence_balance_z", "signed_breadth_z"]
    labels = {"policy_support_z": "政策支持", "company_sentiment_z": "企业景气", "confidence_balance_z": "置信度余额", "signed_breadth_z": "事件广度"}
    long = selected[["trade_date", *channels]].melt("trade_date", var_name="channel", value_name="value")
    long["channel"] = long["channel"].map(labels)
    fig = px.line(
        long, x="trade_date", y="value", color="channel", title="动态指数四个标准化通道",
        color_discrete_sequence=[COLORS["cyan"], COLORS["green"], COLORS["orange"], COLORS["blue"]],
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(159,183,213,0.65)")
    style_figure(fig, height=500)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(selected[["trade_date", "dynamic_sentiment_filtered", "realtime_sentiment_index", *channels]].tail(30), width="stretch", hide_index=True)


def event_timeline(data: dict[str, pd.DataFrame]) -> None:
    company = data["company"][data["company"]["accepted"].astype(bool)].copy()
    policy = data["policy"][data["policy"]["accepted"].astype(bool)].copy()
    company["source"] = "公司公告"
    policy["source"] = "政策文件"
    company["display_title"] = company["name"].fillna("") + " | " + company["title"].fillna("")
    policy["display_title"] = policy["publisher"].fillna("") + " | " + policy["title"].fillna("")
    columns = ["event_date", "source", "event_type", "direction", "intensity", "uncertainty", "display_title", "evidence_span", "label_source"]
    events = pd.concat([company[columns], policy[columns]], ignore_index=True).sort_values("event_date", ascending=False)
    c1, c2, c3 = st.columns(3)
    sources = c1.multiselect("来源", sorted(events["source"].unique()), default=sorted(events["source"].unique()))
    types = c2.multiselect("事件类型", sorted(events["event_type"].unique()), default=[])
    directions = c3.multiselect("方向", sorted(events["direction"].unique()), default=[])
    filtered = events[events["source"].isin(sources)]
    if types:
        filtered = filtered[filtered["event_type"].isin(types)]
    if directions:
        filtered = filtered[filtered["direction"].isin(directions)]
    st.caption(f"当前显示 {len(filtered)} 条事件")
    st.dataframe(filtered, width="stretch", hide_index=True, column_config={"evidence_span": st.column_config.TextColumn(width="large"), "display_title": st.column_config.TextColumn("标题", width="large")})


def inference(data: dict[str, pd.DataFrame]) -> None:
    quantile, lp, irf, placebo = data["quantile"], data["lp"], data["irf"], data["placebo"]
    c1, c2 = st.columns(2)
    qfig = go.Figure()
    qfig.add_trace(go.Scatter(x=quantile["quantile"], y=quantile["coefficient"], mode="lines+markers", name="系数", line=dict(color=COLORS["cyan"], width=2.2), marker=dict(size=8), error_y=dict(type="data", array=1.96 * quantile["std_error"], color=COLORS["muted"])))
    qfig.add_hline(y=0, line_color="rgba(159,183,213,0.65)")
    qfig.update_layout(title="五日收益分位数回归")
    style_figure(qfig, height=400)
    c1.plotly_chart(qfig, width="stretch")
    lpfig = go.Figure(go.Scatter(x=lp["horizon"], y=lp["coefficient"], mode="lines", line=dict(color=COLORS["green"], width=2.1), fill=None, name="系数"))
    lpfig.add_trace(go.Scatter(x=pd.concat([lp["horizon"], lp["horizon"][::-1]]), y=pd.concat([lp["ci_high"], lp["ci_low"][::-1]]), fill="toself", fillcolor="rgba(83,242,174,0.16)", line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    lpfig.add_hline(y=0, line_color="rgba(159,183,213,0.65)")
    lpfig.update_layout(title="局部投影")
    style_figure(lpfig, height=400)
    c2.plotly_chart(lpfig, width="stretch")
    c3, c4 = st.columns(2)
    irffig = px.line(irf, x="horizon", y="excess_return_response_to_sentiment_shock", title="VAR：情绪冲击后的超额收益响应")
    irffig.update_traces(line=dict(color=COLORS["orange"], width=2.2))
    irffig.add_hline(y=0, line_color="rgba(159,183,213,0.65)")
    style_figure(irffig, height=400)
    c3.plotly_chart(irffig, width="stretch")
    c4.dataframe(placebo, width="stretch", hide_index=True)
    placebo_5d = placebo.loc[placebo["horizon"].eq(5), "placebo_p_value_two_sided"]
    placebo_text = f"{placebo_5d.iloc[0]:.3f}" if not placebo_5d.empty else "N/A"
    st.info(f"分位数回归显示非对称关系；局部投影没有稳定平均效应。5日事件研究的随机日期安慰剂 p={placebo_text}，需要与样本扩容和稳健性检验一起解读。")


def prediction(data: dict[str, pd.DataFrame]) -> None:
    metrics, shap_values, panel_metrics = data["metrics"], data["shap"], data["panel_metrics"]
    panel_scores, ranking_metrics = data["panel_scores"], data["ranking_metrics"]
    top_precision = data["top_precision"]
    selective_accuracy, latest_signals = data["selective_accuracy"], data["latest_signals"]
    st.subheader("个股面板模型")
    st.dataframe(panel_metrics.sort_values("auc", ascending=False), width="stretch", hide_index=True)
    best_panel = panel_metrics.sort_values("auc", ascending=False).iloc[0]
    st.success(f"当前最佳面板任务为 {best_panel['target']} / {best_panel['subset']}，AUC={best_panel['auc']:.3f}，95%区间 [{best_panel['auc_ci_low']:.3f}, {best_panel['auc_ci_high']:.3f}]。")
    c1, c2 = st.columns(2)
    c1.dataframe(panel_scores.sort_values("relative_strength_rank"), width="stretch", hide_index=True)
    c2.dataframe(ranking_metrics, width="stretch", hide_index=True)
    st.subheader("日度Top-K风险识别")
    top_view = top_precision.sort_values(["target", "daily_top_fraction", "lift"], ascending=[True, True, False])
    st.dataframe(top_view, width="stretch", hide_index=True)
    best_top = top_precision.sort_values("lift", ascending=False).iloc[0]
    st.info(f"Top-K模块当前最强信号为 {best_top['target']} / Top {best_top['daily_top_fraction']:.0%}，命中率 {best_top['hit_rate']:.1%}，相对基准提升 {best_top['lift']:.2f} 倍。")
    st.subheader("高置信度信号")
    best_selective = selective_accuracy.sort_values("accuracy", ascending=False).groupby("target").head(3)
    st.dataframe(best_selective, width="stretch", hide_index=True)
    st.dataframe(latest_signals.sort_values(["volatility_jump_probability", "relative_strength_probability"], ascending=False), width="stretch", hide_index=True)
    st.subheader("指数方向预测基线")
    st.dataframe(metrics.sort_values("auc", ascending=False), width="stretch", hide_index=True)
    fig = px.bar(shap_values.head(15).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h", title="SHAP 全局重要性", color_discrete_sequence=[COLORS["cyan"]])
    style_figure(fig, height=500)
    st.plotly_chart(fig, width="stretch")
    best = metrics.sort_values("auc", ascending=False).iloc[0]
    st.warning(f"最佳样本外 AUC 为 {best['auc']:.3f}（{best['model']} / {best['feature_set']}）。实时情绪指数没有带来稳定增益，本模块定位为辅助预测与对照实验。")


data = load_data()
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">低空经济动态情绪指数研究与可视化系统</div>
      <div class="hero-subtitle">LLM结构化事件抽取 · 动态因子指数 · 事件研究 / VAR / 分位数回归 · 极端表现与波动风险预警</div>
      <div class="chip-row">
        <span class="chip">60-stock Industry Chain Universe</span>
        <span class="chip">98,761 Stock-day Records</span>
        <span class="chip">LLM Event-driven Index</span>
        <span class="chip">Top-K Risk Warning</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
tabs = st.tabs(["研究总览", "指数拆解", "事件时间线", "统计检验", "预测解释"])
with tabs[0]:
    overview(data)
with tabs[1]:
    decomposition(data)
with tabs[2]:
    event_timeline(data)
with tabs[3]:
    inference(data)
with tabs[4]:
    prediction(data)
