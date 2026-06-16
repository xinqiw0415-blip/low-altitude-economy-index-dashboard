# 长期访问链接部署说明

当前 `trycloudflare.com` 链接是临时隧道，电脑关机、断网或进程退出后会失效。长期链接建议使用下面两种方案之一。

## 推荐方案：Streamlit Community Cloud

适合论文展示、简历项目、临时答辩链接。优点是免费、链接长期稳定、不依赖本机开机。

### 1. 上传到 GitHub

将项目推送到 GitHub。注意：面板不需要以下大文件，已在 `.gitignore` 中排除：

- `data/processed/stock_panel_prediction_oos.csv`
- `data/processed/stock_panel_daily.csv`
- `data/processed/stock_panel_ensemble_oos.csv`
- `data/raw/announcements/pdf/`
- `data/raw/announcements/text/`

### 2. Streamlit Cloud 设置

进入 https://share.streamlit.io/ ，连接 GitHub 仓库，然后设置：

- Repository：你的项目仓库
- Branch：`main`
- Main file path：`dashboard/app.py`

部署完成后会得到一个长期链接，格式通常类似：

```text
https://你的项目名.streamlit.app
```

### 3. 需要随仓库保留的数据

面板当前会读取这些相对较小的结果文件，推送仓库时需要保留：

- `config/stock_universe.csv`
- `data/interim/market_daily.csv`
- `data/interim/company_announcements_metadata.csv`
- `data/interim/company_announcement_documents.csv`
- `data/processed/model_daily_dataset.csv`
- `data/processed/final_company_events_hybrid.csv`
- `data/processed/final_policy_events_hybrid.csv`
- `data/processed/realtime_prediction_metrics.csv`
- `data/processed/realtime_prediction_shap.csv`
- `data/processed/dynamic_index_quantile_regression.csv`
- `data/processed/dynamic_index_local_projections.csv`
- `data/processed/dynamic_index_var_irf.csv`
- `data/processed/dynamic_index_placebo_event_study.csv`
- `data/processed/stock_panel_ensemble_metrics.csv`
- `data/processed/stock_panel_latest_scores.csv`
- `data/processed/stock_panel_ranking_metrics.csv`
- `data/processed/stock_panel_daily_top_precision.csv`
- `data/processed/stock_panel_selective_accuracy.csv`
- `data/processed/stock_panel_latest_high_confidence_signals.csv`

## 备选方案：Cloudflare 固定域名隧道

适合你已经有域名，且愿意让自己的电脑或服务器长期在线的情况。

需要条件：

- 一个自己的域名
- Cloudflare 账号
- 本机或服务器保持开机
- 使用 `cloudflared tunnel create` 创建 named tunnel
- 在 Cloudflare DNS 中绑定固定子域名，例如 `lae-dashboard.yourdomain.com`

这个方案链接稳定，但依赖本机/服务器持续运行；如果只是给老师或同学看，Streamlit Cloud 更省心。

