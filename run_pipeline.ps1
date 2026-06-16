$ErrorActionPreference = "Stop"

python scripts/audit_legacy_data.py
python scripts/collect_market_data.py --start 2019-01-01 --end 2026-06-15
python scripts/validate_market_data.py
python scripts/collect_benchmark_data.py --start 2019-01-01 --end 2026-06-15
python scripts/build_low_altitude_index.py
python scripts/collect_policy_documents.py
python scripts/validate_policy_documents.py
python scripts/extract_policy_events_baseline.py
python scripts/build_event_features.py
python scripts/build_policy_climate_index.py
python scripts/run_baseline_event_study.py
python scripts/export_llm_event_tasks.py
python scripts/create_annotation_workbook.py
python scripts/collect_company_announcements.py
python scripts/build_company_event_candidates.py
python scripts/validate_announcement_data.py
python scripts/export_company_llm_tasks.py
# DeepSeek调用按需单独运行，避免每次重建数据都重复产生API费用：
# python scripts/run_deepseek_extraction.py --task-type company
# python scripts/run_deepseek_extraction.py --task-type policy
python scripts/build_company_event_features.py
python scripts/run_company_event_study.py
python scripts/collect_news_metadata.py --max-records 3000 --start-date 2024-01-01
python scripts/build_news_features.py
python scripts/validate_news_data.py
python scripts/finalize_deepseek_events.py
python scripts/build_deepseek_event_features.py
python scripts/run_deepseek_company_event_study.py
python scripts/create_double_annotation_packets.py
python scripts/build_provisional_final_event_library.py
python scripts/build_provisional_event_features.py
python scripts/build_dynamic_sentiment_index.py
python scripts/build_realtime_sentiment_index.py
python scripts/build_model_dataset.py
python scripts/run_statistical_baselines.py
python scripts/run_dynamic_index_inference.py
python scripts/run_dynamic_index_robustness.py
python scripts/run_prediction_baseline.py
python scripts/run_realtime_prediction.py
python scripts/build_stock_panel_dataset.py
python scripts/run_stock_panel_prediction.py
python scripts/evaluate_stock_panel_uplift.py
python scripts/build_stock_panel_ensembles.py
python scripts/generate_stock_panel_scores.py
python scripts/build_high_confidence_signals.py
python scripts/create_baseline_figures.py
python scripts/generate_data_manifest.py
