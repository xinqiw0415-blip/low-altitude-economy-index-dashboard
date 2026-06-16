# 严格时序预测基线

| model                  | feature_set                       |   observations |      auc |   accuracy |       f1 |    brier |
|:-----------------------|:----------------------------------|---------------:|---------:|-----------:|---------:|---------:|
| logistic               | technical_only                    |           1500 | 0.49428  |   0.488667 | 0.478586 | 0.261286 |
| logistic               | technical_plus_rule_events        |           1500 | 0.513949 |   0.508667 | 0.541952 | 0.270359 |
| logistic               | technical_plus_llm_events         |           1500 | 0.508886 |   0.508    | 0.463663 | 0.28602  |
| logistic               | technical_plus_adjudicated_hybrid |           1500 | 0.50623  |   0.503333 | 0.457393 | 0.285451 |
| hist_gradient_boosting | technical_only                    |           1500 | 0.51292  |   0.51     | 0.5174   | 0.284078 |
| hist_gradient_boosting | technical_plus_rule_events        |           1500 | 0.512019 |   0.521333 | 0.537371 | 0.295913 |
| hist_gradient_boosting | technical_plus_llm_events         |           1500 | 0.48833  |   0.487333 | 0.46782  | 0.30334  |
| hist_gradient_boosting | technical_plus_adjudicated_hybrid |           1500 | 0.488132 |   0.488    | 0.469613 | 0.305991 |

使用5折时间序列切分并设置20个交易日间隔。裁决后混合事件特征由40条双人标注样本及其裁决结果增强，其余公司事件和政策事件仍采用DeepSeek证据过滤。近期新闻仅覆盖2026年4月至6月，未进入长期预测。结果用于检验增量信息，不据此宣称可交易性。
