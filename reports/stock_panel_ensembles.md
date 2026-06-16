# 个股面板概率集成

集成直接平均各基础模型的样本外概率，不训练二级模型。置信区间按交易日自助抽样1000次，以保留同日股票之间的横截面相关性。

| target              | subset           | feature_set             | ensemble                                    |   observations |   positive_share |      auc |   accuracy |   balanced_accuracy |       f1 |    brier |   auc_ci_low |   auc_ci_high |
|:--------------------|:-----------------|:------------------------|:--------------------------------------------|---------------:|-----------------:|---------:|-----------:|--------------------:|---------:|---------:|-------------:|--------------:|
| extreme_relative_5d | all              | technical_global_events | logistic+hist_gradient_boosting+extra_trees |          31559 |         0.513926 | 0.536687 |   0.53297  |            0.530944 | 0.570554 | 0.252056 |     0.52815  |      0.545466 |
| extreme_relative_5d | event_window_20d | technical_global_events | logistic+hist_gradient_boosting+extra_trees |            331 |         0.486405 | 0.654914 |   0.610272 |            0.610723 | 0.610272 | 0.237265 |     0.598859 |      0.716147 |
| volatility_jump_5d  | all              | technical_events        | logistic+hist_gradient_boosting+extra_trees |          52061 |         0.240103 | 0.679484 |   0.685869 |            0.61699  | 0.42549  | 0.198547 |     0.665909 |      0.691723 |
| volatility_jump_5d  | event_window_20d | technical_events        | logistic+hist_gradient_boosting+extra_trees |            577 |         0.272097 | 0.685093 |   0.639515 |            0.604807 | 0.44385  | 0.208112 |     0.638678 |      0.733915 |
| volatility_jump_5d  | all              | technical_events        | logistic+extra_trees                        |          52061 |         0.240103 | 0.678794 |   0.602063 |            0.631373 | 0.453536 | 0.229071 |     0.6663   |      0.690749 |
| volatility_jump_5d  | event_window_20d | technical_events        | logistic+extra_trees                        |            577 |         0.272097 | 0.6968   |   0.623917 |            0.649932 | 0.505695 | 0.229429 |     0.654006 |      0.745352 |

集成方案是在本轮探索中比较后选定，属于探索性模型选择；最终论文若将其作为主结果，应使用独立留出期再次确认。
