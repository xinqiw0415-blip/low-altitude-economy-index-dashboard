# 公司事件特征增量检验

针对事件后20日的极端相对强弱目标，比较同一批样本上的技术基线与事件增强模型。置信区间按最近一次公司事件聚类，自助抽样2000次。

## 分时间折结果

| feature_set             |   fold |   observations |      auc |
|:------------------------|-------:|---------------:|---------:|
| technical               |      1 |             75 | 0.427128 |
| technical               |      2 |             56 | 0.356952 |
| technical               |      3 |             43 | 0.521645 |
| technical               |      4 |             56 | 0.642401 |
| technical               |      5 |            101 | 0.73254  |
| technical_events        |      1 |             75 | 0.747475 |
| technical_events        |      2 |             56 | 0.65508  |
| technical_events        |      3 |             43 | 0.489177 |
| technical_events        |      4 |             56 | 0.692209 |
| technical_events        |      5 |            101 | 0.509127 |
| technical_global_events |      1 |             75 | 0.787157 |
| technical_global_events |      2 |             56 | 0.707219 |
| technical_global_events |      3 |             43 | 0.482684 |
| technical_global_events |      4 |             56 | 0.685824 |
| technical_global_events |      5 |            101 | 0.50119  |

## 事件簇自助法

| metric                                        |   estimate |     ci_low |   ci_high |   clusters |   observations | comparison            |
|:----------------------------------------------|-----------:|-----------:|----------:|-----------:|---------------:|:----------------------|
| technical                                     |  0.542565  |  0.441283  |  0.658859 |         30 |            331 | events_without_global |
| technical_events                              |  0.607234  |  0.513297  |  0.704689 |         30 |            331 | events_without_global |
| delta_technical_events_minus_technical        |  0.0646693 | -0.092667  |  0.230503 |         30 |            331 | events_without_global |
| technical                                     |  0.542565  |  0.444308  |  0.65649  |         30 |            331 | events_with_global    |
| technical_global_events                       |  0.631421  |  0.516157  |  0.734086 |         30 |            331 | events_with_global    |
| delta_technical_global_events_minus_technical |  0.0888564 | -0.0901904 |  0.263753 |         30 |            331 | events_with_global    |

若AUC增量置信区间包含0，则不能认为事件特征带来了稳定提升，即使点估计更高。
