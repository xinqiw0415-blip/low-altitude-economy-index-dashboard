# 模型与裁决后金标准比较

> 金标准包含10条A/B一致样本和30条Codex辅助裁决样本。裁决未读取DeepSeek预测，但它不等同于第三位独立人工专家标注。

## 分类指标

| model         | task       |   accuracy |   macro_f1 |    kappa |
|:--------------|:-----------|-----------:|-----------:|---------:|
| DeepSeek      | relevance  |      0.725 |   0.68     | 0.411765 |
| Rule baseline | relevance  |      0.825 |   0.452055 | 0        |
| DeepSeek      | event_type |      0.8   |   0.758377 | 0.769784 |
| Rule baseline | event_type |      0.725 |   0.672593 | 0.678363 |
| DeepSeek      | direction  |      0.875 |   0.651961 | 0.766082 |

## 强度与不确定性

| field       |   mae |   weighted_kappa |
|:------------|------:|-----------------:|
| intensity   |  0.45 |         0.634615 |
| uncertainty |  0.65 |         0.756757 |

DeepSeek与金标准存在分类分歧的样本：12 / 40。
