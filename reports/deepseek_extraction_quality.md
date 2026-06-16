# DeepSeek事件抽取质量报告

- 公司任务：115，证据可回溯：96，自动接纳：33
- 公司规则误报/无关：60
- 政策子事件：136，证据可回溯：123，自动接纳：123

## 公司事件类型

| event_type            |   events |
|:----------------------|---------:|
| project_investment    |       17 |
| government_support    |        5 |
| order_contract        |        5 |
| strategic_cooperation |        3 |
| legal_safety_risk     |        1 |
| product_technology    |        1 |
| performance           |        1 |

## 政策事件类型（统一映射后）

| event_type        |   events |
|:------------------|---------:|
| fiscal_subsidy    |       44 |
| regulation        |       17 |
| infrastructure    |       16 |
| support_measure   |       15 |
| airspace_reform   |       15 |
| strategic_plan    |       10 |
| industry_standard |        4 |
| other             |        2 |

所有事件仍标记为待人工复核。`accepted`只表示通过自动证据与相关性门槛，不代表人工金标准。
