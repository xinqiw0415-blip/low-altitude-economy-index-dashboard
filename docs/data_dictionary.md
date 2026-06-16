# 数据字典 v0.1

## `market_daily`

| 字段 | 含义 |
|---|---|
| `ts_code` | 带交易所后缀的证券代码 |
| `name` | 证券简称，以采集接口返回值为准 |
| `trade_date` | 交易日，ISO 格式 |
| `open/high/low/close` | 前复权开高低收 |
| `volume` | 成交量，接口原始单位 |
| `amount` | 成交额，元 |
| `amplitude_pct` | 振幅，百分数 |
| `pct_change` | 涨跌幅，百分数 |
| `price_change` | 涨跌额 |
| `turnover_pct` | 换手率，百分数 |
| `source` | 数据来源 |
| `fetched_at` | UTC 抓取时间 |

## 后续政策文本表 `policy_document`

至少包含：`document_id`、`title`、`publisher`、`publish_date`、`region`、`url`、`full_text`、`source_domain`、`fetched_at`、`content_sha256`。原文和结构化事件表必须分开保存，LLM 抽取结果不能覆盖原始文本。

## 时间规则

- 行情统一使用交易所交易日。
- 文本同时保存网页公布时间和采集时间。
- 收盘后发布的事件只能进入下一交易日特征，防止未来信息泄漏。
