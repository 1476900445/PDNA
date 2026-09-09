# 数据接口

LLM_DATA 的 train.jsonl 每行是一段完整协商，字段包含 dialogue_id、
scenario_id、turns、agreement 和 final_utility。每个 turn 包含 turn_id、
speaker、utterance、personality（O/C/E/A/N 五维）、intent（0-22）、
strategy（0-9）、politeness、addon_sensitivity、offer 和 metadata。

NEGMAS_DATA 的 offline.jsonl 与 online.jsonl 每行包含：

- state：式(12)的 7 维状态；
- action：下一轮己方报价效用或让步幅度，长度 1；
- reward：式(9)，达成时取效用，中断/超时为 -1，其他为 0；
- next_state：7 维；
- done：是否终止。

示例：

~~~json
{"state":[0,0,0,0,0,0,0.0],"action":[0.85],"reward":0.0,"next_state":[0,0,0,0,0.2,0.9,0.1],"done":false}
~~~

论文仅说明包含 57 个云产品与附加产品场景，没有公开本 PDF 中的完整偏好文件。
因此 scenarios 与 utility_functions 只保留读取位置，不生成替代数据。

