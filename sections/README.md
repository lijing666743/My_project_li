# sections 目录说明

## 方案源与当前实现规格

项目的唯一实施基线为：

```
knowledge/project_plan/方案2.md
```

该文件已经冻结，但位于 AGENTS.md 定义的 protected `knowledge/` 目录，本轮自动任务不得修改。当前环境实现规格以已经冻结的 `sections/2_system_model.md` 为准。`sections/` 中的论文正文不得自行改变任务状态机、资源异构定义、动作空间、信息权限、时隙因果或实验边界。

## 章节状态与职责

| 文件                                   | 当前状态           | 内容边界                                                     | 下一步                               |
| -------------------------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------ |
| `0_abstract_keywords.md`               | 等待核心章节冻结   | 摘要与关键词，不提前写实验结论                               | 在系统模型、方法和实验协议完成后整理 |
| `1_introduction.md`                    | 等待核心章节冻结   | 研究背景、问题、贡献和文章结构                               | 在三章一致性审计后正式撰写           |
| `2_system_model.md`                    | FROZEN / COMPLETE SPECIFICATION | 系统对象、符号、异构性、任务状态机、队列、时隙因果、信道、动作语义和信息权限 | 作为 Section 3/4 接口基线            |
| `3_methods.md`                         | NEXT / NOT YET FROZEN | CA-GAT-MAPPO、Factorized-Action GAT-QMIX、训练与执行流程     | 完成并冻结方法接口                   |
| `4_experimental_protocol.md`           | DRAFT / NOT YET FROZEN | Gate P–Gate 6、实验族 A/B/C、参数校准、统计和日志规范        | 在 Section 3 冻结后完成并冻结         |
| `5_results_template.md`                | 结果占位           | 只接收真实运行结果、图表和统计检验                           | 编码和实验完成后填写                 |
| `6_discussion_limitations_template.md` | 讨论占位           | 结果解释、失败场景、局限性和外推边界                         | 依据真实结果填写                     |
| `7_conclusion.md`                      | 等待核心章节和结果 | 总结研究问题、方法、主要结果和局限性                         | 最后阶段完成                         |

## 核心维护规则

1. `knowledge/project_plan/方案2.md` 是既有方案源；该文件位于 protected `knowledge/` 目录，本轮自动任务不得修改。
2. `2_system_model.md` 是当前环境实现规格源，也是系统对象、符号、资源/负载异构、任务状态机、队列、时隙因果、物理机制、信息权限和动作语义的论文主源。
3. `3_methods.md` 只能说明算法如何处理已经冻结的系统，不得重新定义物理规则、队列更新、动作含义或信息权限。
4. `4_experimental_protocol.md` 只能定义如何校准、训练、评价和统计，不得重新定义系统和算法接口。
5. `route` 与 `tx_select` 必须始终分离：
   - `route` 只处理未绑定任务；
   - `tx_select` 只处理已有传输队列。
6. 主实验使用五个固定资源组，不再使用任意 RB 起始位置。
7. 主实验使用七分支离散动作：
   - `route`
   - `tx_select`
   - `resource_group`
   - `resource_width`
   - `power_level`
   - `cpu_queue`
   - `cpu_frequency`
8. Beta 和 Dirichlet 仅允许出现在可选连续动作扩展中，不得进入主模型。
9. Factorized-Action GAT-QMIX 是分支式多离散变体，不得表述为未经修改的标准 QMIX。
10. 新路由任务、传输完成任务和新到达任务均在槽末入队，下一时隙才允许服务。
11. actor 不读取真实 `lambda_i`，只读取历史任务到达率估计。
12. outage 只在实际被调度且分配非零资源的链路上统计。
13. 半双工由联合 hard-feasible 执行器保证，不能只依赖单智能体本地 mask。
14. 能量不可行时只能执行离散降档或 idle，不得映射为档位集合之外的连续值。
15. 第三 UAV 软遮挡、连续资源动作、Sionna RT、数量 OOD 和类型 embedding 均为可选扩展。
16. 结果和讨论章节只能使用真实运行结果，不得填入虚构数字、曲线、显著性或结论。
17. `AGENTS.md` 暂不修改。只有 `2_system_model.md`、`3_methods.md` 和 `4_experimental_protocol.md` 完成并通过一致性审计后，才结束初始化阶段。

## 推荐工作顺序

```
Section 3
    -> Section 4
    -> cross-section interface audit
    -> implementation-stage transition
```
