# 项目现状审计与完成实施路线（2026-09-15）

## 1. 本次任务与结论

本文件回应组会后的三个问题：最终策略性能怎样、Option-Aware Route Decoder 是否有效、responsibility-terminal credit assignment 是否有效，并给出从当前代码到实验和论文交付的执行顺序。本次完成的是代码、Git、真实运行产物与部分测试的审计及实施规划；没有执行新训练或正式性能评估，没有修改模型与奖励实现。

**当前最合适的工作顺序：修复评估兼容性 → 评估已有 64K 策略 → 同场景双因素消融 → 多训练种子正式实验 → 学习型基线与场景验证 → 论文、复现包和答辩材料。**

本路线中的新实验配置、开发项、统计扩展均为待实施事项。它不自动变更冻结协议，也不把已有诊断升级为性能结论。`.agents/`、`knowledge/` 保持受保护。

## 2. 已核实的项目进展

### 2.1 最新训练的身份与数据

- Run ID：`ca_gat_mappo__small__seed-42__cfg-4ee36a6391fa__git-10df6909bd8d`。
- Git：`10df6909bd8d5af0f004d8292418fc34d2062178`；训练快照记录 `git_dirty=false`。
- 配置：`experiments/cooperative-load/configs/cooperative-load-v2-optionaware-v1-routealign-64k.json`。
- 实际预算：64,000 environment transitions；128 episodes；每 episode 500 slots；250 PPO updates；1,000 PPO epoch diagnostics；64,000 optimized transitions；尾部丢弃为 0。
- 实际方法：`option_aware_v1` + `role_decomposed` + `branch_specific` + `responsibility_terminal`；route credit 使用 `shared_gae`。不是 route-event n-step，也不是 route-specific GAE 消融。
- 完成/过期信用参数均为 0.5；信道模式为 `full`；workload timing 为历史 `legacy_post_route`。
- `final.pt` 实际存在且 payload 类型为 `FINAL_COMPLETED`；SHA-256 为 `fac87c71c61013d6a94a828f717cb73a2c0a236074225d7bcded5479dae3c803`。
- Dashboard 是真实训练诊断图，reward 平滑使用最多 5 episodes 的后向均值；route 遥测最多 4 个 PPO epoch 行做平滑。不能把 1,000 个 PPO epoch 当作 1,000 次独立实验。

从 Dashboard CSV 的 `record_type=episode` 重新计算：

| 训练窗口 | 每 episode 平均 reward | 每 episode 平均完成数 | 每 episode 平均过期数 |
|---|---:|---:|---:|
| 前 20 episodes | -41.2986 | 31.80 | 62.90 |
| 后 20 episodes | 11.4524 | 54.10 | 35.65 |
| 最后 1 episode | 5.7643 | 47 | 34 |

`trajectory_credit_events.jsonl` 中逐 `(episode_id, task_id)` 去重的 terminal 记录为 12,694 个：done=6,345，expired=5,559，truncated=790，三者相加一致。全训练完成占比约 49.984%，但这里混合了从初始化到结束的不同策略，**不能当作最终 checkpoint 的测试完成率**。

原始 aggregate 的 PPO 数值有限检查通过；平均 approx KL 约 0.000699，平均 clip fraction 约 0.00510。均值不能替代逐更新峰值检查。图中远程选择仍存在、奖励后期改善，但后期波动明显，不能宣布收敛或两个改进已经有效。`rl-formal` 是启动配置名称，产物自己的证据标签仍为 `signal-watch` / `smoke-training`；判断证据级别应看实际协议和样本。

### 2.2 场景已发生显式配置变化

最新 run 虽标记 `small`，具体为 cooperative-load-v2：

| 项目 | 初始 small 规范 | 最新运行 |
|---|---|---|
| UAV 数 | 4 | 4 |
| 到达概率，按 UAV ID | [0.04, 0.05, 0.06, 0.05] | [0.04, 0.09, 0.02, 0.05] |
| Compute-rich 最大 CPU 相对参考值 | 1.5 | 2.0 |
| small 候选半径 | 当前场景默认 525 m | 525 m |
| 信道 | 完整模型 | `full` |

两组到达概率之和都是 0.20，但负载在节点之间的分布不同。最新配置强化了资源较差节点的负载，并强化计算富裕节点的能力。因此主表必须写出 scenario variant，不能只写 Small。旧默认场景的 Random/Heuristic 记录不可直接与当前策略排名。

### 2.3 两项修改的准确含义

**解码器链条有三个版本。**

1. `legacy`：由 GRU 表征经固定线性 route head 输出各 route logits。
2. `candidate_aware_v1`：把候选 UAV 的 public encoding 与对应 edge features 显式送入共享 remote scorer；Local/Idle/Defer 仍走 fixed head。
3. `option_aware_v1`：任务、源节点、GAT 协同表征、GRU 状态、Local/Remote endpoint、服务负担和有效性共同参与选项打分；Local 与各 Remote 使用共享最终 scorer，Idle/Defer 使用 control head。候选轴信息可绕过 GRU 聚合后再进入 decoder。

Option-Aware 的实现提交为 `02bc445`。这里更准确的研究表述是“加强任务—执行选项的显式匹配与可比打分”。历史 localization 的结论是 Mixed，包含 decoder representation drift 和 terminal scorer amplification；不宜把“原编码器表达能力不足”当作已经证明的唯一根因。

**信用分配改变的是个体学习信号。**

`500c3a3` 引入/恢复 `reward.credit_assignment_mode={legacy,responsibility_terminal}`：

- legacy 的远程完成收益按任务固有通信/计算参考工作量比例分给源和目的 UAV；过期惩罚由源 UAV 承担。
- responsibility-terminal 把远程完成分配向源/目的平分混合，混合程度由 `beta_completion_credit` 控制。
- 远程任务已进入目的 CPU 且在 deadline 前有合法 CPU 服务窗口时，由 `mu_expiration_credit` 分配目的 UAV 的过期责任；没有这个服务窗口时不转移该责任。判定不以“实际已经执行了多少 cycles”为条件，避免空闲逃避责任。
- Local、尚未绑定、传输中、CPU 阶段和 horizon truncation 必须分别核验。truncate 不变成 expiration。
- 团队完成、过期、workload、能耗及总 reward 保持守恒。因此论文宜写“责任感知的终态信用分配”，不能简单宣称“更换了团队优化目标”。

比较新旧 credit 时两组都固定 `agent_credit_mode=role_decomposed`；否则同时改变 critic 输出和 GAE 信号。`responsibility_terminal` 的配置校验本身也要求 role-decomposed。

### 2.4 已有对照资产及其边界

- 已有 `logs/reward-credit-ablation/` 的 5K、seed=42 smoke 归档；其中 legacy 完成 263，responsibility 完成 267。属于初步学习轨迹比较，不是正式因果或性能结论。
- 已有 legacy 与 Option-Aware 的 32K 配置文件。
- 现存 full-channel legacy 32K run（`cfg-b247b75a2a1b__git-f1c83c1019d0`）与 Option-Aware 32K run（`cfg-83084cbb9304__git-42e168cb9daa`）在快照中的实质差异是 decoder 和 `route_diagnostic_samples_enabled`，另有 Git/provenance 差异。可作历史探索性对照，但正式复用前必须核查代码差异及诊断开关的行为中立性。
- 最新 Option-Aware 64K 与 legacy 32K 不能作为等预算消融。
- 现有 FormalEvaluationRunner 已实现四方法、共同外生轨迹、冻结 actor、masked argmax、NA 处理和独立产物，但本次在 `logs/` 中未找到 `evaluation_manifest.json`，因此未找到已落盘正式评估证据。
- `Factorized-Action GAT-QMIX` 有章节接口和配置；registry 的训练/评估仍落到 unavailable handler，尚不是可用学习型基线。

### 2.5 本次实际验证与阻塞项

本次审计运行环境是 Python 3.14 / Torch 2.12.0+cpu；原训练快照是 Python 3.13.11 / Torch 2.7.1+cu118 / CUDA 11.8。以下结果不能冒充原训练环境的完整回归。

| 验证 | 本次结果 |
|---|---|
| Option-Aware 专项 | 10/10 PASS |
| 原有 agent-credit 专项 | 20/20 PASS |
| Formal evaluation 专项 | 13 项中 7 PASS、6 ERROR |
| 完整 Gate 0 / 全量回归 | 本次未重跑；README 有历史通过记录 |
| responsibility-terminal 专项覆盖 | 在现有 tests 中按该模式、beta、mu 检索未找到直接测试；20 项旧 credit 测试不能替代新增模式测试 |

最初测试另遇到沙箱临时目录权限问题；沙箱外重跑后，6 个正式评估错误均为 `checkpoint training config hash does not match its canonical snapshot`，属于仍需处理的实际兼容性问题。

已定位证据：`RunConfig.resolved_dict()` 为历史 identity 兼容省略 `channel_ablation_mode=full`；snapshot 却携带该字段；`src/evaluation/actor_loader.py::_validate_source_config_identity()` 对整个去 metadata 快照直接 hash。最新快照存储/规范化重算均为 `4ee36a...`，直接 hash 却为 `b69266...`；两种配置表示的差异就是默认 full 字段。此外 `_validate_source_compatibility()` 也直接比较 source environment 与规范化 environment，应一并审查。

## 3. 阶段 A：先建立可靠评估入口

### A1. 固定证据与版本（最高优先级）

1. 建立实验资产清单：训练 run、源配置、完整 Git、dirty 状态、checkpoint 类型/SHA、decoder、credit、场景变体、collected/optimized steps、训练种子、运行设备。
2. 原始 CSV/JSONL/PNG/checkpoint 只读保存；分析输出另建路径；不能重命名 periodic checkpoint 为 final。
3. 建立独立评估配置，保留源 checkpoint 的环境、动作、decoder 与归一化参考，显式设置 evaluation mode/device。
4. 文档中的“下一步生成 final”属于旧进度；已有 64K final 应优先复用。

交付：`sections/` 中的资产/版本说明，`experiments/` 中的评估配置，`logs/` 中的审计结果。验收：每个比较结果能准确追溯到配置与 checkpoint。

### A2. 修复配置 identity 兼容并补充专项测试

修改范围建议：`src/config.py`、`src/evaluation/actor_loader.py` 及对应 `tests/`。用唯一、明确版本规则规范化配置，同时保留原始快照和源 hash。不得删除 hash 校验、篡改旧 checkpoint、静默忽略未知字段，或伪装成原训练 commit。

必须验证：

- 历史缺省 full 与显式 full 的合法 identity 兼容；非默认信道仍严格区分。
- environment、architecture、action 或非默认值被篡改时仍拒绝。
- legacy/candidate/option 三类 actor 正确装载；错误 decoder 拒绝。
- 新旧 credit provenance 可追溯；评估 manifest 补充或验证源训练 seed、预算、decoder、credit、beta/mu、场景变体、训练及评估运行时版本。
- 现有 13 项 evaluation 回归全部通过；已有 64K final 做真实 actor-only load 验证，前后 SHA 不变。
- 增补 responsibility-terminal 的完成分配、CPU 服务窗口边界、传输阶段过期、闲置责任、truncate、零任务、团队逐项守恒测试。
- 固定动作重放下切换 credit 后物理轨迹和团队 reward 相同；仅个体 credit 按定义变化。测试不要求学习后的策略仍相同。

验收：新增模式的行为有直接测试；评估加载通过；没有训练副作用。完成后才进入 A3。

### A3. 用已有 64K final 做第一张性能表

按现有评估契约，在 `{1042,1043,1044,1045,1046}` 各运行 500 slots，同时评估：

1. 最新 CA-GAT-MAPPO actor，masked argmax；
2. Local-only；
3. 已冻结的 Deadline-and-Historical-Link-Aware Heuristic；
4. Random，使用自身独立 policy RNG。

总量为 4 methods × 5 evaluation seeds × 500 slots = 10,000 evaluation slots。四种方法必须使用同一个 cooperative-load-v2 环境，并验证相同 seed 下外生轨迹 hash 一致；动作产生的干扰、队列、测量与耗能不要求相同。

主指标与解释：

| 指标 | 用途与统计边界 |
|---|---|
| completion_ratio | 首要性能；completed/generated |
| expiration_ratio、truncation_ratio | 与完成率一起报告，三种终态不可混淆 |
| completed-task E2E delay | 只含成功任务，单位 s；必须和完成率一起看，避免幸存者偏差 |
| total/tx/cpu energy | 实际主动能耗，不包括推进能耗 |
| energy_per_completed_task | 全部主动能耗/成功任务数；无成功为 NA |
| backlog mean/max/final | 检查拥塞与积压；明确 final 是现有槽初快照语义 |
| outage numerator/denominator/rate | 只含实际发送尝试；Local-only 无尝试为 NA |
| executor rejection/downgrade | 排查 proposal 可执行性，报告计数和曝光量 |
| total reward | 同一团队目标下的辅助指标，不取代以上物理指标 |

交付：现有 runner 的 manifest、episode JSONL、aggregate JSON、evaluation CSV；新增由真实评估生成的完成率、时延、能耗图和中文结果说明。

验收不是“必须赢”：即使输给 Heuristic，只要评估协议正确，这个结果也应保留。若失败，按任务阶段查原因；不能通过选择有利 seed、删掉失败 episode 或调整测试集来制造优势。一个训练 seed 在五个环境 seed 上的结果只说明该模型的评估表现，尚不能说明训练稳定性。

## 4. 阶段 B：用一套双因素消融回答老师的两个问题

### B1. 核心 2×2 矩阵

| 实验组 | route_decoder_mode | reward.credit_assignment_mode | 对应问题 |
|---|---|---|---|
| G00 | legacy | legacy | 统一当前其他实现的双旧组件基线 |
| G10 | option_aware_v1 | legacy | 只增加选项感知解码器 |
| G01 | legacy | responsibility_terminal | 只增加终态责任分配 |
| G11 | option_aware_v1 | responsibility_terminal | 当前完整方法 |

G00 指在同一当前代码中关闭这两个组件，不代表复原整个最早的训练版本。全部组固定 `role_decomposed`、`branch_specific`、`shared_gae`、reward weights、workload timing、entropy 设置及 PPO 超参数。

- G11 对 G01：当前新 credit 下解码器增益。
- G11 对 G10：当前新 decoder 下信用分配增益。
- G10 对 G00、G01 对 G00：两个模块各自独立作用。
- 比较两种 credit 下 decoder 的增益是否不同，检查组件交互；不能把联合提升完全归功于任一单个模块。

如果老师所说“上一版”特指 Candidate-aware，再加 Gmid=`candidate_aware_v1 + responsibility_terminal`。先作为优先补充对照；要完整研究三版本与信用分配交互时，再做 3×2，不能用新增组替代四个核心组。

### B2. 公平性约束

1. 同一 clean Git 实现，通过配置开关切换；每组从头训练，不能把完整方法 checkpoint 关掉模块后直接评估作为消融。
2. 同一场景快照、硬件 realization、task/deadline、channel、CSI、资源、执行器、奖励参考和 episode horizon。
3. 相同训练种子、采样预算、rollout/更新次数、checkpoint 规则和评估种子；相同初始化 RNG 策略。不同网络结构不能要求全部参数逐值相同，但应检查共享 trunk/其他 heads 的初始化一致，并记录参数量。
4. 保持 route stability guard、route entropy schedule、n-step、workload timing fix 等额外开关一致。最新 run 未启用的额外修复不混入主消融。
5. 诊断 sidecar 在所有组统一开关；先验明不改变动作/RNG/优化，再考虑减少正式 run 的日志量。
6. 按环境步数比较训练曲线；累计 reward 会随预算变化，不直接比较 32K 与 64K 总回报。
7. 测试集不能用于选择最优 checkpoint。主结果预先固定最终预算的 FINAL；若需要验证集选点，先增加独立 validation seeds 并写入协议。

### B3. 执行顺序与预算

- 首先做四组配置解析、identity、共同条件差异审计及短 smoke；短 smoke 只用于发现接线错误。
- 然后做四组 seed=42、64K 等预算 pilot。每组 128 episodes / 250 updates / 无尾部丢弃。当前 G11 可作探索性参照；正式纳入匹配组前需完成版本等价性审计，否则四组统一重跑。
- 每个 pilot 的 FINAL 用同一评估集比较。若后期仍明显变化，先在开发阶段统一决定更长预算，再冻结正式协议，不按每组测试结果分别调预算。
- 正式采用冻结的训练 seeds `{42,43,44,45,46}`。500K 是现有正式预算候选，能否充分学习需由 pilot 判断，不能把它当作自动收敛保证。
- 500,000/256 存在 32 transitions 尾部：应记录 optimized=499,968、完整 updates=1,953；四组一致。若另选整除预算，先记录协议修订。

四组 × 五训练种子 × 500K = 10,000,000 collected transitions（单场景）。不能把五个 eval seeds 当作五次独立训练；同一模型反复评估不能替代训练 seed。

### B4. 解码器机制证据

性能指标之外，记录 Local/Remote/Defer 的选择分布、合法 remote 条件下选择率、不同目的地分布、源 1→计算富裕 UAV 2 的选择与终态、logit margin、route entropy、任务尺寸/slack/负载分层下的选择。

优先复用 `src/evaluation/route_option_alignment.py`，但它的预设窗口是 32K 诊断窗口；64K/正式实验需显式扩展窗口，不能漏掉后半程而声称完成整程分析。DRA/utility alignment 是对已有代理效用的对齐指标，不是真实最优性证明。遇到重构失败应记录有效样本数、失败数和原因。

必要时复用 same-state/oracle 工具做小规模机制诊断；旧的 4K/60K paired-state 合约不应直接挪作新三种网络的完整比较。需先验证 actor 输入/候选对齐、hook 名称和 recurrent hidden 处理。Zero-hidden probe 与实际携带历史的部署策略分开解释。

重点问题是“是否在需要协作且有可行候选时选得更好”，不是远程比例是否越大越好。模型参数量、固定 batch 下推理耗时和额外特征成本也需要报告。

### B5. 信用分配机制证据

对固定任务生命周期案例和真实训练任务分别记录：源/目的完成奖励、源/目的过期责任、是否有 CPU 服务窗口、route 时 source-agent advantage、后续 tx/CPU 服务和终态。

重点区分：

- 未绑定/本地任务；远程传输尚未完成即过期；进入目的 CPU 且有服务机会后过期；远程成功完成；horizon truncated。
- 收益增加来自更合理目的地选择，还是接收端更及时服务、减少空闲或降低积压。
- beta/mu 的直接数值变化是设计规定，不能单靠“奖励分得不同”证明学习有效；必须观察后续物理性能。

主消融先固定 beta=mu=0.5。只有主结论清楚且稿件需要时再补 beta/mu 敏感性，避免过早把实验扩展成大网格。

## 5. 阶段 C：完成正式性能、学习型基线和场景证据

### C1. 先冻结主张与场景

将 cooperative-load-v2 定义为明确的协作负载场景，同时保留原默认 small 作为低协作需求参照。不要只展示人为强化协作需求的单场景。若主论文覆盖 Small/Medium/Large，应分别在对应场景训练和评估；当前 loader 严格拒绝环境变更，不能拿 small checkpoint 加 `--scenario-id medium` 就宣称跨规模泛化。

单场景四组主消融完成后，再安排主方法与必要基线在 medium、large；只有要宣称模块跨规模普遍有效时才补齐相应规模的消融。所有三场景四组五种子都做 500K 将达到 30,000,000 transitions，需先测吞吐与存储。

### C2. 完成已约定学习型对比

当前非学习基线足以回答第一次性能检查，但 Section 3 已写入 Factorized-Action GAT-QMIX 学习型比较；若最终保留该主张，该实现仍是项目未完成项。

实施内容：共享合法观察接口的 GATv2+GRU、七分支 Q heads、合法自回归选择、factorized utility、monotonic mixer、recurrent replay、target network、TD loss、epsilon schedule、独立 optimizer、训练/评估 registry 和 checkpoint 接口。沿用冻结 Section 3/4，不私自换为枚举联合动作或未修改标准 QMIX。

验收：分支 mask/依赖正确、replay 序列与 boundary 正确、TD target/bootstrap 正确、mixer 单调性与梯度、target 更新、CPU smoke、固定 seed 可复现、FINAL greedy 评估与共同外生轨迹验证。训练预算与调参预算记录清楚。

QMIX 以同一团队物理目标为基准；不要未经方法设计就把 MAPPO 的 per-agent credit/critic 直接移植到 QMIX。其实现与公平性说明完成后，进入五种子比较。若决定从稿件移除 QMIX，需要先记录并确认研究范围变更，不能在本计划中默认取消。

### C3. 环境适用范围与敏感性

按论文最终主张选择少量有解释力的实验轴，主模型保持 full channel：

1. 负载从轻到重：保持硬件不变，统一缩放到达概率，所有方法使用同一网格。
2. 资源/负载错配：保持总负载一致，比较均衡分布与当前不对称分布。
3. CSI 陈旧程度或误差：改变一个因素，其余不变。
4. 带宽/能量约束：仅在资源效率主张需要时增加。

区分两种实验：每个配置重新训练的适配性能，和固定策略在变化环境下的鲁棒性。后者当前 loader 不支持任意环境改变，需要显式的跨环境评估契约、allowlist 和测试，不能直接去掉兼容性校验。`deterministic` 简化信道只用作诊断/敏感性组，不能替代真实主模型。

若论文还要声称 GAT/GRU 本身带来收益，才追加对应结构消融；这些属于额外待设计内容，不能把已有 decoder 消融当作 GAT 消融。

### C4. 统计协议

- 每训练种子先得到跨评估种子的指标，再对五个训练种子报告均值、标准差和逐种子点；同时保留 pooled task 统计，两种口径分别标注。
- 使用成对种子差值比较组别，附不确定区间。区间可采用按训练种子为簇的 bootstrap；若同时重采样环境 seed，保持四组配对和层次结构。五个训练 seed 的区间本身也可能不稳定，应如实呈现。
- 不把全部任务、PPO epoch、四次重复使用的 route samples 当作独立训练样本。Defer 可对同一任务重复发生，不能把所有 route-outcome 行当作互斥任务终态。
- 当前评估器的 latency aggregate 是 pooled completed-task 样本；要得到训练种子层面的性能不确定性，应另做层次汇总，不直接套用该字段 std。
- 预先指定主要指标和对比，避免大规模多重比较后只报告显著结果；报告改善幅度、失败比例、计算成本，而不只报告 p 值。
- 当前五个 eval seeds 若用于开发诊断后再大量调参，就不再是严格未见测试集。应在正式前记录其用途；需要新锁定 holdout 时新增互不重叠种子并修订协议，不能悄悄替换。

多 seed 与区间报告的方法依据可参阅 [Agarwal 等，Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)。这里的具体 seed 集合来自项目 Section 4，并非该论文替本项目规定的充分样本数。MAPPO 原论文与实现可用于核对算法定位：[论文](https://arxiv.org/abs/2103.01955)、[官方实现](https://github.com/zoeyuchao/mappo)；本项目带有 branch-specific/role-credit 扩展，不应称为未修改标准 MAPPO。

## 6. 阶段 D：论文、图表与复现交付

### D1. 写作与证据同步

- Section 2：保持已冻结物理模型，清楚区分团队 reward 与个体信用分配；配置变体另有表，不混入默认值。
- Section 3：补充 Option-Aware 的真实信息流、特征来源、共享 scorer、信息权限及训练/推理；补充 responsibility-terminal 生命周期规则与 beta/mu；将最终实际启用的 branch-specific PPO、role-decomposed critic/GAE 写清楚。
- Section 4：写出场景变体、软件/硬件、预算、train/validation/test 划分、FINAL 选择、基线和统计口径。
- Section 5：按性能、解码器贡献、信用贡献、场景适用范围组织，不按调试历史组织。
- Section 6：讨论未获益场景、采样与 argmax 差异、能耗范围、固定硬件/建筑布局、单跳与仿真边界、计算代价及统计不确定性。
- Section 1、摘要、结论最后更新；根据真实结果决定贡献强度。不提前写收敛保证、全局最优或普遍优越性。

具体公式、符号表与算法伪代码在实施写作阶段依据真实代码整理并编译核对；本计划不新增理论结果。

### D2. 建议最终图表

1. 系统/队列时序图；2. Option-Aware 网络结构图；3. 终态信用分配生命周期图。
4. 多训练种子学习曲线（统一步数、同一平滑规则，保留原始值与 seed 范围）。
5. 各场景完成率/过期率主性能图；6. 时延与能耗联合结果图。
7. 2×2 消融表或点图，附逐 seed 点与区间；8. 目的地选择与任务终态机制图。
9. 与论文主张对应的少量负载/CSI 曲线；10. 参数量、训练时间、推理时间表。

每张数据图有源 CSV/JSON、生成代码、单位、分母、seed、版本和图注。科研数据图由 Python 生成；源代码在 `src/`，图片在 `plots/`，Dashboard CSV 在 `dashboard_logs/`，普通实验记录在 `logs/`。不手工改数据曲线或造误差条。

### D3. 复现与最终验收

- 干净版本和依赖锁定；说明 CUDA 训练与 CPU 评估的具体版本，禁止静默 fallback。
- 轻量 `main.py`，可用交互入口和配置运行；批量 runner 在 `src/`，测试在 `tests/`。
- 训练、评估、聚合、绘图各有可执行命令；路径不存在时清楚报错，产物不可静默覆盖。
- README 只放宏观状态和入口；详细实验/学术内容保留在 `sections/`。
- 从一个新输出目录实际重放最小复现链；检查图表与结果章节数值一致，checkpoint 和数据 SHA 可追溯。
- 保留负结果与失败 run 原因；大日志不重复拷贝；正式训练前估计 sidecar/CSV/checkpoint 存储。
- 论文引用核验、术语/公式/图表一致性、编译和答辩/组会 PPT 在数值冻结后完成。

**项目完成标准**：当前功能与新组件直接测试通过；至少主场景有五训练种子的正式性能与四组消融；稿件承诺的学习型基线/场景已完成或范围变更已确认；两项改进是否有效有可复核答案；图表来自真实结果；论文与复现命令完整。结果未证明某项收益时，应缩小主张或移除相应贡献，不能把“完成项目”等同于“必须实验胜出”。

## 7. 任务分解、顺序与交付

以下为工作量估计，不是已测算训练工期。GPU 用时应在同一设备测得吞吐后计算，并额外计入 checkpoint、遥测和评估 I/O。

| 顺序 | 工作包 | 依赖 | 核心交付 | 估计人工时间 |
|---|---|---|---|---|
| 01 | 资产清单、canonical identity 修复、新增 credit 测试 | 无 | 兼容性与测试报告 | 1–3 工作日 |
| 02 | 64K FINAL 四方法性能评估 | 01 | 第一张性能表与图 | 1–2 日＋运行 |
| 03 | 四组消融配置与公平性 preflight | 01 | G00/G10/G01/G11 配置和差异清单 | 1–2 日 |
| 04 | 四组 64K pilot 与机制分析 | 02、03 | 初步模块效应与预算决定 | 2–4 日＋训练 |
| 05 | 五种子主实验和统计 | 04 | 主性能、2×2 消融与区间 | 2–4 日＋正式训练 |
| 06 | QMIX 实现、验证与正式比较 | 02 后可独立推进 | 学习型 baseline | 5–10 日＋训练 |
| 07 | 必要规模/负载/CSI 实验 | 05；比较需 06 | 适用边界与敏感性 | 3–5 日＋运行 |
| 08 | 方法、协议、结果、讨论与图表 | 方法可提前；结果依赖 05–07 | 完整论文初稿 | 5–8 日 |
| 09 | 全链复现、论文检查与答辩材料 | 08 | 冻结版论文与复现包 | 2–4 日 |

首次组会最值得交付的三份材料是：①当前 64K FINAL 与三条非学习基线的性能表；②统一场景的四组消融矩阵及已经完成的 pilot 数据；③两个模块的结构/生命周期解释与尚待验证的假设。无需先把 500K 全部跑完才开始回答性能问题。

## 8. 紧接本计划的可执行任务

建议下一项工作明确限定为：**“修复 Formal Evaluation 的 canonical-config/hash 兼容问题，补上责任终态 credit 专项测试，验证现有 64K FINAL 可加载，再在 cooperative-load-v2 上运行四方法、五个 eval seeds 的只推理评估并生成性能报告。”**

修复完成后的 CLI 入口模板如下；本次没有执行此命令，当前兼容性问题解决前不应期待成功：

```powershell
python main.py --mode evaluation --method-id ca_gat_mappo --config experiments/cooperative-load/configs/cooperative-load-v2-optionaware-v1-routealign-64k.json --evaluate-from logs/ca_gat_mappo__small__seed-42__cfg-4ee36a6391fa__git-10df6909bd8d/checkpoints/final.pt --evaluation-device cpu
```

执行前需核对解析后 mode、launch profile、actor architecture、完整环境 identity 和源 checkpoint 兼容；`cpu` 是显式评估选择，不是训练时自动降级。正式运行使用已记录并验证的运行时。随后再实施四组消融，不从 64K FINAL 直接 resume 到 500K，也不先盲目扩大训练预算。
