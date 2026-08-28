# 动态异构多 UAV U2U-MEC 协同卸载与资源编排

## 项目定位

本项目面向外生移动和动态 U2U 邻居环境，研究在 UAV 资源异构、业务负载异构、建筑物遮挡、有限频谱、不完美 CSI、跨时隙传输/计算队列和任务截止期共同存在时的单跳 U2U-MEC 协同卸载与资源调度问题。

项目的既有方案源为：

```
knowledge/project_plan/方案2.md
```

该文件已冻结为“修订版可实施实验方案（编码冻结版）”，但位于 AGENTS.md 定义的 protected `knowledge/` 目录，本轮自动任务不得修改。当前环境实现规格以已经冻结的 `sections/2_system_model.md` 为准；后续 `sections/`、配置文件、代码、测试和实验报告均须与冻结章节保持一致。

## 研究主线

1. UAV 资源异构与业务负载异构分离；
2. 外生移动与动态候选邻居；
3. 建筑物几何遮挡；
4. 空间相关阴影与 Rician/Rayleigh 衰落；
5. 不完美 CSI 与 CSI AoI；
6. 五个固定资源组及同频干扰；
7. 任务级 EDF 队列；
8. 跨时隙传输与计算服务；
9. 能量约束、离散降档与联合半双工解析；
10. CTDE 图多智能体协同卸载。

## 冻结方法

### 主方法

**CA-GAT-MAPPO**

核心结构为：

```
局部节点、队列与边特征
    -> 一层边感知 GATv2
    -> GRU
    -> 七个 categorical 动作头
    -> 动作 mask
    -> 联合 hard-feasible 执行器
    -> 集合式集中 critic
```

### 保底方法

**Factorized-Action GAT-QMIX**

该方法是面向多分支离散动作的 GAT-QMIX 变体，不等同于未经修改的标准 QMIX。它与主方法共享环境转移、观测权限、动作分支、动作 mask、联合半双工解析器、离散降档、奖励和评价指标。

## 主动作空间

主实验使用七分支因子化离散动作：

1. `route`：只处理槽初未绑定任务；
2. `tx_select`：只选择槽初已有的非空传输队列；
3. `resource_group`：选择固定资源组；
4. `resource_width`：选择一个或两个相邻资源组；
5. `power_level`：选择离散功率档位；
6. `cpu_queue`：选择本地或远程 CPU 队列；
7. `cpu_frequency`：选择离散 CPU 频率档位。

主实验不再使用任意 RB 起始位置，也不使用 Beta 或 Dirichlet 连续资源动作。

## 当前进度

- Git 基线已经建立；
- 本地备份与缓存文件已通过 `.gitignore` 排除；
- 系统模型第一项 P0 修订已完成：采用硬截止期语义，取消延迟完成类别；
- 系统模型第二项 P0 修订已完成：区分 unbound 与 local 的目的地语义，并规定 local 绑定时清零剩余待传输量；
- 系统模型第三项 P0 修订已完成：任务目的地在 route 决策生效并进入 local 或 tx 队列时立即锁定；
- 系统模型第四项 P0 修订已完成：已统一槽末到达、下一槽 route、槽末绑定、下一槽服务、槽末完成和 deadline 结算的时间边界；
- 系统模型第五项 P0 修订已完成：通信和 CPU 实际能耗按实际活跃时间计算；
- 系统模型第六项 P0 修订已完成：actor 仅使用陈旧 CSI、历史干扰摘要和历史干扰链路质量代理量；
- 系统模型第七项 P0 修订已完成：outage 仅统计执行器实际接受、非零功率、非零资源且队列非空的实际传输尝试；主动 idle、执行器拒绝、零功率或零资源占用均记为 NA，不进入无线 outage 分母；
- 最终一致性审计发现的 P0-01 已修复：真实 SINR 保持线性表示，-3 dB 门限已补充对应的线性转换；
- 已统一有效速率门控和 RB 级 outage 比较所使用的线性 SINR 门限；
- 最终一致性审计发现的 P0-02 已修复：历史到达率估计在时隙 $t$ 仅使用 $A_i(0),\ldots,A_i(t-1)$，槽末生成的 $A_i(t)$ 不进入当前槽 actor 观测，最早在 $t+1$ 槽初产生影响；
- 历史到达率已补充短窗口、$t=0$ 缺省值 $0$ 与可用性 mask 的统一定义；
- 最终一致性审计发现的 P0-03 已修复：episode reset 显式初始化位置、阴影、真实信道与陈旧 CSI 边界，位移和阴影递推不再访问负时隙历史；
- 最终一致性审计发现的 P1-01/P1-02 已修复：七分支资源动作与资源构造变量已统一映射，inactive width 使用既有定义域中的 canonical encoding，不新增 idle 类别；
- 已拆分 actor 的 proposed power、executor 内部 candidate power 与最终 executed power，并将 per-RU power、真实 SINR、实际服务、实际通信能耗和 outage 绑定到 executed power；
- 通信能量预留明确使用 candidate power；executor 拒绝、降档和最终执行语义已写入系统模型的事件顺序；
- 已补充上述动作接口与功率阶段语义的计划测试，测试代码与实验尚未实现；
- 已统一 $\Delta s_{ij}(0)=0$、$X_{ij}(0)\sim\mathcal N(0,\sigma_s^2)$、无负索引 CSI 历史、缺省值与可用性 mask，以及当 $\ell_{ij}^{\mathrm{CSI}}(t)=0$ 时首次读取 $h_{ij,r}(0)$ 的语义，并补充对应计划测试；
- 最终一致性审计发现的 P1-03 已修复：联合执行器通信冲突按 $(\operatorname{slack},-\text{历史链路质量},i,j)$ 升序字典序仲裁，半双工逐项接受与拒绝结果唯一确定；
- 已冻结通信功率优先降档、CPU 频率后降档以及每次选择最高可行档位的顺序，执行器不使用随机 tie-break 或当前真实 SINR 排序；
- 已补充 P1-03 确定性、半双工冲突、最高可行通信档位和 CPU 后降档的计划测试；测试代码与实验仍未实现。
- 最终一致性审计发现的 P1-04 已修订：已冻结等宽 RU 的 $B_{\mathrm{RU}}=B^{\mathrm{tot}}/R$、噪声 PSD/噪声系数到线性 RU 噪声功率的单位链、executed-action 干扰测量及历史更新时间；
- 已补齐 CSI 陈旧偏移量、CSI 误差 dB 域与作用对象、干扰历史 mask、消息 AoI 的 refresh/increment 和当前槽测量不得泄漏给 actor 的可执行语义；测试代码与实验尚未实现。
- 最终一致性审计发现的 P1-05/P1-06 已完成：reward timing 已冻结为 post-service / post-settlement，normalization scales 已冻结为 episode 开始前固定的 reference constants，episode horizon 与 terminal truncation 已定义；
- 最终冻结审计发现的 P1-01/P1-07/P1-08 已完成本轮修订：已闭合 deadline slot transition/settlement，补齐 reset 状态，并将 zero-power executed communication 统一为 canonical idle；
- 审计中原有 P0/P1 修订项至此全部处理完成；
- 本轮最终冻结审计已收口 P1-02 至 P1-06：移动边界、reset 安全距离、small-scale fading、CSI error、interference measurement mask 和 historical-quality scalar 的实现规格均已唯一化；
- Section 2 已补齐固定采样顺序、fixed-seed 可复现性、当前测量不得泄漏给 actor、measured zero 与 missing 区分，以及 proposed resource set 上的 valid-RU arithmetic-mean 聚合；
- 当前最终 READY TO FREEZE 审计唯一剩余的 P1 已修复：消除 scalar historical quality 观测执行器循环，actor 只读取 proposal 形成前已存在的 per-RU historical quality 及其有效性 mask；
- executor-only scalar historical quality 仅在 actor 形成 $\mathcal S_i^{\mathrm{prop}}(t)$ 后计算，并继续使用既有 valid-RU masked arithmetic mean、fallback $0$ 和 $(\operatorname{slack},-\widehat\Gamma_{ij}^{\mathrm{hist}},i,j)$ 优先级键；
- Section 2 系统模型已完成最终冻结验收：FROZEN；P0 = 0；P1 = 0；
- Section 2、Section 3 和 Section 4 的实现契约均已冻结并通过当前跨章节接口检查；三章状态均为 FROZEN，P0 = 0，P1 = 0；
- 项目阶段已切换为 Environment Implementation；允许开始环境、配置、运行器和 Gate 0 相关代码实现；当前实现、测试和实验结果仍未产生，不得据此声称代码或实验已经完成；
- 已完成 01-Config-CLI-Runner：新增 typed `RunConfig`、Section 4 场景默认值、配置文件/CLI/交互覆盖优先级、seed stream 元数据、registry 和统一 runner；
- 已接通 `python main.py` 无参数交互菜单与 `python main.py --mode random --seed 42` 直连入口；当前尚无 environment backend，未生成伪造 metrics、dashboard 或 checkpoint；
- 已新增 Config/CLI/Runner 单元测试并通过 6 项；后续仍需实现 environment sanity、Gate 0、random policy 和 heuristic policy 的真实后端与 raw metrics；
- 已完成 02-Task-Queue-Lifecycle：新增 typed Task、确定性 task ID、EDF 队列、单一队列所有权、route/service 时序、目的地锁定、剩余 bit/cycle bookkeeping、hard deadline、done/expired 与 horizon-only truncated 结算；新增 Task/Queue/Lifecycle 测试并与 Implementation 01 回归测试合计通过 29 项；
- 已完成 03-Mobility-Topology-Channel：新增统一 seed stream、Gauss-Markov 移动与安全 reset、coordinate-wise specular reflection、动态候选邻居、建筑物遮挡、相关阴影、directed per-RU Rician/Rayleigh 真实信道、stale CSI/CSI error、历史干扰 EMA/消息 AoI、历史 denominator、per-RU 历史质量代理量与 validity masks；新增 18 项测试并与既有 29 项回归测试合计通过 47 项；
- 已完成 04-Executor-Service-Energy：新增七分支 proposal adapter、确定性 half-duplex executor、power-first/CPU-after 离散能量降档、同 RU 复用下的真实 interference/SINR/rate、跨任务 EDF 传输、单队首 CPU service、实际活跃时间能耗、剩余能量扣除、I_meas 与 outage NA 语义；新增 23 项测试并与既有 47 项回归测试合计通过 70 项；
- 已完成 05-Full-Environment-Integration：统一接通确定性 reset/step、槽末到达、route、执行器与物理服务、deadline settlement、因果 observation/action masks、centralized state、固定参考 reward、history、horizon truncation、metrics 以及 runner/registry/CLI environment backend；新增 28 项测试并与既有 70 项回归测试合计通过 98 项；
- Full Gate 0 已由 G0-01 至 G0-21 的逐项可复现测试全部验证通过；environment sanity、direct CLI 与 interactive CLI 均调用同一真实环境 backend；
- 已完成 06-Random-and-Heuristic-Rollout：RandomPolicy 与 Deadline-and-Historical-Link-Aware Lexicographic Heuristic 均通过统一 RolloutRunner、确定性 executor、direct CLI 和 interactive menu 运行；Heuristic 只使用 actor-visible 槽初信息与 action masks，并保持七分支 proposal/executed 分离；
- 已完成 08-Local-only 基础基线实现与短程工程测试：`LocalOnlyPolicy` 已接入 baseline launcher，并通过统一 `RolloutRunner` 验证本地路由、通信 idle、本机 CPU 源、零 TX 能量与守恒；尚未开展正式 baseline 性能实验，MAPPO/QMIX training 仍暂停且未实现；
- 全量回归测试现为 119/119 PASS，Full Gate 0 继续 PASS；small 场景、500 slots、seed=42 的 Random 与 Heuristic 单种子工程 smoke 已真实完成并保留 raw/aggregate/dashboard CSV，未生成 plot；
- 上述 rollout 仅用于工程正确性与 sanity 对比，不是正式论文性能结果；MAPPO、QMIX 和 RL training 仍未实现；
- 当前真实 SINR 仅在联合执行动作确定后由环境计算；
- 已消除 actor 读取当前联合干扰的未来信息风险；
- 联合执行器使用保守能量预留上界保证执行前硬可行性；
- 剩余能量只扣除实际能耗，未使用的预留能量在时隙结束时释放；
- 已修正端到端时延及 slack 的 off-by-one 语义；
- 已绑定任务不允许重新路由；
- Section 2 系统模型已完成冻结验收，当前不再新增系统模型语义；
- Section 2 系统模型、Section 3 方法接口和 Section 4 实验协议共同构成后续实现的冻结基线；
- 最终项目方案已经冻结；
- 项目状态文档正在与冻结章节同步；
- 完整 environment backend 与 Random/Heuristic rollout pipeline 已实现；CA-GAT-MAPPO 的 GAT/MAPPO、Rollout、GAE、PPO Objective/Update、Production Trainer 和 Exact Resume Checkpoint 已实现；QMIX、正式论文绘图和正式论文实验结果仍未完成；
- CA-GAT-MAPPO RL CLI Gate 已修复：交互菜单与直连 CLI 均经 registry 路由到真实 `CAGATMAPPOTrainer.train()` handler；`unavailable` 与 `failed` 状态返回非零退出码；
- 已新增 `rl-smoke`（CPU、256 transitions、1 个完整 rollout）、`rl-long-smoke`（CUDA、small、seed=42、50000 transitions）与 `rl-formal`（CUDA、冻结正式预算）profile，并修复带类型注解的 PyTorch CUDA provenance 解析；
- CA-GAT-MAPPO Smoke Training Gate 已真实通过：CPU、seed=42、256 transitions、8 episodes、1 个完整 rollout、1 次 PPO update，CLI 退出码为 0；
- 已新增 Trainer 外围训练诊断 writer，不修改 environment、reward 公式、PPO 算法或 Trainer 核心生命周期；真实运行已生成 config snapshot、raw JSONL、aggregate JSON、dashboard CSV 和 reward/loss dashboard PNG；
- 本次真实 smoke 的均值为 reward=-0.0615874、actor_loss=0.876723、critic_loss=0.599072、entropy=0.843768；数值均有限，reward 公式恒等式与 PPO update accounting 均通过；
- Smoke Training Gate 状态为 `pass`，但 RL signal gate 为 `insufficient-horizon`：仅有 8 episodes / 4 个 PPO epoch 诊断点，且 completion component 为 0、penalty 主导；该产物只验证训练链路、日志和绘图，不证明收敛、稳定性、性能或优越性；
- 相关 RL CLI、Trainer、PPO Update 与 artifact 回归测试现为 92/92 PASS（其中 2 项 CUDA-only 测试因当前无 CUDA 跳过）；当前仍没有正式论文实验结果。
- CA-GAT-MAPPO Extended Smoke Training 已在 CPU、small、seed=42、5000 transitions、10 episodes 下真实完成：19 个完整 rollout/PPO update，completion component 已出现，signal gate 为 `signal-watch`；该单种子结果仅支持 basic learning signal 诊断，不证明收敛、稳定性、性能或优越性；
- `rl-long-smoke` 已完成配置准备与聚焦测试：默认 `training_device` 已由 CPU 调整为 CUDA；保持 Extended Smoke 的 500-slot environment、reward 和冻结 PPO 配置，仅将预算扩展为 100 episodes / 50000 transitions，沿用统一 metrics CSV 与 dashboard writer；该 profile 的 single-seed Long Smoke 已真实完成并归档。
- CA-GAT-MAPPO 已新增 episode 级实时 console progress logger：每个真实完成边界输出 episode/预算、collected transitions/预算、PPO update 计数、episode reward、完成/过期计数、最近一次完整 PPO update 的 actor loss、critic loss、entropy 均值（尚无 update 时为 `n/a`）及实际 device；仅复用既有诊断数据，不修改 artifact writer、返回值、随机种子、训练预算或算法逻辑；新增 2 项 logger 测试，Trainer/CLI 联合回归 72/72 PASS，artifact/checkpoint 实现测试 14 项与 8 个 subtests PASS；本轮未启动正式训练，也未生成新训练结果。
- 已将 CA-GAT-MAPPO single-seed Long Smoke（small、seed=42）真实运行产物归档至 `experiments/long_smoke/small_seed42/`，包括 aggregate metrics、配置快照、metrics CSV 与 dashboard PNG；本轮仅执行结果归档，未运行训练、未修改算法代码，该单 seed 产物不证明收敛、稳定性、性能或优越性。
- Formal checkpoint 生命周期、显式 `--resume-from`、Artifact Non-Overwrite Gate 和 Formal Evaluation Runner 均已完成；真实 `128 → periodic checkpoint → CLI resume → 256 → final checkpoint` 验证已通过；最新完整测试为 405 passed、119 subtests passed。
- 尚未运行小规模真实 final-checkpoint evaluation，也尚未运行任何 500,000-transition 正式实验；当前仍未生成正式实验结果。
- Post-Validation-V1 Fix/V2（分支 `codex/heuristic-fix-v2`）已完成 Fix Package 1 工程修复：未新增 observation/schema 字段，Heuristic 仅使用当前 actor tensor 已编码的队列聚合/队首、self resource、neighbor public、stale CSI、历史质量与 last-rate/mask 信息。远端路由在公开目的端状态有效时执行 `f_max + queued_cycles` coarse gate；冷启动时仅允许自身 `f_max` 不高于 reference 的 actor 使用自身 `f_max` 检查新任务本身，并以 stale CSI 的 noise-floor rate 解除 prior-TX 循环依赖，compute-rich actor 不向未知能力目的端盲目外送；历史链路质量仅在通过 gate 的候选间排序。DVFS 改为基于 `task_count + total/head cycles + head slack` 的保守离散 service-slot prefix approximation，无可行正档时保持最大正档 fallback 并暴露 infeasible telemetry。targeted Heuristic 24/24、lifecycle/conservation/energy 52/52、全量 437/437 均通过；seed 1042 同一冻结外部轨迹上 Fix-V2 为 82 completed、13 expired、reward 63.2674，zero-based `UAV1->UAV2=7`、`UAV2->UAV1=0`，外部 trace SHA-256 仍为 `752cf520a2eb735fc1d05863ea6aef17443ed9177a953cd0848e8a53d844294e`。历史 Validation V1 checkpoint、manifest、protocol、evaluation artifacts 与 run IDs 均未修改，本轮未训练、未运行 Seeds 43–46、未覆盖旧结果；后续若重新评估，必须使用新的 config/protocol identity 以及新的 run/artifact。

- Fix Package 2（分支 `codex/route-telemetry-v2`）已完成 CA-GAT-MAPPO Route Training Telemetry：新增 route-specific sample/probability/entropy、advantage/return attribution、route-head gradient、`approx_kl`/clip fraction，以及 diagnostics schema v2 的逐 PPO epoch、rolling-window、final-summary 与 Dashboard 持久化；CSV 空值与 JSON `null` 表示不可用，旧 schema CSV 和 Checkpoint V1 仍可读取且 checkpoint identity/字段不变。遥测仅消费既有 rollout/evaluation/gradient，采集点位于 backward 后、gradient clipping/optimizer step 前，不改变 sampling、loss、PPO objective、网络结构、observation、reward 或 environment dynamics。targeted 10/10、相关回归 137/137、全量 447/447 均通过（5 项 CUDA 条件跳过）；启用/禁用遥测的动作、外部 generator、全局 Torch RNG、loss/common diagnostics 和更新后参数均一致。本轮未训练、未调参、未运行 Seeds 43–46、未修改 Fix Package 1、未覆盖既有实验产物、未提交 commit。
- Diagnostic Package 3 已进入 Environment Implementation 的细粒度 route credit telemetry 阶段：仅追加 route-active 概率分解、route-only PPO/advantage/TD 分布、route-head 行级梯度方向、shared trunk 可观测性标记、route→task→destination→service→terminal 生命周期关联与 team-level reward timing；诊断 schema 升级为 v3，旧产物与 Checkpoint V1 保持兼容。本轮未训练、未调参、未运行 Seeds 43–46、未覆盖旧 artifact、未提交 commit，待人工复核后再决定后续门禁。

- Algorithm Fix Package 5 已完成 Conservation-Preserving Per-Agent Credit 实现：新增 training.mappo.agent_credit_mode={team,role_decomposed}，默认 team 继续消隐于 canonical config identity 并保持标量 reward、标量 critic/GAE/PPO 与 Checkpoint V1 legacy 行为；role_decomposed 在不改变 team reward 公式的前提下，将本地完成归属 source、远端完成按固定参考 bit/cycle 工作量拆分给 source/destination、过期归属 source、workload 的 bits/cycles 分别归属 source/CPU owner、实际 TX/CPU 能耗分别归属 sender/executor，并以 1e-9 tolerance 对 completion、expiration、workload、energy 和最终 reward 逐项执行守恒硬 Gate。role 路径使用 critic [B,T,A]、rollout/GAE [T,A]、actor advantage [B,T,A] 及 Fix-4 branch advantage [B,T,A,7]；diagnostics schema 保持 v3，仅追加 per-agent value/TD/advantage/return、route×agent、remote completion、destination balance 与 per-head critic loss，旧 artifact 缺字段仍可读取；跨 team/role resume 要求 new run，actor-only Formal Evaluation 记录 credit-mode provenance。Fix-5 targeted 20/20、综合相关回归 402/402、full unittest 493/493 PASS；Fix-5 control/treatment long-smoke 尚未执行，本轮未启动训练、未运行 Seeds 42–46、未生成新训练 artifact、未提交 commit，待人工审核。

## 下一步

1. 生成当前最终 commit 对应的小规模 `FINAL_COMPLETED` checkpoint，并运行一次小规模真实 final-checkpoint evaluation；
2. evaluation 通过后启动 seed 42 的 500,000-transition Formal 训练；第一个 `step_50000.pt` 正常后继续其余 seeds。

- `knowledge/project_plan/方案2.md` 仍包含尚未同步的旧系统模型语义，因为该文件位于 AGENTS.md 定义的 protected `knowledge/` 目录，当前自动任务不得修改。当前环境实现规格以已经冻结的 `sections/2_system_model.md` 为准（PROTECTED SYNC DEBT）。

## 可选扩展

以下内容只有在主实验闸门通过后才考虑：

- 第三 UAV Fresnel 软遮挡；
- 连续资源动作；
- Sionna RT 离线回放；
- UAV 数量 OOD；
- 离散硬件类型与类型 embedding。

## 明确不做

主论文不实施：

- UAV 轨迹优化；
- RIS；
- 多跳路由；
- DAG 任务；
- 完整 MIMO；
- 实时射线追踪；
- 完整 HARQ 协议栈；
- ns-3 包级联动。

## 证据边界

结果和讨论章节只能使用真实程序运行产生的数据、日志和图表。不得虚构实验结果、训练曲线、统计显著性或算法优势。


## 结果绘图目录

- `results/scripts/plot_training_curves.py`：从真实 Long Smoke 指标生成论文结果曲线，默认读取 `experiments/long_smoke/small_seed42/metrics.csv`，不启动训练且不修改原始实验数据。
- `results/figures/`：保存可由脚本重新生成的四幅 PNG 图：`reward_curve.png`、`completion_penalty_curve.png`、`ppo_loss_curve.png` 和 `policy_diagnostics_curve.png`。
- reward 图包含 episode reward 原始曲线与 window=10 的 trailing moving average；组件图使用 completion component 与 expiration penalty 的原始记录；PPO 图使用按 update/epoch 排序的 actor loss、critic loss、entropy 和 ratio 记录。
- 当前图表来自单 seed（seed=42） Long Smoke 诊断产物，只用于训练信号与日志检查，不支持收敛、稳定性、性能、优越性或多 seed 泛化结论。
- 在项目根目录运行 `python results/scripts/plot_training_curves.py` 可独立重新生成上述图表。
- Diagnostic Package 3.1 已完成 Return/TD residual 分布持久化补全：route-active、local/remote/defer 与 legal-remote subset 均写出 valid sample count、mean、std、median、p25、p75、positive/negative fraction；统计复用既有 detached collector 的 finite/NA/quantile 语义，schema 保持 v3，旧 artifact 未改写。本轮未训练、未改 checkpoint interval/reward/GAE/PPO/entropy/LR/network/observation/action mask/environment/Heuristic，未运行 Seeds 43–46，targeted 5/5、Package 3 11/11、PPO 39/39、update 17/17、GAE 28/28、persistence 13/13、full unittest 445/445 PASS，待人工审核。
- Algorithm Fix Package 3 已完成 Route-slot Workload Timing Neutralization：新增 `legacy_post_route` 与 `route_slot_pre_route` 两种可配置模式，默认值和旧配置/Checkpoint V1 canonical hash 继续保持 legacy；新模式只对当槽实际发生的 `unbound→local/remote` 绑定使用绑定前的 task-level bit/cycle 快照计算 workload penalty，Defer、后续时隙、真实任务状态、物理服务、能耗、完成/过期、observation、PPO 输入与 RNG 轨迹均保持原语义。未新增 telemetry/schema 字段，targeted 10/10、相关回归 327/327、full unittest 455/455 PASS；本轮未启动正式 RL training、未生成正式 artifact、未运行 Seeds 42–46、未提交 commit，待人工审核。
- Algorithm Fix Package 4 已完成 Branch-Specific PPO Ratio：新增 `training.mappo.actor_ratio_mode={joint,branch_specific}`，默认 `joint` 继续消隐于 canonical config identity 并逐操作保持 legacy joint-ratio objective；`branch_specific` 对 `[B,L,A,7]` 的 active branches 分别构造 PPO ratio/clipped surrogate，再按每个 time-agent 的 active-count 归一化，team scalar advantage、critic/GAE、entropy、reward、Fix-3 `legacy_post_route` 环境语义均未修改。diagnostics schema 保持 v3，以向后兼容追加七分支 active-count/ratio/KL/clip/surrogate/advantage 统计和逐 rollout timestep `[A,7]` activity matrix；Checkpoint V1 字段集合不变，旧 checkpoint 缺字段按 `joint`，跨模式 resume 明确要求新 run，Formal Evaluation manifest 记录训练时 ratio provenance。Fix-4 targeted 14/14、综合相关回归 334/334、full unittest 471/471 PASS（0 failure、0 error、0 skip）；本轮未启动训练、未生成正式 RL artifact、未运行 Seeds 42–46、未提交 commit，待人工审核。
