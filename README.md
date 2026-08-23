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
- 完整 environment backend 与 Random/Heuristic rollout pipeline 已实现；CA-GAT-MAPPO 的 GAT/MAPPO、Rollout、GAE、PPO Objective/Update 和 Production Trainer 已实现；QMIX、正式论文绘图和正式论文实验结果仍未完成；
- CA-GAT-MAPPO RL CLI Gate 已修复：交互菜单与直连 CLI 均经 registry 路由到真实 `CAGATMAPPOTrainer.train()` handler；`unavailable` 与 `failed` 状态返回非零退出码；
- 已新增 `rl-smoke`（CPU、256 transitions、1 个完整 rollout）、`rl-long-smoke`（CPU、small、seed=42、50000 transitions）与 `rl-formal`（CUDA、冻结正式预算）profile，并修复带类型注解的 PyTorch CUDA provenance 解析；
- CA-GAT-MAPPO Smoke Training Gate 已真实通过：CPU、seed=42、256 transitions、8 episodes、1 个完整 rollout、1 次 PPO update，CLI 退出码为 0；
- 已新增 Trainer 外围训练诊断 writer，不修改 environment、reward 公式、PPO 算法或 Trainer 核心生命周期；真实运行已生成 config snapshot、raw JSONL、aggregate JSON、dashboard CSV 和 reward/loss dashboard PNG；
- 本次真实 smoke 的均值为 reward=-0.0615874、actor_loss=0.876723、critic_loss=0.599072、entropy=0.843768；数值均有限，reward 公式恒等式与 PPO update accounting 均通过；
- Smoke Training Gate 状态为 `pass`，但 RL signal gate 为 `insufficient-horizon`：仅有 8 episodes / 4 个 PPO epoch 诊断点，且 completion component 为 0、penalty 主导；该产物只验证训练链路、日志和绘图，不证明收敛、稳定性、性能或优越性；
- `rl-long-smoke` 已完成配置准备与聚焦测试：保持 Extended Smoke 的 500-slot environment、reward 和冻结 PPO 配置，仅将预算扩展为 100 episodes / 50000 transitions，沿用统一 metrics CSV 与 dashboard writer；本轮未启动 Long Smoke 训练，也未生成 Long Smoke artifact。

## 下一步

1. `rl-long-smoke` profile 已准备但尚未执行；仅在新的明确授权下启动 single-seed Long Smoke，并继续把输出限定为诊断证据；
2. 当前不得自动启动 `rl-formal` 或 500000 transitions 训练；在正式多 seed 评估与对应证据门通过前，不作收敛、性能、稳定性或优越性结论。

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
