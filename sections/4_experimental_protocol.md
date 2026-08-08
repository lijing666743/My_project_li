# 4 极速仿真契约

> 状态：SECTION 4 READY FOR INTERFACE AUDIT
>
> 本章是进入 Python 实现前的最小实验契约，不是实验结果、收敛证明或物理层标定报告。本文所有数值均为已冻结的 Section 2 数值、Section 3 接口数值化结果，或明确标记为 IMPLEMENTATION DEFAULT — Section 4 的实现默认值。
>
> 最后更新：2026-08-08

## 4.1 适用范围与来源优先级

Section 4 只规定如何实例化环境、运行基线、训练、评估、记录和复现，不重新定义 Section 2 的系统状态、任务状态机、时隙因果、物理机制、信息权限或奖励语义，也不重新定义 Section 3 的算法接口。

参数和接口的唯一来源按以下顺序解释：

1. sections/2_system_model.md 的冻结语义和已给出的明确数值；
2. sections/3_methods.md 的七分支动作、mask、proposal/execution、recurrent、MAPPO 和 Factorized-Action GAT-QMIX 接口；
3. 本章明确标记的 IMPLEMENTATION DEFAULT — Section 4；
4. 单次运行解析后的 `RunConfig` 快照。

配置文件、命令行和交互式菜单不得产生三套不同的实验逻辑。配置解析完成后，所有执行路径都必须得到同一种 `RunConfig`，并调用同一个 `registry`、`runner` 和 environment backend。未在本章或 Section 2/3 中定义的参数，不得由实现者临时猜测。

本章默认值可以在后续 sensitivity/ablation 中由一个新的、显式记录的配置覆盖，但不得在正式 run 中使用未记录的隐式覆盖。默认值未经真实仿真验证，不得在 Section 5/6 中写成实验结论。

## 4.2 Experiment configuration contract

### 4.2.1 `RunConfig` 唯一入口

所有实验由统一配置对象 `RunConfig` 驱动。它至少包含以下顶层字段：

| 字段 | 唯一含义 | 必须记录 |
|---|---|---|
| `config_version` | `section4.v1` | 是 |
| `mode` | `environment_sanity`、`gate0`、`random`、`heuristic`、`baseline`、`rl`、`evaluation`、`ablation` 或 `plot` | 是 |
| `method_id` | `environment`、`random`、`heuristic`、`ca_gat_mappo` 或 `factorized_action_gat_qmix` | 是 |
| `scenario_id` | `small`、`medium` 或 `large` | 是 |
| seed | 本次运行的 master seed | 是 |
| environment | 场景、移动、任务、信道、资源和能量参数 | 是 |
| action | 七分支 domain、canonical 值和 mask 规则的数值配置 | 是 |
| reproducibility | seed stream、版本和快照设置 | 是 |
| training | 方法专属网络、优化和预算参数 | 是 |
| evaluation | 评估 seed、推理规则和统计参数 | 是 |
| output | run ID 与输出路径 | 是 |

YAML/config 文件的每一个叶字段必须一一映射到 `RunConfig` 的 typed field；字段名、单位、枚举值和默认值不得在 YAML 与 Python 中分别维护。未知字段、缺失必填字段和非法枚举值均在 `runner` 启动前报错。解析后的配置必须保存为 canonical snapshot，快照中同时保留原始配置路径、CLI 覆盖和最终解析值。

### 4.2.2 覆盖优先级

配置覆盖优先级从低到高固定为：

Section 4 defaults < YAML/config file < explicit CLI flags < interactive selections。

交互式选择只负责产生与 CLI 等价的显式字段；菜单不得直接修改环境或训练状态。`--seed 42` 高于 YAML 中的 `seed`；没有显式覆盖时使用配置文件中的 `seed`；没有配置文件时使用 42。同一个 `RunConfig` 只能有一个最终值，解析后不得在 `runner` 内再次读取环境变量或随机生成未记录参数。

interactive CLI 和 argparse/direct CLI 的区别只在输入方式：

- `python main.py` 显示菜单，收集 `mode`、`method_id`、`scenario_id`、配置路径和覆盖项，构造 `RunConfig`；
- `python main.py --mode random --config <config> --seed 42` 直接构造同样的 `RunConfig`；
- 两条路径都调用同一个 `registry[mode/method_id]`、同一个 `runner` 和同一个环境后端；
- 不允许为 interactive mode 和 batch/direct mode 编写两套业务执行逻辑。

## 4.3 Scenario contract

### 4.3.1 场景标识和共同参数

本轮实现只注册 small、medium、large 三个场景。三者共享 Section 2 的系统语义，并使用以下共同数值。

| 配置项 | 唯一值 | 来源与说明 |
|---|---:|---|
| episode horizon $T$ | 500 slot | Section 2 编码初始值；固定 horizon |
| slot duration $\Delta t$ | 0.020 s | Section 2 已冻结的 20 ms |
| 固定高度 $H$ | 80 m | Section 2 已冻结 |
| 区域 bounds | $x,y\in[0,1000]\,\mathrm{m}$ | Section 2 的 1000 m × 1000 m 轴对齐区域 |
| 边界规则 | coordinate-wise specular reflection | Section 2 已冻结；不使用 clamp |
| reset safe distance | 50 m | IMPLEMENTATION DEFAULT — Section 4 |
| UAV 初始速度均值 $\bar{\mathbf v}$ | $(0,0)\,\mathrm{m/s}$ | IMPLEMENTATION DEFAULT — Section 4 |
| UAV 速度创新标准差 | $(1,1)\,\mathrm{m/s}$ | IMPLEMENTATION DEFAULT — Section 4，逐坐标 Gaussian |
| Gauss-Markov coefficient $\alpha$ | 0.85 | Section 2 编码初始值 |
| candidate neighbor radius $R_{\mathrm{cand}}$ | 500 m | Section 2 编码初始值 |
| total bandwidth $B_{\mathrm{tot}}$ | 20 MHz | Section 2 参数表；不得改写 |
| RU count $R$ | 20 | Section 2 参数表；不得改写 |
| RU bandwidth $B_{\mathrm{RU}}$ | 1 MHz | 由 $B_{\mathrm{tot}}/R$ 唯一得到 |
| fixed resource-group count $G$ | 5 | Section 2 已冻结 |
| RU per group | 4 | $R/G$；组内 RU 按 1-based 连续编号 |
| carrier frequency $f_c$ | 3.5 GHz | Section 2 编码初始值 |
| reference transmit power $P_{\mathrm{ref}}$ | 1 W | Section 2 编码初始值 |
| reference CPU frequency $f_{\mathrm{ref}}$ | 2.0e9 cycle/s | Section 2 编码初始值 |
| reference CPU coefficient $\kappa_{\mathrm{ref}}$ | 1.0e-28 | Section 2 编码初始值 |
| reference initial energy $E_{\mathrm{ref}}^0$ | 30 J | Section 2 编码初始值 |
| reference rate $R_{\mathrm{ref}}$ | 20e6 bit/s | Section 2 编码初始值 |
| antenna gain | 2 dBi at both ends | Section 2 编码初始值 |
| noise PSD $N_0$ | -174 dBm/Hz, converted to linear W/Hz | IMPLEMENTATION DEFAULT — Section 4 |
| receiver noise figure $F_{\mathrm{dB}}$ | 7 dB | Section 2 编码初始值 |
| building loss $L_B$ | 15 dB when blocked, 0 dB otherwise | Section 2 主模型编码初始值 |
| LoS Rician $K_{\mathrm{dB}}$ | 6 dB | Section 2 已给出 |
| shadowing std $\sigma_s$ | 4 dB | Section 2 编码初始值 |
| shadowing correlation distance $d_{\mathrm{corr}}$ | 50 m | Section 2 编码初始值 |
| fixed CSI AoI | 3 slot | IMPLEMENTATION DEFAULT — Section 4；whole episode constant |
| CSI error std $\sigma_{\mathrm{CSI}}$ | 2 dB | IMPLEMENTATION DEFAULT — Section 4 |
| interference EMA $\beta_I$ | 0.8 | IMPLEMENTATION DEFAULT — Section 4 |
| initial interference value $I_{\mathrm{def}}$ | 0 W with availability mask 0 | IMPLEMENTATION DEFAULT — Section 4；不是有效零测量 |
| outage threshold | -3 dB, 0.501187... linear | Section 2 已给出 dB/linear conversion |
| minimum task slack clip | 10 slot | Section 2 deadline formula |
| maximum task slack clip | 150 slot | Section 2 deadline formula |

$F=10^{F_{\mathrm{dB}}/10}$，$P_{\mathrm{noise,RU}}=N_0B_{\mathrm{RU}}F$。所有噪声、功率、SINR 和 rate 计算在同一功率域中完成；不得将 dB 值直接代入线性公式。

### 4.3.2 三个场景

下表冻结每个场景的代码读取值。$\lambda_i$ 是每槽最多一个任务时的 Bernoulli 到达概率；同一数组按 UAV ID 从 0 到 $N-1$ 使用。它们均标记为 IMPLEMENTATION DEFAULT — Section 4，不表示已经过 Gate P 或实验验证。

| 场景 | $N$ | $\lambda_i$（按 UAV ID） | profile assignment（按 UAV ID） |
|---|---:|---|---|
| small | 4 | [0.04, 0.05, 0.06, 0.05] | [Balanced, Resource-poor, Compute-rich, Energy-limited] |
| medium | 6 | [0.03, 0.04, 0.05, 0.06, 0.05, 0.07] | [Balanced, Resource-poor, Compute-rich, Energy-limited, Communication-rich, Balanced] |
| large | 8 | [0.02, 0.04, 0.06, 0.08, 0.02, 0.04, 0.06, 0.08] | [Resource-poor, Balanced, Compute-rich, Energy-limited, Communication-rich, Resource-poor, Balanced, Compute-rich] |

三个场景的 horizon 均为 $T=500$；场景规模差异只由 $N$、任务到达向量和 profile assignment 产生，不引入未声明的 horizon 变化。

每个 UAV 的资源参数由 Section 2 的 profile 比例乘以参考值，再使用独立、可复现的逐参数扰动 $\delta_{i,m}\sim\operatorname{Uniform}(-0.05,0.05)$ 生成。五类 profile 的比例固定如下：

| profile | $f_{\max}/f_{\mathrm{ref}}$ | $P_{\max}/P_{\mathrm{ref}}$ | $E^0/E_{\mathrm{ref}}^0$ | $\kappa/\kappa_{\mathrm{ref}}$ |
|---|---:|---:|---:|---:|
| Resource-poor | 0.7 | 0.8 | 0.8 | 1.20 |
| Balanced | 1.0 | 1.0 | 1.0 | 1.00 |
| Compute-rich | 1.5 | 1.0 | 1.1 | 0.90 |
| Energy-limited | 1.0 | 0.8 | 0.5 | 1.10 |
| Communication-rich | 1.0 | 1.5 | 1.1 | 1.00 |

### 4.3.3 任务、移动和环境几何

任务生成和 deadline 公式直接复用 Section 2：

- $D_n\sim\operatorname{Uniform}(0.25,1.5)\,\mathrm{Mbit}$，代入传输时间前转换为 bit；
- $C_n/D_n\sim\operatorname{Uniform}(300,1000)\,\mathrm{cycle/bit}$；
- $\eta_n\sim\operatorname{Uniform}(1.5,3.0)$；
相对 deadline 时隙预算为：

$$
L_n^{\mathrm{ddl}}
=
\operatorname{clip}
\left(
\left\lceil
\frac{
\eta_n
\left(
D_n/R_{\mathrm{ref}}+C_n/f_{\mathrm{ref}}
\right)
}{
\Delta t
}
\right\rceil,
10,
150
\right).
$$

截止时隙索引为：

$$
t_n^{\mathrm{ddl}}
=
t_n^{\mathrm{arr}}+L_n^{\mathrm{ddl}}.
$$
- 到达在槽末生成，下一槽可见并 route，再下一槽才可 service。

reset 时所有队列为空，task-ID counter 为 0，新任务按槽末 UAV-ID 升序、同一 UAV 内稳定生成顺序分配 0,1,2,...。任务目的地在 route 槽末绑定后锁定；local 绑定立即清零剩余待传输 bit；已绑定任务不得重新 route。

移动使用 Section 2 的 Gauss-Markov 更新和 coordinate-wise specular reflection。reset 采用 UAV-ID 升序 rejection sampling，候选位置必须在 $[0,1000]^2$ 且两两距离不小于 50 m。运行期不加入 collision controller、clamp 或随机重置。

为使建筑物几何也有唯一的实现输入，第一版使用以下确定性 `building_layout_v1`，该布局是 IMPLEMENTATION DEFAULT — Section 4：

| building | `x_range` m | `y_range` m | height m |
|---|---|---|---:|
| B1 | [150,250] | [150,350] | 120 |
| B2 | [400,500] | [150,350] | 120 |
| B3 | [550,650] | [650,850] | 120 |
| B4 | [750,850] | [650,850] | 120 |

建筑物按 Section 2 的三维线段相交规则判定遮挡；不使用随机建筑物、不使用实时射线追踪、不加入第三 UAV Fresnel 软遮挡。

## 4.4 Seven-branch action configuration

### 4.4.1 固定 domain、canonical inactive value 和 activity

动作严格保持 Section 3 的七个分支，不新增 g/w 独立分支，不新增第八分支：

| 顺序 | branch | fixed domain | canonical inactive value | active indicator |
|---:|---|---|---|---|
| 1 | `route` | $\{\mathrm{idle},\mathrm{local},\mathrm{defer}\}\cup(\mathcal U\setminus\{i\})$ | `idle` | $\mathbf{1}[Q_i^{\mathrm{unb}}\ne\varnothing]$ |
| 2 | `tx_select` | $\{\mathrm{idle}\}\cup(\mathcal U\setminus\{i\})$ | `idle` | $\mathbf{1}[\exists j\ne i:Q_{i\rightarrow j}^{\mathrm{tx}}\ne\varnothing]$ |
| 3 | `resource_group` | $\{\mathrm{idle},1,2,3,4,5\}$ | `idle` | $\mathbf{1}[\widetilde a_i^{\mathrm{tx}}\ne\mathrm{idle}]$ |
| 4 | `resource_width` | $\{1,2\}$ | 1 | $\mathbf{1}[\widetilde a_i^{\mathrm{tx}}\ne\mathrm{idle}]\mathbf{1}[\widetilde a_i^{\mathrm{rg}}\ne\mathrm{idle}]$ |
| 5 | `power_level` | $\{0,0.25,0.5,1\}$ | 0 | $b_i^{\mathrm{width}}$ |
| 6 | `cpu_queue` | $\{\mathrm{idle}\}\cup\mathcal U$ | `idle` | $\mathbf{1}[Q_i^{\mathrm{loc}}\ne\varnothing\lor\exists k\ne i:Q_{k\rightarrow i}^{\mathrm{cpu}}\ne\varnothing]$ |
| 7 | `cpu_frequency` | $\{0,0.25,0.5,1\}$ | 0 | $\mathbf{1}[\widetilde a_i^{\mathrm{cpuq}}\ne\mathrm{idle}]$ |

`resource_width=1` 在 inactive 时只是既有 domain 中的 canonical encoding，不是新的 `idle` action；`power_level=0` 仍是既有功率档位。active branch 即使合法采样到 0 power，也仍计为该 branch 的显式决策；父通信分支 inactive 时的 canonical 0 不计入 policy contribution。

### 4.4.2 资源组数值化和 sampling order

固定资源组按 1-based RU 编号冻结为：

- $G_1=\{1,2,3,4\}$；
- $G_2=\{5,6,7,8\}$；
- $G_3=\{9,10,11,12\}$；
- $G_4=\{13,14,15,16\}$；
- $G_5=\{17,18,19,20\}$。

active 通信 proposal 只允许一个组或两个相邻组。`resource_group=5` 与 `resource_width=2` 的组合由 mask 排除；策略网络不直接输出任意 RU 子集。`resource_group + resource_width` 只形成 $S_{\mathrm{prop}}$，不代表已经执行占用。

七个 branch 的唯一采样/选择顺序为：

`route -> tx_select -> resource_group -> resource_width -> power_level -> cpu_queue -> cpu_frequency`。

mask 只可读取槽初 actor-visible state、历史特征、feature masks 和已经采样的合法前置 branch。每个 branch 至少保留一个合法 action：无任务或无队列时保留 canonical action，width 保留 1，power 保留 0，CPU frequency 保留 0。mask 不得读取当前真实 channel、current SINR、current interference、当前 measurement、联合 proposal 或 executor 输出。

### 4.4.3 proposal、executor 和 physical action

每个 slot 先形成 actor proposal，再按 Section 2 的顺序执行：

1. `resource_group + resource_width` 构造 $S_{\mathrm{prop}}$；
2. `power_level` 构造 $p_{\mathrm{prop}}$，其数值为该档位与 $P_i^{\max}$ 的乘积；
3. raw $p_{\mathrm{prop}}=0$ 立即 canonicalize 为 communication `idle`；
4. 只在 proposal 形成后由 executor 计算 valid-RU historical-quality scalar；
5. 按 `(slack, -historical_quality, sender_id, receiver_id)` 升序进行确定性 half-duplex 仲裁；
6. 按 $1.0\rightarrow0.5\rightarrow0.25\rightarrow0$ 的离散 candidate power 顺序进行 power-first energy downgrade；
7. 必要时按 Section 2 的既有 CPU frequency downgrade 顺序选择最高可行档位；
8. 形成最终 $y$、$x$、$S_{\mathrm{exec}}$、$p_{\mathrm{exec}}$、CPU frequency、service、actual energy 和 outage。

executor 不改变 proposal，不生成第八个 action branch，不使用随机 tie-break，不依赖 Python set/dict 遍历顺序，不因最终降至零功率而 backtrack 或 re-arbitrate。`proposal_action[7]`、`executed_action_summary` 和 `rejection_or_downgrade_summary` 必须分开记录；后两者不是 policy action，也不得用于重算 old log-prob 或 PPO ratio。

## 4.5 Reproducibility contract

### 4.5.1 seed ownership

`RunConfig.seed` 是唯一 master seed；实现不得使用未记录的系统时间作为随机源。各随机源的 ownership 固定如下：

| stream | owner | derived stream id |
|---|---|---:|
| reset position/velocity and mobility | environment | 10 |
| task arrival | environment | 20 |
| task size/compute/deadline | environment | 30 |
| shadowing and small-scale fading | environment | 40 |
| CSI error | environment | 50 |
| interference measurement/history noise if enabled | environment | 60 |
| Python utility RNG | launcher/runner only | 70 |
| Torch model initialization | method implementation | 100 |
| Torch policy sampling | method implementation | 110 |
| QMIX epsilon exploration | method implementation | 120 |

每个 stream 使用 `SeedSequence([master_seed, stream_id])` 派生独立 generator；环境使用自己的 NumPy generator，模型不得调用环境 generator，环境不得调用 Torch generator。Python random 只使用 stream 70，不能成为任务、信道或 executor 的隐式随机源。Torch CPU/CUDA generator、Python RNG 和 NumPy generator 均须在 snapshot 中记录其派生关系。

固定 master seed、场景、配置和 git commit 时，reset 状态、task IDs、arrival sequence、mobility、shadowing、fading、CSI error、measurement、random policy proposal 和训练初始化必须可复现。确定性 executor 不使用随机 tie-break。

### 4.5.2 train/eval seeds、snapshot 和 run ID

默认单次调试 `seed` 为 42。正式多 seed 训练集合冻结为 `{42,43,44,45,46}`；评估集合冻结为 `{1042,1043,1044,1045,1046}`，两集合不重叠。random 和 heuristic baseline 使用与被比较方法相同的 evaluation seed 集合，并保留独立 `method_id`。

每次运行必须记录：

- canonical `RunConfig` snapshot 和所有显式覆盖；
- `run_id`、`method_id`、`scenario_id`、master seed 和所有 derived stream IDs；
- git branch、完整 git commit、dirty/clean 状态；
- Python、NumPy、Torch、CUDA 和 Miniconda 版本与依赖清单；
- 环境 backend、算法 backend 和配置 schema 版本。

`run_id` 唯一构造为：

`<method_id>__<scenario_id>__seed-<seed>__cfg-<sha256(canonical_config)[0:12]>__git-<HEAD[0:12]>`。

Python、PyTorch、CUDA 和 Miniconda 的真实版本在当前文档阶段尚未验证，必须原样标记为：TO VERIFY AT IMPLEMENTATION PREFLIGHT。不得在代码或论文中猜写版本号。

## 4.6 Gate 0：RL 前环境一致性闸门

Gate 0 是开始 MAPPO/QMIX 正式训练前的硬闸门。Gate 0 未通过时，只允许修复环境、运行单元测试和诊断 rollout，禁止正式 RL training、超参数搜索和算法优势结论。每一项都必须有可复现输入、实际执行输出和 pass/fail 记录。

| ID | 必须验证的契约 | 通过条件 |
|---|---|---|
| G0-01 | reset reproducibility | 相同 config/seed 的 reset observation、state、位置、队列、history 和 masks 完全一致 |
| G0-02 | deterministic seed ownership | 不同 stream 不互相消费；重复运行的 stream sequence 一致 |
| G0-03 | task lifecycle | 任务只能沿 Section 2 合法状态转移，终止态无后继转移 |
| G0-04 | `arrival -> route -> service timing` | 槽末 arrival 在下一槽 route，route 槽末 binding，下一槽才首次 service |
| G0-05 | hard deadline | deadline slot 可接受最后一次合法 service；未完成任务立即 expired，不能过期后继续 service |
| G0-06 | done/expired mutual exclusivity | 每个任务最多标记一个 done 或 expired，二者不共存 |
| G0-07 | truncated semantics | horizon terminal 的 truncated 不计入 expired，不伪造 completion、$s_T$ 或额外 reward |
| G0-08 | no fake `s_T` | $t=T-1$ 后无 actor、service、post-horizon drain 或 bootstrap state |
| G0-09 | destination locking | route 绑定目的地后保持不变，越出新邻居范围不触发 reroute |
| G0-10 | queue bookkeeping | task 只存在于正确队列；transfer completion 在槽末入 CPU 队列，不能同槽重复计算 |
| G0-11 | bit conservation | 初始输入 bit = 已服务 bit + 剩余 bit + 已结算未完成 bit，允许的清零只限 local binding 语义 |
| G0-12 | CPU-cycle conservation | 初始 cycles = 已服务 cycles + 剩余 cycles + 已结算未完成 cycles |
| G0-13 | tx energy | 通信能耗只由最终 executed nonzero power 和实际活跃时间产生，reservation 未使用部分释放 |
| G0-14 | CPU energy | CPU 能耗只由最终 executed frequency 和实际 CPU 活跃时间产生 |
| G0-15 | executor deterministic ordering | 固定 proposal/state 下仲裁、拒绝、降档和 tie-break 结果唯一 |
| G0-16 | proposal/executed separation | proposal、executed action 和 rejection/downgrade metadata 分离保存，executor 不反写 proposal |
| G0-17 | zero-power physical idle | zero power 不占资源、不占 half-duplex、不产生 interference、service、tx energy、attempt 或 outage |
| G0-18 | half-duplex consistency | 任意 UAV 不同时发送和接收；CPU 与无线并行语义保持不变 |
| G0-19 | actor observation leakage | actor/mask/graph 不读取 current true SINR、current interference、current measurement、future state 或 executor scalar |
| G0-20 | historical CSI/interference causality | slot-start history 只使用已结束 slot；current measurement 最早在下一槽可见；stale CSI 不访问负索引 |
| G0-21 | outage denominator and NA | outage 只由实际 accepted nonzero transmission attempt 进入分母；无样本返回 NA，不得写为 0 |

Gate 0 的统计结果必须同时区分 terminated、expired、truncated。truncated 是 episode boundary/accounting 标签，不是任务 expired 状态；最后一个槽末生成且未进入 service 的任务也只计 truncated。

## 4.7 First executable baselines

第一批真实环境结果的执行顺序冻结为：

`environment sanity -> Gate 0 -> random policy rollout -> heuristic policy rollout -> RL`。

### 4.7.1 Random policy

random policy 只在当前 branch 的 valid mask 集合内抽样，按固定七分支顺序形成 proposal，并始终经过同一个 deterministic executor、half-duplex、energy downgrade、service、settlement 和 history update。random policy 不得绕过执行器，也不得直接把 proposal 当作 physical action。其输出使用 `method_id=random`，不能与 RL 结果混名。

### 4.7.2 Heuristic policy

heuristic policy 只可读取 actor-visible information：槽初本地队列、合法邻居、历史 CSI、历史干扰摘要、历史质量代理、AoI、历史到达率、资源/能量状态和 masks。它不得读取 current true channel、current SINR、current interference、current measurement、future arrival、future queue 或 executor-only scalar。heuristic proposal 仍必须经过相同 deterministic executor。

第一版 heuristic 只使用已有 EDF 顺序、合法 destination/queue preference 和 valid-mask fallback；本章不引入复杂优化器、look-ahead、MILP、oracle 或新的方法分支。

RL 只有在 Gate 0、random rollout 和 heuristic rollout 均留下真实 raw metrics 后才允许启动；早期实现菜单可以显示 RL，但在算法未实现时必须返回明确的 unavailable/not implemented 状态，不得伪装成已可用训练。

## 4.8 Metrics contract

第一批环境仿真每个 run、每个 episode 和 aggregate level 至少输出以下指标：

| 指标 | 定义 | NA/边界语义 |
|---|---|---|
| task completion rate | `completed_task_count / generated_task_count` | generated 为零时为 NA |
| task expiration rate | `expired_task_count / generated_task_count` | truncated 不进入 expired numerator |
| successful completed task count | done 任务的计数 | 整数，不把 truncated 算作 done |
| E2E latency | 对每个 done 任务计算 $(t_{\mathrm{cmp}}-t_{\mathrm{arr}})\Delta t$，再汇总 | 只对 completed task；无 completed sample 为 NA |
| energy consumption | $\sum(E_{\mathrm{tx}}+E_{\mathrm{cpu}})$ 的实际主动能耗 | 不包括推进能耗；只计当前 episode service slots |
| outage rate | `sum(outage_sample=1) / len(actual_attempt_samples)` | 无 actual-attempt sample 为 NA，不能写为 0 |
| queue backlog | 每个 slot-start 的 active task 数之和，并报告 mean/final | 不把 done/expired/truncated 计入 active backlog |

`generated_task_count` 包括 $t=0,\ldots,T-1$ 槽末实际生成的任务；最后槽生成但未进入 route/service 的任务计入 generated 和 truncated，不计入 expired。E2E latency、outage rate 等有 sample gate 的指标必须同时保存 mean、std 和 `valid_sample_count`；aggregate 不能把 NA 样本隐式转换成零。

每个 raw slot record 还必须能审计 terminated、expired、truncated、actual attempt denominator、outage numerator、queue backlog、actual tx/CPU energy、proposal/executed summary 和 task counters。指标定义不得因方法或场景改变。

## 4.9 Output / logging contract

本项目的目录约束优先于通用 skill 默认值：

- `logs/`：运行日志、每次运行的配置快照、raw metrics、aggregate metrics 和未来 checkpoint；
- `plots/`：只保存由真实 raw/aggregate metrics 生成的 dashboard 和 figure input data；
- `dashboard_logs/`：Dashboard CSV；
- `tests/`：Gate 0、环境和接口测试代码；本轮不创建测试文件。

单次 run 的输出路径冻结为：

| artifact | 路径 |
|---|---|
| config snapshot | `logs/<run_id>/config_snapshot.yaml` |
| raw slot/episode metrics | `logs/<run_id>/raw_metrics.jsonl` |
| aggregate metrics | `logs/<run_id>/aggregate_metrics.json` |
| dashboard CSV | `dashboard_logs/<run_id>_metrics.csv` |
| figure input data | `plots/<run_id>_figure_input.csv` |
| future RL checkpoint | `logs/<run_id>/checkpoints/step_<k>.pt` |
| future dashboard image | `plots/<run_id>_dashboard.png` |

只有真实程序运行成功后才能生成这些产物；本轮不得创建 fake CSV、fake PNG、fake checkpoint 或伪造指标。raw metrics 是 aggregate 和 dashboard 的唯一数据源；dashboard 不能反向成为实验数据源。每个 artifact 的 header/字段必须包含 `run_id`、`method_id`、`scenario_id`、`seed`、`git_commit` 和 `config_hash`。

## 4.10 CLI contract

无参数入口固定为：

`python main.py`

交互式菜单文本和逻辑顺序固定为：

U2U-MEC Experiment Launcher

[1] Environment sanity check
[2] Gate 0 tests
[3] Random policy rollout
[4] Heuristic policy rollout
[5] Baseline experiment
[6] RL training
[7] Evaluation
[8] Ablation
[9] Plot results
[0] Exit

非交互入口至少支持：

`python main.py --mode random --config <config> --seed 42`

`--mode`、`--config` 和 `--seed` 必须进入同一个 `RunConfig` 解析器。菜单项 [5] 至 [9] 可以在早期实现中显示为未实现或依赖未满足，但不得返回伪造成功；[1] 至 [4] 是进入 RL 前的最小实现面。

CLI 不得复制 environment backend、registry、runner 或指标逻辑。interactive、direct、测试调用和未来 batch 调用都只允许经由同一个执行入口。

## 4.11 CA-GAT-MAPPO numerical training contract

以下数值是 IMPLEMENTATION DEFAULT — Section 4，用于使代码可以开始实现；它们不是已验证的最优超参数。正式 run 必须在 config snapshot 中显式写出这些值。

| 项目 | 唯一实现默认值 |
|---|---:|
| encoder hidden dimension | 128 |
| CA-GATv2 layer count | 1 |
| attention head count | 4 |
| GRU hidden dimension | 128 |
| actor learning rate | 3.0e-4 |
| critic learning rate | 3.0e-4 |
| optimizer | Adam |
| $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| PPO clip $\epsilon$ | 0.20 |
| entropy coefficient $c_H$ | 0.01 |
| value coefficient $c_V$ | 0.50 |
| rollout length | 256 environment slots |
| recurrent chunk length | 32 slots |
| sequence minibatch size | 8 contiguous chunks |
| PPO update epochs | 4 |
| gradient clipping | global norm 0.5 |
| max training episodes | 1000 |
| max training environment steps | 500000 |
| budget stop rule | first reached among the two max budgets |
| evaluation interval | every 50000 training environment steps |
| checkpoint interval | every 50000 training environment steps |

训练使用 Section 3 的 active-branch joint log-prob、active-branch entropy、post-service/post-settlement reward 和 contiguous recurrent minibatch。固定 horizon no-bootstrap contract 必须保持：$t=T-1$、terminated 和 truncated 的 $b_t^{\mathrm{boot}}=0$；不得创建虚构 $s_T$。本文固定 NO BURN-IN；chunk initial hidden 唯一来自 rollout buffer 对应位置保存的 `hidden_in`，不得重新估计或重建。

MAPPO actor 输出 proposal，环境执行器输出 executed action；PPO ratio 只使用 proposal 的 masked categorical log-prob，不使用 executed summary、当前 SINR 或 post-action outcome。

## 4.12 Factorized-Action GAT-QMIX numerical training contract

以下数值同样是 IMPLEMENTATION DEFAULT — Section 4，并保持 Factorized-Action GAT-QMIX 而非 flat joint-action QMIX：

| 项目 | 唯一实现默认值 |
|---|---:|
| encoder hidden dimension | 128 |
| CA-GATv2 layer count | 1 |
| attention head count | 4 |
| GRU hidden dimension | 128 |
| learning rate | 5.0e-4 |
| $\gamma$ | 0.99 |
| optimizer | Adam |
| replay capacity | 10000 contiguous sequences |
| recurrent sequence length | 32 slots |
| sequence batch size | 32 sequences |
| $\epsilon$ start | 1.00 |
| $\epsilon$ end | 0.05 |
| epsilon decay | linear over 100000 environment steps |
| target update interval | every 2000 optimizer updates |
| max training episodes | 1000 |
| max training environment steps | 500000 |
| budget stop rule | first reached among the two max budgets |
| checkpoint interval | every 50000 training environment steps |

QMIX 训练规则固定为 masked epsilon-greedy，评估规则固定为 masked greedy；所有 branch 选择都使用当前 valid mask，所有 proposal 都进入同一个 deterministic executor。replay 必须保存 episode 内 contiguous sequence、proposal、masks、active indicators、boundary、hidden、next-state 和 executed summary；不得打散成独立 transition，不得枚举 flat joint action。mixer 使用 non-negative mixing weights，保持 Section 3 的 monotonic mixer。固定 horizon no-bootstrap、NO BURN-IN、七分支 proposal contract 和 half-duplex/energy executor 均与 MAPPO 共享。

## 4.13 Evaluation contract

评估必须满足以下规则：

- train seeds `{42,43,44,45,46}` 与 eval seeds `{1042,1043,1044,1045,1046}` 不重叠；
- evaluation 不更新网络、不写 optimizer state、不改变 checkpoint；
- MAPPO evaluation 使用每个有效 branch 的 masked argmax，按固定 order 形成 proposal；
- QMIX evaluation 使用 masked greedy，按固定 order 形成 proposal；
- random 和 heuristic 使用独立 `method_id`，不得伪装成 trained method；
- 所有方法共享同一 environment、executor、reward timing、task/deadline、metric 和 NA 语义；
- 每个 metric 至少保存 mean、std 和 `valid_sample_count`；
- latency、outage 和其他无有效 sample 的统计只在有效样本上计算，无样本输出 NA，不以零填充；
- evaluation 的 actor 仍不得读取 current true SINR/interference/future state；共享 environment 的内部真值只可用于执行、settlement、critic training metadata 和审计；
- terminated、expired 和 truncated 在 evaluation accounting 中分别报告，truncated 不计入 expired。

当前章节只冻结实现契约和默认参数，不产生训练曲线、算法排名、统计显著性或性能优势声明。

## 4.14 Implementation preflight and audit checklist

开始写环境接口前，必须完成以下只读/验证项：

1. 读取并锁定 Section 2/3 的 commit 版本和当前 Section 4 snapshot schema；
2. 实测记录 Python、NumPy、Torch、CUDA、Miniconda 和依赖版本，并替换 TO VERIFY AT IMPLEMENTATION PREFLIGHT；
3. 由 `RunConfig` 解析同一份 `small/medium/large` 配置，检查所有叶字段有唯一值和单位；
4. 先实现 environment sanity 和 Gate 0，再运行 random 与 heuristic rollout；
5. Gate 0 未通过时停止在环境审计阶段，不启动 MAPPO/QMIX；
6. 所有输出只来自真实运行，任何 missing/NA 指标必须保留其语义；
7. 通过 interface audit 后，才允许进入代码实现和后续 RL 训练。

本章冻结的实现面为：scenario/config、action discretization、reproducibility、Gate 0、random/heuristic、metrics、logging、CLI、MAPPO hyperparameters、QMIX hyperparameters 和 evaluation。除版本值需在实现前实测外，本契约不保留关键参数二选一，也不开放新的动作分支、物理机制、状态或方法创新问题。
