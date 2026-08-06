# 2 系统模型与问题建模

## 2.1 系统概述

​      本文考虑由 $N$ 架 UAV 组成的动态异构多 UAV U2U-MEC 系统。UAV 既可以产生任务，也可以作为 U2U 发送节点、接收节点、本地计算节点或远程协同计算节点。系统在外生移动、动态候选邻居、资源异构、业务负载异构、建筑物遮挡、同频干扰、不完美 CSI、跨时隙队列和任务截止期共同存在的条件下运行。研究对象是单跳协同卸载与资源编排：任务只能从源 UAV 直接传输到一个目的 UAV，不能拆分到多个目的地，也不允许经由第三架 UAV 多跳转发。

​      系统时间被离散为长度为 $\Delta t$ 的时隙。槽内传输和计算服务只作用于槽初已经存在的队列；槽内新路由任务、传输完成任务和新到达任务均在槽末入队，并在下一时隙才具有服务资格。该安排保留传输中断、排队等待、CPU 竞争和截止期违约的端到端因果关系。任务结果回传暂不单独建模，因此远程计算完成表示任务服务过程结束；若后续研究需要刻画结果数据回传，可在主模型通过附加的反向服务队列扩展，但不改变本章的单跳输入卸载接口。

​      本章的主模型包括 UAV 状态、任务生命周期、EDF 队列、外生移动、轻量 U2U 物理层评价、固定资源组、跨时隙传输和计算服务、主动通信及 CPU 能耗、七分支离散动作、联合半双工解析和 Dec-POMDP 信息边界。第三 UAV 软遮挡、高保真传播回放、UAV 数量 OOD 和离散硬件类型属于主实验通过后再考虑的可选扩展；连续资源动作同样不进入主模型。它们不改变主模型中任务不可多目的地拆分、单跳、固定高度、固定资源组和槽边界等冻结规则。

​      本章刻意限定建模声明边界。该模型不是完整飞控系统，不包含六自由度飞行动力学、旋翼推力、姿态控制或飞控回路；不是完整通信协议栈，不包含包级重传、HARQ、随机接入和控制信令细节；不是实测信道、NS-3 包级联动或实时射线追踪系统。无线部分属于 modeled physical-layer evaluation：它用几何遮挡、路径损耗、相关阴影、槽级衰落、不完美 CSI 和同频干扰生成可解释的链路评价量，用于比较卸载和资源编排行为，而不宣称替代实测或高保真物理层仿真。

## 2.2 符号与基本定义

​      系统采用离散时隙、有限 UAV 集合和显式任务记录。下表集中给出跨小节重复使用的符号；首次出现时仍在局部文字中补充其物理含义和单位。

| 分组 | 符号 | 含义 | 单位或定义域 |
|---|---|---|---|
| UAV 网络 | $\mathcal U$ | UAV 集合 | $\{1,\ldots,N\}$ |
| UAV 网络 | $i,j,k$ | UAV 索引 | $\mathcal U$ |
| UAV 网络 | $N$ | UAV 数量 | 正整数 |
| UAV 网络 | $\mathcal N_i(t)$ | UAV $i$ 在时隙 $t$ 的新任务候选邻居集合 | $\mathcal U\setminus\{i\}$ 的子集 |
| 时间与空间 | $t$ | 时隙索引 | $\{0,\ldots,T-1\}$ |
| 时间与空间 | $\Delta t$ | 时隙长度 | s |
| 时间与空间 | $T$ | episode 时隙数 | 正整数 |
| 时间与空间 | $\mathbf p_i(t)$ | 水平位置 | $\mathrm{m}^2$ |
| 时间与空间 | $\mathbf q_i(t)$ | 三维 UAV 位置 | $\mathrm{m}^3$ |
| 时间与空间 | $\mathbf v_i(t)$ | 三维速度 | $\mathrm{m/s}$ |
| 时间与空间 | $H$ | 固定飞行高度 | m |
| 时间与空间 | $d_{ij}(t)$ | UAV $i$ 与 $j$ 的三维距离 | m |
| UAV 平台 | $f_i^{\max}$ | UAV $i$ 最大 CPU 频率 | cycle/s |
| UAV 平台 | $P_i^{\max}$ | UAV $i$ 最大通信发射功率 | W |
| UAV 平台 | $E_i^0$ | episode 初始主动能量预算 | J |
| UAV 平台 | $E_i^{\mathrm{res}}(t)$ | 剩余主动能量 | J |
| UAV 平台 | $\kappa_i$ | CPU 动态能耗系数 | 由 $E=\kappa\Delta t f^3$ 定义 |
| 任务与队列 | $\tau_n$ | 任务 $n$ 的完整记录 | 记录元组 |
| 任务与队列 | $D_n,C_n$ | 任务输入数据量和计算量 | bit，cycle |
| 任务与队列 | $D_n^{\mathrm{rem}},C_n^{\mathrm{rem}}$ | 剩余待传输 bit 和待执行 cycle | bit，cycle |
| 任务与队列 | $t_n^{\mathrm{arr}},t_n^{\mathrm{ddl}}$ | 到达时隙和截止时隙 | slot |
| 任务与队列 | $d_n^{\mathrm{dst}}$ | 已锁定目的 UAV | $\mathcal U\cup\{\varnothing\}$ |
| 任务与队列 | $s_n(t)$ | 任务生命周期状态 | 见 2.5 |
| 任务与队列 | $Q_i^{\mathrm{unb}}$ | UAV $i$ 的未绑定任务队列 | 任务列表 |
| 任务与队列 | $Q_i^{\mathrm{loc}}$ | UAV $i$ 的本地待计算任务队列 | 任务列表 |
| 任务与队列 | $Q_{i\rightarrow j}^{\mathrm{tx}}$ | 从 $i$ 到 $j$ 的已绑定传输队列 | 任务列表 |
| 任务与队列 | $Q_{i\rightarrow j}^{\mathrm{cpu}}$ | 来源为 $i$、在 $j$ 上执行的 CPU 队列 | 任务列表 |
| 任务与队列 | $\operatorname{slack}_n(t)$ | 任务截止期余量 | slot |
| 无线信道 | $f_c$ | 载频 | Hz |
| 无线信道 | $\mathcal B$ | 建筑物集合 | 三维柱体集合 |
| 无线信道 | $I_{ij}^{\mathrm{bld}}(t)$ | 几何遮挡指示量 | $\{0,1\}$ |
| 无线信道 | $PL_{ij}(t)$ | $i\rightarrow j$ 路径损耗 | dB |
| 无线信道 | $X_{ij}(t)$ | 相关阴影项 | dB |
| 无线信道 | $h_{ij,r}(t)$ | 资源单元 $r$ 上的复信道增益 | 复数 |
| 无线信道 | $\hat h_{ij,r}(t)$ | actor 可用的 CSI 估计 | 复数 |
| 无线信道 | $a_{ij}^{\mathrm{CSI}}(t)$ | CSI AoI | slot |
| 无线信道 | $\mathrm{SINR}_{ij,r}(t)$ | 资源单元级信干噪比 | 线性值 |
| 无线信道 | $R_{ij}^{\mathrm{eff}}(t)$ | 链路有效速率 | bit/s |
| 资源与能量 | $R$ | 等效资源单元总数 | 正整数 |
| 资源与能量 | $G$ | 固定资源组数 | $G=5$ |
| 资源与能量 | $\mathcal G_g$ | 第 $g$ 个固定资源组 | 等效资源单元集合 |
| 资源与能量 | $x_{ij,r}(t)$ | 链路对等效 RB 的占用指示 | $\{0,1\}$ |
| 资源与能量 | $p_{ij,r}(t)$ | 链路在资源单元上的功率 | W |
| 资源与能量 | $E_i^{\mathrm{tx}}(t)$ | 主动通信能耗 | J |
| 资源与能量 | $E_i^{\mathrm{cpu}}(t)$ | CPU 动态能耗 | J |
| 资源与能量 | $E_i^{\mathrm{act}}(t)$ | 主动通信与计算总能耗 | J |
| 动作接口 | $a_i(t)$ | UAV $i$ 的联合动作 | 七元组 |
| 动作接口 | $a_i^{\mathrm{route}}(t)$ | 新任务路由分支 | 离散候选集合 |
| 动作接口 | $a_i^{\mathrm{tx}}(t)$ | 已有传输队列选择分支 | 离散候选集合 |
| 动作接口 | $a_i^{\mathrm{rg}}(t)$ | 固定资源组分支 | $\{\mathrm{idle},1,\ldots,5\}$ |
| 动作接口 | $a_i^{\mathrm{width}}(t)$ | 相邻资源组宽度分支 | $\{1,2\}$ |
| 动作接口 | $a_i^{\mathrm{pow}}(t)$ | 通信功率档位分支 | $\{0,0.25,0.5,1\}$ |
| 动作接口 | $a_i^{\mathrm{cpuq}}(t)$ | CPU 队列分支 | 本地、远程或 idle |
| 动作接口 | $a_i^{\mathrm{cpuf}}(t)$ | CPU 频率档位分支 | $\{0,0.25,0.5,1\}$ |
| Dec-POMDP | $s_t$ | 全局系统状态 | 状态空间 $\mathcal S$ |
| Dec-POMDP | $o_i(t)$ | UAV $i$ 的局部观测 | 观测空间 $\mathcal O_i$ |
| Dec-POMDP | $\mathbf a_t$ | 联合动作 | $\prod_i\mathcal A_i$ |
| Dec-POMDP | $P$ | 状态转移核 | $\mathcal S\times\mathcal A\rightarrow\mathcal P(\mathcal S)$ |
| Dec-POMDP | $r_t$ | 团队即时奖励 | 实数 |
| Dec-POMDP | $\gamma$ | 折扣因子 | $(0,1)$ |

​      默认参数用于编码初始接口和 Gate P 校准前的可复现实验起点。它们不等同于已完成的物理层标定结果。

| 参数 | 编码初始值 | 单位 | 状态 |
|---|---:|---|---|
| $N$ | $6$ | 架 | 敏感性参数，主规模集合为 $\{4,6,8\}$ |
| 区域边长 | $1000\times1000$ | m | 敏感性参数 |
| $H$ | $80$ | m | 已冻结 |
| $\Delta t$ | $20$ | ms | 已冻结 |
| $T$ | $500$ | slot | 敏感性参数 |
| $f_c$ | $3.5$ | GHz | Gate P 待校准 |
| 总带宽 | $20$ | MHz | Gate P 待校准 |
| $R$ | $20$ | 等效资源单元 | Gate P 待校准 |
| $G$ | $5$ | 组 | 已冻结 |
| $P_{\mathrm{ref}}$ | $1$ | W | Gate P 待校准 |
| $f_{\mathrm{ref}}$ | $2.0\times10^9$ | cycle/s | Gate P 待校准 |
| $\kappa_{\mathrm{ref}}$ | $1.0\times10^{-28}$ | 由能耗式确定 | Gate P 待校准 |
| $E_{\mathrm{ref}}^0$ | $30$ | J | Gate P 待校准 |
| $R_{\mathrm{ref}}$ | $20$ | Mbit/s | Gate P 待校准 |
| 天线增益 | $2$ | dBi | Gate P 待校准 |
| 噪声系数 | $7$ | dB | Gate P 待校准 |
| $\gamma_{\min}$ | $-3$ | dB | 敏感性参数 |
| $\sigma_s$ | $4$ | dB | 敏感性参数 |
| $d_{\mathrm{corr}}$ | $50$ | m | 敏感性参数 |
| $R_{\mathrm{cand}}$ | $500$ | m | Gate P 待校准 |
| $a^{\mathrm{CSI}}$ | $0$ | slot | 敏感性参数，主场景可设为 $\{0,3,5\}$ |

​      “已冻结”表示本章不得改变其语义；“Gate P 待校准”表示必须通过链路预算、deadline 和能量 pilot 后再固化数值；“敏感性参数”表示在主模型不变的前提下用于机制或鲁棒性分析。

## 2.3 UAV平台与节点功能模型

### 2.3.1 UAV空间状态

UAV 采用固定高度的三维位置表示。令 $\mathbf p_i(t)=[x_i(t),y_i(t)]^{\mathrm T}$ 为水平位置，则完整位置定义为：

$$
\mathbf q_i(t)
=
\left[
x_i(t),
y_i(t),
H
\right]^{\mathrm T}.
$$

其中 $x_i(t)$ 和 $y_i(t)$ 的单位为 m，$H$ 为固定高度，$\mathbf q_i(t)$ 用于后续距离、建筑物几何和信道评价。高度不随时隙变化，因此移动模块只更新水平位置。

对应的三维速度为：

$$
\mathbf v_i(t)
=
\left[
v_i^x(t),
v_i^y(t),
0
\right]^{\mathrm T}.
$$

其中 $v_i^x(t)$ 和 $v_i^y(t)$ 的单位为 m/s，竖直速度恒为零。该速度由外生移动模型产生，不是 actor 的动作分支，也不参与轨迹优化。

任意两架 UAV 之间的三维欧氏距离定义为：

$$
d_{ij}(t)
=
\left\|
\mathbf q_i(t)-\mathbf q_j(t)
\right\|_2.
$$

该距离统一用于候选邻居筛选、建筑物路径几何和链路大尺度评价。因而后续通信模块不将水平距离替代为完整的三维距离。

### 2.3.2 机载计算、通信、能量与队列模块

每架 UAV 由四个相互关联但功能可区分的模块组成：机载 CPU 模块、U2U 无线通信模块、主动通信和计算能量模块，以及任务与队列管理模块。CPU 模块服务本地队列或远程来源队列；无线模块服务已绑定的单跳传输链路；能量模块扣除通信发射和 CPU 动态运行产生的主动能耗；队列管理模块负责任务状态、目的地锁定、EDF 排序和槽边界更新。

UAV 的资源能力向量定义为：

$$
\boldsymbol{\chi}_i^{\mathrm{res}}
=
\left[
f_i^{\max},
P_i^{\max},
E_i^0,
\kappa_i
\right].
$$

其中，$f_i^{\max}$ 的单位是 cycle/s，表示 CPU 的最大等效处理频率；$P_i^{\max}$ 的单位是 W，表示无线发射功率上限；$E_i^0$ 的单位是 J，表示一个 episode 可用于主动通信和计算的初始能量预算；$\kappa_i$ 是 CPU 动态能耗系数，使得频率越高时单位时间能耗按三次方增长。该向量描述 UAV 本身的资源异构性，不包含业务到达率。

### 2.3.3 UAV节点角色与并行服务能力

同一架 UAV 的角色由当前任务和队列状态动态决定，而不是由固定类型标签预先决定。它可以同时承担任务源节点、U2U 发送节点、U2U 接收节点、本地计算节点和远程协同计算节点。CPU 模块与无线模块允许并行工作，因此一架 UAV 可以在同一时隙执行 CPU 服务并参与一条无线传输链路。

无线部分采用半双工约束。同一 UAV 在同一时隙不能同时发送和接收；联合执行器负责在所有 UAV 的候选传输边之间解析冲突。每个发送 UAV 每槽至多服务一个目的传输链路，每个计算 UAV 每槽至多选择一个 CPU 队列。一个链路服务可以按照 EDF 顺序连续扣除同一传输队列中的多个小任务，但不能在同一槽切换到另一个目的链路。

### 2.3.4 UAV动态节点状态

为了描述环境内部的节点状态，定义：

$$
\mathbf z_i^{\mathrm{uav}}(t)
=
\left[
\mathbf q_i(t),
\mathbf v_i(t),
E_i^{\mathrm{res}}(t),
\boldsymbol{\chi}_i^{\mathrm{res}},
\hat{\lambda}_i(t),
\mathbf q_i^{\mathrm{queue}}(t)
\right].
$$

其中 $\mathbf q_i^{\mathrm{queue}}(t)$ 是队列摘要向量，至少包括未绑定任务数、本地任务数、传输任务数、CPU 任务数、队首剩余 bit、队首剩余 cycle 和队首 slack。$\hat{\lambda}_i(t)$ 是由历史到达记录得到的估计量，而不是真实业务参数。系统级节点状态还可以保留真实建筑物、真实信道、完整任务记录和队列真值，但这些量不自动进入 actor 的局部观测。

### 2.3.5 UAV模型边界

本模型不建模六自由度飞行动力学、旋翼推力、姿态、飞控回路或轨迹优化，也不统计推进能耗。所有比较方法使用同一外生移动轨迹和同一轨迹随机数，因此性能差异不来自轨迹控制。主模型只统计卸载和资源编排直接产生的主动通信能耗与 CPU 动态能耗；不能将该能耗报告为完整 UAV 总能耗。

## 2.4 UAV资源异构与业务负载模型

资源异构和业务负载异构分开定义。资源异构来自硬件能力和能量预算，业务负载异构来自外部任务到达过程。业务负载向量写为：

$$
\boldsymbol{\chi}_i^{\mathrm{traffic}}
=
[\lambda_i].
$$

其中 $\lambda_i$ 表示 UAV $i$ 的任务到达概率或强度，只用于环境生成。若每个 UAV 每槽最多产生一个任务，则可采用：

$$
A_i(t)\sim\operatorname{Bernoulli}(\lambda_i).
$$

这里 $A_i(t)\in\{0,1\}$ 是槽末生成任务的到达指示量。负载异构实验改变各 UAV 的 $\lambda_i$ 离散程度，同时保持全局平均到达率的设定不变。

actor 不能读取真实 $\lambda_i$。执行期可用的负载线索是历史窗口内的经验估计：

$$
\hat{\lambda}_i(t)
=
\frac{1}{W_\lambda}
\sum_{\tau=t-W_\lambda+1}^{t}
A_i(\tau).
$$

其中 $W_\lambda$ 是历史窗口长度，单位为 slot。槽初不足一个完整窗口时采用已观测样本数进行边界处理，并附带可用性标志；该处理不改变真实业务过程。

硬件参数采用可解释 profile 加小扰动生成，而不是把所有资源参数完全独立采样。对 profile $c(i)$，资源向量可表示为：

$$
\boldsymbol{\chi}_i^{\mathrm{res}}
=
\boldsymbol{\chi}_{c(i)}^{\mathrm{profile}}
\odot
\left(\mathbf 1+\boldsymbol{\delta}_i\right),
\qquad
\delta_{i,m}\in[-0.05,0.05].
$$

其中 $\odot$ 表示逐元素乘法，$\boldsymbol{\delta}_i$ 是由独立随机种子生成的可复现小扰动。五类硬件 profile 的相对参数如下：

| Profile | $f_i^{\max}/f_{\mathrm{ref}}$ | $P_i^{\max}/P_{\mathrm{ref}}$ | $E_i^0/E_{\mathrm{ref}}^0$ | $\kappa_i/\kappa_{\mathrm{ref}}$ | 物理含义 |
|---|---:|---:|---:|---:|---|
| Resource-poor | $0.7$ | $0.8$ | $0.8$ | $1.20$ | 计算、通信和能效均较弱 |
| Balanced | $1.0$ | $1.0$ | $1.0$ | $1.00$ | 参考 profile |
| Compute-rich | $1.5$ | $1.0$ | $1.1$ | $0.90$ | CPU 能力较强且相对节能 |
| Energy-limited | $1.0$ | $0.8$ | $0.5$ | $1.10$ | 能量预算较紧 |
| Communication-rich | $1.0$ | $1.5$ | $1.1$ | $1.00$ | 通信发射能力较强 |

主场景使用上述五类 profile，并分别设置 Homogeneous、Mild-Heterogeneous 和 Strong-Heterogeneous 三档资源/负载组合。profile 只描述资源结构，不能代替 $\lambda_i$ 的业务配置。

## 2.5 任务、生命周期与队列模型

每个任务由唯一标识、源 UAV、输入数据量、计算量、到达时隙、截止时隙、剩余工作量、目的地和生命周期状态构成。任务记录定义为：

$$
\tau_n
=
\left(
id_n,
i_n,
D_n,
C_n,
t_n^{\mathrm{arr}},
t_n^{\mathrm{ddl}},
D_n^{\mathrm{rem}},
C_n^{\mathrm{rem}},
d_n^{\mathrm{dst}},
s_n
\right).
$$

其中 $D_n$ 的单位为 bit，$C_n$ 的单位为 cycle，$D_n^{\mathrm{rem}}$ 和 $C_n^{\mathrm{rem}}$ 分别表示未完成的传输和计算工作量；$d_n^{\mathrm{dst}}=\varnothing$ 表示尚未锁定目的 UAV。任务生成时令 $D_n^{\mathrm{rem}}=D_n$、$C_n^{\mathrm{rem}}=C_n$，随后仅通过实际服务量递减。

输入数据量和单位 bit 计算量使用方案 2 的初始化范围：

$$
D_n\sim U(0.25,1.5)\ \mathrm{Mbit}.
$$

该式给出任务输入规模的编码初始分布，正式实验前仍需经过 Gate P 的非退化性检查。计算量与数据量的比值定义为：

$$
\frac{C_n}{D_n}
\sim U(300,1000)\ \mathrm{cycle/bit}.
$$

deadline 根据参考传输时间和参考计算时间生成。令 $\eta_n\sim U(1.5,3.0)$ 为服务裕量因子，则：

$$
t_n^{\mathrm{ddl}}
=
t_n^{\mathrm{arr}}
+
\left\lceil
\frac{
\eta_n
\left(
D_n/R_{\mathrm{ref}}+C_n/f_{\mathrm{ref}}
\right)
}{
\Delta t
}
\right\rceil.
$$

其中 $R_{\mathrm{ref}}$ 的单位为 bit/s，$f_{\mathrm{ref}}$ 的单位为 cycle/s，$\Delta t$ 的单位为 s。分布表中的 Mbit 在代入传输时间前统一换算为 bit。生成后的 deadline 在实现中限制为 10 至 150 个时隙，以避免出现不受控的极端期限。

任务状态集合为：

$$
\mathcal S_{\mathrm{task}}
=
\{
\mathrm{unbound},
\mathrm{local},
\mathrm{tx},
\mathrm{cpu},
\mathrm{done},
\mathrm{expired}
\}.
$$

其中，unbound 表示任务已经到达但尚未绑定本地或远程目的地；local 表示任务已绑定源 UAV 的本地 CPU 队列；tx 表示任务已绑定远程目的 UAV 并处于传输队列；cpu 表示任务正在本地或远程 CPU 队列中接受计算服务；done 表示在截止期判定前完成；expired 表示在截止时隙结束时仍未完成或完成时间超过截止时隙。

合法生命周期转移由以下集合概括：

$$
\mathcal T_{\mathrm{task}}
=
\{
\mathrm{unbound}\rightarrow\mathrm{local},
\mathrm{unbound}\rightarrow\mathrm{tx},
\mathrm{unbound}\rightarrow\mathrm{expired},
\mathrm{local}\rightarrow\mathrm{cpu},
\mathrm{local}\rightarrow\mathrm{expired},
\mathrm{tx}\rightarrow\mathrm{cpu},
\mathrm{tx}\rightarrow\mathrm{expired},
\mathrm{cpu}\rightarrow\mathrm{done},
\mathrm{cpu}\rightarrow\mathrm{expired}
\}.
$$

该集合不包含已发送任务重新绑定到其他目的 UAV 的转移，也不包含多跳转发。任务在首 bit 发送前仍可由 route 重新决策；一旦进入传输服务并发送首 bit，目的地锁定。新生成任务、本槽路由任务和本槽传输完成任务均在槽末进入相应队列，因此不可能在一个时隙内完成路由、传输和计算。

若任务在时隙 $t_n^{\mathrm{cmp}}$ 完成，其端到端时延定义为：

$$
T_n^{\mathrm{E2E}}
=
\left(
t_n^{\mathrm{cmp}}-t_n^{\mathrm{arr}}+1
\right)\Delta t.
$$

若任务在截止时隙结束时仍未完成，或其完成时隙晚于截止时隙，则计为过期。该判定可写为：

$$
\operatorname{expired}_n
=
\mathbb 1
\left\{
t_n^{\mathrm{cmp}}>t_n^{\mathrm{ddl}}
\ \text{或任务在 }t_n^{\mathrm{ddl}}\text{ 结束时仍未完成}
\right\}.
$$

该定义区分按时完成、延迟完成和未完成过期三种结算情形，并与后续按时完成率和过期率保持一致。

## 2.6 任务队列与EDF调度

环境显式保存四类队列。未绑定队列为：

$$
Q_i^{\mathrm{unb}}(t)
=
\left(
\tau_n:\ i_n=i,\ s_n(t)=\mathrm{unbound}
\right).
$$

本地队列为：

$$
Q_i^{\mathrm{loc}}(t)
=
\left(
\tau_n:\ i_n=i,\ s_n(t)\in\{\mathrm{local},\mathrm{cpu}\},\ d_n^{\mathrm{dst}}=\varnothing
\right).
$$

从源 UAV $i$ 到目的 UAV $j$ 的传输队列为：

$$
Q_{i\rightarrow j}^{\mathrm{tx}}(t)
=
\left(
\tau_n:\ i_n=i,\ d_n^{\mathrm{dst}}=j,\ s_n(t)=\mathrm{tx}
\right).
$$

来源为 $i$、在目的 UAV $j$ 上执行的远程 CPU 队列为：

$$
Q_{i\rightarrow j}^{\mathrm{cpu}}(t)
=
\left(
\tau_n:\ i_n=i,\ d_n^{\mathrm{dst}}=j,\ s_n(t)=\mathrm{cpu}
\right).
$$

每类队列均按队首 deadline 优先的 EDF 顺序排列。任务 $n$ 在槽初的 deadline 余量定义为：

$$
\operatorname{slack}_n(t)
=
t_n^{\mathrm{ddl}}-t.
$$

余量越小表示任务越紧迫。相同 slack 时采用固定任务标识作为可复现的 tie-break，不把随机打散引入队列语义。

route 只处理槽初 $Q_i^{\mathrm{unb}}(t)$ 中的最高优先级任务，并在 local、defer 或当前候选远程 UAV 中选择；它负责绑定目的地，不直接产生本槽传输服务。tx_select 只从槽初已有的 $Q_{i\rightarrow j}^{\mathrm{tx}}(t)$ 中选择一个非空传输队列，并负责本槽服务预算分配。两者即使在同一槽分别输出，也不能使新路由任务获得本槽传输服务。

## 2.7 外生移动与动态U2U图模型

### 2.7.1 Gauss-Markov水平移动

移动轨迹由环境外生生成。令 $\bar{\mathbf v}_i$ 为平均水平速度，$\boldsymbol{\varepsilon}_i(t)$ 为零均值扰动，水平速度采用：

$$
\mathbf v_i^{\mathrm{hor}}(t+1)
=
\alpha\mathbf v_i^{\mathrm{hor}}(t)
+
(1-\alpha)\bar{\mathbf v}_i
+
\sqrt{1-\alpha^2}\,
\boldsymbol{\varepsilon}_i(t).
$$

其中 $\alpha=0.85$ 为编码初始相关系数，$\mathbf v_i^{\mathrm{hor}}(t)=[v_i^x(t),v_i^y(t)]^{\mathrm T}$。边界处理采用反射或速度反向，最小安全距离由轨迹生成器保证。

令 $\mathcal A$ 为水平移动区域，位置更新使用区域投影：

$$
\mathbf p_i(t+1)
=
\Pi_{\mathcal A}
\left(
\mathbf p_i(t)+\Delta t\,\mathbf v_i^{\mathrm{hor}}(t+1)
\right).
$$

投影算子 $\Pi_{\mathcal A}$ 保证位置留在区域内。更新后的三维位置重新写为：

$$
\mathbf q_i(t+1)
=
\left[
\mathbf p_i(t+1)^{\mathrm T},H
\right]^{\mathrm T}.
$$

上述移动不属于动作空间，所有算法共享同一外生轨迹和轨迹随机数。

### 2.7.2 新任务候选邻居与服务边

新任务候选邻居由三维距离阈值定义：

$$
\mathcal N_i(t)
=
\left\{
j\in\mathcal U\setminus\{i\}:
d_{ij}(t)\le R_{\mathrm{cand}}
\right\}.
$$

其中 $R_{\mathrm{cand}}$ 是候选通信半径，初始值为 500 m，正式实验前结合链路预算校准。可选的距离滞回只用于减少边界抖动，不改变单跳语义。

新任务边只用于决定未绑定任务当前可选的目的 UAV。服务边则保存已经锁定的 $i\rightarrow j$ 关系；即使锁定任务因移动暂时超出新任务候选范围，仍保留其传输队列和剩余 bit，等待链路恢复或按过期规则结算。outage 不删除已绑定服务边。所有邻居、建筑物路径和信道计算均以 $\mathbf q_i(t)$ 的三维位置为输入。

## 2.8 建模的U2U物理层评价

### 2.8.1 建筑物几何遮挡

建筑物表示为带高度的二维多边形柱体。令 $\mathcal B_b$ 为第 $b$ 个建筑物的三维实体，$\zeta\in[0,1]$ 为线段参数，则 UAV $i$ 到 UAV $j$ 的传播线段为：

$$
\boldsymbol{\ell}_{ij}(\zeta,t)
=
(1-\zeta)\mathbf q_i(t)+\zeta\mathbf q_j(t),
\qquad
\zeta\in[0,1].
$$

只要存在至少一个建筑物与该闭线段相交，就判定链路发生几何遮挡：

$$
I_{ij}^{\mathrm{bld}}(t)
=
\mathbb 1
\left\{
\exists b\in\mathcal B:
\boldsymbol{\ell}_{ij}([0,1],t)
\cap\mathcal B_b
\ne\varnothing
\right\}.
$$

这里显式使用存在量词，以保证遮挡判定是“任一建筑物相交即遮挡”，而不是对建筑物进行平均或随机抽样。该指示量是环境内部真实状态，actor 不直接读取其真值。

### 2.8.2 路径损耗和建筑物附加损耗

自由空间基准路径损耗可写为：

$$
PL_{\mathrm{FS}}(d,f_c)
=
20\log_{10}
\left(
\frac{4\pi f_c d}{c_0}
\right),
$$

其中 $c_0$ 为光速，$d$ 使用三维距离，$f_c$ 使用载频。建筑物附加损耗按遮挡指示量加入：

$$
L_{\mathrm{bld}}(t)
=
L_{\mathrm{B}} I_{ij}^{\mathrm{bld}}(t).
$$

主模型的编码初始值为 LoS 时附加损耗为 0 dB、发生遮挡时 $L_{\mathrm{B}}=15$ dB，并将 10、20、30 dB 作为敏感性参数。含相关阴影的总路径损耗为：

$$
PL_{ij}(t)
=
PL_{\mathrm{FS}}\left(d_{ij}(t),f_c\right)
+
L_{\mathrm{bld}}(t)
+
X_{ij}(t).
$$

该式只提供轻量链路评价所需的大尺度项，不等同于实测路径损耗或实时射线追踪输出。

### 2.8.3 空间相关阴影

空间相关性通过相邻时隙内两端 UAV 的水平位移建立。位移尺度为：

$$
\Delta s_{ij}(t)
=
\frac{
\left\|\mathbf p_i(t)-\mathbf p_i(t-1)\right\|_2
+
\left\|\mathbf p_j(t)-\mathbf p_j(t-1)\right\|_2
}{2}.
$$

设 $d_{\mathrm{corr}}$ 为相关距离，则相关系数为：

$$
\rho_{ij}(t)
=
\exp
\left(
-\frac{\Delta s_{ij}(t)}{d_{\mathrm{corr}}}
\right).
$$

阴影项按槽间递推：

$$
X_{ij}(t)
=
\rho_{ij}(t)X_{ij}(t-1)
+
\sqrt{1-\rho_{ij}^2(t)}\,
\sigma_s\epsilon_{ij}(t).
$$

其中 $\sigma_s$ 的单位为 dB，$\epsilon_{ij}(t)$ 是标准化零均值扰动。该模型保持短期移动连续性，并使遮挡和链路质量变化不被人为地设置为完全独立。

### 2.8.4 Rician/Rayleigh槽级衰落

主模型按遮挡状态选择槽级小尺度衰落类型。LoS 链路使用 Rician 衰落，NLoS 链路使用 Rayleigh 衰落。资源单元 $r$ 上的复信道功率增益写为：

$$
\left|h_{ij,r}(t)\right|^2
=
G_tG_r
10^{-PL_{ij}(t)/10}
\left|g_{ij,r}(t)\right|^2.
$$

其中 $G_t$ 和 $G_r$ 为发送与接收增益的线性值，$g_{ij,r}(t)$ 是单位化小尺度衰落项。当 $I_{ij}^{\mathrm{bld}}(t)=0$ 时使用 Rician 参数 $K=6$ dB；当 $I_{ij}^{\mathrm{bld}}(t)=1$ 时使用 Rayleigh 特例 $K=0$。主训练环境使用槽级平均功率增益，以避免将不必要的瞬时波形细节引入决策接口。

### 2.8.5 不完美CSI和CSI AoI

actor 使用陈旧且带估计误差的信道信息。令 $a_{ij}^{\mathrm{CSI}}(t)$ 为 CSI AoI，$\xi_{ij,r}(t)$ 为正的乘性估计误差，则：

$$
\hat h_{ij,r}(t)
=
h_{ij,r}
\left(
t-a_{ij}^{\mathrm{CSI}}(t)
\right)
\xi_{ij,r}(t).
$$

误差幅度以 dB 域建模为：

$$
10\log_{10}\xi_{ij,r}(t)
\sim
\mathcal N
\left(
0,\sigma_{\mathrm{CSI}}^2
\right).
$$

actor 可使用 $\hat h_{ij,r}(t)$ 派生的估计 SINR、上一槽有效速率、已调度历史 outage 率、CSI AoI 和特征可用性 mask；真实 $h_{ij,r}(t)$ 只用于环境转移以及训练期允许的集中状态。CSI AoI 的更新是环境事件，不是 actor 可直接控制的动作。

### 2.8.6 模型边界和第三UAV软遮挡扩展

主模型只包含两端 UAV 与建筑物几何遮挡，不把第三 UAV 对 Fresnel 区域的软遮挡加入训练环境。只有主模型通过 Gate P、Gate 0 及后续机制闸门后，才可将第三 UAV 软遮挡作为几何启发式消融或离线高保真回放扩展。该扩展只能用于敏感性和趋势验证，不能被表述为实测遮挡模型，也不能反向改变主模型的 actor 信息权限。

## 2.9 固定资源组、SINR、有效速率和outage

系统将 $R$ 个等效资源单元预先划分为 $G=5$ 个连续、互不重叠的固定资源组：

$$
\mathcal G
=
\left\{
\mathcal G_1,\mathcal G_2,\mathcal G_3,\mathcal G_4,\mathcal G_5
\right\},
\qquad
\mathcal G_g\cap\mathcal G_{g'}=\varnothing\ (g\ne g'),
\qquad
|\mathcal G_g|=\frac{R}{G}.
$$

主场景 $R=20$ 时每组含 4 个等效资源单元；$R=10$ 和 $R=40$ 时分别为 2 个和 8 个。资源选择以固定组为粒度，不暴露单个等效资源单元的自由索引，也不使用连续带宽份额。

若动作选择组索引 $g_i(t)$ 和宽度 $w_i(t)\in\{1,2\}$，则所用组集合为：

$$
\mathcal W_i(t)
=
\{g_i(t),g_i(t)+1,\ldots,g_i(t)+w_i(t)-1\},
\qquad
\mathcal S_i(t)
=
\bigcup_{g\in\mathcal W_i(t)}\mathcal G_g.
$$

当 $g_i(t)=5$ 且 $w_i(t)=2$，该组合由动作 mask 禁止；当通信分支为 idle 时令 $\mathcal S_i(t)=\varnothing$。实际发射功率由离散功率分支确定：

$$
p_i(t)
\in
\{0,0.25,0.5,1.0\}P_i^{\max}.
$$

因此资源宽度只表示一个组或两个相邻组，不改变固定资源组边界；功率分支也不会产生离散集合之外的连续值。

若链路 $i\rightarrow j$ 被联合执行器接受，且资源单元 $r$ 属于 $\mathcal S_i(t)$，则占用指示为：

$$
x_{ij,r}(t)
=
\mathbb 1\{r\in\mathcal S_i(t)\};
$$

否则 $x_{ij,r}(t)=0$。链路在已分配资源上的总功率按资源单元数平均分配：

$$
p_{ij,r}(t)
=
\frac{p_i(t)}
{\sum_{r'=1}^{R}x_{ij,r'}(t)}
\quad
\text{当 }\sum_{r'=1}^{R}x_{ij,r'}(t)>0.
$$

当分母为零时链路未获得资源，约定所有 $p_{ij,r}(t)=0$，且不计算该链路的有效服务。

在接收噪声功率为 $N_0B_{\mathrm{RU}}F$ 时，资源单元级 SINR 定义为：

$$
\mathrm{SINR}_{ij,r}(t)
=
\frac{
x_{ij,r}(t)p_{ij,r}(t)|h_{ij,r}(t)|^2
}{
N_0B_{\mathrm{RU}}F
+
\sum_{(k,l)\ne(i,j)}
x_{kl,r}(t)p_{kl,r}(t)|h_{kj,r}(t)|^2
}.
$$

其中 $N_0$ 是噪声谱密度，$B_{\mathrm{RU}}$ 是单个等效资源单元带宽，$F$ 是由噪声系数换算得到的线性因子。分母中的求和仅包含同一资源单元上被联合执行器接受的其他传输边。

设 $\gamma_{\min}$ 为最小可用 SINR 阈值，则链路有效速率为：

$$
R_{ij}^{\mathrm{eff}}(t)
=
\sum_{r=1}^{R}
B_{\mathrm{RU}}
\log_2
\left(
1+\mathrm{SINR}_{ij,r}(t)
\right)
\mathbb 1
\left\{
\mathrm{SINR}_{ij,r}(t)\ge\gamma_{\min}
\right\}.
$$

$\gamma_{\min}$ 的初始值为 $-3$ dB，并可在 $-6$、$0$ 和 $3$ dB 之间做敏感性分析。有效速率是环境真实服务计算的输入，不应与 actor 只能获得的估计 SINR 混淆。

outage 仅在实际调度且获得非零资源的链路上统计。其定义为：

$$
O_{ij}(t)
=
\begin{cases}
1,
&
a_i^{\mathrm{tx}}(t)=j,\
|\mathcal S_i(t)|>0,\
R_{ij}^{\mathrm{eff}}(t)=0,
\\
0,
&
a_i^{\mathrm{tx}}(t)=j,\
|\mathcal S_i(t)|>0,\
R_{ij}^{\mathrm{eff}}(t)>0,
\\
\mathrm{NA},
&
\text{本槽未调度链路 }i\rightarrow j.
\end{cases}
$$

主动暂停、没有传输任务或没有分配资源均不计为信道 outage。对历史上实际被调度的时隙集合 $\mathcal H_{ij}(t)$，outage 率估计为：

$$
\hat q_{ij}^{\mathrm{out}}(t)
=
\frac{
\sum_{\tau\in\mathcal H_{ij}(t)}
\mathbb 1\{O_{ij}(\tau)=1\}
}{
|\mathcal H_{ij}(t)|
}.
$$

当 $\mathcal H_{ij}(t)$ 为空时，使用缺省值并附带可用性 mask，不把 NA 当作失败事件。等效 RB 级失效率另行记录为：

$$
O_{ij}^{\mathrm{RB}}(t)
=
\frac{
\sum_{r:x_{ij,r}(t)=1}
\mathbb 1\{\mathrm{SINR}_{ij,r}(t)<\gamma_{\min}\}
}{
\sum_{r=1}^{R}x_{ij,r}(t)
},
$$

且仅当分母大于零时计算。该口径使链路 outage、等效 RB 级失效率和未调度状态保持可区分。

## 2.10 跨时隙传输服务模型

tx_select 的候选集合只来自槽初已经存在的非空传输队列：

$$
a_i^{\mathrm{tx}}(t)
\in
\{\mathrm{idle}\}
\cup
\left\{
j:
Q_{i\rightarrow j}^{\mathrm{tx}}(t)\ne\varnothing
\right\}.
$$

该动作只选择已有服务边的目的 UAV，不负责新任务目的地绑定。新任务的 route 输出即使在同一时隙存在，也只能在槽末进入本地或传输队列。

对被选择的链路，先聚合槽初队列中所有任务的剩余 bit：

$$
Q_{ij}^{\mathrm{bit}}(t)
=
\sum_{n\in Q_{i\rightarrow j}^{\mathrm{tx}}(t)}
D_n^{\mathrm{rem}}(t).
$$

根据有效速率得到本时隙链路服务预算：

$$
S_{ij}^{\mathrm{tx}}(t)
=
\min
\left\{
Q_{ij}^{\mathrm{bit}}(t),
R_{ij}^{\mathrm{eff}}(t)\Delta t
\right\}.
$$

服务预算沿 $Q_{i\rightarrow j}^{\mathrm{tx}}(t)$ 的 EDF 顺序逐任务扣除：先服务队首任务，扣除其剩余 bit 与当前预算的较小值；预算仍有剩余时继续处理下一任务；直到预算耗尽或队列为空。对任务 $n$，令 $b_n^{\mathrm{tx}}(t)$ 为本槽实际扣除的 bit，则更新关系为：

$$
D_n^{\mathrm{rem}}(t+1)
=
D_n^{\mathrm{rem}}(t)-b_n^{\mathrm{tx}}(t),
\qquad
0\le b_n^{\mathrm{tx}}(t)\le D_n^{\mathrm{rem}}(t).
$$

一条被选择的链路可以在同槽完成多个小任务，但不能同时服务另一个目的链路。传输完成的任务在槽末进入目的 UAV 的远程 CPU 队列，计算量仍为其完整 $C_n$；本槽新进入 CPU 队列的任务下一时隙才可计算。因此传输完成任务不能同槽计算。

## 2.11 计算与能耗模型

每个计算 UAV 每槽最多选择一个槽初非空 CPU 队列。CPU 队列动作候选集合为：

$$
a_j^{\mathrm{cpuq}}(t)
\in
\{\mathrm{idle},\mathrm{local}\}
\cup
\left\{
i:
Q_{i\rightarrow j}^{\mathrm{cpu}}(t)\ne\varnothing
\right\}.
$$

其中 local 表示 $Q_j^{\mathrm{loc}}(t)$，远程索引 $i$ 表示来源为 $i$ 的队列 $Q_{i\rightarrow j}^{\mathrm{cpu}}(t)$。CPU 频率采用离散档位：

$$
f_j(t)
\in
\{0,0.25,0.5,1.0\}f_j^{\max}.
$$

设被选 CPU 队列的 EDF 队首任务为 $n^\star$，其本槽服务量为：

$$
S_j^{\mathrm{cpu}}(t)
=
\min
\left\{
C_{n^\star}^{\mathrm{rem}}(t),
f_j(t)\Delta t
\right\}.
$$

第一版主模型每个 UAV 每槽只服务一个 CPU 队首任务，不把剩余 CPU 服务预算继续分配给第二个 CPU 队列。若服务量为 $c_j^{\mathrm{cpu}}(t)$，则队首剩余 cycle 更新为：

$$
C_{n^\star}^{\mathrm{rem}}(t+1)
=
C_{n^\star}^{\mathrm{rem}}(t)-c_j^{\mathrm{cpu}}(t),
\qquad
0\le c_j^{\mathrm{cpu}}(t)\le S_j^{\mathrm{cpu}}(t).
$$

主动通信能耗由实际接受的传输边、其资源占用和每资源单元功率决定：

$$
E_i^{\mathrm{tx}}(t)
=
\Delta t
\sum_{j\in\mathcal U}
\sum_{r=1}^{R}
x_{ij,r}(t)p_{ij,r}(t).
$$

CPU 动态能耗按实际使用频率计算：

$$
E_i^{\mathrm{cpu}}(t)
=
\kappa_i\Delta t\,f_i^3(t).
$$

当 $f_i(t)=0$ 时，CPU 动态能耗为零。UAV 的主动能耗和剩余能量更新为：

$$
E_i^{\mathrm{act}}(t)
=
E_i^{\mathrm{tx}}(t)+E_i^{\mathrm{cpu}}(t),
\qquad
E_i^{\mathrm{act}}(t)
\le
E_i^{\mathrm{res}}(t),
$$

$$
E_i^{\mathrm{res}}(t+1)
=
E_i^{\mathrm{res}}(t)-E_i^{\mathrm{act}}(t).
$$

上述约束只保证主动通信与计算的能量守恒，不包含推进、悬停、姿态或机载传感器能耗。初始能量、参考 CPU 和参考速率必须在 Gate P pilot 后校准，不能把未校准数值直接解释为实际硬件功耗。

若联合动作的能量不可行，执行器只在预定义离散档位中降级，降档序列为：

$$
1.0\rightarrow0.5\rightarrow0.25\rightarrow0.
$$

执行顺序是先屏蔽明显超过剩余能量的单分支档位，再对联合通信—CPU 组合重新计算能耗；若仍不可行，按固定优先规则降低通信功率或 CPU 频率；若所有非零组合均不可行，则相应分支执行 idle，并记录降档原因、次数以及执行前后的动作。该规则不把动作映射到离散档位集合之外的连续值。

## 2.12 七分支离散动作与硬约束

每个 UAV 的联合动作固定为七个离散分支：

$$
a_i(t)
=
\left(
a_i^{\mathrm{route}}(t),
a_i^{\mathrm{tx}}(t),
a_i^{\mathrm{rg}}(t),
a_i^{\mathrm{width}}(t),
a_i^{\mathrm{pow}}(t),
a_i^{\mathrm{cpuq}}(t),
a_i^{\mathrm{cpuf}}(t)
\right).
$$

七个分支的语义如下。route 只处理槽初未绑定 EDF 任务，候选为 local、defer 或 $\mathcal N_i(t)$ 中的一个远程目的地；tx 只选择槽初已有的非空传输队列；resource group 选择 idle 或五个固定资源组之一；resource width 选择一个或两个相邻资源组；power level 选择 $0$、$0.25P_i^{\max}$、$0.5P_i^{\max}$ 或 $P_i^{\max}$；CPU queue 选择 local、一个远程 CPU 队列或 idle；CPU frequency 选择 $0$、$0.25f_i^{\max}$、$0.5f_i^{\max}$ 或 $f_i^{\max}$。

动作 mask 在槽初根据可用队列和资源状态生成：

- 没有未绑定任务时，route 固定为 idle；
- 不在 $\mathcal N_i(t)$ 中的远程目的地不可选；
- 已发送首 bit 的任务不得重新路由；
- 没有传输队列时，tx、resource group、width 和 power 均固定为 idle；
- 第五组不能与宽度 2 组合；
- 没有本地或远程 CPU 队列时，CPU queue 和 CPU frequency 固定为 idle；
- 明显超过剩余能量的离散档位提前 mask；
- 联合半双工冲突不由单个 actor 的本地 mask 假定解决，而由联合执行器解析。

为明确联合解析的硬约束，令 $y_{ij}(t)\in\{0,1\}$ 表示传输边 $i\rightarrow j$ 是否被接受，令 $u_{jk}(t)\in\{0,1\}$ 表示 UAV $j$ 是否选择来源为 $k$ 的 CPU 队列，其中 $k=j$ 表示本地队列。则每个发送 UAV 和每个计算 UAV 分别满足：

$$
\sum_{j\in\mathcal U\setminus\{i\}}y_{ij}(t)\le1,
\qquad
\sum_{k\in\mathcal U}u_{ik}(t)\le1.
$$

半双工约束要求任意 UAV 不得同时作为发送端和接收端：

$$
\sum_{j\ne i}y_{ij}(t)
+
\sum_{k\ne i}y_{ki}(t)
\le1,
\qquad
\forall i\in\mathcal U.
$$

CPU 与无线通信模块被视为并行资源，因此半双工约束不限制 CPU 和无线动作之间的并行性。联合执行器收集所有非 idle 候选传输边，按 EDF slack、估计链路质量和固定 UAV 标识的可复现优先级排序，依次接受不产生半双工冲突的边；冲突边退化为通信 idle，并记录冲突与修正次数。

## 2.13 时隙事件顺序

每个时隙严格按以下顺序执行，以保持任务、bit、cycle 和能量的因果一致性：

1. 槽初读取已有任务、四类队列、候选邻居、估计 CSI 和延迟消息；
2. 根据槽初局部观测生成七分支离散动作；
3. 对一个未绑定 EDF 任务执行 route 决策；
4. 从槽初已有的非空传输队列中执行 tx_select；
5. 选择固定资源组、资源宽度和功率档位；
6. 选择一个槽初 CPU 队列和 CPU 频率档位；
7. 将所有候选动作送入联合 hard-feasible 执行器；
8. 根据真实槽级信道计算 SINR、有效速率和通信服务；
9. 执行本地或远程 CPU 服务；
10. 更新任务剩余 bit、剩余 cycle 和主动能量；
11. 结算完成任务、延迟完成任务和过期任务；
12. 将本槽 route 任务、本槽传输完成任务和新到达任务在槽末入队；
13. 更新外生移动、真实信道、CSI AoI 和消息 AoI，并进入下一槽。

该顺序带来三条不可绕过的可用性规则：新 route 任务不能在本槽由 tx_select 服务；本槽传输完成任务不能在本槽计算；槽末生成的新任务不能在本槽再次参与 route。因而任何算法都共享相同的服务边界，不能通过改变网络输出顺序获得额外的槽内服务。

## 2.14 Dec-POMDP、信息边界与优化问题

### 2.14.1 状态、局部观测与联合动作

全局状态包含位置、速度、真实信道、建筑物几何、真实业务参数、任务记录、队列、资源和能量等环境变量，可抽象为：

$$
s_t
=
\left(
\{\mathbf q_i(t),\mathbf v_i(t),\boldsymbol{\chi}_i^{\mathrm{res}},E_i^{\mathrm{res}}(t)\}_{i\in\mathcal U},
\mathcal T_t,
\mathcal Q_t,
\mathcal H_t,
\mathcal B,
\boldsymbol{\lambda}
\right).
$$

其中 $\mathcal T_t$ 是活动任务集合，$\mathcal Q_t$ 是四类队列集合，$\mathcal H_t$ 是真实信道与 CSI 历史，$\boldsymbol{\lambda}$ 是环境业务到达参数。该状态用于定义环境转移和集中训练期的全局信息，不表示每个 actor 都能读取所有分量。

UAV $i$ 的局部观测由状态和可用消息历史共同决定：

$$
o_i(t)
=
\mathcal O_i
\left(
s_t,
m_i(t)
\right).
$$

局部观测至少包括自身剩余能量比例、归一化资源能力、历史到达率估计、本地/传输/CPU 队列摘要、候选传输队列的任务数与剩余 bit、CPU 队列的任务数与剩余 cycle、队首 slack、上一槽动作和资源利用率；邻居特征包括相对位置、相对速度、广播剩余能量、广播资源能力、CPU 负载摘要和消息 AoI；边特征包括估计距离、估计 SINR、上一槽有效速率、仅基于已调度历史的 outage 率、CSI AoI、相对速度和特征可用性 mask。

actor 不得读取以下信息：

- 真实建筑物遮挡标签；
- 真实瞬时信道 $h_{ij,r}(t)$；
- 未来位置或未来移动轨迹；
- 真实业务到达参数 $\lambda_i$；
- 尚未生成的未来任务；
- 其他 UAV 的完整私有队列与完整任务记录。

集中训练期的 critic 或等价全局评价模块可以使用真实资源参数、所有任务和队列真值、真实信道与遮挡状态、真实业务到达参数以及建筑物和全局统计，但执行期 actor 的权限不因此扩大。

联合动作是各 UAV 局部动作的笛卡尔积：

$$
\mathbf a_t
=
\left(
a_1(t),\ldots,a_N(t)
\right)
\in
\mathcal A
=
\prod_{i\in\mathcal U}\mathcal A_i.
$$

动作 mask 和联合 hard-feasible 执行器共同把原始联合动作映射为实际可执行动作，但不改变七分支的语义。

### 2.14.2 状态转移、团队奖励与折扣目标

在外生任务到达、移动、信道演化和联合服务规则共同作用下，状态转移可写为：

$$
s_{t+1}
\sim
P
\left(
\cdot\mid s_t,\mathbf a_t
\right).
$$

转移核包含真实信道和遮挡对服务量的影响、传输与 CPU 队列更新、能量扣除、任务完成/过期结算、槽末入队、外生移动和 CSI AoI 更新。它不把 actor 的局部估计直接当作真实环境状态。

为了刻画截止期和资源代价，定义活动任务的紧迫工作量：

$$
W(t)
=
\sum_{n\in\mathcal T_t^{\mathrm{active}}}
\frac{
D_n^{\mathrm{rem}}(t)/R_{\mathrm{ref}}
+
C_n^{\mathrm{rem}}(t)/f_{\mathrm{ref}}
}{
\max\{1,t_n^{\mathrm{ddl}}-t\}
}.
$$

其中 $\mathcal T_t^{\mathrm{active}}$ 是尚未处于 done 或 expired 的任务集合。设 $N_{\mathrm{on}}(t)$ 为本槽按 deadline 完成的任务数，$N_{\mathrm{exp}}(t)$ 为本槽结算的过期任务数，$E^{\mathrm{act}}(t)$ 为全体 UAV 主动能耗，则团队即时奖励定义为：

$$
r_t
=
w_c\widetilde N_{\mathrm{on}}(t)
-
w_d\widetilde N_{\mathrm{exp}}(t)
-
w_q\widetilde W(t)
-
w_e\widetilde E^{\mathrm{act}}(t).
$$

波浪号表示使用预先固定的参考尺度归一化，不使用当前 batch 均值动态归一化。$w_c,w_d,w_q,w_e$ 的编码初始值为 $1.0,1.0,0.2,0.05$，仅作为调试起点，不作为已验证的最优权重。

Dec-POMDP 可用元组表示为：

$$
\mathcal M
=
\left(
\mathcal S,
\{\mathcal O_i\}_{i\in\mathcal U},
\{\mathcal A_i\}_{i\in\mathcal U},
P,
r,
\gamma
\right).
$$

其中 $\gamma\in(0,1)$ 为折扣因子。给定共享环境和各 actor 的局部策略集合 $\boldsymbol{\pi}$，优化目标写为：

$$
J(\boldsymbol{\pi})
=
\mathbb E_{\boldsymbol{\pi},P}
\left[
\sum_{t=0}^{T-1}
\gamma^t r_t
\right],
\qquad
\max_{\boldsymbol{\pi}}J(\boldsymbol{\pi}).
$$

该目标只定义受约束的团队决策问题，不推出全局最优、收敛保证或部署性能。具体学习器、策略网络和价值更新属于后续方法章节，本章不写入其更新公式。

### 2.14.3 约束集合与执行边界

联合决策必须满足队列存在性、固定资源组合法性、离散功率/频率档位、能量可行性、单目的链路服务、CPU 单队列服务和联合半双工约束。可以将本章的实际可行集合概括为：

$$
\mathcal A_t^{\mathrm{feas}}
=
\left\{
\mathbf a_t:
\begin{array}{l}
\text{队列和候选邻居存在性约束成立},\\
\text{固定资源组与相邻宽度约束成立},\\
E_i^{\mathrm{act}}(t)\le E_i^{\mathrm{res}}(t),\ \forall i,\\
\text{每个发送端至多一条服务边},\\
\text{半双工约束成立},\\
\text{每个 CPU 至多一个队列}
\end{array}
\right\}.
$$

动作 mask 处理可由局部信息提前识别的非法分支；联合执行器处理必须看到联合候选动作后才能判断的半双工和联合能量冲突。两者共同保证执行结果属于 hard-feasible 集合，但不把非法动作伪装成连续资源值。

## 2.15 假设、局限性与实现映射

### 2.15.1 主要建模假设

1. UAV 高度固定，水平移动由外生 Gauss-Markov 过程产生，所有算法使用相同移动轨迹。
2. 任务输入不可拆分到多个目的 UAV，不允许多跳转发；目的地在首 bit 发送后锁定。
3. 传输和 CPU 服务跨时隙持续，槽末入队的任务下一时隙才可服务。
4. 计算模块与无线模块可以并行，但无线模块满足联合半双工。
5. 主通信资源由五个固定资源组组成，资源组宽度只取一个或两个相邻组，功率和 CPU 频率只取离散档位。
6. 建筑物以三维柱体表示，遮挡通过线段与建筑物的几何相交判定。
7. 轻量信道模型仅服务于趋势评价；真实信道用于环境转移，actor 使用估计信道、AoI 和历史统计。
8. 初始能量、参考速率、deadline 和路径损耗参数在 Gate P pilot 后校准。
9. 主动能耗只包含无线发射和 CPU 动态能耗，不包含推进和其他飞行器功耗。
10. 结果回传暂不建模，远程 CPU 完成即视为任务计算完成。

### 2.15.2 局限性和声明边界

本模型适合研究动态异构、单跳 U2U-MEC、离散资源编排和截止期队列之间的耦合机制，但不应被解释为完整 UAV 通信系统的数字孪生。固定高度和外生移动排除了轨迹控制与飞行动力学影响；轻量物理层不提供实测信道或实时射线追踪级别的真实性保证；未建模结果回传、完整协议栈、HARQ、MIMO、RIS、DAG 任务和多跳路由；第三 UAV 软遮挡及高保真传播回放只能作为后续扩展。性能结论必须限定在已实现的环境、参数校准范围、测试场景和统计协议内。

本章的公式定义服务接口、约束和评价量，不证明任务完成率的理论最优边界，不证明策略对未测试规模或未测试分布的泛化，也不把经验趋势写成定理。任何数值、曲线、显著性或算法优越性结论都必须留给真实程序运行后的结果章节。

### 2.15.3 系统规则到代码和测试的映射

下表只描述后续实现与测试应覆盖的接口，不声称这些模块当前已经实现。由于项目初始化阶段不编写系统实现代码，本表中的文件名是规划映射。

| 系统规则 | 形式化对象或不变量 | 计划实现位置 | 计划测试或 Gate |
|---|---|---|---|
| 任务记录与生命周期 | $\tau_n$、$\mathcal S_{\mathrm{task}}$、$\mathcal T_{\mathrm{task}}$ | src/env/tasks.py | tests/test_task_lifecycle.py，Gate 0 |
| bit 守恒 | $D_n^{\mathrm{rem}}$ 按实际传输服务递减且不为负 | src/env/queues.py | bit 守恒测试，Gate 0 |
| cycle 守恒 | $C_n^{\mathrm{rem}}$ 按实际 CPU 服务递减且不为负 | src/env/queues.py | cycle 守恒测试，Gate 0 |
| 状态机合法性 | 只允许 $\mathcal T_{\mathrm{task}}$ 中的转移 | src/env/tasks.py | 零非法状态转移测试，Gate 0 |
| route/tx 分离 | route 仅操作 $Q_i^{\mathrm{unb}}$；tx_select 仅操作 $Q_{i\rightarrow j}^{\mathrm{tx}}$ | src/env/action_space.py，src/env/queues.py | 同槽因果测试，Gate 0 |
| 槽边界 | 新路由、新到达和传输完成任务下一槽才可服务 | src/env/u2u_mec_env.py | 槽序列测试，Gate 0 |
| 固定高度 | $\mathbf q_i(t)=[x_i(t),y_i(t),H]^{\mathrm T}$ | src/env/mobility.py | 高度不变测试，Gate 0 |
| 动态候选邻居 | $\mathcal N_i(t)$ 与三维距离阈值 | src/env/topology.py | 邻居边界测试，Gate 1 |
| 建筑物遮挡 | 存在量词几何相交判定 | src/env/buildings.py | 遮挡几何测试，Gate 1 |
| 固定资源组 | $\mathcal G_1,\ldots,\mathcal G_5$ 及宽度 mask | src/env/resource_groups.py | 资源组边界测试，Gate 0 |
| SINR 与有效速率 | $\mathrm{SINR}_{ij,r}$、$R_{ij}^{\mathrm{eff}}$ | src/env/channel.py | 干扰和速率单元测试，Gate 1 |
| outage 统计 | 未调度链路记为 NA，仅统计实际调度链路 | src/env/channel.py，src/evaluation/ | outage 口径测试，Gate 1 |
| 能量守恒 | $E_i^{\mathrm{res}}(t+1)=E_i^{\mathrm{res}}(t)-E_i^{\mathrm{act}}(t)$ | src/env/energy.py | energy 守恒与非负测试，Gate 0 |
| 离散能量降档 | $1.0\rightarrow0.5\rightarrow0.25\rightarrow0$ | src/env/energy.py，src/agents/action_mask.py | 档位闭集测试，Gate 0 |
| 联合半双工 | 同一 UAV 不得同时发送和接收 | src/env/half_duplex_resolver.py | 冲突解析测试，Gate 0 |
| 信息权限 | actor 只使用 $o_i(t)$，禁止读取真实遮挡、信道、未来位置、真实 $\lambda_i$ 和完整私有队列 | src/agents/observation.py | 信息泄漏测试，Gate 0 |
| Gate P 参数校准 | 链路预算、deadline 与能量处于非退化区间 | configs/ 与 experiments/（待建） | Gate P pilot |
| Gate 0 环境正确 | 任务、bit、cycle、energy 守恒；route/tx 分离；槽因果正确 | tests/（待建） | Gate 0 |

### 2.15.4 本章与后续章节的关系

本章是系统对象、符号、状态机、队列、物理层评价、动作接口和信息权限的唯一主源。后续方法章节只能在这些接口上说明策略网络、训练流程、集中训练和分散执行方式，不得重新定义任务状态、route/tx_select 语义、固定资源组、半双工、离散降档或 actor 信息边界。实验协议章节只能使用本章的参数状态、Gate P/Gate 0 和评价口径组织校准、训练、测试、统计与复现，不得通过实验设置改变系统规则。

在编码实现前，必须先以不调用学习器的完整 episode 验证任务、bit、cycle 和能量守恒、零非法状态转移、槽边界、outage 统计、离散降档和联合半双工；只有 Gate 0 通过后，才可将本章定义的环境接口连接到后续策略和价值模块。



