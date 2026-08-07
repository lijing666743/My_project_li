# 2 系统模型与问题建模

## 2.1 系统概述

​      本文考虑由 $N$ 架 UAV 组成的动态异构多 UAV U2U-MEC 系统。UAV 既可以产生任务，也可以作为 U2U 发送节点、接收节点、本地计算节点或远程协同计算节点。系统在外生移动、动态候选邻居、资源异构、业务负载异构、建筑物遮挡、同频干扰、不完美 CSI、跨时隙队列和任务截止期共同存在的条件下运行。研究对象是单跳协同卸载与资源编排：任务只能从源 UAV 直接传输到一个目的 UAV，不能拆分到多个目的地，也不允许经由第三架 UAV 多跳转发。

​      系统时间被离散为长度为 $\Delta t$ 的时隙，时隙 $t$ 对应连续时间区间 $[t\Delta t,(t+1)\Delta t)$。槽初表示该区间开始的边界，槽内只对槽初已经存在的队列执行传输和计算服务，槽末表示本槽服务完成后的边界并承载到达、绑定、转队和完成等事件。新任务在时隙 $t_n^{\mathrm{arr}}$ 的槽末生成并到达，记录其到达事件所属时隙索引 $t_n^{\mathrm{arr}}$，因此在 $t_n^{\mathrm{arr}}+1$ 的槽初才首次进入 actor 可见的未绑定队列并允许 route；本槽 route 决策在槽末绑定目的地，任务从下一时隙起才可接受传输或 CPU 服务。传输完成任务同样在槽末转入 CPU 队列，并在下一时隙首次计算。该安排保留传输中断、排队等待、CPU 竞争和截止期违约的端到端因果关系。任务结果回传暂不单独建模，因此远程计算完成表示任务服务过程结束；若后续研究需要刻画结果数据回传，可在主模型通过附加的反向服务队列扩展，但不改变本章的单跳输入卸载接口。

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
| UAV 平台 | $\kappa_i$ | CPU 动态能耗系数 | 由 $E=\kappa f^3\tau$ 定义 |
| 任务与队列 | $\tau_n$ | 任务 $n$ 的完整记录 | 记录元组 |
| 任务与队列 | $D_n,C_n$ | 任务输入数据量和计算量 | bit，cycle |
| 任务与队列 | $D_n^{\mathrm{rem}},C_n^{\mathrm{rem}}$ | 剩余待传输 bit 和待执行 cycle | bit，cycle |
| 任务与队列 | $t_n^{\mathrm{arr}}$ | 槽末到达事件所属的时隙索引 | slot |
| 任务与队列 | $t_n^{\mathrm{cmp}}$ | 完成发生并在槽末记录的服务时隙索引 | slot |
| 任务与队列 | $t_n^{\mathrm{ddl}}$ | 最后允许接受传输或计算服务的时隙索引 | slot |
| 任务与队列 | $L_n^{\mathrm{ddl}}$ | clip 后的相对 deadline 时隙预算 | slot |
| 任务与队列 | $d_n^{\mathrm{dst}}$ | 任务当前绑定的执行目的 UAV | $\mathcal U\cup\{\varnothing\}$ |
| 任务与队列 | $s_n(t)$ | 任务生命周期状态 | 见 2.5 |
| 任务与队列 | $Q_i^{\mathrm{unb}}$ | UAV $i$ 的未绑定任务队列 | 任务列表 |
| 任务与队列 | $Q_i^{\mathrm{loc}}$ | UAV $i$ 的本地待计算任务队列 | 任务列表 |
| 任务与队列 | $Q_{i\rightarrow j}^{\mathrm{tx}}$ | 从 $i$ 到 $j$ 的已绑定传输队列 | 任务列表 |
| 任务与队列 | $Q_{i\rightarrow j}^{\mathrm{cpu}}$ | 来源为 $i$、在 $j$ 上执行的 CPU 队列 | 任务列表 |
| 任务与队列 | $\operatorname{slack}_n(t)$ | 任务截止期余量 | slot |
| 业务到达 | $A_i(t)$ | UAV $i$ 在时隙 $t$ 槽末生成并到达的任务数 | 任务数；Bernoulli 情况为 $\{0,1\}$ |
| 业务到达 | $\lambda_i$ | UAV $i$ 的环境任务到达概率或按时隙计的强度 | task/slot；Bernoulli 概率为无量纲 |
| 业务到达 | $\hat{\lambda}_i(t)$ | 时隙 $t$ 槽初基于已结束时隙到达量形成的历史到达率估计 | task/slot |
| 业务到达 | $W_\lambda$ | 历史到达率估计的最大窗口长度 | slot |
| 业务到达 | $K_\lambda(t)$ | 时隙 $t$ 槽初实际可用的历史样本数 | 样本数 |
| 业务到达 | $m_i^\lambda(t)$ | 历史到达率估计的可用性 mask | $\{0,1\}$ |
| 无线信道 | $f_c$ | 载频 | Hz |
| 无线信道 | $\mathcal B$ | 建筑物集合 | 三维柱体集合 |
| 无线信道 | $I_{ij}^{\mathrm{bld}}(t)$ | 几何遮挡指示量 | $\{0,1\}$ |
| 无线信道 | $PL_{ij}(t)$ | $i\rightarrow j$ 路径损耗 | dB |
| 无线信道 | $\Delta s_{ij}(t)$ | 两端 UAV 在相邻时隙间的平均水平位移；episode reset 时 $\Delta s_{ij}(0)=0$ | m |
| 无线信道 | $X_{ij}(t)$ | 相关阴影项 | dB |
| 无线信道 | $h_{ij,r}(t)$ | 资源单元 $r$ 上的复信道增益 | 复数 |
| 无线信道 | $\hat h_{ij,r}(t)$ | actor 可用的 CSI 估计 | 复数 |
| 无线信道 | $a_{ij}^{\mathrm{CSI}}(t)$ | CSI AoI | slot |
| 无线信道 | $\widehat I_{j,r}^{\mathrm{hist}}(t)$ | 接收 UAV $j$ 的历史干扰摘要 | W |
| 无线信道 | $\ell_{ij}^{\mathrm{CSI}}(t)$ | 陈旧 CSI 的原始历史索引，$t-a_{ij}^{\mathrm{CSI}}(t)$ | 整数，可为负 |
| 无线信道 | $m_{ij}^{\mathrm{CSI}}(t)$ | 陈旧 CSI 历史索引的可用性 mask | $\{0,1\}$ |
| 无线信道 | $\widehat Z_{j,r}^{\mathrm{hist}}(t)$ | 历史干扰加噪声摘要 | W |
| 无线信道 | $I_{j,r}^{\mathrm{meas}}(t)$ | 已结束时隙中的可用干扰测量 | W |
| 无线信道 | $\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)$ | 历史干扰链路质量代理量 | 线性值 |
| 无线信道 | $p_r^{\mathrm{ref}}$ | 固定参考每资源单元功率，$P_{\mathrm{ref}}/R$ | W |
| 无线信道 | $\beta_I$ | 历史干扰指数滑动平均系数 | $[0,1)$ |
| 无线信道 | $I_{j,r}^{\mathrm{def}}$ | episode 初始历史干扰缺省值 | W |
| 无线信道 | $a_j^{\mathrm{msg}}(t)$ | 接收 UAV $j$ 广播摘要的消息 AoI | slot |
| 无线信道 | $m_{j,r}^{I}(t)$ | 历史干扰摘要可用性 mask | $\{0,1\}$ |
| 无线信道 | $\mathrm{SINR}_{ij,r}(t)$ | 资源单元级信干噪比 | 线性值 |
| 无线信道 | $\gamma_{\min}^{\mathrm{dB}}$ | 无线 SINR 门限的 dB 表示 | dB |
| 无线信道 | $\gamma_{\min}^{\mathrm{lin}}$ | 由 dB 门限转换得到的线性 SINR 门限 | 线性无量纲值 |
| 无线信道 | $R_{ij}^{\mathrm{eff}}(t)$ | 链路有效速率 | bit/s |
| outage 统计 | $\chi_{ij}^{\mathrm{att}}(t)$ | 实际传输尝试指示量 | $\{0,1\}$ |
| outage 统计 | $o_{ij}(t)$ | 单次实际传输尝试的 outage 样本 | $\{0,1,\mathrm{NA}\}$ |
| outage 统计 | $\mathcal A^{\mathrm{att}}$ | 实际传输尝试样本集合 | 时隙、发送端和接收端三元组集合 |
| outage 统计 | $\widehat q_{ij}^{\mathrm{out}}(t)$ | 仅由历史实际传输尝试计算的 outage 特征 | $[0,1]$ 或缺省值 |
| outage 统计 | $m_{ij}^{\mathrm{out}}(t)$ | 历史 outage 特征可用性 mask | $\{0,1\}$ |
| outage 统计 | $\mathrm{NA}$ | 没有实际传输尝试时的不可用标记，不参与数值统计 | 非数值标记 |
| 资源与能量 | $R$ | 等效资源单元总数 | 正整数 |
| 资源与能量 | $G$ | 固定资源组数 | $G=5$ |
| 资源与能量 | $\mathcal G_g$ | 第 $g$ 个固定资源组 | 等效资源单元集合 |
| 资源与能量 | $\mathcal R$ | 等效资源单元索引集合 | $\{1,\ldots,R\}$ |
| 资源与能量 | $\mathcal S_i^{\mathrm{prop}}(t)$ | actor 动作提案为发送 UAV $i$ 申请的资源单元集合 | 资源单元集合 |
| 资源与能量 | $\mathcal S_i^{\mathrm{exec}}(t)$ | 联合执行器仲裁后发送 UAV $i$ 实际占用的资源单元集合 | 资源单元集合 |
| 资源与能量 | $x_{ij,r}(t)$ | 链路 $i\rightarrow j$ 对资源单元 $r$ 的最终实际占用指示 | $\{0,1\}$ |
| 资源与能量 | $p_i(t)$ | UAV $i$ 在已分配资源上的总发射功率 | W |
| 资源与能量 | $p_{ij,r}(t)$ | 链路在资源单元上的功率 | W |
| 资源与能量 | $\tau_{ij}^{\mathrm{tx}}(t)$ | 链路 $i\rightarrow j$ 的通信实际活跃时间 | s |
| 资源与能量 | $\tau_i^{\mathrm{cpu}}(t)$ | UAV $i$ 的 CPU 实际活跃时间 | s |
| 资源与能量 | $\overline\tau_i^{\mathrm{cpu}}(t)$ | CPU 预留实际活跃时间上界 | s |
| 资源与能量 | $E_i^{\mathrm{tx}}(t)$ | 主动通信能耗 | J |
| 资源与能量 | $E_i^{\mathrm{cpu}}(t)$ | CPU 动态能耗 | J |
| 资源与能量 | $E_i^{\mathrm{act}}(t)$ | 主动通信与计算总能耗 | J |
| 资源与能量 | $\overline E_i^{\mathrm{tx}}(t)$ | 通信预留能量上界 | J |
| 资源与能量 | $\overline E_i^{\mathrm{cpu}}(t)$ | CPU 预留能量上界 | J |
| 资源与能量 | $\overline E_i^{\mathrm{act}}(t)$ | 联合主动能量预留上界 | J |
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

任务目的地字段统一定义为：

$$
d_n^{\mathrm{dst}}(t)
=
\begin{cases}
\varnothing, & \text{任务尚未绑定，即处于 unbound 状态},\\
i_n, & \text{任务已绑定在源 UAV 本地执行},\\
j,\ j\ne i_n, & \text{任务已绑定到远程 UAV }j\text{ 执行}.
\end{cases}
$$

因此，$\varnothing$ 仅表示任务尚未绑定目的地；local 任务和 remote 任务的目的 UAV 分别为源 UAV $i_n$ 和远程 UAV $j\ne i_n$，字段定义域仍为 $\mathcal U\cup\{\varnothing\}$。

route 决策在槽末生效并使任务进入 local 或 tx 队列时，任务才完成目的地绑定；绑定后目的地字段保持不变。

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
| $\gamma_{\min}^{\mathrm{dB}}$ | $-3$ | dB | 敏感性参数 |
| $\gamma_{\min}^{\mathrm{lin}}$ | $10^{-3/10}\approx0.5012$ | 线性无量纲值 | 由 dB 门限转换 |
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
W_\lambda\in\mathbb N_{+},
\qquad
K_\lambda(t)=\min\{W_\lambda,t\}.
$$

其中 $W_\lambda$ 是历史窗口长度，单位为 slot，$K_\lambda(t)$ 是时隙 $t$ 槽初实际可用的历史样本数。历史到达率估计和可用性 mask 联合定义为：

$$
\hat{\lambda}_i(t)
=
\begin{cases}
0, & t=0,\\[2pt]
\displaystyle
\frac{\sum_{\tau=t-K_\lambda(t)}^{t-1}A_i(\tau)}
{K_\lambda(t)}, & t\ge 1,
\end{cases}
\qquad
m_i^\lambda(t)=\mathbb I\!\left[K_\lambda(t)>0\right].
$$

因此，$t=0$ 时没有已结束时隙的到达样本，使用固定缺省值 $\hat{\lambda}_i(0)=0$ 和 $m_i^\lambda(0)=0$；对 $t\ge1$，$m_i^\lambda(t)=1$。求和上界严格为 $t-1$：当 $1\le t<W_\lambda$ 时使用当前实际存在的短历史窗口，当 $t\ge W_\lambda$ 时使用最近 $W_\lambda$ 个已结束时隙。该估计的单位沿用按时隙计的 task/slot，不引入额外的 $\Delta t$ 换算，也不改变真实业务到达分布。

特别地，$A_i(t)$ 在时隙 $t$ 槽末才生成，因此不属于 $\hat{\lambda}_i(t)$ 的历史样本，也不得进入时隙 $t$ 的 actor 观测；它最早在时隙 $t+1$ 槽初通过历史估计或队列状态产生影响。

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

其中 $D_n$ 的单位为 bit，$C_n$ 的单位为 cycle，$D_n^{\mathrm{rem}}$ 表示剩余待传输输入 bit，$C_n^{\mathrm{rem}}$ 表示剩余待执行 cycle；$d_n^{\mathrm{dst}}=\varnothing$ 表示任务尚未绑定目的 UAV。任务生成时处于 unbound 状态，并初始化为 $D_n^{\mathrm{rem}}=D_n$、$C_n^{\mathrm{rem}}=C_n$ 和 $d_n^{\mathrm{dst}}=\varnothing$。远程任务的 $D_n^{\mathrm{rem}}$ 只按实际 U2U 传输服务量递减；任务绑定为 local 时则必须按下述生命周期规则立即清零该字段。$C_n^{\mathrm{rem}}$ 保持为 $C_n$，直到实际 CPU 服务使其递减。

输入数据量和单位 bit 计算量使用方案 2 的初始化范围：

$$
D_n\sim U(0.25,1.5)\ \mathrm{Mbit}.
$$

该式给出任务输入规模的编码初始分布，正式实验前仍需经过 Gate P 的非退化性检查。计算量与数据量的比值定义为：

$$
\frac{C_n}{D_n}
\sim U(300,1000)\ \mathrm{cycle/bit}.
$$

deadline 根据参考传输时间和参考计算时间生成。令 $\eta_n\sim U(1.5,3.0)$ 为服务裕量因子，先定义 clip 后的相对 deadline 时隙预算：

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

随后定义绝对 deadline 时隙索引为：

$$
t_n^{\mathrm{ddl}}
=
t_n^{\mathrm{arr}}+L_n^{\mathrm{ddl}}.
$$

其中 $L_n^{\mathrm{ddl}}$ 表示从任务到达槽末到截止槽末的总相对时隙预算，其中包含 route 决策所占用的一个时隙，不表示纯传输或 CPU 可服务时隙数量；$R_{\mathrm{ref}}$ 的单位为 bit/s，$f_{\mathrm{ref}}$ 的单位为 cycle/s，$\Delta t$ 的单位为 s。分布表中的 Mbit 在代入传输时间前统一换算为 bit，10 至 150 只限制相对预算 $L_n^{\mathrm{ddl}}$，不表示绝对时隙索引。

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

其中，unbound 表示任务已经到达但尚未绑定本地或远程目的地，此时 $d_n^{\mathrm{dst}}=\varnothing$；local 表示任务已绑定到源 UAV $i_n$ 的本地 CPU 队列，此时 $d_n^{\mathrm{dst}}=i_n$；tx 表示任务已绑定到远程 UAV $j\ne i_n$ 并处于传输队列，此时 $d_n^{\mathrm{dst}}=j$；cpu 表示任务正在本地或远程 CPU 队列中接受计算服务，其目的地字段必须等于实际执行 UAV，即本地执行时为 $i_n$、远程执行时为 $j\ne i_n$；done 表示任务不晚于截止时隙末完成；expired 表示任务在截止时隙服务结束、且本槽服务量和剩余工作量更新后仍未完成。

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

done 和 expired 均为终止状态，该集合不包含由二者出发的转移，尤其不包含 $\mathrm{expired}\rightarrow\mathrm{done}$。

当 route 决策在槽末生效并触发 $\mathrm{unbound}\rightarrow\mathrm{local}$ 转移时，绑定操作必须同时执行：

$$
d_n^{\mathrm{dst}}\leftarrow i_n,
\qquad
D_n^{\mathrm{rem}}\leftarrow 0.
$$

当 route 决策在槽末生效并触发 $\mathrm{unbound}\rightarrow\mathrm{tx}$ 转移时，绑定操作必须同时执行：

$$
d_n^{\mathrm{dst}}\leftarrow j,
\qquad
j\ne i_n.
$$

本地任务不需要 U2U 输入传输，因此进入本地 CPU 队列后必须始终满足 $D_n^{\mathrm{rem}}=0$；其 $C_n^{\mathrm{rem}}=C_n$，直到实际 CPU 服务使其递减。远程任务绑定到 $j\ne i_n$ 后，$D_n^{\mathrm{rem}}$ 继续表示尚未完成的 U2U 输入传输量，并只按实际传输服务量递减。

该集合不包含已绑定任务改变目的 UAV 的转移，也不包含多跳转发。新任务在槽末到达并记录 $t_n^{\mathrm{arr}}$，下一时隙槽初才首次进入未绑定队列并允许 route；route 决策在槽末生效时，unbound 任务转入 local 或 tx 队列并立即锁定目的地，下一时隙才首次接受 CPU 或传输服务。此后处于 local、tx 或 cpu 状态的任务不得再次执行 route，也不得改变 $d_n^{\mathrm{dst}}$。本槽传输完成任务在槽末进入远程 CPU 队列，下一时隙才首次计算；因此新生成任务、本槽路由任务和本槽传输完成任务均不能在同一时隙获得下一阶段服务。

对于处于 done 状态的任务，若其在服务时隙 $t_n^{\mathrm{cmp}}$ 的服务过程中完成，则在该时隙槽末记录完成，完成边界为 $(t_n^{\mathrm{cmp}}+1)\Delta t$。由于到达边界为 $(t_n^{\mathrm{arr}}+1)\Delta t$，其端到端时延定义为：

$$
T_n^{\mathrm{E2E}}
=
\left(
t_n^{\mathrm{cmp}}-t_n^{\mathrm{arr}}
\right)\Delta t.
$$

合法完成记录必须满足 $t_n^{\mathrm{cmp}}\le t_n^{\mathrm{ddl}}$。任务可以在其截止时隙 $t_n^{\mathrm{ddl}}$ 内继续接受本槽传输或计算服务；仅在该槽的服务量和剩余工作量更新完成后进行终止结算。若任务在该结算点仍未全部完成，则立即转为 expired。该判定可写为：

$$
\operatorname{expired}_n
=
\mathbb 1
\left\{
\text{任务在 }t_n^{\mathrm{ddl}}\text{ 的本槽服务量和剩余工作量更新后仍未全部完成}
\right\}.
$$

若任务在相同结算点已全部完成，则转为 done，并计为按时完成。系统仅保留按时完成和过期两类结算结果；expired 任务在 $t_n^{\mathrm{ddl}}$ 槽末结算后立即从所有活动队列中移除，因此在 $t_n^{\mathrm{ddl}}+1$ 的槽初不再出现在任何活动队列中，也不再接受传输或计算服务。

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
\tau_n:\ i_n=i,\ s_n(t)\in\{\mathrm{local},\mathrm{cpu}\},\ d_n^{\mathrm{dst}}=i
\right).
$$

这里的 $i$ 同时是任务源 UAV 和本地执行 UAV；队列内任务均满足 $D_n^{\mathrm{rem}}=0$。对 $j\in\mathcal U\setminus\{i\}$，从源 UAV $i$ 到远程目的 UAV $j$ 的传输队列为：

$$
Q_{i\rightarrow j}^{\mathrm{tx}}(t)
=
\left(
\tau_n:\ i_n=i,\ d_n^{\mathrm{dst}}=j,\ s_n(t)=\mathrm{tx}
\right).
$$

对 $j\in\mathcal U\setminus\{i\}$，来源为 $i$、在远程目的 UAV $j$ 上执行的 CPU 队列为：

$$
Q_{i\rightarrow j}^{\mathrm{cpu}}(t)
=
\left(
\tau_n:\ i_n=i,\ d_n^{\mathrm{dst}}=j,\ s_n(t)=\mathrm{cpu}
\right).
$$

因此，两类远程队列均要求 $j\ne i$；目的地为源 UAV 的本地任务不会被收入远程 CPU 队列。

每类队列均按队首 deadline 优先的 EDF 顺序排列。任务 $n$ 在槽初、且仍处于活动状态并满足 $t\le t_n^{\mathrm{ddl}}$ 时的 deadline 余量定义为：

$$
\operatorname{slack}_n(t)
=
t_n^{\mathrm{ddl}}-t+1.
$$

余量越小表示任务越紧迫；在最后允许服务的时隙 $t=t_n^{\mathrm{ddl}}$，slack 等于 $1$，活动任务不应出现 $0$ 或负数 slack。相同 slack 时采用固定任务标识作为可复现的 tie-break，不把随机打散引入队列语义。

route 只处理槽初 $Q_i^{\mathrm{unb}}(t)$ 中的最高优先级任务，并在 local、defer 或当前候选远程 UAV 中选择；选择 local 或远程目的地时，route 决策在槽末生效并一次性绑定目的地，不直接产生本槽传输服务。tx_select 只从槽初已有的 $Q_{i\rightarrow j}^{\mathrm{tx}}(t)$ 中选择一个非空传输队列，并负责本槽服务预算分配；该动作只服务已绑定的传输队列，不参与目的地选择或改变 $d_n^{\mathrm{dst}}$。两者即使在同一槽分别输出，也不能使新路由任务获得本槽传输服务。

## 2.7 外生移动与动态U2U图模型

### 2.7.1 Gauss-Markov水平移动

移动轨迹由环境外生生成。每个 episode reset 时直接生成或载入初始水平位置和速度 $\mathbf p_i(0)$、$\mathbf v_i(0)$，以及移动模型所需的其他初始状态；不定义也不访问 $\mathbf p_i(-1)$。令 $\bar{\mathbf v}_i$ 为平均水平速度，$\boldsymbol{\varepsilon}_i(t)$ 为零均值扰动，水平速度采用：

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

其中 $\alpha=0.85$ 为编码初始相关系数，$\mathbf v_i^{\mathrm{hor}}(t)=[v_i^x(t),v_i^y(t)]^{\mathrm T}$，该式对 $t\ge0$ 使用。边界处理采用反射或速度反向，最小安全距离由轨迹生成器保证。

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

位置更新式对 $t\ge0$ 使用已定义的 $\mathbf p_i(t)$；需要引用上一时隙位置的位移量从 $t=1$ 开始定义，初始时刻不通过负时隙历史反推移动状态。
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

新任务边只用于决定未绑定任务当前可选的目的 UAV。服务边则保存已经锁定的 $i\rightarrow j$ 关系；即使锁定任务因移动暂时超出新任务候选范围，仍保留其传输队列和剩余 bit，等待链路恢复或按过期规则结算，链路 outage、移动或超出候选范围均不触发目的地变更。outage 不删除已绑定服务边。所有邻居、建筑物路径和信道计算均以 $\mathbf q_i(t)$ 的三维位置为输入。

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

空间相关性通过相邻时隙内两端 UAV 的水平位移建立。episode reset 时直接采样初始阴影项：

$$
X_{ij}(0)
\sim
\mathcal N(0,\sigma_s^2),
$$

其中 $X_{ij}(0)$ 是阴影衰落过程的显式初始样本，不通过未定义的 $X_{ij}(-1)$ 递推获得。初始位移约定为 $\Delta s_{ij}(0)=0$。对 $t\ge1$，位移尺度为：

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

该递推式仅对 $t\ge1$ 使用。其中 $\sigma_s$ 的单位为 dB，$\epsilon_{ij}(t)$ 是标准化零均值扰动。该模型保持短期移动连续性，并使遮挡和链路质量变化不被人为地设置为完全独立；阴影项在 dB 域保持既有定义，不将其直接当作线性功率相加。

### 2.8.4 Rician/Rayleigh槽级衰落

主模型按遮挡状态选择槽级小尺度衰落类型。episode reset 时，环境依据初始位置或距离、初始路径损耗、$X_{ij}(0)$、初始小尺度衰落样本以及当前 LoS/NLoS 状态生成有效的首个真实信道样本 $h_{ij,r}(0)$。后续 $h_{ij,r}(t)$ 按现有信道模型由时隙 $t$ 的环境状态生成，不访问任何负时隙信道历史。

LoS 链路使用 Rician 衰落，NLoS 链路使用 Rayleigh 衰落。资源单元 $r$ 上的复信道功率增益写为：

$$
\left|h_{ij,r}(t)\right|^2
=
G_tG_r
10^{-PL_{ij}(t)/10}
\left|g_{ij,r}(t)\right|^2.
$$

其中 $G_t$ 和 $G_r$ 为发送与接收增益的线性值，$g_{ij,r}(t)$ 是单位化小尺度衰落项，且该式对 $t\ge0$ 使用。当 $I_{ij}^{\mathrm{bld}}(t)=0$ 时使用 Rician 参数 $K=6$ dB；当 $I_{ij}^{\mathrm{bld}}(t)=1$ 时使用 Rayleigh 特例 $K=0$。主训练环境使用槽级平均功率增益，以避免将不必要的瞬时波形细节引入决策接口。

### 2.8.5 不完美CSI和CSI AoI

actor 使用陈旧且带估计误差的信道信息。令 $a_{ij}^{\mathrm{CSI}}(t)$ 为 CSI AoI，$\xi_{ij,r}(t)$ 为正的乘性估计误差。先定义陈旧 CSI 的原始历史索引及可用性 mask：

$$
\ell_{ij}^{\mathrm{CSI}}(t)
=
t-a_{ij}^{\mathrm{CSI}}(t),
\qquad
m_{ij}^{\mathrm{CSI}}(t)
=
\mathbb I
\left[
\ell_{ij}^{\mathrm{CSI}}(t)\ge0
\right].
$$

陈旧 CSI 特征定义为：

$$
\hat h_{ij,r}(t)
=
\begin{cases}
h_{ij,r}\!\left(\ell_{ij}^{\mathrm{CSI}}(t)\right)\xi_{ij,r}(t),
&m_{ij}^{\mathrm{CSI}}(t)=1,
\\[6pt]
0,
&m_{ij}^{\mathrm{CSI}}(t)=0.
\end{cases}
$$

当 $\ell_{ij}^{\mathrm{CSI}}(t)<0$ 时，$m_{ij}^{\mathrm{CSI}}(t)=0$，不访问负时隙信道；此时的零值只表示缺省占位，并不表示真实零信道。只有当 $\ell_{ij}^{\mathrm{CSI}}(t)\ge0$ 时才读取已经存在的历史样本。当当前时隙满足 $\ell_{ij}^{\mathrm{CSI}}(t)=0$ 时，首次读取 $h_{ij,r}(0)$。若 AoI 固定为常数 $d$，则在 $t=d$ 时有 $\ell_{ij}^{\mathrm{CSI}}(t)=0$，首次读取 $h_{ij,r}(0)$。本轮不生成 pre-episode 信道历史，也不通过 clamp 或当前真实 $h_{ij,r}(t)$ 回填指定 AoI 的历史样本。

误差幅度以 dB 域建模为：

$$
10\log_{10}\xi_{ij,r}(t)
\sim
\mathcal N
\left(
0,\sigma_{\mathrm{CSI}}^2
\right).
$$

actor 使用 $\hat h_{ij,r}(t)$ 时必须同时读取对应的 $m_{ij}^{\mathrm{CSI}}(t)$；缺省值不能脱离 mask 单独解释为真实测量。
上述 $\hat h_{ij,r}(t)$ 只描述期望链路 $i\rightarrow j$ 的陈旧信道估计，不包含本时隙其他 UAV 尚未确定的资源选择、功率或执行器接受结果；actor 也不得读取当前真实信道 $h_{ij,r}(t)$。为给槽初决策提供可实现的干扰线索，接收 UAV $j$ 对每个资源单元维护只由上一时隙或更早信息形成的历史干扰摘要。对 $t\ge1$，采用指数滑动平均：

$$
\widehat I_{j,r}^{\mathrm{hist}}(t)
=
\begin{cases}
\beta_I\widehat I_{j,r}^{\mathrm{hist}}(t-1)
+(1-\beta_I)I_{j,r}^{\mathrm{meas}}(t-1),
&
\text{上一时隙有环境规定的可用测量},\\[4pt]
\widehat I_{j,r}^{\mathrm{hist}}(t-1),
&
\text{上一时隙无可用测量}.
\end{cases}
$$

其中 $I_{j,r}^{\mathrm{meas}}(t-1)$ 是接收端在已结束时隙形成的干扰测量，只在环境规定为可观测时参与更新；无测量时保持上一历史值，并使相应摘要的 AoI 增加。episode 初始时设置 $\widehat I_{j,r}^{\mathrm{hist}}(0)=I_{j,r}^{\mathrm{def}}$、$m_{j,r}^{I}(0)=0$，并将消息 AoI $a_j^{\mathrm{msg}}(0)$ 置为预先约定的不可用哨值。该缺省值不是测量结果，actor 必须同时读取 $m_{j,r}^{I}(t)$；有可用历史摘要时 $m_{j,r}^{I}(t)=1$，若 episode 尚无任何历史测量则保持 $m_{j,r}^{I}(t)=0$，后续无新测量时不把已有历史值伪装成当前测量。接收 UAV $j$ 通过既有控制消息广播该摘要，actor 可获得其消息 AoI $a_j^{\mathrm{msg}}(t)$。

历史摘要先与噪声功率合成为：

$$
\widehat Z_{j,r}^{\mathrm{hist}}(t)
=
N_0B_{\mathrm{RU}}F
+
\widehat I_{j,r}^{\mathrm{hist}}(t).
$$

为避免把该历史量误称为当前真实 SINR，定义固定参考每资源单元功率下的历史干扰链路质量代理量：

$$
\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)
=
\frac{
p_r^{\mathrm{ref}}
\left|\hat h_{ij,r}(t)\right|^2
}{
\widehat Z_{j,r}^{\mathrm{hist}}(t)
},
\qquad
p_r^{\mathrm{ref}}
=
\frac{P_{\mathrm{ref}}}{R}.
$$

其中 $p_r^{\mathrm{ref}}$ 只用于构造跨时隙、跨动作可比较的观测特征，不是 actor 在当前时隙最终选择的实际功率；$\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)$ 也不使用当前其他 UAV 的动作，因而不等于当前真实 $\mathrm{SINR}_{ij,r}(t)$。actor 可使用该代理量或其固定资源组聚合值、上一槽已实现的有效速率、仅基于过去实际传输尝试的 outage 率、CSI AoI、消息 AoI 和可用性 mask；$\mathrm{SINR}_{ij,r}(t)$ 仍只由环境在联合动作确定后计算。CSI AoI 和历史摘要/消息 AoI 的更新是环境事件，不是 actor 可直接控制的动作。

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

若 actor 动作选择组索引 $g_i(t)$ 和宽度 $w_i(t)\in\{1,2\}$，则动作提案申请的组集合为：

$$
\mathcal W_i(t)
=
\{g_i(t),g_i(t)+1,\ldots,g_i(t)+w_i(t)-1\},
\qquad
\mathcal S_i^{\mathrm{prop}}(t)
=
\bigcup_{g\in\mathcal W_i(t)}\mathcal G_g.
$$

$\mathcal S_i^{\mathrm{prop}}(t)$ 只表示发送 UAV $i$ 在 actor 动作提案中申请的资源单元集合，在联合执行器运行前即可由 $g_i(t),w_i(t)$ 确定，不表示资源已经实际占用。$g_i(t)=5$ 且 $w_i(t)=2$ 时该组合由动作 mask 禁止；当通信分支为 idle 时令 $\mathcal S_i^{\mathrm{prop}}(t)=\varnothing$。即使执行器最终拒绝发送，也允许 $\mathcal S_i^{\mathrm{prop}}(t)\ne\varnothing$。

实际发射功率由离散功率分支确定：

$$
p_i(t)
\in
\{0,0.25,0.5,1.0\}P_i^{\max}.
$$

因此资源宽度只表示一个组或两个相邻组，不改变固定资源组边界；功率分支也不会产生离散集合之外的连续值。

在当前模型中，联合执行器对已选择的固定资源组内部资源单元不做部分删除、增加或重新分配；对提案链路只决定是否接受。于是链路 $i\rightarrow j$ 的最终实际资源占用指示为：

$$
x_{ij,r}(t)
=
y_{ij}(t)\mathbb 1\left[r\in\mathcal S_i^{\mathrm{prop}}(t)\right].
$$

其中 $y_{ij}(t)=1$ 表示链路经联合执行器接受；当 $y_{ij}(t)=0$ 时，$x_{ij,r}(t)=0$ 对所有 $r$ 成立，即使提案集合非空。链路在已分配资源上的总功率按实际占用资源单元数平均分配：

$$
p_{ij,r}(t)
=
\frac{p_i(t)}
{\sum_{r'=1}^{R}x_{ij,r'}(t)}
\quad
\text{当 }\sum_{r'=1}^{R}x_{ij,r'}(t)>0.
$$

联合执行器仲裁后的实际资源集合定义为：

$$
\mathcal S_i^{\mathrm{exec}}(t)
=
\left\{
 r\in\mathcal R:
 \sum_{j\in\mathcal U,\ j\ne i}x_{ij,r}(t)>0
\right\}.
$$

它表示发送 UAV $i$ 在执行器完成硬约束仲裁后的最终实际占用资源单元集合。由于每个发送 UAV 每槽至多有一条满足 $y_{ij}(t)=1$ 的实际接受发送链路，若该唯一链路为 $i\rightarrow j$，则：

$$
\mathcal S_i^{\mathrm{exec}}(t)
=
\left\{r\in\mathcal R:x_{ij,r}(t)=1\right\}.
$$

若发送 UAV $i$ 没有任何被接受的发送链路，则 $\mathcal S_i^{\mathrm{exec}}(t)=\varnothing$；此时 $\mathcal S_i^{\mathrm{prop}}(t)$ 仍可能非空。对满足 $y_{ij}(t)=1$ 的唯一实际接受链路，有：

$$
\left|\mathcal S_i^{\mathrm{exec}}(t)\right|>0
\Longleftrightarrow
\sum_{r=1}^{R}x_{ij,r}(t)>0,
\qquad y_{ij}(t)=1.
$$

链路未被接受时，实际资源集合为空；后续实际服务、通信能耗和 outage 资格均使用 $\mathcal S_i^{\mathrm{exec}}(t)$ 或等价的 $x_{ij,r}(t)$，不使用 $\mathcal S_i^{\mathrm{prop}}(t)$。

当分母为零时链路未获得资源，约定所有 $p_{ij,r}(t)=0$，且不计算该链路的有效服务。

在接收噪声功率为 $N_0B_{\mathrm{RU}}F$ 时，资源单元级 SINR 定义如下。这里的 $x_{ij,r}(t)$、$p_{ij,r}(t)$ 和 $y_{ij}(t)$ 是联合执行器处理后的实际接受结果：只有在所有 UAV 生成动作提案、执行器完成半双工、能量等硬约束处理，且实际接受变量、资源占用、功率和当前真实信道均确定后，环境才计算当前真实 SINR；该量不进入槽初 actor 的局部观测。

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

其中 $N_0$ 是噪声谱密度，$B_{\mathrm{RU}}$ 是单个等效资源单元带宽，$F$ 是由噪声系数换算得到的线性因子。分母中的求和仅包含同一资源单元上被联合执行器接受的其他传输边。该真实 SINR 始终是线性无量纲比值，不是 dB 值；它仅用于环境状态转移、有效速率、实际服务量、outage 以及集中训练期允许使用的真实全局状态。

设 $\gamma_{\min}^{\mathrm{dB}}$ 为 dB 表示的最小可用 SINR 阈值，并定义对应的线性阈值为：

$$
\gamma_{\min}^{\mathrm{lin}}
=
10^{\gamma_{\min}^{\mathrm{dB}}/10}.
$$

本模型中 $\gamma_{\min}^{\mathrm{dB}}=-3\ \mathrm{dB}$，因此 $\gamma_{\min}^{\mathrm{lin}}=10^{-3/10}\approx0.5012$。所有与线性真实 SINR 的数值比较均使用 $\gamma_{\min}^{\mathrm{lin}}$；dB 阈值仅用于参数说明、报告和可读性展示。链路有效速率为：

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
\mathrm{SINR}_{ij,r}(t)\ge\gamma_{\min}^{\mathrm{lin}}
\right\}.
$$

$\gamma_{\min}^{\mathrm{dB}}$ 可在 $-6$、$0$ 和 $3$ dB 之间做敏感性分析，并按上述转换同步得到对应的 $\gamma_{\min}^{\mathrm{lin}}$。有效速率是环境真实服务计算的输入，不应与 actor 只能获得的历史干扰链路质量代理量混淆。

outage 样本只由联合执行器最终接受并实际发起传输的链路产生。$a_i^{\mathrm{tx}}(t)=j$ 仅表示 actor 的传输动作提案，不能直接生成 outage 样本。令槽初已绑定传输队列中的剩余 bit 为：

$$
Q_{ij}^{\mathrm{bit}}(t)
=
\sum_{n\in Q_{i\rightarrow j}^{\mathrm{tx}}(t)}D_n^{\mathrm{rem}}(t).
$$

实际传输尝试指示量定义为：

$$
\chi_{ij}^{\mathrm{att}}(t)
=
\mathbb 1\left[y_{ij}(t)=1\right]
\mathbb 1\left[p_i(t)>0\right]
\mathbb 1\left[\left|\mathcal S_i^{\mathrm{exec}}(t)\right|>0\right]
\mathbb 1\left[Q_{ij}^{\mathrm{bit}}(t)>0\right].
$$

在 $y_{ij}(t)=1$ 时，$\left|\mathcal S_i^{\mathrm{exec}}(t)\right|>0$ 与 $\sum_{r=1}^{R}x_{ij,r}(t)>0$ 等价，因此也可写为：

$$
\chi_{ij}^{\mathrm{att}}(t)
=
\mathbb 1\left[y_{ij}(t)=1\right]
\mathbb 1\left[p_i(t)>0\right]
\mathbb 1\left[\sum_{r=1}^{R}x_{ij,r}(t)>0\right]
\mathbb 1\left[Q_{ij}^{\mathrm{bit}}(t)>0\right].
$$

其中 $y_{ij}(t)$、$p_i(t)$ 和 $x_{ij,r}(t)$ 均指联合执行器处理后的实际接受、实际功率和实际资源占用结果。只有 $\chi_{ij}^{\mathrm{att}}(t)=1$ 才生成一个实际传输尝试样本；同一发送 UAV 每槽至多生成一个此类样本。actor 主动选择 idle、执行器拒绝或因半双工冲突退化为 idle、能量降档至零功率、实际资源占用为空或传输队列为空时，$\chi_{ij}^{\mathrm{att}}(t)=0$，这些情况不属于无线 outage。

对单次实际传输尝试，沿用当前真实 SINR、有效速率和既有无线 outage 判定条件：

$$
o_{ij}(t)
=
\begin{cases}
1,
&
\chi_{ij}^{\mathrm{att}}(t)=1,\
R_{ij}^{\mathrm{eff}}(t)=0,
\\
0,
&
\chi_{ij}^{\mathrm{att}}(t)=1,\
R_{ij}^{\mathrm{eff}}(t)>0,
\\
\mathrm{NA},
&
\chi_{ij}^{\mathrm{att}}(t)=0.
\end{cases}
$$

因此，实际非零功率和非零资源均已发出数据、但真实有效速率为零时，仍按现有无线 outage 条件记为 $o_{ij}(t)=1$；$o_{ij}(t)=\mathrm{NA}$ 不得在任何统计中转为 0 或 1。对当前时隙 $t$，历史 outage 特征只使用已结束时隙中的实际传输尝试集合：

$$
\mathcal H_{ij}^{\mathrm{att}}(t)
=
\left\{
\tau:\,0\le\tau\le t-1,\ \chi_{ij}^{\mathrm{att}}(\tau)=1
\right\}.
$$

当 $|\mathcal H_{ij}^{\mathrm{att}}(t)|>0$ 时，outage 特征为：

$$
\hat q_{ij}^{\mathrm{out}}(t)
=
\frac{
\sum_{\tau\in\mathcal H_{ij}^{\mathrm{att}}(t)}
\mathbb 1\{o_{ij}(\tau)=1\}
}{
|\mathcal H_{ij}^{\mathrm{att}}(t)|
}.
$$

当 $|\mathcal H_{ij}^{\mathrm{att}}(t)|=0$ 时，使用固定缺省值并令 $m_{ij}^{\mathrm{out}}(t)=0$；有历史实际传输尝试时令该 mask 为 1。缺省值不表示 outage，NA 也不进入上述分子或分母。累计 outage 指标的实际尝试样本集合定义为：

$$
\mathcal A^{\mathrm{att}}
=
\left\{
(t,i,j):\chi_{ij}^{\mathrm{att}}(t)=1
\right\},
$$

并按实际传输尝试计算：

$$
P_{\mathrm{out}}
=
\begin{cases}
\displaystyle
\frac{
\sum_{(t,i,j)\in\mathcal A^{\mathrm{att}}}o_{ij}(t)
}{|\mathcal A^{\mathrm{att}}|},
& |\mathcal A^{\mathrm{att}}|>0,\\[1.2ex]
\mathrm{NA},
& |\mathcal A^{\mathrm{att}}|=0.
\end{cases}
$$

按链路、episode、场景或滑动窗口分别统计时，分母均只包含对应窗口内的实际传输尝试次数。等效 RB 级失效率另行记录为：

$$
O_{ij}^{\mathrm{RB}}(t)
=
\frac{
\sum_{r:x_{ij,r}(t)=1}
\mathbb 1\{\mathrm{SINR}_{ij,r}(t)<\gamma_{\min}^{\mathrm{lin}}\}
}{
\sum_{r=1}^{R}x_{ij,r}(t)
},
$$

且仅当 $\chi_{ij}^{\mathrm{att}}(t)=1$ 时计算；否则记为 NA。该口径使链路 outage、等效 RB 级失效率和未实际传输状态保持可区分。

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

对联合执行器接受的链路，通信实际活跃时间定义为：

$$
\tau_{ij}^{\mathrm{tx}}(t)
=
\begin{cases}
\min\left\{
\Delta t,
\dfrac{Q_{ij}^{\mathrm{bit}}(t)}
{R_{ij}^{\mathrm{eff}}(t)}
\right\},
&
y_{ij}(t)=1,\ |\mathcal S_i^{\mathrm{exec}}(t)|>0,\ p_i(t)>0,\
R_{ij}^{\mathrm{eff}}(t)>0,
\\[8pt]
\Delta t,
&
y_{ij}(t)=1,\ |\mathcal S_i^{\mathrm{exec}}(t)|>0,\ p_i(t)>0,\
R_{ij}^{\mathrm{eff}}(t)=0,
\\[4pt]
0,
&
\text{其他情况}.
\end{cases}
$$

其中 $y_{ij}(t)=1$ 表示链路经联合执行器接受。有效速率大于零且整个传输队列在槽内提前清空时，发送端立即停止主动发射，因而实际活跃时间小于 $\Delta t$；有效速率为零但已以非零功率占用资源尝试发射时，视为整槽发射尝试。未被接受、通信 idle、功率为零或没有实际资源占用的链路均不产生通信活跃时间，也不产生 outage 样本；本定义与 $\chi_{ij}^{\mathrm{att}}(t)$ 的实际执行口径一致。

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

设被选 CPU 队列的 EDF 队首任务为 $n^\star$，其本槽实际执行 cycle 为：

$$
c_j^{\mathrm{cpu}}(t)
=
\begin{cases}
\min\left\{
C_{n^\star}^{\mathrm{rem}}(t),
f_j(t)\Delta t
\right\},
&
f_j(t)>0\text{ 且选择了有效非空 CPU 队列},
\\[8pt]
0,
&
\text{其他情况}.
\end{cases}
$$

第一版主模型每个 UAV 每槽只服务一个 CPU 队首任务，不把剩余 CPU 服务预算继续分配给第二个 CPU 队列。队首剩余 cycle 更新为：

$$
C_{n^\star}^{\mathrm{rem}}(t+1)
=
C_{n^\star}^{\mathrm{rem}}(t)-c_j^{\mathrm{cpu}}(t),
\qquad
0\le c_j^{\mathrm{cpu}}(t)\le C_{n^\star}^{\mathrm{rem}}(t).
$$

CPU 实际活跃时间定义为：

$$
\tau_j^{\mathrm{cpu}}(t)
=
\begin{cases}
\dfrac{c_j^{\mathrm{cpu}}(t)}{f_j(t)},
&
f_j(t)>0\text{ 且选择了有效非空 CPU 队列},
\\[8pt]
0,
&
\text{其他情况}.
\end{cases}
$$

因此 $0\le\tau_j^{\mathrm{cpu}}(t)\le\Delta t$。任务在槽内提前完成时，CPU 随即停止本任务的动态计算活动；剩余槽时间不转而服务第二个任务，也不产生 CPU 动态能耗。

实际通信能耗由接受的传输边、其总发射功率和实际活跃时间决定：

$$
E_i^{\mathrm{tx}}(t)
=
\sum_{j\in\mathcal U}
y_{ij}(t)\,p_i(t)\,\tau_{ij}^{\mathrm{tx}}(t).
$$

其中 $p_i(t)$ 是链路在全部已分配资源上的总发射功率；由于每个发送 UAV 每槽至多有一条接受链路，上式不会重复累计多条发送活跃时间。虽然仍有 $\sum_r p_{ij,r}(t)=p_i(t)$，实际通信能耗不再无条件乘以完整 $\Delta t$。

CPU 动态能耗按实际活跃时间计算：

$$
E_j^{\mathrm{cpu}}(t)
=
\kappa_j f_j^3(t)\tau_j^{\mathrm{cpu}}(t)
=
\kappa_j f_j^2(t)c_j^{\mathrm{cpu}}(t).
$$

当 $f_j(t)=0$ 或 CPU queue 为 idle 时，$c_j^{\mathrm{cpu}}(t)=0$、$\tau_j^{\mathrm{cpu}}(t)=0$ 且 $E_j^{\mathrm{cpu}}(t)=0$。在真实有效速率和最终执行结果形成前，联合执行器不能知道通信实际活跃时间，因此对候选动作使用基于提案阶段的保守能量预留上界。此阶段只读取候选传输动作、槽初已绑定队列、$\mathcal S_i^{\mathrm{prop}}(t)$ 和候选功率；$y_{ij}(t)$、$x_{ij,r}(t)$ 与 $\mathcal S_i^{\mathrm{exec}}(t)$ 尚未形成：

$$
\overline E_i^{\mathrm{tx}}(t)
=
\begin{cases}
p_i(t)\Delta t,
&
\begin{gathered}
\exists j:\ a_i^{\mathrm{tx}}(t)=j,\ Q_{i\rightarrow j}^{\mathrm{tx}}(t)\ne\varnothing,\ Q_{ij}^{\mathrm{bit}}(t)>0,\\
|\mathcal S_i^{\mathrm{prop}}(t)|>0,\ p_i(t)>0
\end{gathered},
\\[4pt]
0,
&
\text{其他情况}.
\end{cases}
$$

上式中的 $p_i(t)$ 表示执行器当前正在检查的候选功率档位，而不是对最终接受结果的预判；预留条件不以 $y_{ij}(t)=1$、$|\mathcal S_i^{\mathrm{exec}}(t)|>0$ 或 $\sum_r x_{ij,r}(t)>0$ 为前提。若候选发送被拒绝、因半双工冲突被丢弃或最终降档为通信 idle，则不保留通信预留，$\mathcal S_i^{\mathrm{exec}}(t)=\varnothing$ 且 $E_i^{\mathrm{tx}}(t)=0$。若候选被接受，未实际消耗的预留在槽末释放，剩余能量只按实际主动能耗扣除。

CPU 预留可以根据槽初已知的队首任务和频率确定：

$$
\overline\tau_j^{\mathrm{cpu}}(t)
=
\begin{cases}
\min\left\{
\Delta t,
\dfrac{C_{n^\star}^{\mathrm{rem}}(t)}{f_j(t)}
\right\},
&
f_j(t)>0\text{ 且 CPU 队列有效},
\\[8pt]
0,
&
\text{其他情况},
\end{cases}
$$

$$
\overline E_j^{\mathrm{cpu}}(t)
=
\kappa_j f_j^3(t)\overline\tau_j^{\mathrm{cpu}}(t).
$$

联合主动能量预留为：

$$
\overline E_i^{\mathrm{act}}(t)
=
\overline E_i^{\mathrm{tx}}(t)+\overline E_i^{\mathrm{cpu}}(t),
\qquad
\overline E_i^{\mathrm{act}}(t)
\le
E_i^{\mathrm{res}}(t).
$$

在形成最终 $y_{ij}(t)$、$x_{ij,r}(t)$ 和 $\mathcal S_i^{\mathrm{exec}}(t)$ 之前，联合执行器先使用上述候选预留上界检查接受或离散降档组合的硬可行性；真实服务完成后，实际主动能耗和剩余能量更新为：

$$
E_i^{\mathrm{act}}(t)
=
E_i^{\mathrm{tx}}(t)+E_i^{\mathrm{cpu}}(t),
\qquad
E_i^{\mathrm{act}}(t)
\le
\overline E_i^{\mathrm{act}}(t)
\le
E_i^{\mathrm{res}}(t),
$$

$$
E_i^{\mathrm{res}}(t+1)
=
E_i^{\mathrm{res}}(t)-E_i^{\mathrm{act}}(t).
$$

预留但未实际消耗的能量在时隙结束时释放，不从剩余能量中扣除。上述约束只保证主动通信与计算的能量守恒，不包含推进、悬停、姿态或机载传感器能耗。初始能量、参考 CPU 和参考速率必须在 Gate P pilot 后校准，不能把未校准数值直接解释为实际硬件功耗。

若联合动作的预留能量不可行，执行器只在预定义离散档位中降级，降档序列为：

$$
1.0\rightarrow0.5\rightarrow0.25\rightarrow0.
$$

执行顺序是先按预留能量上界屏蔽明显不可行的单分支档位，再对联合通信—CPU 组合重新计算预留能量；若仍不可行，按固定优先规则降低通信功率或 CPU 频率；若所有非零组合均不可行，则相应分支执行 idle，并记录降档原因、次数以及执行前后的动作。该规则不把动作映射到离散档位集合之外的连续值。

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
- 已绑定到 local 或 remote 的任务不得再次进入 route 候选；route 候选始终只来自 $Q_i^{\mathrm{unb}}(t)$；
- 没有传输队列时，tx、resource group、width 和 power 均固定为 idle；
- 第五组不能与宽度 2 组合；
- 没有本地或远程 CPU 队列时，CPU queue 和 CPU frequency 固定为 idle；
- 按预留能量上界明显超过剩余能量的离散档位提前 mask；
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

CPU 与无线通信模块被视为并行资源，因此半双工约束不限制 CPU 和无线动作之间的并行性。联合执行器收集所有非 idle 候选传输边，按 EDF slack、估计链路质量和固定 UAV 标识的可复现优先级排序，依次接受不产生半双工冲突的边；冲突边退化为通信 idle，并记录冲突与修正次数；退化后的 $y_{ij}(t)=0$，不生成 outage 样本。

## 2.13 时隙事件顺序

每个 episode 在时隙 $0$ 开始前先执行 reset：初始化 $\mathbf p_i(0)$、$\mathbf v_i(0)$ 及其他移动状态，设置 $\Delta s_{ij}(0)=0$，直接采样 $X_{ij}(0)$，并依据初始环境状态生成首个真实信道样本 $h_{ij,r}(0)$；随后按 2.8.5 计算 $t=0$ 的 CSI 历史索引、可用性 mask 和陈旧 CSI 特征。reset 不生成 pre-episode 信道历史，也不定义 $\mathbf p_i(-1)$、$X_{ij}(-1)$ 或 $h_{ij,r}(-1)$。

每个时隙严格按以下顺序执行，以保持任务、bit、cycle 和能量的因果一致性。槽初读取已有队列，槽内完成服务，槽末统一处理状态更新和事件入队：

1. 槽初读取已有任务、四类队列、候选邻居、陈旧 CSI、历史干扰摘要和延迟消息；历史到达率估计按 2.4 节仅使用 $A_i(0),\ldots,A_i(t-1)$ 及对应的 $m_i^\lambda(t)$，$t=0$ 使用固定缺省值和无效 mask；
2. actor 根据槽初局部观测生成七分支动作提案；
3. 对一个未绑定 EDF 任务执行 route 决策；
4. 从槽初已有的非空传输队列中执行 tx_select；
5. 解析固定资源组、资源宽度和候选功率档位，形成 $\mathcal S_i^{\mathrm{prop}}(t)$ 与候选 $p_i(t)$；此时 $y_{ij}(t)$、$x_{ij,r}(t)$ 和 $\mathcal S_i^{\mathrm{exec}}(t)$ 尚未形成；
6. 选择一个槽初 CPU 队列和 CPU 频率档位；
7. 联合执行器先依据 $\mathcal S_i^{\mathrm{prop}}(t)$、候选 $p_i(t)$、槽初传输队列数据和 CPU 预留计算通信及联合主动能量预留，并检查硬可行性；该阶段不读取 $y_{ij}(t)$、$x_{ij,r}(t)$ 或 $\mathcal S_i^{\mathrm{exec}}(t)$；
8. 通过预留检查后，联合执行器完成半双工等硬约束仲裁和既有功率/频率降档，确定最终 $y_{ij}(t)$、$x_{ij,r}(t)$ 和 $p_i(t)$，再构造 $\mathcal S_i^{\mathrm{exec}}(t)$；随后环境结合当前真实槽级信道计算当前真实干扰、SINR、有效速率、通信服务量和通信实际活跃时间；
9. 执行本地或远程 CPU 服务并计算 CPU 实际活跃时间；
10. 更新任务剩余 bit、剩余 cycle，并计算和扣除实际主动能耗；随后仅按 $\chi_{ij}^{\mathrm{att}}(t)$ 生成单次 outage 样本 $o_{ij}(t)$，并释放未实际消耗的预留；
11. 在本槽服务量和剩余工作量更新完成后，结算按时完成任务和过期任务；
12. 将本槽 route 任务、本槽传输完成任务和新到达任务在槽末入队；新到达任务数记为 $A_i(t)$，并记录 $t_n^{\mathrm{arr}}=t$，在下一时隙首次可 route。本槽 route 决策在此时生效并锁定目的地，绑定为 local 的 route 任务同步设置 $d_n^{\mathrm{dst}}\leftarrow i_n$ 和 $D_n^{\mathrm{rem}}\leftarrow0$，绑定为远程 UAV $j$ 的 route 任务同步设置 $d_n^{\mathrm{dst}}\leftarrow j$（$j\ne i_n$），本槽传输完成任务在下一时隙首次可计算；$A_i(t)$ 不影响时隙 $t$ 的 actor 观测，最早在时隙 $t+1$ 槽初进入历史到达率或队列观测；
13. 时隙结束后，对环境规定可观测的接收端形成 $I_{j,r}^{\mathrm{meas}}(t)$，供未来时隙更新历史干扰摘要；无可用测量时保持上一历史值并增加其 AoI，已有历史值仍保持可用性 mask，只有 episode 尚无任何历史测量时 mask 为 $0$。
14. 更新外生移动、真实信道、CSI AoI 和消息 AoI，并进入下一槽；这些更新从已定义的 $t\ge0$ 状态开始，不引入负时隙索引。

该顺序带来三条不可绕过的可用性规则：新 route 任务不能在本槽由 tx_select 服务；本槽传输完成任务不能在本槽计算；槽末生成的新任务不能在本槽再次参与 route。完成和过期结算均发生在本槽服务量及剩余工作量更新之后，且 deadline 时隙仍允许服务。因而任何算法都共享相同的服务边界，不能通过改变网络输出顺序获得额外的槽内服务。

时间轴示例：时隙 $t$ 槽末任务到达，记录 $t_n^{\mathrm{arr}}=t$；时隙 $t+1$ 任务首次可 route，槽末完成绑定；时隙 $t+2$ 首次可接受传输或 CPU 服务。若在时隙 $t+2$ 槽末完成，则 $t_n^{\mathrm{cmp}}=t+2$，端到端时延为 $((t+2)-t)\Delta t=2\Delta t$。

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

局部观测至少包括自身剩余能量比例、归一化资源能力、历史到达率估计 $\hat{\lambda}_i(t)$ 及其可用性 mask $m_i^\lambda(t)$、本地/传输/CPU 队列摘要、候选传输队列的任务数与剩余 bit、CPU 队列的任务数与剩余 cycle、队首 slack、上一槽动作和资源利用率；邻居特征包括相对位置、相对速度、广播剩余能量、广播资源能力、CPU 负载摘要和消息 AoI；边特征包括估计距离、陈旧期望链路增益或其归一化形式、历史干扰链路质量代理量 $\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)$ 或固定资源组聚合值、上一槽有效速率、仅基于过去实际传输尝试的 outage 率、CSI AoI、历史干扰摘要 AoI、相对速度和特征可用性 mask。上一槽有效速率只表示已结束时隙的实际结果；未调度链路没有该观测时使用缺省值加 mask，NA 不视为失败；outage 率只使用 $\tau\le t-1$ 的 $\chi_{ij}^{\mathrm{att}}(\tau)=1$ 样本，不读取当前时隙尚未产生的真实 SINR 或执行结果。

actor 不得读取以下信息：

- 真实建筑物遮挡标签；
- 真实瞬时信道 $h_{ij,r}(t)$；
- 当前真实 $\mathrm{SINR}_{ij,r}(t)$ 和当前真实干扰；
- 当前其他 UAV 尚未确定的资源、功率及执行器接受结果；
- 未来位置或未来移动轨迹；
- 真实业务到达参数 $\lambda_i$；
- 当前时隙槽末才生成的到达量 $A_i(t)$；
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
\max\{1,t_n^{\mathrm{ddl}}-t+1\}
}.
$$

其中 $\mathcal T_t^{\mathrm{active}}$ 是尚未处于 done 或 expired 的任务集合。本地绑定任务因满足 $D_n^{\mathrm{rem}}=0$，不计入通信剩余工作量。设 $N_{\mathrm{on}}(t)$ 为本槽按 deadline 完成的任务数，$N_{\mathrm{exp}}(t)$ 为本槽结算的过期任务数，$E^{\mathrm{act}}(t)$ 为全体 UAV 主动能耗，则团队即时奖励定义为：

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
\overline E_i^{\mathrm{act}}(t)\le E_i^{\mathrm{res}}(t),\ \forall i,\\
\text{每个发送端至多一条服务边},\\
\text{半双工约束成立},\\
\text{每个 CPU 至多一个队列}
\end{array}
\right\}.
$$

动作 mask 处理可由局部信息提前识别的非法分支；联合执行器处理必须看到联合候选动作后才能判断的半双工和联合能量冲突。两者共同保证执行结果属于 hard-feasible 集合，但不把非法动作伪装成连续资源值。

能量硬约束在执行前使用预留主动能量上界；服务完成后，实际能耗满足 $E_i^{\mathrm{act}}(t)\le\overline E_i^{\mathrm{act}}(t)$，未使用的预留能量释放。奖励中的能耗项仍使用实际主动能耗 $E^{\mathrm{act}}(t)$，不使用预留能量。

## 2.15 假设、局限性与实现映射

### 2.15.1 主要建模假设

1. UAV 高度固定，水平移动由外生 Gauss-Markov 过程产生，所有算法使用相同移动轨迹。
2. 任务输入不可拆分到多个目的 UAV，不允许多跳转发；目的地在 route 决策生效并进入 local 或 tx 队列时锁定。
3. 传输和 CPU 服务跨时隙持续，槽末入队的任务下一时隙才可服务。
4. 计算模块与无线模块可以并行，但无线模块满足联合半双工。
5. 主通信资源由五个固定资源组组成，资源组宽度只取一个或两个相邻组，功率和 CPU 频率只取离散档位。
6. 建筑物以三维柱体表示，遮挡通过线段与建筑物的几何相交判定。
7. 轻量信道模型仅服务于趋势评价；真实信道用于环境转移，actor 使用陈旧期望链路信道估计、历史干扰摘要、历史干扰链路质量代理量、相应 AoI 和可用性 mask。
8. episode reset 直接初始化 $\mathbf p_i(0)$、$\mathbf v_i(0)$、$X_{ij}(0)$ 和 $h_{ij,r}(0)$，并以 $\Delta s_{ij}(0)=0$ 作为初始位移约定；不生成或访问负时隙历史，CSI 不可用时使用缺省值与 mask。
9. 初始能量、参考速率、deadline 和路径损耗参数在 Gate P pilot 后校准。
10. 主动能耗只包含按实际传输活跃时间和 CPU 动态活跃时间统计的无线发射与 CPU 动态能耗，不包含推进和其他飞行器功耗；预留能量只用于执行前的硬可行性检查，不作为实际能耗指标。
11. 结果回传暂不建模，远程 CPU 完成即视为任务计算完成。

### 2.15.2 局限性和声明边界

本模型适合研究动态异构、单跳 U2U-MEC、离散资源编排和截止期队列之间的耦合机制，但不应被解释为完整 UAV 通信系统的数字孪生。固定高度和外生移动排除了轨迹控制与飞行动力学影响；轻量物理层不提供实测信道或实时射线追踪级别的真实性保证；未建模结果回传、完整协议栈、HARQ、MIMO、RIS、DAG 任务和多跳路由；第三 UAV 软遮挡及高保真传播回放只能作为后续扩展。性能结论必须限定在已实现的环境、参数校准范围、测试场景和统计协议内。

本章的公式定义服务接口、约束和评价量，不证明任务完成率的理论最优边界，不证明策略对未测试规模或未测试分布的泛化，也不把经验趋势写成定理。任何数值、曲线、显著性或算法优越性结论都必须留给真实程序运行后的结果章节。

历史干扰摘要是接收端经既有控制消息广播的抽象槽级控制信息，不展开完整控制信令协议、测量误差硬件或信令能耗；$\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)$ 是历史干扰链路质量代理量，不代表当前真实 SINR。计划测试应覆盖当前干扰信息泄漏、历史量只使用 $t-1$ 或更早数据、无历史时的缺省值与 mask、AoI 更新以及 actor 观测不含当前真实 SINR 等信息边界，但这些测试目前尚未实现，也不构成已完成的测试结果。

### 2.15.3 系统规则到代码和测试的映射

下表只描述后续实现与测试应覆盖的接口，不声称这些模块当前已经实现。由于项目初始化阶段不编写系统实现代码，本表中的文件名是规划映射。本节的时间边界不变量是：槽末到达的任务下一槽可 route，槽末绑定或转队的任务下一槽可服务，deadline 时隙服务结束后再结算。计划测试应覆盖这些 off-by-one 与槽边界用例；以下仅为测试要求，不代表测试已经实现。

outage 口径的计划测试至少包括：

- proposed send 被执行器拒绝时，outage 为 NA；
- actor 主动选择通信 idle 时，outage 为 NA；
- 零功率或零实际资源占用时，outage 为 NA；
- 实际非零功率、非零资源且队列非空，但有效速率为零时，outage 为 1；
- 成功实际传输时，沿用既有判定得到 outage 为 0；
- 累计指标的分母只包含实际传输尝试；
- 没有实际传输尝试时，累计 outage 为 NA；
- 历史 outage 特征只使用 $t-1$ 或更早的实际传输尝试样本。
- SINR 门限量纲测试：确认 $\gamma_{\min}^{\mathrm{dB}}=-3\ \mathrm{dB}$ 转换为 $\gamma_{\min}^{\mathrm{lin}}\approx0.5012$，并覆盖线性 SINR 等于、略低于和略高于该值时的有效速率门控与 outage 判定边界；

历史到达率的计划测试至少包括：

- $t=0$ 时验证 $K_\lambda(0)=0$、$\hat{\lambda}_i(0)=0$ 和 $m_i^\lambda(0)=0$；
- $t=1$ 时只使用 $A_i(0)$，不访问负时隙索引；
- $1<t<W_\lambda$ 时只使用全部已存在的历史样本，不包含 $A_i(t)$；
- $t\ge W_\lambda$ 时只使用最近 $W_\lambda$ 个已结束时隙的到达量；
- 修改尚未在槽末生成的 $A_i(t)$ 不得改变时隙 $t$ 的 actor 观测；
- $A_i(t)$ 最早只能影响时隙 $t+1$ 的历史到达率或队列观测；
- 缺省值 $0$ 必须始终与 $m_i^\lambda(t)=0$ 的不可用标志配对，不能被解释为真实零到达率。

物理层初始时刻的计划测试至少包括：

- $t=0$ 时 reset 直接提供 $\mathbf p_i(0)$ 和 $\mathbf v_i(0)$，不得读取 $\mathbf p_i(-1)$；
- $\Delta s_{ij}(0)=0$，且 $t\ge1$ 时才使用相邻时隙位置计算位移；
- $X_{ij}(0)$ 直接从 $\mathcal N(0,\sigma_s^2)$ 采样，不读取 $X_{ij}(-1)$；
- 阴影递推仅从 $t=1$ 开始；
- 对任意 $a_{ij}^{\mathrm{CSI}}(t)>t$，不得访问负历史索引；
- CSI 不可用时，缺省值与 $m_{ij}^{\mathrm{CSI}}(t)=0$ 配套；
- 当 $\ell_{ij}^{\mathrm{CSI}}(t)=0$ 时，首次可读取已生成的 $h_{ij,r}(0)$；
- 缺省 CSI 不得被解释为真实零信道，且不得通过当前真实信道回填历史；
- actor 观测不包含当前时隙的真实 $h_{ij,r}(t)$。

| 系统规则 | 形式化对象或不变量 | 计划实现位置 | 计划测试或 Gate |
|---|---|---|---|
| 任务记录与生命周期 | $\tau_n$、$\mathcal S_{\mathrm{task}}$、$\mathcal T_{\mathrm{task}}$；unbound→local/tx 时一次性绑定目的地，且 local、tx、cpu 状态下目的地不可变 | src/env/tasks.py | 目的地标记与任务生命周期测试，Gate 0 |
| bit 守恒 | local 绑定后 $D_n^{\mathrm{rem}}=0$；远程任务按实际传输服务递减且不为负 | src/env/queues.py | local 绑定清零与远程 bit 守恒测试，Gate 0 |
| cycle 守恒 | $C_n^{\mathrm{rem}}$ 按实际 CPU 服务递减且不为负 | src/env/queues.py | cycle 守恒测试，Gate 0 |
| 状态机合法性 | 只允许 $\mathcal T_{\mathrm{task}}$ 中的转移 | src/env/tasks.py | 零非法状态转移测试，Gate 0 |
| route/tx 分离 | route 仅操作 $Q_i^{\mathrm{unb}}$ 并一次性绑定；tx_select 仅操作 $Q_{i\rightarrow j}^{\mathrm{tx}}$ 且不改变目的地 | src/env/action_space.py，src/env/queues.py | 同槽因果测试，Gate 0 |
| 槽边界 | 新路由、新到达和传输完成任务下一槽才可服务 | src/env/u2u_mec_env.py | 槽序列测试，Gate 0 |
| 固定高度 | $\mathbf q_i(t)=[x_i(t),y_i(t),H]^{\mathrm T}$ | src/env/mobility.py | 高度不变测试，Gate 0 |
| 动态候选邻居 | $\mathcal N_i(t)$ 与三维距离阈值 | src/env/topology.py | 邻居边界测试，Gate 1 |
| 建筑物遮挡 | 存在量词几何相交判定 | src/env/buildings.py | 遮挡几何测试，Gate 1 |
| 固定资源组 | $\mathcal G_1,\ldots,\mathcal G_5$ 及宽度 mask | src/env/resource_groups.py | 资源组边界测试，Gate 0 |
| SINR 与有效速率 | $\mathrm{SINR}_{ij,r}$、$R_{ij}^{\mathrm{eff}}$ | src/env/channel.py | 干扰和速率单元测试，Gate 1 |
| outage 统计 | 仅对 $y_{ij}=1$、$p_i>0$、实际资源占用非空且队列剩余 bit 大于零的实际传输尝试统计；其余为 NA | src/env/channel.py，src/evaluation/ | outage 口径测试，Gate 1 |
| 能量守恒 | $\overline E_i^{\mathrm{act}}(t)\le E_i^{\mathrm{res}}(t)$；$E_i^{\mathrm{res}}(t+1)=E_i^{\mathrm{res}}(t)-E_i^{\mathrm{act}}(t)$ | src/env/energy.py | 预留硬约束、实际能耗守恒与非负测试，Gate 0 |
| 实际服务与能耗 | $\tau_{ij}^{\mathrm{tx}}$、$\tau_i^{\mathrm{cpu}}$；提前完成后按实际活跃时间扣除，outage 非零功率尝试按整槽计 | src/env/energy.py，src/env/queues.py | 提前完成能耗、outage 整槽发射尝试、实际能耗不超过预留、剩余能量按实际能耗更新，均为计划测试，Gate 0 |
| 离散能量降档 | $1.0\rightarrow0.5\rightarrow0.25\rightarrow0$ | src/env/energy.py，src/agents/action_mask.py | 档位闭集测试，Gate 0 |
| 联合半双工 | 同一 UAV 不得同时发送和接收 | src/env/half_duplex_resolver.py | 冲突解析测试，Gate 0 |
| 信息权限 | actor 只使用 $o_i(t)$，禁止读取真实遮挡、当前真实信道/SINR/干扰、当前联合动作结果、未来位置、真实 $\lambda_i$ 和完整私有队列 | src/agents/observation.py | 信息泄漏、历史量时序、缺省值与 mask、AoI 更新测试，Gate 0 |
| Gate P 参数校准 | 链路预算、deadline 与能量处于非退化区间 | configs/ 与 experiments/（待建） | Gate P pilot |
| Gate 0 环境正确 | 任务、bit、cycle、energy 守恒；route/tx 分离；槽因果正确 | tests/（待建） | Gate 0 |

### 2.15.4 本章与后续章节的关系

本章是系统对象、符号、状态机、队列、物理层评价、动作接口和信息权限的唯一主源。后续方法章节只能在这些接口上说明策略网络、训练流程、集中训练和分散执行方式，不得重新定义任务状态、route/tx_select 语义、固定资源组、半双工、离散降档或 actor 信息边界。实验协议章节只能使用本章的参数状态、Gate P/Gate 0 和评价口径组织校准、训练、测试、统计与复现，不得通过实验设置改变系统规则。

在编码实现前，必须先以不调用学习器的完整 episode 验证任务、bit、cycle 和能量守恒、零非法状态转移、槽边界、outage 统计、离散降档和联合半双工；只有 Gate 0 通过后，才可将本章定义的环境接口连接到后续策略和价值模块。



