### Task 2

一、核心目标\
量化 “系绳摇摆、火箭故障、电梯损坏” 等非完美工况对运输方案的影响，通过概率建模 + 鲁棒优化修正基础模型，分析成本、时间的变化幅度，并提出应对策略。

二、关键假设\
故障均为独立随机事件

三、故障量化参数设置
| 运输系统 | 故障类型 | 量化参数 (数值) |
| :--- | :--- | :--- |
| **太空电梯** | 1. 系绳摇摆 | 故障概率 $p_{E1} = 5\%$；运力折减系数 $\beta_{E1} = 0.3$ |
| | 2. 攀爬器故障（停运） | 故障概率 $p_{E2} = 3\%$；故障持续时间 30 天 / 次；修复成本 5000 万美元 / 次 |
| **火箭系统** | 1. 发射失败 | 故障概率 $p_{R1} = 3\%$ |
| | 2. 发射场维护（停运） | 故障概率 $p_{R2} = 4\%$；故障持续时间 60 天 / 场 |

后续会用到这些参数

四、场景a
1. 决策变量\
（1）连续变量：T a有效运输时间（年）\
（2）辅助变量：Q E,eff有效运力（t / 年），E[N E2]攀爬器故障次数期望
2. 目标函数\
$\min Z_a' = \omega_C \times C_a' + \omega_T \times T_a'$
其中：\
权重确定：topsis/AHP\
期望总成本C a′：\
$C_a' = 3 \times C_{E,f} \times T_a' + M \times C_{E,u} + 3 \times C_{E,\text{repair}} \times E[N_{E2}]$
C E,f为太空电梯固定成本，C E,u为太空电梯单位运输成本，C E,repair=5000万美元/次，M=$10^8$ t为建设总材料量\
$E[N_{E2}] = p_{E2} \times T_a'$（3 个港口独立）\
p E2=3%\
有效运力Q E,eff（考虑两种故障叠加）：\
$Q_{E,\text{eff}} = Q_{E,\text{total}} \times (1 - p_{E1} \times \beta_{E1}) \times (1 - p_{E2} \times t_{E2} / 365)$
（$t_{E2} / 365$ 为故障导致的年运力损失比例）
p E1=5%，β E1=0.3，p E2=3%，t E2=30天/次为攀爬器故障持续时间\
有效运输时间： $T_a' = \frac{M}{Q_{E,\text{eff}}}$
3. 约束条件\
（1）有效运力非负约束：Q E,eff>0\
（2）故障次数非负约束：E[N E2]≥0\
（3）时间非负约束：T a′≥0

五、场景b
1. 决策变量\
（1）0-1 变量：\
y i∈{0,1}（i=1..10），i=1表示是否选中第i个发射场\
（2）连续变量：\
x i（i=1..10），表示i个发射场的年发射次数（未选中时x i=0）\
（3）辅助变量：\
Q R,total′为火箭总有效运力，N R,i′为第i个发射场的期望发射总次数，含3% 发射失败补发
2. 目标函数\
$\min Z_b' = \omega_C \times C_b' + \omega_T \times T_b'$
其中：\
权重确定：topsis/AHP\
期望总成本C b′（含发射失败补发成本）：$C_b' = \sum_{i=1}^{10} \left( y_i \times C_{R,l} \times T_b' \right) + \sum_{i=1}^{10} \left( y_i \times N'_{R,i} \times C_{R,i} \right)$
C R,l为每个发射场火箭的固定成本，c R,i为每个发射场火箭单位运输成本\
第i个发射场的期望发射总次数：$N'_{R,i} = M_R \times \left( \frac{y_i \times q_{R,i}}{Q_{R,\text{total\_single}}} \right) \times \frac{1}{q_{R,i}} \times \frac{1}{(1 - p_{R1})}$，推导得：$N'_{R,i} = \left( \frac{M \times y_i}{\sum_{i=1}^{10} \left( y_i \times Q_R \times (1 - p_{R1}) \right)} \right) \times \frac{1}{(1 - p_{R1})}$
解释：
* 项1：$M_R \times \left( \frac{y_i \times q_{R,i}}{Q_{R,\text{total\_single}}} \right)$ 为第i个发射场要承担的运输量，项2：$frac{1}{q_{R,i}}$ 为完美工况下（无发射失败），要完成项 1 的运输量需要发射多少箭，项3 $frac{1}{(1 - p_{R1})}$ 为发射失败补发修正，在非完美工况下，发射会失败（失败的箭无法交付物资），需要多发射箭来 “补发” 失败的部分，确保实际交付的物资满足运输需求，几何分布（见微信群）。
* $Q_{R,\text{total\_single}} = \sum_{j=1}^{10} y_j \times q_{R,j} = \sum_{j=1}^{10} y_j \times Q_R \times (1 - p_{R1})$
p R1=3%，Q R=100-150 t为每个发射场的火箭有效载荷\
火箭总有效运力：$Q'_{R,\text{total}} = \sum_{i=1}^{10} \left[ y_i \times x_i \times Q_R \times (1 - p_{R1}) \times \left(1 - p_{R2} \times \frac{t_{R2}}{365}\right) \right]$
p R1=3%，p R2=4%，t R2=60天/场，为发射场维护持续时间\
有效运输时间：$T_b' = \frac{M}{Q'_{R,\text{total}}}$
3. 约束条件\
（1）发射场数量约束：$\sum_{i=1}^{10} y_i \le 10$
（2）发射次数约束：未选中的发射场发射次数为 0，选中的不超过年最大次数 $0 \le x_i \le x_{\text{max},i} \times y_i \quad (i=1..10)$
（3）有效运力约束：$Q'_{R,\text{total}} \ge \frac{M}{T_b'}$
（4）0-1 约束：y i∈{0,1}\
（5）非负约束：x i≥0，T b′≥0

六、场景c
1. 决策变量\
（1）0-1 变量：\
y i∈{0,1}（i=1..10），i=1表示是否选中第i个发射场\
（2）连续变量：\
α，太空电梯运输占比，0≤α≤1\
（3）连续变量：\
x i（i=1..10），表示i个发射场的年发射次数（未选中时x i=0）\
（4）辅助变量：\
Q E,eff电梯有效运力（t / 年），Q R,total′火箭总有效运力（t / 年），N R,i′第i个发射场的期望发射总次数，含 3% 发射失败补发
2. 目标函数\
$\min Z_c' = \omega_C \times C_c' + \omega_T \times T_c'$
其中：\
权重确定：topsis/AHP\
期望总成本C c′：按顺序拆解为“电梯港口和发射场固定成本 + 电梯运输成本 + 火箭发射成本 + 电梯故障修复成本”\
$C_c' = \left(3 \times C_{E,f} + \sum_{i=1}^{10} y_i \times C_{R,l} \times T_c'\right) + \alpha M \times C_{E,u} + \sum_{i=1}^{10} \left(y_i \times N'_{R,i} \times C_{R,i}\right) + 3 \times C_{E,\text{repair}} \times p_{E2} \times T_c'$
C E,f为太空电梯固定成本，C R,l为每个发射场火箭的固定成本，C E,u为太空电梯单位运输成本，c R,i为每个发射场火箭单位运输成本，C E,repair=5000万美元/次，p E2=3%\
第i个发射场的期望发射总次数： $N'_{R,i} = \frac{(1-\alpha) M \times y_i}{\sum_{j=1}^{10} y_j \times Q_R \times (1 - p_{R1})} \times \frac{1}{(1 - p_{R1})}$
Q R=100-150 t为每个发射场的火箭有效载荷，p R1=3%\
运输时间：太空电梯与火箭运输时间同步（避免物资积压）\
$T_c' = \frac{\alpha M}{Q_{E,\text{eff}}} = \frac{(1-\alpha) M}{Q'_{R,\text{total}}}$
推导得： $\alpha = \frac{Q_{E,\text{eff}}}{Q_{E,\text{eff}} + Q'_{R,\text{total}}}$
火箭总有效运力： $Q'_{R,\text{total}} = \sum_{i=1}^{10} \left[ y_i \times x_i \times Q_R \times (1 - p_{R1}) \times \left(1 - p_{R2} \times \frac{t_{R2}}{365}\right) \right]$
p R1=3%，p R2=4%，t R2=60天/场，为发射场维护持续时间，Q R=100-150 t为每个发射场的火箭有效载荷\
3. 约束条件\
（1）运力同步约束（电梯与火箭运输时间一致）：$T_c' = \frac{\alpha M}{Q_{E,\text{eff}}} = \frac{(1-\alpha) M}{Q'_{R,\text{total}}}$，推导得：$\alpha = \frac{Q_{E,\text{eff}}}{Q_{E,\text{eff}} + Q'_{R,\text{total}}}$
（2）发射场约束：未选中的发射场发射次数为 0，选中的不超过年最大次数 $0 \le x_i \le x_{\text{max},i} \times y_i \quad (i=1..10)$
（3）发射场数量约束：$\sum_{i=1}^{10} y_i \le 10$
（4）0-1 约束：y i∈{0,1}（i=1..10）\
（5）运输占比约束：0≤α≤1\
（6）非负约束：x i≥0,T c′≥0

七、三种场景比较\
对比三种场景的目标函数，选择最小的场景

八，敏感性分析
使用 monte-carlo 方法进行分析