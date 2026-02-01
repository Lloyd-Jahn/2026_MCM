### Task 4

一、核心目标\
构建四目标优化模型（成本最小化、时效最大化、可靠性最大化、环境影响最小化），完整覆盖纯电梯（a）、纯火箭（b）、混合方案（c）三大场景，量化各场景对地球环境的全维度影响，给出可落地的模型调整策略与分场景最优解，支撑长期运营决策。

二、核心四目标体系\
目标 1：成本最小化\
目标 2：时效最大化，“最小化时间”\
目标 3：可靠性最大化	可靠性 = 1 - 运输中断概率（中断概率 = 故障概率 × 故障时长 / 年运营时长）\
目标 4：环境影响最小化	环境影响综合指数 = 加权求和（大气污染、碳排放、资源消耗、生态破坏）\
*** 加权合并为单目标函数，权重确定：AHP/topsis

三、通用参数回顾
1. M=$10^8$ t 为建设总材料量
2. Mw = $3.05125×10^5$ t 为总需运输水量（1 年内）
3. Mtotal = M + Mw 为总运输量（建筑材料 + 水）
4. T oper = 10 年（3650 天）为长期运营周期，数字可以更改，以 10 年为例
5. Q R = 100-150 t 为每个发射场的火箭有效载荷
6. Q E = 1.79×$10^5$ t/年 为单银河港口年运力
7. p E1 = 5% 为系绳摇摆概率
8. β E1 = 0.3 为运力折减系数
9. p E2 = 3% 为太空电梯攀爬器故障概率
10. p R1 = 3% 为火箭发射失败率
11. p R2 = 4% 为发射场维护概率
12. t E2 = 30 天/次 为攀爬器故障持续时间
13. t R2 = 60 天/场，为发射场维护持续时间
14. C E,f为太空电梯固定成本，C E,u为太空电梯单位运输成本
15. C E,repair = 5000 万美元/次
16. C R,l为每个发射场火箭的固定成本
17. c R,i为每个发射场火箭单位运输成本

四、场景a
1. 决策变量\
（1）s E,elec：电梯电力来源占比（可再生能源占比），0.8~1.0\
（2）m E：电梯年维护频率（单位：次 / 港 / 年），整数
2. 目标函数\
目标 1：总运输成本：\
$C_a = 3 \times C_{E,f} \times \frac{T_a}{365} + M_{\text{total}} \times C_{E,u} + 3 \times p_{E2} \times e^{-0.2 \times m_E} \times \frac{T_a}{365} \times C_{E,\text{repair}}$
其中：\
Ta/365是将运输时间（天）折算为年数，因C E,f和 “年维护频率m E” 均以年为单位，确保成本维度一致\
m E 越高，故障概率修正因子 e^(−0.2×m E) 越小，修复成本越低\
目标 2：平均运输时间\
$T_a = \frac{M_{\text{total}}}{Q_{E,\text{eff}}} \times \left(1 + p_{E2} \times e^{-0.2 \times m_E} \times \frac{t_{E2}}{365}\right)$
$Q_{E,\text{eff}} = Q_{E,\text{total}} \times \left(1 - p_{E1} \times \beta_{E1}\right) \times \left(1 - p_{E2} \times \frac{t_{E2}}{365}\right)$
逻辑：总运输时间 = 总运输量 / 有效运力 ×（1 + 故障延误率）\
目标 3：系统可靠性\
$R_a = 1 - \frac{p_{E2} \times e^{-0.2 \times m_E}}{3} \times \frac{t_{E2}}{365}$
逻辑：电梯港固定 3 个，冗余由固定数量提供，降低故障概率，共同提升可靠性\
目标 4：环境影响综合指数\
$EI_a = \omega_{PM2.5} \times \frac{E_{PM2.5}^a}{E_{PM2.5}^{\max}} + \omega_{CO2} \times \frac{E_{CO2}^a \times \left(1 - s_{E,\text{elec}}\right)}{E_{CO2}^{\max}} + \omega_{reso} \times \frac{E_{reso}^a}{E_{reso}^{\max}} + \omega_{eco} \times \frac{E_{eco}^a}{E_{eco}^{\max}}$
（符号说明见群）\
***\
成本归一化：$F_1^k = \frac{C_k}{C_{\max}}$\
时效归一化：$F_2^k = \frac{T_k}{T_{\max}}$\
可靠性归一化：$F_3^k = 1 - \frac{R_k}{R_{\max}}$\
环境影响归一化：$F_4^k = \frac{EI_k}{EI_{\max}}$\
$\min F_k = \omega_1 \times F_1^k + \omega_2 \times F_2^k - \omega_3 \times F_3^k + \omega_4 \times F_4^k$\
k为场景a/b/c\
权重确定：AHP/topsis
3. 约束条件\
（1）运力约束：Q E,eff ≥ 0\
（2）维护约束：m E ≥ 1\
（3）环境约束：s E,elec ∈ (0.8,1.0)\
（4）时间约束：Ta > 0

五、场景b
1. 决策变量\
（1）y i∈{0,1}（i=1..10），i=1表示选中第i个发射场\
（2）f R,fuel：燃料类型修正系数（传统煤油 = 1.0，绿色燃料 = 0.6），0.6~1.0\
（3）x i（i=1..10），表示i个发射场的年发射次数（未选中时x i=0）
2. 目标函数\
目标 1：总运输成本\
$C_b = \left(\sum_{i=1}^{10} y_i \times C_{R,l}\right) \times \frac{T_b}{365} + N'_{R,\text{total}} \times \left(C_{R,i} + 2000 \times f_{R,\text{fuel}} \times Q_R / 1000\right) + \sum_{i=1}^{10} \left( y_i \times p_{R1} \times s_{R,\text{safe},i} \times \frac{T_b}{365} \times C_{R,\text{repair}} \right)$
其中：\
s R,safe,i 是第 i 个发射场的安全系数（低风险 = 1.0，高风险 = 1.5），由 y i 选择的发射场类型决定\
目标 2：平均运输时间\
$T_b = \frac{M_{\text{total}}}{\sum_{i=1}^{10} y_i \times x_i \times Q_R \times \left(1 - p_{R1} \times s_{R,\text{safe},i}\right)} \times \left(1 + p_{R1} \times \overline{s_{R,\text{safe}}} \times \frac{t_{R2}}{365}\right)$
$\overline{s_{R,\text{safe}}}$ 是选中发射场的平均安全系数\
目标 3：系统可靠性\
$R_b = 1 - p_{R1} \times \overline{s_{R,\text{safe}}} \times e^{-0.15 \times \sum y_i} \times \frac{t_{R2}}{365}$
目标 4：环境影响综合指数\
$EI_b = \omega_{PM2.5} \times \frac{E_{PM2.5}^b}{E_{PM2.5}^{\max}} + \omega_{CO2} \times \frac{E_{CO2}^b \times f_{R,\text{fuel}}}{E_{CO2}^{\max}} + \omega_{reso} \times \frac{E_{reso}^b \times f_{R,\text{fuel}}}{E_{reso}^{\max}} + \omega_{eco} \times \frac{E_{eco}^b \times \overline{s_{R,\text{safe}}}}{E_{eco}^{\max}}$
***\
成本归一化：$F_1^k = \frac{C_k}{C_{\max}}$\
时效归一化：$F_2^k = \frac{T_k}{T_{\max}}$\
可靠性归一化：$F_3^k = 1 - \frac{R_k}{R_{\max}}$\
环境影响归一化：$F_4^k = \frac{EI_k}{EI_{\max}}$\
$\min F_k = \omega_1 \times F_1^k + \omega_2 \times F_2^k - \omega_3 \times F_3^k + \omega_4 \times F_4^k$\
k为场景a/b/c\
权重确定：AHP/topsis
3. 约束条件\
（1）发射场数量约束：$\sum_{i=1}^{10} y_i \le 10$\
（2）发射次数约束：未选中的发射场发射次数为 0，选中的不超过年最大次数 $0 \le x_i \le x_{\text{max},i} \times y_i \quad (i=1..10)$\
（3）运力约束：$\sum_{i=1}^{10} y_i \times x_i \times Q_R \times \left(1 - p_{R1} \times s_{R,\text{safe},i}\right) \times \frac{T_b}{365} \ge M_{\text{total}}$\
（4）环境约束：f R,fuel ∈ (0.6,1.0)\
（5）0-1 约束：y i∈{0,1}

六、场景c
1. 决策变量\
（1）α，太空电梯运输占比，0≤α≤1\
（2）y i∈{0,1}（i=1..10），i=1表示选中第i个发射场\
（3）m E：电梯年维护频率（单位：次 / 港 / 年），整数
2. 目标函数\
目标 1：总运输成本\
$C_c = \left(3 \times C_{E,f} + \sum_{i=1}^{10} y_i \times C_{R,l}\right) \times \frac{T_c}{365} + \alpha \times M_{\text{total}} \times C_{E,u} + \frac{(1-\alpha) M_{\text{total}}}{Q_R \times (1 - p_{R1})^2} \times C_{R,i}$\
逻辑：电梯港固定 3 个，火箭发射场由 y i 选择，固定成本按 Tc/365 折算\
目标 2：总运输时间\
$T_E = \frac{\alpha \times M_{\text{total}}}{Q_{E,\text{eff}} / 365}$\
$T_R = \frac{(1 - \alpha) \times M_{\text{total}}}{Q'_{R,\text{total}} / 365}$\
Q R,total′为火箭总有效运力，$Q'_{R,\text{total}} = \sum_{i=1}^{10} \left[ y_i \times x_i \times Q_R \times (1 - p_{R1}) \times \left(1 - p_{R2} \times \frac{t_{R2}}{365}\right) \right]$\
$T_c = \max(T_E, T_R)$\
目标 3：系统可靠性\
$R_c = \alpha \times \left(1 - \frac{p_{E2} \times e^{-0.2 \times m_E}}{3} \times \frac{t_{E2}}{365}\right) + (1 - \alpha) \times \left(1 - p_{R1} \times e^{-0.15 \times \sum y_i} \times \frac{t_{R2}}{365}\right)$\
目标 4：环境影响综合指数\
$EI_c = \omega_{PM2.5} \times \frac{(1-\alpha) \times E_{PM2.5}^b}{E_{PM2.5}^{\max}} + \omega_{CO2} \times \frac{\alpha \times E_{CO2}^a + (1-\alpha) \times E_{CO2}^b}{E_{CO2}^{\max}} + \omega_{reso} \times \frac{\alpha \times E_{reso}^a + (1-\alpha) \times E_{reso}^b}{E_{reso}^{\max}} + \omega_{eco} \times \frac{\alpha \times E_{eco}^a + (1-\alpha) \times E_{eco}^b \times \overline{s_{R,\text{safe}}}}{E_{eco}^{\max}}$\
***\
成本归一化：$F_1^k = \frac{C_k}{C_{\max}}$\
时效归一化：$F_2^k = \frac{T_k}{T_{\max}}$\
可靠性归一化：$F_3^k = 1 - \frac{R_k}{R_{\max}}$\
环境影响归一化：$F_4^k = \frac{EI_k}{EI_{\max}}$\
$\min F_k = \omega_1 \times F_1^k + \omega_2 \times F_2^k - \omega_3 \times F_3^k + \omega_4 \times F_4^k$\
k为场景a/b/c\
权重确定：AHP/topsis
3. 约束条件\
（1）发射场约束：未选中的发射场发射次数为 0，选中的不超过年最大次数 $0 \le x_i \le x_{\text{max},i} \times y_i \quad (i=1..10)$\
（2）发射场数量约束：$\sum_{i=1}^{10} y_i \le 10$\
（3）运力合理性约束：$Q_{E,\text{eff}} \times \frac{T_c}{365} \ge \alpha \times M_{\text{total}}$ 且 $Q'_{R,\text{total}} \times \frac{T_c}{365} \ge (1 - \alpha) \times M_{\text{total}}$\
（4）0-1 约束：y i∈{0,1}（i=1..10）\
（5）运输占比约束：0≤α≤1

七、三种场景比较\
对比三种场景的目标函数，选择最小的场景