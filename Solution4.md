# Task 4: 环境影响评估与多目标优化模型

## 一、核心目标

构建四目标优化模型（成本最小化、时效最大化、可靠性最大化、环境影响最小化），完整覆盖纯电梯（a）、纯火箭（b）、混合方案（c）三大场景，量化各场景对地球环境的全维度影响，给出可落地的模型调整策略与分场景最优解，支撑长期运营决策。

## 二、核心四目标体系

1. **目标 1：成本最小化**
   最小化总运输成本（固定成本 + 可变成本 + 故障修复成本）

2. **目标 2：时效最大化**
   最小化运输时间，确保月球殖民地建设按期完成

3. **目标 3：可靠性最大化**
   最大化系统可靠性，定义为：
   $$R = 1 - P_{\text{中断}}$$
   其中中断概率 $P_{\text{中断}} = p_{\text{故障}} \times \frac{t_{\text{故障}}}{T_{\text{运营}}}$

4. **目标 4：环境影响最小化**
   最小化环境影响综合指数：
   $$EI = \sum_{j} \omega_j \times E_j$$
   考虑大气污染（PM2.5）、碳排放（CO₂）、资源消耗、生态破坏四个维度

**多目标整合方法**：采用**层次分析法（AHP）**确定权重，通过**加权求和法**将四个目标合并为单一优化目标函数。

---

## 三、通用参数设定

### 3.1 任务参数（继承自 Task 1-3）

| 序号 | 参数符号 | 参数值 | 说明 |
|:---:|:---|:---|:---|
| 1 | $M$ | $10^8$ t | 建设总材料量 |
| 2 | $M_w$ | $3.05125 \times 10^5$ t | 年度水资源需求量 |
| 3 | $M_{\text{total}}$ | $M + M_w$ | 总运输量（材料 + 水） |
| 4 | $T_{\text{oper}}$ | 10 年（3650 天） | 长期运营周期基准 |

### 3.2 运输系统参数

**太空电梯系统**：
| 参数 | 符号 | 数值 | 来源 |
|:---|:---|:---|:---|
| 港口数量 | $N_E$ | 3 | Task 1 |
| 单港口年运力 | $Q_E$ | $1.79 \times 10^5$ t/年 | 题面 |
| 总名义运力 | $Q_{E,\text{total}}$ | $5.37 \times 10^5$ t/年 | $3 \times Q_E$ |
| 固定年成本（每港口） | $C_{E,f}$ | $3.0 \times 10^9$ USD/年 | Task 1 |
| 单位转运成本 | $C_{E,u}$ | $92,700$ USD/t | Task 1 |
| 攀爬器修复成本 | $C_{E,\text{repair}}$ | $5.0 \times 10^7$ USD/次 | Task 2 |

**火箭系统**：
| 参数 | 符号 | 数值 | 来源 |
|:---|:---|:---|:---|
| 单次有效载荷 | $Q_R$ | $150$ t | 题面上限 |
| 直接运输成本 | $c_R$ | $1,500,000$ USD/t | Task 1 |
| 单次发射成本 | $C_{R,i}$ | $2.25 \times 10^8$ USD/次 | $Q_R \times c_R$ |
| 发射场固定成本 | $C_{R,l}$ | 各场不同 | 见 Task 1 表 9.1 |
| 发射场修复成本 | $C_{R,\text{repair}}$ | $1.0 \times 10^8$ USD/次 | 设定值 |

### 3.3 故障参数（继承自 Task 2）

**太空电梯故障模式**：
- **FE1 - 系绳摇摆**：$p_{E1} = 5\%$（年概率），运力折减系数 $\beta_{E1} = 0.3$
- **FE2 - 攀爬器故障**：$p_{E2} = 3\%$（年概率/港口），停运时间 $t_{E2} = 30$ 天/次

**火箭系统故障模式**：
- **FR1 - 发射失败**：$p_{R1} = 3\%$（每次发射）
- **FR2 - 发射场维护**：$p_{R2} = 4\%$（年概率/场），停运时间 $t_{R2} = 60$ 天/次

### 3.4 归一化基准值

| 参数 | 符号 | 数值 | 说明 |
|:---|:---|:---|:---|
| 成本上界 | $C_{\max}$ | $1.5 \times 10^{14}$ USD | Task 1 设定 |
| 时间上界 | $T_{\max}$ | $350$ 年 | Task 1 设定 |
| 可靠性上界 | $R_{\max}$ | $1.0$ | 理想无故障状态 |
| 环境影响上界 | $EI_{\max}$ | 待计算 | 纯火箭方案加权和 |

---

## 四、权重确定方法：层次分析法（AHP）

### 4.1 初步权重设定（专家判断）

**环境影响子权重**（四个环境指标之间）：
$$\omega_{PM2.5} = 0.3, \quad \omega_{CO2} = 0.4, \quad \omega_{\text{eco}} = 0.2, \quad \omega_{\text{reso}} = 0.1$$

**多目标主权重**（四大目标之间）：
$$\omega_1 = 0.3 \text{ (成本)}, \quad \omega_2 = 0.2 \text{ (时效)}, \quad \omega_3 = 0.2 \text{ (可靠性)}, \quad \omega_4 = 0.3 \text{ (环境)}$$

### 4.2 AHP 判断矩阵构建方法

**步骤 1：构建成对比较矩阵**
对于 4 个目标 $\{C, T, R, EI\}$，构建 $4 \times 4$ 判断矩阵 $A$：
$$A = [a_{ij}], \quad a_{ij} = \frac{\text{目标 } i \text{ 的重要性}}{\text{目标 } j \text{ 的重要性}}$$

采用 **Saaty 1-9 标度**：
- $a_{ij} = 1$：两目标同等重要
- $a_{ij} = 3$：目标 $i$ 稍微重要
- $a_{ij} = 5$：目标 $i$ 明显重要
- $a_{ij} = 7$：目标 $i$ 强烈重要
- $a_{ij} = 9$：目标 $i$ 极端重要

### 4.3 AHP 权重计算结果

#### 4.3.1 主准则层权重 (Main Objective Weights)

**判断矩阵**（基于 Saaty 1-9 标度）：

| 评价目标 | 成本 (C) | 时间 (T) | 可靠性 (R) | 环境 (EI) |
|:---|:---:|:---:|:---:|:---:|
| **成本 (C)** | 1.000 | 3.000 | 5.000 | 0.200 |
| **时间 (T)** | 0.333 | 1.000 | 3.000 | 0.143 |
| **可靠性 (R)** | 0.200 | 0.333 | 1.000 | 0.111 |
| **环境 (EI)** | 5.000 | 7.000 | 9.000 | 1.000 |

**判断理由**：
- **环境 vs 其他**：鉴于月球基地建设的长期可持续性，环境影响被赋予最高优先级（极端重要）。
- **成本 vs 时间/可靠性**：成本控制比时间进度和基础可靠性更具决策权重（稍微重要到明显重要）。
- **时间 vs 可靠性**：由于技术成熟度限制，时间进度相对于纯粹的可靠性指标略占优势。

**计算结果**：

| 评价目标 | 符号 | 权重值 | 百分比 |
|:---|:---:|:---:|:---:|
| 成本 | $\omega_1$ | 0.2190 | 21.90% |
| 时间 | $\omega_2$ | 0.1012 | 10.12% |
| 可靠性 | $\omega_3$ | 0.0508 | 5.08% |
| 环境 | $\omega_4$ | 0.6290 | 62.90% |

**一致性检验**：
- 最大特征值：$\lambda_{\max} = 4.2268$
- 一致性指标：$CI = 0.0756$
- 一致性比率：$CR = 0.0840$ （通过检验，$CR < 0.1$）

---

#### 4.3.2 环境子准则权重 (Environmental Sub-Weights)

**判断矩阵**：

| 环境指标 | PM2.5 | CO2 | 生态 (eco) | 资源 (reso) |
|:---|:---:|:---:|:---:|:---:|
| **PM2.5** | 1.000 | 0.200 | 3.000 | 7.000 |
| **CO2** | 5.000 | 1.000 | 5.000 | 9.000 |
| **生态 (eco)** | 0.333 | 0.200 | 1.000 | 3.000 |
| **资源 (reso)** | 0.143 | 0.111 | 0.333 | 1.000 |

**判断理由**：
- **CO2 vs 其他**：碳排放是环境评价的核心指标（明显到极端重要）。
- **PM2.5 vs 生态/资源**：大气污染物排放对封闭基地内循环系统影响显著，重要性较高。
- **生态 vs 资源**：局部生态平衡的维持优先于原材料的消耗速率。

**计算结果**：

| 环境指标 | 符号 | 权重值 | 百分比 |
|:---|:---:|:---:|:---:|
| PM2.5 排放 | $\omega_{PM2.5}$ | 0.1792 | 17.92% |
| CO2 排放 | $\omega_{CO2}$ | 0.6608 | 66.08% |
| 生态影响 | $\omega_{\text{eco}}$ | 0.1087 | 10.87% |
| 资源消耗 | $\omega_{\text{reso}}$ | 0.0513 | 5.13% |

**一致性检验**：
- 最大特征值：$\lambda_{\max} = 4.2144$
- 一致性指标：$CI = 0.0715$
- 一致性比率：$CR = 0.0794$ （通过检验，$CR < 0.1$）

---

#### 4.3.3 权重汇总与目标函数

**最终采用权重**：

**主准则层权重**：
$$\omega_1 = 0.2190, \quad \omega_2 = 0.1012, \quad \omega_3 = 0.0508, \quad \omega_4 = 0.6290$$

**环境子准则权重**：
$$\omega_{PM2.5} = 0.1792, \quad \omega_{CO2} = 0.6608, \quad \omega_{\text{eco}} = 0.1087, \quad \omega_{\text{reso}} = 0.0513$$

**综合目标函数**：
$$\min\, F_k = 0.2190 \times F_1^k + 0.1012 \times F_2^k + 0.0508 \times F_3^k + 0.6290 \times F_4^k$$

其中环境影响项 $F_4^k$ 定义为：
$$F_4^k = \frac{EI_k}{EI_{\max}}, \quad EI_k = 0.1792 \times E_{PM2.5}^k + 0.6608 \times E_{CO2}^k + 0.1087 \times E_{\text{eco}}^k + 0.0513 \times E_{\text{reso}}^k$$

---

## 五、环境影响评估体系

### 5.1 环境参数基准表

以传统煤油火箭方案为基准最大值 ($E^{\max} = E^b$)，纯电梯方案为对比值 ($E^a$)：

| 环境指标 | 符号 | 纯电梯值 $E^a$ | 纯火箭值 $E^b = E^{\max}$ | 单位 | 数据来源 |
|:---|:---|:---:|:---:|:---|:---|
| **PM2.5 排放** | $E_{PM2.5}$ | 0 | 0.85 | kg/t | 电梯零排放，火箭燃烧排放 |
| **CO₂ 排放** | $E_{CO2}$ | 1.2 | 1860 | kg/t | 电梯电力生产，火箭燃料燃烧 |
| **资源消耗** | $E_{\text{reso}}$ | 0.35 | 200 | kg/t | 电梯维护，火箭燃料制造 |
| **生态影响指数** | $E_{\text{eco}}$ | 1.2 | 8.5 | 无量纲 (0-10) | 发射场生态破坏 |

**注**：$E^{\max}$ 为归一化分母，计算环境影响上界 $EI_{\max}$ 时需使用加权和：
$$EI_{\max} = \omega_{PM2.5} \times 1 + \omega_{CO2} \times 1 + \omega_{\text{reso}} \times 1 + \omega_{\text{eco}} \times 1 = 1.0$$

### 5.2 环境影响综合指数计算公式

对于场景 $k \in \{a, b, c\}$，环境影响综合指数定义为：
$$EI_k = \sum_{j} \omega_j \times \frac{E_j^k}{E_j^{\max}}$$

---

## 六、场景 A：纯太空电梯方案

### 6.1 决策变量

| 符号 | 含义 | 取值范围 | 默认值/初始值 |
|:---|:---|:---|:---|
| $s_{E,\text{elec}}$ | 电梯电力可再生能源占比 | $[0.8, 1.0]$ | 0.90 |
| $m_E$ | 电梯年维护频率（次/港口/年） | $\mathbb{Z}^+$ | 2 |

### 6.2 目标函数

**目标 1：总运输成本**
$$C_a = 3 \times C_{E,f} \times \frac{T_a}{365} + M_{\text{total}} \times C_{E,u} + 3 \times p_{E2} \times e^{-0.2 \times m_E} \times \frac{T_a}{365} \times C_{E,\text{repair}}$$

**成本组成**：
- **固定运营成本**：3 个港口 × 固定年成本 × 运营年数
- **可变转运成本**：总运输量 × 单位转运成本（锚点 → 月球）
- **故障修复成本**：期望修复次数 × 单次修复成本
  - 维护频率 $m_E$ 通过指数衰减因子 $e^{-0.2 m_E}$ 降低故障概率

**目标 2：运输时间**
$$T_a = \frac{M_{\text{total}}}{Q_{E,\text{eff}}} \times \left(1 + p_{E2} \times e^{-0.2 \times m_E} \times \frac{t_{E2}}{365}\right)$$

其中，**有效运力**考虑系绳摇摆和攀爬器故障的复合影响：
$$Q_{E,\text{eff}} = Q_{E,\text{total}} \times \underbrace{\left(1 - p_{E1} \times \beta_{E1}\right)}_{\text{摇摆折减}} \times \underbrace{\left(1 - p_{E2} \times \frac{t_{E2}}{365}\right)}_{\text{停运折减}}$$

**目标 3：系统可靠性**
$$R_a = 1 - \frac{p_{E2} \times e^{-0.2 \times m_E}}{3} \times \frac{t_{E2}}{365}$$

**可靠性建模逻辑**：
- 3 个港口提供系统冗余，单港口故障不导致系统完全中断
- 故障概率被维护频率修正（$e^{-0.2 m_E}$）
- 中断概率为调整后故障概率 × 年停运时间占比

**目标 4：环境影响综合指数**
$$EI_a = \omega_{PM2.5} \times \frac{E_{PM2.5}^a}{E_{PM2.5}^{\max}} + \omega_{CO2} \times \frac{E_{CO2}^a \times \left(1 - s_{E,\text{elec}}\right)}{E_{CO2}^{\max}} + \omega_{\text{reso}} \times \frac{E_{\text{reso}}^a}{E_{\text{reso}}^{\max}} + \omega_{\text{eco}} \times \frac{E_{\text{eco}}^a}{E_{\text{eco}}^{\max}}$$

**环境影响建模要点**：
- **PM2.5**：电梯零排放（$E_{PM2.5}^a = 0$）
- **CO₂**：仅来自电力生产，受可再生能源占比 $s_{E,\text{elec}}$ 调节
  - 传统电网 $(s_{E,\text{elec}} = 0)$：全额排放 1.2 kg/t
  - 纯可再生能源 $(s_{E,\text{elec}} = 1.0)$：零排放
- **资源消耗**：电梯维护材料（石墨烯等）
- **生态影响**：港口区域占地，影响较小

### 6.3 归一化与综合目标函数

对四个目标进行归一化处理（统一转化为"越小越好"）：
$$
\begin{aligned}
F_1^a &= \frac{C_a}{C_{\max}} \\
F_2^a &= \frac{T_a}{T_{\max}} \\
F_3^a &= 1 - \frac{R_a}{R_{\max}} \quad &\text{（可靠性取反）} \\
F_4^a &= \frac{EI_a}{EI_{\max}}
\end{aligned}
$$

**综合目标函数**（通过 AHP 权重加权求和）：
$$\min\, F_a = \omega_1 \times F_1^a + \omega_2 \times F_2^a + \omega_3 \times F_3^a + \omega_4 \times F_4^a$$

### 6.4 约束条件

1. **运力正性约束**：$Q_{E,\text{eff}} \geq 0$
2. **维护频率约束**：$m_E \geq 1$（至少年维护 1 次）
3. **可再生能源约束**：$s_{E,\text{elec}} \in [0.8, 1.0]$
4. **时间非负约束**：$T_a > 0$

---

## 七、场景 B：纯火箭方案

### 7.1 决策变量

| 符号 | 含义 | 取值范围 | 说明 |
|:---|:---|:---|:---|
| $y_i$ | 第 $i$ 个发射场选择指示变量 | $\{0, 1\}$ | $y_i = 1$ 表示选中该发射场 |
| $x_i$ | 第 $i$ 个发射场年发射次数 | $[0, x_{\text{max},i}]$ | 未选中时 $x_i = 0$ |
| $f_{R,\text{fuel}}$ | 燃料类型成本修正系数 | $[0.6, 1.5]$ | 传统煤油 = 1.0，绿色燃料 = 1.5 |
| $f_{R,\text{env}}$ | 燃料类型环境修正系数 | $\{0, 1.0\}$ | 传统煤油 = 1.0，绿色燃料 = 0 |

**燃料选择说明**：
- **传统煤油**：成本基准（$f_{R,\text{fuel}} = 1.0$），环境影响全额（$f_{R,\text{env}} = 1.0$）
- **绿色燃料**（液氢/甲烷）：成本增加 50%（$f_{R,\text{fuel}} = 1.5$），环境影响显著降低（$f_{R,\text{env}} = 0$）

### 7.2 发射场安全系数表

| 编号 $i$ | 发射场名称 | 风险类型 | 安全系数 $s_{R,\text{safe},i}$ | 分类依据 |
|:---:|:---|:---:|:---:|:---|
| 1 | 法属圭亚那 | 低风险 | 1.0 | 赤道偏远沿海区，人口极少，生态干扰小 |
| 2 | 美国佛罗里达（肯尼迪） | 低风险 | 1.0 | 沿海空旷区，运营成熟，人口密度低 |
| 3 | 印度萨蒂什·达万 | 低风险 | 1.0 | 沿海偏远区，人口稀疏，生态干扰小 |
| 4 | 新西兰马希亚半岛 | 低风险 | 1.0 | 孤立沿海半岛，人口极少，无生态敏感区 |
| 5 | 美国得克萨斯（博卡奇卡） | 低风险 | 1.0 | 偏远沿海区域，人口密度极低 |
| 6 | 美国阿拉斯加 | 高风险 | 1.5 | 高纬度寒冷区，生态脆弱（冻土带） |
| 7 | 美国加利福尼亚（范登堡） | 高风险 | 1.5 | 靠近加州沿海人口区，发射噪音/污染影响大 |
| 8 | 美国弗吉尼亚（沃洛普斯） | 高风险 | 1.5 | 靠近东海岸人口密集区，生态干扰风险高 |
| 9 | 哈萨克斯坦 | 高风险 | 1.5 | 内陆人口相对密集区，草原生态脆弱 |
| 10 | 中国太原 | 高风险 | 1.5 | 内陆生态敏感区（黄土高原边缘） |

**安全系数影响机制**：
- 低风险发射场（$s_{R,\text{safe},i} = 1.0$）：基准故障率
- 高风险发射场（$s_{R,\text{safe},i} = 1.5$）：故障率提升 50%，反映复杂运营环境

### 7.3 目标函数

**目标 1：总运输成本**
$$C_b = \underbrace{\left(\sum_{i=1}^{10} y_i \times C_{R,l,i}\right) \times \frac{T_b}{365}}_{\text{发射场固定成本}} + \underbrace{N'_{R,\text{total}} \times C_{R,i} \times f_{R,\text{fuel}}}_{\text{发射可变成本}} + \underbrace{\sum_{i=1}^{10} y_i \times p_{R1} \times s_{R,\text{safe},i} \times \frac{T_b}{365} \times C_{R,\text{repair}}}_{\text{故障修复成本}}$$

其中，**期望总发射次数**（考虑 3% 失败率补发）：
$$N'_{R,\text{total}} = \frac{M_{\text{total}}}{Q_R \times (1 - p_{R1})^2}$$

推导依据：几何分布期望（见 Task 2 第 5.2.1 节）

**目标 2：运输时间**
$$T_b = \frac{M_{\text{total}}}{\sum_{i=1}^{10} y_i \times x_i \times Q_R \times \left(1 - p_{R1} \times s_{R,\text{safe},i}\right)} \times \left(1 + p_{R1} \times \overline{s_{R,\text{safe}}} \times \frac{t_{R2}}{365}\right)$$

其中，$\overline{s_{R,\text{safe}}}$ 为选中发射场的平均安全系数：
$$\overline{s_{R,\text{safe}}} = \frac{\sum_{i=1}^{10} y_i \times s_{R,\text{safe},i}}{\sum_{i=1}^{10} y_i}$$

**目标 3：系统可靠性**
$$R_b = 1 - p_{R1} \times \overline{s_{R,\text{safe}}} \times e^{-0.15 \times \sum_{i=1}^{10} y_i} \times \frac{t_{R2}}{365}$$

**可靠性建模逻辑**：
- 发射场数量提供冗余，$e^{-0.15 \sum y_i}$ 表示冗余效应
- 选择更多发射场（$\sum y_i$ 增大）→ 指数项减小 → 可靠性提升

**目标 4：环境影响综合指数**
$$EI_b = \omega_{PM2.5} \times \frac{E_{PM2.5}^b}{E_{PM2.5}^{\max}} + \omega_{CO2} \times \frac{E_{CO2}^b \times f_{R,\text{env}}}{E_{CO2}^{\max}} + \omega_{\text{reso}} \times \frac{E_{\text{reso}}^b \times f_{R,\text{env}}}{E_{\text{reso}}^{\max}} + \omega_{\text{eco}} \times \frac{E_{\text{eco}}^b \times \overline{s_{R,\text{safe}}}}{E_{\text{eco}}^{\max}}$$

**环境影响建模要点**：
- **PM2.5 和 CO₂**：受燃料类型修正系数 $f_{R,\text{env}}$ 调节
  - 传统煤油：全额排放（$f_{R,\text{env}} = 1.0$）
  - 绿色燃料：零排放（$f_{R,\text{env}} = 0$）
- **生态影响**：受发射场平均安全系数 $\overline{s_{R,\text{safe}}}$ 调节
  - 选择低风险发射场 → 生态影响降低

### 7.4 归一化与综合目标函数

$$
\begin{aligned}
F_1^b &= \frac{C_b}{C_{\max}}, \quad F_2^b = \frac{T_b}{T_{\max}} \\
F_3^b &= 1 - \frac{R_b}{R_{\max}}, \quad F_4^b = \frac{EI_b}{EI_{\max}}
\end{aligned}
$$

$$\min\, F_b = \omega_1 \times F_1^b + \omega_2 \times F_2^b + \omega_3 \times F_3^b + \omega_4 \times F_4^b$$

### 7.5 约束条件

1. **发射场数量约束**：$\sum_{i=1}^{10} y_i \leq 10$
2. **发射次数约束**：$0 \leq x_i \leq x_{\text{max},i} \times y_i \quad (i=1,\ldots,10)$
   - 未选中的发射场：$x_i = 0$
   - 选中的发射场：不超过年最大发射次数 $x_{\text{max},i}$（见 Task 1 表 9.1）
3. **运力可行性约束**：
   $$\sum_{i=1}^{10} y_i \times x_i \times Q_R \times \left(1 - p_{R1} \times s_{R,\text{safe},i}\right) \times \frac{T_b}{365} \geq M_{\text{total}}$$
4. **0-1 约束**：$y_i \in \{0, 1\} \quad (i=1,\ldots,10)$

---

## 八、场景 C：混合方案（电梯 + 火箭并行）

### 8.1 决策变量

| 符号 | 含义 | 取值范围 | 初始值 |
|:---|:---|:---|:---|
| $\alpha$ | 太空电梯运输质量占比 | $[0, 1]$ | 0.50 |
| $y_i$ | 第 $i$ 个发射场选择指示变量 | $\{0, 1\}$ | 待优化 |
| $x_i$ | 第 $i$ 个发射场年发射次数 | $[0, x_{\text{max},i}]$ | 待优化 |
| $m_E$ | 电梯年维护频率（次/港口/年） | $\mathbb{Z}^+$ | 2 |

### 8.2 目标函数

**目标 1：总运输成本**
$$
\begin{aligned}
C_c = &\underbrace{\left(3 \times C_{E,f} + \sum_{i=1}^{10} y_i \times C_{R,l,i}\right) \times \frac{T_c}{365}}_{\text{固定成本（电梯 + 火箭）}} \\
&+ \underbrace{\alpha \times M_{\text{total}} \times C_{E,u}}_{\text{电梯转运成本}} + \underbrace{\frac{(1-\alpha) \times M_{\text{total}}}{Q_R \times (1 - p_{R1})^2} \times C_{R,i}}_{\text{火箭发射成本}} \\
&+ \underbrace{3 \times p_{E2} \times e^{-0.2 m_E} \times \frac{T_c}{365} \times C_{E,\text{repair}}}_{\text{电梯修复成本}} + \underbrace{\sum_{i=1}^{10} y_i \times p_{R1} \times s_{R,\text{safe},i} \times \frac{T_c}{365} \times C_{R,\text{repair}}}_{\text{火箭修复成本}}
\end{aligned}
$$

**成本建模逻辑**：
- 电梯和火箭系统同时运行，固定成本叠加
- 可变成本按运输质量占比 $\alpha$ 分配
- 故障修复成本各自独立计算

**目标 2：运输时间（同步约束）**

混合方案要求电梯和火箭**同时完成**运输任务，避免资源闲置：

**电梯运输时间**：
$$T_E = \frac{\alpha \times M_{\text{total}}}{Q_{E,\text{eff}}}$$

**火箭运输时间**：
$$T_R = \frac{(1 - \alpha) \times M_{\text{total}}}{Q'_{R,\text{total}}}$$

其中，火箭总有效运力为：
$$Q'_{R,\text{total}} = \sum_{i=1}^{10} \left[ y_i \times x_i \times Q_R \times (1 - p_{R1}) \times \left(1 - p_{R2} \times \frac{t_{R2}}{365}\right) \right]$$

**同步条件**（两系统同时完工）：
$$T_c = \max(T_E, T_R)$$

最优情况下应满足 $T_E = T_R$，此时：
$$\alpha = \frac{Q_{E,\text{eff}}}{Q_{E,\text{eff}} + Q'_{R,\text{total}}}$$

**目标 3：系统可靠性（加权平均）**
$$R_c = \alpha \times \underbrace{\left(1 - \frac{p_{E2} \times e^{-0.2 m_E}}{3} \times \frac{t_{E2}}{365}\right)}_{R_a} + (1 - \alpha) \times \underbrace{\left(1 - p_{R1} \times \overline{s_{R,\text{safe}}} \times e^{-0.15 \times \sum_{i=1}^{10} y_i} \times \frac{t_{R2}}{365}\right)}_{R_b}$$

**可靠性建模逻辑**：
- 混合方案的可靠性为两系统按质量占比 $\alpha$ 的加权平均
- 反映了两系统的互补效应：单一系统故障时，另一系统可承接部分负载

**目标 4：环境影响综合指数**
$$
\begin{aligned}
EI_c = &\omega_{PM2.5} \times \frac{(1-\alpha) \times E_{PM2.5}^b}{E_{PM2.5}^{\max}} \\
&+ \omega_{CO2} \times \frac{\alpha \times E_{CO2}^a \times (1 - s_{E,\text{elec}}) + (1-\alpha) \times E_{CO2}^b \times f_{R,\text{env}}}{E_{CO2}^{\max}} \\
&+ \omega_{\text{reso}} \times \frac{\alpha \times E_{\text{reso}}^a + (1-\alpha) \times E_{\text{reso}}^b \times f_{R,\text{env}}}{E_{\text{reso}}^{\max}} \\
&+ \omega_{\text{eco}} \times \frac{\alpha \times E_{\text{eco}}^a + (1-\alpha) \times E_{\text{eco}}^b \times \overline{s_{R,\text{safe}}}}{E_{\text{eco}}^{\max}}
\end{aligned}
$$

**环境影响建模逻辑**：
- **PM2.5**：仅来自火箭系统（电梯零排放）
- **CO₂**：电梯部分受可再生能源占比调节，火箭部分受燃料类型调节
- **资源消耗**：按质量占比加权
- **生态影响**：电梯影响较小，火箭影响受发射场选择影响

### 8.3 归一化与综合目标函数

$$
\begin{aligned}
F_1^c &= \frac{C_c}{C_{\max}}, \quad F_2^c = \frac{T_c}{T_{\max}} \\
F_3^c &= 1 - \frac{R_c}{R_{\max}}, \quad F_4^c = \frac{EI_c}{EI_{\max}}
\end{aligned}
$$

$$\min\, F_c = \omega_1 \times F_1^c + \omega_2 \times F_2^c + \omega_3 \times F_3^c + \omega_4 \times F_4^c$$

### 8.4 约束条件

1. **发射场选择约束**：$\sum_{i=1}^{10} y_i \leq 10$
2. **发射次数约束**：$0 \leq x_i \leq x_{\text{max},i} \times y_i \quad (i=1,\ldots,10)$
3. **运力合理性约束**（确保两系统能完成各自分配的运输任务）：
   $$Q_{E,\text{eff}} \times \frac{T_c}{365} \geq \alpha \times M_{\text{total}}$$
   $$Q'_{R,\text{total}} \times \frac{T_c}{365} \geq (1 - \alpha) \times M_{\text{total}}$$
4. **运输占比约束**：$0 \leq \alpha \leq 1$
5. **0-1 约束**：$y_i \in \{0, 1\} \quad (i=1,\ldots,10)$

---

## 九、三场景比较与决策方法

### 9.1 评估流程

对于每个场景 $k \in \{a, b, c\}$：

**步骤 1：求解优化问题**
- **场景 A**：优化决策变量 $(s_{E,\text{elec}}, m_E)$，最小化 $F_a$
- **场景 B**：优化 0-1 组合 $(y_1, \ldots, y_{10})$ 和发射频率 $(x_1, \ldots, x_{10})$，最小化 $F_b$
  - 采用**枚举法**或**遗传算法**求解混合整数规划
- **场景 C**：优化 $(\alpha, y_1, \ldots, y_{10}, x_1, \ldots, x_{10}, m_E)$，最小化 $F_c$

**步骤 2：计算四个原始目标**
得到优化解后，计算：
- 总成本 $C_k$（USD）
- 运输时间 $T_k$（天）
- 系统可靠性 $R_k$（无量纲，0-1）
- 环境影响指数 $EI_k$（无量纲，0-1）

**步骤 3：归一化处理**
计算归一化目标值 $F_1^k, F_2^k, F_3^k, F_4^k$

**步骤 4：综合评分**
计算综合目标函数值 $F_k$

### 9.2 场景选择准则

**主要准则**：选择综合目标函数值最小的场景
$$k^* = \arg\min_{k \in \{a,b,c\}} F_k$$

**辅助分析**：
1. **Pareto 前沿分析**：绘制成本-时间-可靠性-环境四维空间的 Pareto 非支配解
2. **敏感性分析**：考察权重 $(\omega_1, \omega_2, \omega_3, \omega_4)$ 变化对最优场景的影响
3. **情景分析**：
   - 预算约束情景：$C_k \leq C_{\text{budget}}$
   - 时间约束情景：$T_k \leq T_{\text{deadline}}$
   - 环境约束情景：$EI_k \leq EI_{\text{limit}}$

### 9.3 环境影响最小化策略

基于目标 4 的敏感性分析，提出针对性优化策略：

**场景 A 优化方向**：
- 提高可再生能源占比 $s_{E,\text{elec}} \to 1.0$
- 降低 CO₂ 排放（电梯的主要环境影响）

**场景 B 优化方向**：
- 采用绿色燃料（$f_{R,\text{env}} \to 0$），显著降低 PM2.5 和 CO₂ 排放
- 优先选择低风险发射场（$s_{R,\text{safe},i} = 1.0$），减少生态破坏

**场景 C 优化方向**：
- 增大电梯占比 $\alpha$（电梯环境影响远低于火箭）
- 火箭部分采用绿色燃料和低风险发射场

---

## 十、模型求解方法

### 10.1 场景 A 求解（连续优化）

**求解器**：非线性规划（NLP）
- 决策变量：$(s_{E,\text{elec}}, m_E)$（2 维）
- 目标函数：$\min F_a(s_{E,\text{elec}}, m_E)$
- 约束：线性约束 + 边界约束
- 推荐算法：**SLSQP**（Sequential Least Squares Programming）

**求解步骤**：
1. 初始化：$s_{E,\text{elec}}^{(0)} = 0.9$，$m_E^{(0)} = 2$
2. 调用 `scipy.optimize.minimize` 求解
3. 验证约束满足情况

### 10.2 场景 B 求解（混合整数规划）

**求解器**：0-1 组合优化
- 决策变量：$(y_1, \ldots, y_{10})$ 为整数，$(x_1, \ldots, x_{10})$ 为连续
- 搜索空间：$2^{10} = 1024$ 种发射场组合

**求解方法**：
1. **枚举法**（适用于小规模问题）：
   - 遍历所有 1024 种组合
   - 对每种组合，优化连续变量 $(x_1, \ldots, x_{10})$
   - 选择目标函数值最小的组合

2. **遗传算法**（适用于大规模问题）：
   - 编码：二进制染色体 $(y_1, \ldots, y_{10})$
   - 适应度函数：$1 / F_b$
   - 交叉、变异操作，迭代 100-200 代

### 10.3 场景 C 求解（混合整数非线性规划）

**求解器**：分层优化策略
- 外层：优化 0-1 变量 $(y_1, \ldots, y_{10})$（枚举法或遗传算法）
- 内层：给定发射场组合后，优化连续变量 $(\alpha, x_1, \ldots, x_{10}, m_E)$（NLP）

**求解步骤**：
1. 外层枚举：遍历发射场组合
2. 内层优化：
   - 固定 $(y_1, \ldots, y_{10})$
   - 初始化：$\alpha^{(0)} = 0.5$，$m_E^{(0)} = 2$
   - 使用 SLSQP 求解内层 NLP
3. 记录最优解

---