# universe-model 宇宙模型

# MOMO Universe Model (MUM)

# 🌌 The First Universe Model — 第一个宇宙模型

> **世界模型在教 AI 理解一个房间、一段视频、一场游戏。**
> **我们在教 AI——和所有人——理解宇宙本身。**
> >
> *World models teach AI to understand a room, a video, a game. We are building the model that understands the Universe itself.*
> >
> **The first open-source Universe Model built for human understanding — reproducible, interactive, and assumption-explicit.**

**状态：v0.1 先导版已跑通（真实权重、真实验证、真实数字）。** 这不是一份白皮书，不是一个演示动画，而是一个已经训练完成、可以独立推理、每一项声明都能被你亲手复现的学习型宇宙动力学模型。

---

## 🌌 MOMO Universe Model 3D 演示

单文件 Three.js/WebGL 实现

| 场景 |演示| 实现 | 验证状态 |
|---|---|---|---|
| **三体问题** |<img width="1257" height="1262" alt="image" src="https://github.com/user-attachments/assets/3bedbaf3-6f7c-48fc-af48-366f72accbd1" />| 真实 RK4 积分 + 8字形周期解/混沌三星预置 + ε扰动滑杆 + 1200点渐隐拖尾 | ✅ 截图确认 |
| **黑洞·引力透镜** | <img width="1558" height="1240" alt="img_v3_0216e_be5c5f6e-4336-467f-bb8e-cee0dd5c7bdg" src="https://github.com/user-attachments/assets/220b8b95-b335-4366-b679-df0980cb4778" />|100步光线步进着色器：纯黑阴影、细亮光子环、橙红吸积盘、多普勒不对称、盘背面翻折、星空透镜扭曲 | ✅ 截图确认（修复过一次严重过曝） |
| **费米悖论** |<img width="1556" height="1274" alt="image" src="https://github.com/user-attachments/assets/b88e16a1-7919-4434-bbd9-840fdf8faae7" />|2万恒星螺旋星系 + 德雷克蒙特卡洛：文明点亮-存活-熄灭，默认参数下"大沉默" | ✅ 截图确认 |
| **洛希极限** |<img width="1285" height="1253" alt="image" src="https://github.com/user-attachments/assets/40d391e6-5b82-485e-8635-4833a14dcbdc" />|真实洛希公式红色虚线环 + 越限后潮汐拉伸→700碎片开普勒剪切散成碎屑弧 | ✅ 截图确认 |
| **引力弹弓** |<img width="1542" height="1266" alt="20261013231557_rec_-convert" src="https://github.com/user-attachments/assets/46fbfd5c-f1cb-4c9c-8d8a-086c4b156b7f" />|真实限制性三体积分 + 速度矢量箭头 + Δv实测（参数经 Python 网格验证：b=0.9 可获 ~50% 加速） | ✅ 截图确认 |

**几个亮点设计：**
- **每个场景右上角都有"假设与边界"标注**（黑洞标注"示意渲染非GR数值模拟"、费米标注"非外星人存在概率"）——和项目的科学红线完全一致
- 三个真实模拟场景（三体/洛希/弹弓）直接GitHub 仓库里 MUM-Gravity 的能力叙事

---

## 为什么是现在：世界模型之上，还空着一个位置

2020 年代，AI 学会了建模"世界"——但"世界"被用得越来越小：一个驾驶场景叫世界模型，一个游戏关卡叫世界模型，一段视频也叫世界模型。

我们把尺度阶梯补齐：

| 模型层级 | 模拟对象 | 现状 |
| --- | --- | --- |
| 世界模型 | 局部环境（游戏、驾驶、房间） | 巨头混战，极度拥挤 |
| 地球模型 | 天气、气候 | 大玩家已就位 |
| **宇宙模型** | **轨道、恒星、星系、宇宙演化** | **🟢 本项目** |

宇宙模拟的科学工作并不少——但它们是锁在论文和超算中心的**科研工具**。从来没有人做过这样一件事：**把宇宙变成一个开源的、可训练的、每个人都可以亲手运行和验证的模型。** 我们做了。

## 宇宙模型是什么

> **宇宙模型 = 一个可学习的宇宙动力学模型 + 一套让任何人都能提问、实验、验证的开放系统。**

它有三条不可妥协的原则：

**1. 模型，不是动画。** 核心是一个真实训练的神经网络：明确的输入输出、保存的权重、独立的推理入口——拔掉求解器、拔掉 LLM，它依然能凭自己的权重推演天体的未来。

**2. 边界，先于规模。** 每个输出都标注：这是数值预测、近似估计、示意可视化，还是假设推演的后果。我们不假装模拟了整条物理链，我们精确地告诉你模型算到了哪里、误差多大、从哪里开始不可信。

**3. 实验，先于叙事。** 宇宙模型不是"给你讲宇宙"，而是**给你一个可以亲手改假设的宇宙**：改一个初值，看三体系统崩坏；拉一个滑杆，看银河系从热闹到死寂。

## 宇宙模型要回答的问题

人类仰望星空时问的，其实只有四类问题——宇宙模型把它们全部变成可运行的实验：

### 🔥 出圈问题：每个人听过的宇宙之问

> **三体问题** —— 三颗恒星的混沌之舞，《三体》的数学原点。一般解不存在解析式，只能靠计算。改 0.1% 的初值，看一个"文明"提前多少圈毁灭。
> >
> **费米悖论** —— "他们在哪？" 宇宙这么大，为什么一片寂静？蒙特卡洛推演亿万种文明的可能命运，让大过滤器、黑暗森林从辩论变成参数。
> >
> **洛希极限** —— 《流浪地球》里地球冲向木星的那根生死线。亲手把它算出来。
> >
> **引力弹弓** —— 《火星救援》的"免费加速"是真的吗？用同一套引擎证明：混沌不可预测，但引力可以被精确利用。

### 🧠 理解问题：为什么宇宙是这样运作的

为什么初值差一丝，结局天差地别？为什么刚性戴森球没有"自动回中"的能力，戴森云才是正解？运行一次实验，直觉被永久改变——这是"理解"真正发生的时刻。

### 🔬 科研问题：这组假设下，宇宙会怎样

可复现的参数实验、经过验证的学习代理模型、诚实的误差报告——让宇宙学从"少数超算的特权"变成"一块消费级 GPU 上的开放科学"。

### 🌠 终极问题：我们在宇宙中处于什么位置

宇宙模型不给终极答案。但它给每个人一件前所未有的东西：**亲手检验自己宇宙观的能力**。

## 我们与其他一切"宇宙项目"的区别

|  | 天文可视化 | 科研宇宙模拟 | LLM 科幻问答 | **MUM** |
| --- | --- | --- | --- | --- |
| 好看 | ✅ | ❌ | — | ✅ |
| 真实物理 | ❌ | ✅ | ❌ | ✅ |
| 可学习模型权重 | ❌ | ❌ | ❌ | ✅ |
| 人人可运行验证 | ❌ | ❌ | ❌ | ✅ |
| 标注假设与误差边界 | ❌ | 部分 | ❌ | ✅ |

## 已经证明了什么（v0.1，全部真实数字）

Talk is cheap. **以下每一项你都可以 clone 后亲手复现**（命令见 [RESULTS.md](RESULTS.md)）：

| 验证项 | 结果 | 含义 |
| --- | --- | --- |
| 教师积分器 vs Kepler 解析解 | 误差 **2.1e-13** | 训练数据地基如磐石 |
| 二体模型 rollout vs Kepler | 误差中位 **6.3e-05** | 模型真正学会了引力动力学 |
| 三体 16 步 rollout（test 集） | 误差中位 **2.4e-04**，p90 9.5e-04 | 混沌系统的短窗预测可信 |
| 动量 / 角动量守恒 | 漂移 ~**1e-7**（构造性保证） | 对称力架构白送牛顿第三定律 |
| 独立性验收 | 无求解器、无 LLM、无远程服务 | "独立模型"不是营销词 |
| 参数量 / 训练成本 | 464,641 参数 / 34 分钟 CPU | 宇宙模型不需要超算 |

我们也如实报告失败——因为我们相信**诚实的边界比漂亮的曲线更有价值**：

- ❌ 能量严格守恒：**不成立**（漂移中位 3.2e-03）→ 0.2 版做势能头消融
- ❌ 大步长加速假说：**不成立**（dt=0.05 时模型 5.6e-03 vs 解析 5.8e-05）→ 0.2 版改学大步长 flow map
- ⚠️ 近遇奇点、长时混沌：已知最难的两个物理关卡，失败案例全部公开

一个开源项目最硬的底气，不是宣称自己多强，而是**把失败案例也写进模型卡**。

## 路线图：从引力到宇宙

```javascript
MUM-Gravity-0.1   ✅ 有限多体引力动力学（本仓库：权重 + 验证 + 失败案例全公开）
MUM-Gravity-0.2   → 更广质量比与天体数 · 三体失稳时间/结局预测头 · 大步长 flow map
MUM-Fields        → 物质场与宇宙学结构演化的学习模型
MUM-Lab           → 面向全人类的交互实验层：三体 · 费米悖论 · 洛希极限 · 引力弹弓
MUM-Multimodal    → 连接模拟与真实天文观测
MUM-Universe      → 多尺度、多任务的统一宇宙模型（长期北极星）
```

> **北极星指标：被完成的宇宙实验次数**——一个陌生人改了一个假设、跑完一次计算、看到了结果和它的边界。

## 仓库结构

```javascript
momo-universe-model/
├── model/                  # 模型本体
│   ├── normalization.py    #   质心共动 + 无量纲化（rollout 内固定基准）
│   ├── encoders.py         #   节点/边不变量特征
│   ├── message_passing.py  #   消息传递（排列不变聚合）
│   ├── pair_force_head.py  #   对称成对力头（动量/角动量构造守恒）
│   ├── state_update.py     #   kick-drift-kick 更新
│   └── gravity.py          #   MUMGravityModel：predict_acceleration / step / rollout
├── data/
│   ├── integrators.py      #   自写教师：RK4 / KDK / Kepler 解析解（float64）
│   ├── generators.py       #   二体/三体初态采样与轨迹生成
│   └── splits.py           #   按轨迹 seed 划分 train/val/test
├── training/
│   ├── losses.py           #   L_force / L_state / L_rollout / L_energy
│   └── train.py            #   分阶段课程训练
├── inference/
│   ├── load.py             #   权重加载
│   └── rollout.py          #   strict 模式推理（范围检查 + 停止原因）
├── evaluation/
│   ├── symmetry_tests.py   #   排列/旋转/平移一致性
│   ├── two_body_check.py   #   Kepler 解析对照
│   ├── conservation.py     #   守恒量漂移
│   ├── large_dt.py         #   大步长判决实验
│   └── run_eval.py         #   评测总入口
├── tests/                  #   纯断言测试（test_integrators / test_tensors / test_symmetry）
├── configs/                #   训练配置
├── checkpoints/            #   三阶段权重 + config
├── results/                #   metrics.json / 图表 / 训练日志 / 数据集 + manifest
├── RESULTS.md              #   真实评测报告（含失败案例）
└── SPEC.md                 #   实现规格（单一事实来源）
```

## 环境说明

本先导版在无网络环境开发：教师积分器为自写实现（经 Kepler 解析解验收，误差 2.1e-13），替代原计划的 REBOUND；权重以 `.pt` 保存。所有偏离及其验收数字均在 RESULTS.md 中声明。

## License

待定（见 LICENSE）。教师数据为自生成，无第三方数据许可问题。

---

> **从三体到费米悖论：把关于宇宙的好问题，变成可以修改假设、运行计算和检验解释的实验。**
> >
> **From the three-body problem to the Fermi paradox — turning humanity's best questions about the universe into experiments you can run yourself.**
