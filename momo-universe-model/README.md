# MUM-Gravity-0.1（先导版）

学习引力相互作用的对称成对力图网络先导模型。单一事实来源：`SPEC.md`。
详细真实结果（含环境偏离、全部评测数字、已知限制）：`RESULTS.md`。

## 环境

- Python 3.12，torch 2.8（CPU），numpy 2.2，matplotlib 3.10
- 无第三方占星/积分库：教师积分器为自写 numpy float64（RK4 + KDK + Kepler 解析二体）
- 配置为 JSON（`configs/pilot.json`，环境无 pyyaml）

## 快速开始

```bash
# 1. 单元测试（纯断言脚本）
python tests/test_integrators.py     # 教师质量验收：RK4 vs Kepler < 1e-6
python tests/test_tensors.py         # 张量/padding/mask 合同
python tests/test_symmetry.py        # 力头对称性（排列/旋转/平移/动量守恒）

# 2. 生成教师数据（二体 400 + 三体 800 条，约 5 秒）
python -m data.generators --config configs/pilot.json

# 3. 分阶段训练 S1(force) -> S2(+state) -> S3(+rollout+energy)
python -m training.train --config configs/pilot.json

# 4. 评测五件套（test 集仅用这一次），写 results/metrics.json + PNG
python -m evaluation.run_eval

# 5. 推理独立性验收（inference/ 与 model/ 不 import data/）
python evaluation/independence_check.py
```

## 推理（纯模型路径）

```python
from inference.load import load_model
from inference.rollout import strict_rollout
import numpy as np

model, meta = load_model("checkpoints/mum_gravity_pilot_stage3.pt")
res = strict_rollout(
    model,
    mu=np.array([[0.5], [0.3], [0.2]]),          # 归一化质量 [N,1]
    pos0=np.random.randn(3, 3), vel0=np.random.randn(3, 3) * 0.3,
    mask=np.ones(3, dtype=bool),
    dt=0.01, steps=100,
    limits={"max_steps": 10000, "min_separation": 0.02, "max_radius": 10.0},
)
print(res.pos.shape, res.stopped, res.stop_reason)
```

## 结构

- `model/` 归一化、不变量编码器、消息传递、对称成对力头、KDK 步进、带 limits 的 rollout
- `data/` 自写教师积分器（RK4/KDK/Kepler 解析解）、初态采样、轨迹生成、seed 划分
- `training/` 归一化损失（L_force/L_state/L_rollout/L_energy）、分阶段训练 CLI
- `inference/` 权重加载、strict rollout（不 import `data/`）
- `evaluation/` 对称性、二体 Kepler 对照、三体短窗、守恒漂移、大步长判决实验
- `tests/` 三个纯断言测试脚本；`results/` 数据与评测产物；`checkpoints/` 权重
