# Known Issues & Handoff Notes
# palr_habitat — Fetch Continual Rearrangement

---

## 可以直接用的部分（逻辑正确，无需修改）

| 文件 | 说明 |
|---|---|
| `src/plasticity_metrics_cnn.py` | 纯 PyTorch/numpy，死滤波器 + 有效秩计算 |
| `src/palr_resnet_encoder.py` | 标准 torchvision ResNet-18，梯度缩放 + 扰动接口 |
| `src/palr_fetch_policy.py` | Actor-Critic 策略架构（视觉+本体感知+GRU）|
| `src/fetch_curriculum.py` | 课程切换逻辑（纯 Python，与 habitat 无关）|
| `src/plot_results.py` | 画图脚本（需 tensorboard + matplotlib）|
| `setup_conda.sh` | conda 安装命令正确 |

---

## 会报错 / 需要手动修复的问题

### 问题 1：Episode 数据集路径是占位符

**位置**：`configs/ddppo_palr_fetch.yaml` 和 `configs/ddppo_baseline_fetch.yaml`

以下路径是不准确的，实际文件不存在：
```yaml
dataset: "data/datasets/rearrange/apple_pick_train.json.gz"
dataset: "data/datasets/rearrange/bowl_pick_train.json.gz"
dataset: "data/datasets/rearrange/fridge_train.json.gz"
dataset: "data/datasets/rearrange/sink_place_train.json.gz"
```

**修复方法**：
1. 先跑 `bash download_data.sh`，看数据下载到哪里
2. 查看 `data/datasets/rearrange/` 目录下实际文件名
3. 把 YAML 里的路径改成真实路径

---

### 问题 2：Task 名称未验证

**位置**：两个 YAML configs 里

以下 task 名称不确定是否在 habitat-lab 0.2.5 中注册：
```yaml
RearrangeOpenFridgeTask-v0
RearrangePlaceTask-v0
```

**验证方法**：
```bash
conda activate palr_habitat
python -c "import habitat; print(list(habitat.registry._task_map.keys()))"
```
对照输出，把 YAML 里的 task 名改成实际存在的。

---

### 问题 3：`palr_trainer.py` 里的 habitat API 路径错误

**位置**：`src/palr_trainer.py`，`make_envs()` 函数

```python
# 错误（此路径在 habitat-lab 0.2.5 中不存在）：
from habitat.utils.env_utils import construct_envs, make_env_fn

# 正确应为：
from habitat_baselines.utils.env_utils import construct_envs
```

另外 `construct_envs` 的调用接口与当前代码不一致，需要参照
`habitat_baselines/utils/env_utils.py` 的实际签名修改。

---

### 问题 4：缺少 `configs/rearrange_base.yaml`

YAML 里引用了一个不存在的基础配置：
```yaml
BASE_TASK_CONFIG_PATH: "configs/rearrange_base.yaml"
```

**修复方法**：
从 habitat-lab 安装目录复制官方的基础配置：
```bash
cp $(python -c "import habitat_baselines; import os; \
    print(os.path.dirname(habitat_baselines.__file__))")\
/config/rearrange/ddppo_pick.yaml \
configs/rearrange_base.yaml
```
然后按需修改。

---

### 问题 5：`download_data.sh` 中的数据集 UID 未验证

以下 UID 是估计的，不保证正确：
```bash
--uids replica_cad_baked_lighting
--uids ycb
--uids rearrange_pick_dataset_v0
```

**验证方法**：
```bash
python -m habitat_sim.utils.datasets_download --list
```
查看所有可用 UID，对照后修改脚本。

---

## 建议：推荐的接入方式

### 方案 A（推荐）：先跑通官方示例，再插入 PALR

1. 先用 habitat-baselines 官方示例确认环境正常：
   ```bash
   python -m habitat_baselines.run \
       --exp-config habitat_baselines/config/rearrange/ddppo_pick.yaml \
       --run-type train
   ```

2. 跑通之后，只需把两个文件插入：
   - `src/palr_resnet_encoder.py` — 替换官方的视觉编码器
   - `src/plasticity_metrics_cnn.py` — 在 update 后调用

3. `fetch_curriculum.py` 单独管理任务切换，不依赖 habitat 内部逻辑。

### 方案 B：修复 `palr_trainer.py` 的 habitat API 调用

如果需要完全独立的训练脚本（不依赖 habitat_baselines），
需要把 `palr_trainer.py` 中 `make_envs()` 的实现
替换为正确的 `habitat.VectorEnv` 构建方式。
可以参考：
`habitat_baselines/train.py` 或 `habitat_baselines/rl/ddppo/ddppo_trainer.py`
中的环境初始化代码。

---

## 环境要求摘要

- Python 3.9
- habitat-sim 0.2.5（必须用 conda，无 pip wheel）
- habitat-lab 0.2.5 + habitat-baselines 0.2.5
- PyTorch 2.1.2 + CUDA 11.8
- 4× NVIDIA A40（或等效 40 GB GPU）
- 无需显示器（EGL headless）
