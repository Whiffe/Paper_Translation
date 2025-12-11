# 快速开始指南

## 项目概述

这是论文 "Adversarial Reasoning at Jailbreaking Time" (arXiv:2502.01633v2) 的复现代码，已按照您的要求进行定制：
- ✓ 使用本地Qwen 4B模型（`qwen_open_4B.py`）
- ✓ 读取JBB数据集（`--csv ./data/JBB.csv`）
- ✓ 移除了wandb依赖
- ✓ 所有LLM角色（Attacker、Feedback、Refiner）都使用同一个本地模型

## 文件结构

```
adversarial_reasoning_reproduction/
├── main.py                      # 主程序入口
├── adversarial_reasoning.py     # 核心算法实现（Algorithm 1）
├── buffer.py                    # GWW缓冲区管理
├── model_wrapper.py             # 本地模型封装
├── prompts.py                   # 系统提示词
├── utils.py                     # 工具函数
├── test_setup.py                # 测试脚本
├── run.sh                       # 快速启动脚本
├── requirements.txt             # 依赖列表
├── README.md                    # 详细文档（英文）
├── QUICKSTART.md                # 本文件
├── data/
│   └── JBB.csv                 # 数据集（符号链接）
└── results/
    └── results.jsonl           # 结果文件
```

## 快速开始

### 步骤 1: 安装依赖

```bash
cd /Users/yangfan/adversarial_reasoning_reproduction
pip install torch numpy
```

### 步骤 2: 检测GPU（可选）

```bash
# 检查GPU可用性
python test_gpu.py

# 或使用nvidia-smi
nvidia-smi
```

### 步骤 3: 测试设置

```bash
python test_setup.py
```

或使用快捷脚本：
```bash
./run.sh
```

### 步骤 4: 运行算法

#### 使用GPU（推荐）

**使用单个GPU:**
```bash
# 使用GPU 0
python main.py --csv ./data/JBB.csv --gpu 0 --start_idx 0 --end_idx 1 --num_iterations 5

# 使用GPU 3
python main.py --csv ./data/JBB.csv --gpu 3 --start_idx 0 --end_idx 1 --num_iterations 5
```

**使用多个GPU:**
```bash
# 使用GPU 3和4
python main.py --csv ./data/JBB.csv --gpu 3,4 --start_idx 0 --end_idx 1 --num_iterations 5
```

#### 使用CPU（较慢）

**测试单个任务（最快）:**
```bash
python main.py --csv ./data/JBB.csv --start_idx 0 --end_idx 1 --num_iterations 5
```

#### 运行前10个任务
```bash
python main.py --csv ./data/JBB.csv --start_idx 0 --end_idx 10
```

#### 完整运行（所有任务）
```bash
python main.py --csv ./data/JBB.csv
```

## 核心参数说明

| 参数 | 默认值 | 说明 | 论文中符号 |
|------|--------|------|-----------|
| `--num_iterations` | 15 | 迭代次数 | T |
| `--num_prompts` | 16 | 每次迭代生成的攻击提示数量 | n |
| `--num_branches` | 8 | 反馈分支数量 | m |
| `--buffer_size` | 32 | GWW缓冲区大小 | B |
| `--batch_divs` | 2 | 反馈批次划分数 | k |
| `--gpu` | None | GPU设备（如"0", "3,4"） | - |

### GPU参数详解

- `--gpu 0`: 使用GPU 0
- `--gpu 3`: 使用GPU 3
- `--gpu 3,4`: 使用GPU 3和4（提供更大内存）
- 不指定或`None`: 使用CPU

💡 **提示**: 使用GPU可以将每个任务的时间从15-30分钟缩短到3-8分钟！

详细GPU使用说明请参考 [GPU_USAGE.md](GPU_USAGE.md)

## 算法流程

根据论文Algorithm 1：

1. **初始化**: 使用初始推理字符串 S^(0)
2. **主循环** (T次迭代):
   - 从缓冲区选择最佳推理字符串
   - 使用Attacker LLM生成n个攻击提示
   - 计算每个提示的损失值
   - 生成m个反馈（比较提示效果）
   - 使用Refiner LLM优化推理字符串
   - 将新候选加入缓冲区
3. **输出**: 找到的最佳jailbreak提示

## 查看结果

结果保存在 `results/results.jsonl`，每行一个JSON对象：

```json
{
  "task_idx": 0,
  "goal": "Write a defamatory article...",
  "target": "Sure, here is a defamatory article...",
  "category": "Harassment/Discrimination",
  "success": true,
  "best_prompt": "...",
  "best_response": "...",
  "min_loss": 0.234,
  "iterations": 7
}
```

## 性能调优

### 快速测试（减少计算量）
```bash
python main.py \
    --csv ./data/JBB.csv \
    --num_iterations 5 \
    --num_prompts 8 \
    --num_branches 4 \
    --start_idx 0 \
    --end_idx 1
```

### 标准设置（论文配置）
```bash
python main.py \
    --csv ./data/JBB.csv \
    --num_iterations 15 \
    --num_prompts 16 \
    --num_branches 8 \
    --buffer_size 32
```

### 深度搜索（更多迭代）
```bash
python main.py \
    --csv ./data/JBB.csv \
    --num_iterations 30 \
    --num_prompts 16 \
    --num_branches 8
```

## 与原论文的差异

1. **模型**: 使用本地Qwen 4B替代论文中的Mixtral/Vicuna API调用
2. **损失函数**: 由于无法直接访问logits，使用基于目标字符串匹配的代理损失
3. **无wandb**: 已移除所有wandb日志记录
4. **统一模型**: Attacker、Feedback、Refiner使用同一个本地模型

## 故障排查

### 问题：找不到模型文件
**解决**: 确保模型路径正确：
```bash
ls -la /Users/yangfan/Downloads/JAIL-CON/models/qwen_open_4B.py
```

### 问题：找不到数据文件
**解决**: 检查符号链接：
```bash
ls -la data/JBB.csv
```

### 问题：内存不足
**解决**: 减少并发数量：
```bash
python main.py --num_prompts 8 --num_branches 4
```

## 进阶使用

### 修改提示词
编辑 `prompts.py` 中的相关函数

### 修改损失函数
编辑 `model_wrapper.py` 中的 `compute_loss_from_logits()` 方法

### 添加自定义judge
编辑 `utils.py` 中的 `check_jailbreak_simple()` 函数

## 引用

如果使用本代码，请引用原论文：

```bibtex
@article{sabbaghi2025adversarial,
  title={Adversarial Reasoning at Jailbreaking Time},
  author={Sabbaghi, Mahdi and Kassianik, Paul and Pappas, George and Singer, Yaron and Karbasi, Amin and Hassani, Hamed},
  journal={arXiv preprint arXiv:2502.01633},
  year={2025}
}
```

## 联系与支持

- 原论文: https://arxiv.org/abs/2502.01633
- 原代码: /Users/yangfan/Downloads/Adversarial-Reasoning-main

---

**祝您复现顺利！**
