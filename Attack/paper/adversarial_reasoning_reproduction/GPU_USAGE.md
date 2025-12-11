# GPU 使用指南

## 功能说明

新增 `--gpu` 参数，支持灵活指定GPU设备。

## 使用方法

### 1. 使用单个GPU

```bash
# 使用 GPU 0
python main.py --csv ./data/JBB.csv --gpu 0

# 使用 GPU 2
python main.py --csv ./data/JBB.csv --gpu 2

# 使用 GPU 3
python main.py --csv ./data/JBB.csv --gpu 3
```

### 2. 使用多个GPU

```bash
# 使用 GPU 0 和 1
python main.py --csv ./data/JBB.csv --gpu 0,1

# 使用 GPU 3 和 4
python main.py --csv ./data/JBB.csv --gpu 3,4

# 使用 GPU 2, 3, 4
python main.py --csv ./data/JBB.csv --gpu 2,3,4
```

### 3. 使用CPU（不指定GPU）

```bash
# 默认CPU模式
python main.py --csv ./data/JBB.csv

# 或明确指定
python main.py --csv ./data/JBB.csv --gpu None
```

## 工作原理

`--gpu` 参数通过设置 `CUDA_VISIBLE_DEVICES` 环境变量来控制可见的GPU设备：

```python
# 例如：--gpu 3,4
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

# 这样物理GPU 3 会被映射为 cuda:0
# 物理GPU 4 会被映射为 cuda:1
```

## 示例命令

### 完整示例 1: 使用GPU 2运行单个任务
```bash
python main.py \
    --csv ./data/JBB.csv \
    --gpu 2 \
    --start_idx 0 \
    --end_idx 1 \
    --num_iterations 5
```

### 完整示例 2: 使用GPU 3,4运行前10个任务
```bash
python main.py \
    --csv ./data/JBB.csv \
    --gpu 3,4 \
    --start_idx 0 \
    --end_idx 10 \
    --num_iterations 15
```

### 完整示例 3: 使用GPU 0运行标准配置
```bash
python main.py \
    --csv ./data/JBB.csv \
    --gpu 0 \
    --num_iterations 15 \
    --num_prompts 16 \
    --num_branches 8 \
    --buffer_size 32
```

## 查看可用GPU

在运行脚本前，可以查看系统中的GPU：

```bash
# 使用 nvidia-smi
nvidia-smi

# 使用 Python 查看
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

## 程序输出示例

当使用 `--gpu 3,4` 时，程序会输出：

```
================================================================================
GPU CONFIGURATION
================================================================================
Using GPU(s): 3,4
Number of visible GPUs: 2
  GPU 0: NVIDIA A100-SXM4-80GB
  GPU 1: NVIDIA A100-SXM4-80GB
================================================================================
```

注意：即使指定了 `3,4`，程序内部会使用 `cuda:0` 来访问（因为设置了 `CUDA_VISIBLE_DEVICES`）。

## 注意事项

1. **模型自动设备管理**: 你的 `qwen_open_4B.py` 使用 `device_map="auto"`，模型会自动管理设备分配

2. **内存考虑**:
   - Qwen 4B 模型大约需要 8-10GB GPU内存
   - 确保指定的GPU有足够内存

3. **多GPU使用**:
   - 目前代码主要使用单GPU（`cuda:0`）
   - 多GPU主要用于增加可用内存或为后续扩展预留

4. **性能**: 使用GPU会显著加快推理速度

## 故障排查

### 问题 1: CUDA not available
```
WARNING: CUDA not available, falling back to CPU
```
**解决**:
- 检查PyTorch是否安装了CUDA支持
- 运行 `python -c "import torch; print(torch.cuda.is_available())"`

### 问题 2: GPU out of memory
```
RuntimeError: CUDA out of memory
```
**解决**:
- 使用更大内存的GPU
- 减少批次大小: `--num_prompts 8 --num_branches 4`
- 指定多个GPU: `--gpu 0,1`

### 问题 3: Invalid GPU ID
```
Invalid device id
```
**解决**:
- 检查GPU ID是否存在: `nvidia-smi`
- 使用正确的ID（从0开始）

## 性能对比

| 配置 | 每个任务预计时间 |
|------|----------------|
| CPU | 15-30 分钟 |
| Single GPU | 3-8 分钟 |
| Multi-GPU | 3-8 分钟（内存更充足） |

## 高级用法

### 指定特定GPU运行多个实验

```bash
# 终端1: 使用GPU 0运行任务0-25
python main.py --csv ./data/JBB.csv --gpu 0 --start_idx 0 --end_idx 25 &

# 终端2: 使用GPU 1运行任务25-50
python main.py --csv ./data/JBB.csv --gpu 1 --start_idx 25 --end_idx 50 &

# 终端3: 使用GPU 2运行任务50-75
python main.py --csv ./data/JBB.csv --gpu 2 --start_idx 50 --end_idx 75 &
```

这样可以并行加速整个数据集的处理。
