# 训练检查点功能使用说明

本文档说明如何使用新添加的检查点保存和恢复功能。

## 功能特性

1. **自动保存检查点**：在指定的训练步数自动保存模型权重
2. **断点续训**：可以从任意检查点恢复训练
3. **灵活配置**：可以自定义保存检查点的步数

## 使用方法

### 1. 基本训练（默认保存检查点）

```bash
cd /home/zdx/github/VSAHDC/CompressARC-fork
python train.py --task_name 0a2355a6
```

默认会在以下步数保存检查点：400, 800, 1200, 1600, 2000

### 2. 自定义保存步数

```bash
python train.py --task_name 0a2355a6 --save_steps 200 500 1000 1500 2000
```

### 3. 从最新检查点恢复训练

```bash
python train.py --task_name 0a2355a6 --resume
```

### 4. 从指定检查点恢复训练

```bash
python train.py --task_name 0a2355a6 --checkpoint 0a2355a6/checkpoint_step_800.pth
```

### 5. 训练不同任务

```bash
python train.py --task_name 1a2355b7 --save_steps 400 800 1200
```

## 检查点文件

检查点文件保存在任务文件夹中，命名格式为：`checkpoint_step_{step_number}.pth`

例如：
- `0a2355a6/checkpoint_step_400.pth`
- `0a2355a6/checkpoint_step_800.pth`
- `0a2355a6/checkpoint_step_1200.pth`

## 检查点内容

每个检查点文件包含：
- 模型状态字典（model_state_dict）
- 优化器状态字典（optimizer_state_dict）
- 当前训练步数（train_step）
- 任务名称（task_name）
- RuleLayer状态字典（如果使用）
- 使用规则标志（use_rule）

## 注意事项

1. 检查点文件大小可能较大，请确保有足够的磁盘空间
2. 从检查点恢复时，确保模型结构与保存时一致
3. 如果修改了模型结构，可能无法正确加载旧的检查点
4. 建议定期清理不需要的检查点文件以节省空间

## 错误处理

如果遇到检查点加载错误：
1. 检查检查点文件是否存在
2. 确认模型结构是否与保存时一致
3. 检查CUDA设备是否可用（如果使用GPU训练）

## 示例训练流程

```bash
# 开始训练，保存检查点
python train.py --task_name 0a2355a6 --save_steps 400 800

# 如果训练中断，从最新检查点恢复
python train.py --task_name 0a2355a6 --resume

# 或者从特定步数恢复
python train.py --task_name 0a2355a6 --checkpoint 0a2355a6/checkpoint_step_400.pth
```
