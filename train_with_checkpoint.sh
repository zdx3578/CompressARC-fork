#!/bin/bash

# ARC训练脚本 - 带检查点功能
# 使用说明：./train_with_checkpoint.sh [task_name] [options]

set -e  # 遇到错误时退出

# 默认参数
TASK_NAME="0a2355a6"
SAVE_STEPS="400 800 1200 1600 2000"
RESUME=""
CHECKPOINT=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--task)
            TASK_NAME="$2"
            shift 2
            ;;
        -s|--save-steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME="--resume"
            shift
            ;;
        -c|--checkpoint)
            CHECKPOINT="--checkpoint $2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -t, --task TASK_NAME          任务名称 (默认: 0a2355a6)"
            echo "  -s, --save-steps \"STEPS\"      保存检查点的步数 (默认: \"400 800 1200 1600 2000\")"
            echo "  -r, --resume                  从最新检查点恢复训练"
            echo "  -c, --checkpoint PATH         从指定检查点恢复训练"
            echo "  -h, --help                    显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                    # 默认训练"
            echo "  $0 -t 1a2355b7                      # 训练指定任务"
            echo "  $0 -t 0a2355a6 -r                   # 从最新检查点恢复"
            echo "  $0 -t 0a2355a6 -s \"200 500 1000\"   # 自定义保存步数"
            echo "  $0 -c 0a2355a6/checkpoint_step_800.pth  # 从指定检查点恢复"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 $0 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 切换到脚本目录
cd "$(dirname "$0")"

echo "========================================"
echo "ARC模型训练 - 带检查点功能"
echo "========================================"
echo "任务名称: $TASK_NAME"
echo "保存步数: $SAVE_STEPS"
echo "恢复选项: $RESUME $CHECKPOINT"
echo "========================================"

# 构建Python命令
PYTHON_CMD="python train.py --task_name $TASK_NAME --save_steps $SAVE_STEPS"

if [[ -n "$RESUME" ]]; then
    PYTHON_CMD="$PYTHON_CMD $RESUME"
fi

if [[ -n "$CHECKPOINT" ]]; then
    PYTHON_CMD="$PYTHON_CMD $CHECKPOINT"
fi

echo "执行命令: $PYTHON_CMD"
echo "========================================"

# 执行训练
eval $PYTHON_CMD

echo "========================================"
echo "训练完成！"
echo "检查点文件保存在: $TASK_NAME/"
echo "========================================"
