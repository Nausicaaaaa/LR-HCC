#!/bin/bash
# 增量训练脚本：使用 Dataset001 的预训练模型 + Dataset003 的数据训练 Dataset004

set -e  # 遇到错误立即退出

echo "=========================================="
echo "增量训练流程"
echo "=========================================="
echo ""

# 设置环境变量
export nnUNet_raw="/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_preprocessed"
export nnUNet_results="/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_results"

# 步骤 0: 检查 Dataset001 模型是否存在
echo "[步骤 0] 检查 Dataset001 预训练模型..."
PRETRAIN_MODEL_PATH="${nnUNet_results}/Dataset001_HCC/nnUNetTrainer_smallROI__nnUNetPlans__2d/fold_0/checkpoint_final.pth"

if [ ! -f "$PRETRAIN_MODEL_PATH" ]; then
    echo "错误: 未找到 Dataset001 的预训练模型"
    echo "路径: $PRETRAIN_MODEL_PATH"
    echo ""
    echo "请先训练 Dataset001:"
    echo "  python main.py --dataset_id 001 --trainer nnUNetTrainer_smallROI --train_all_folds"
    exit 1
fi

echo "✓ 找到预训练模型: $PRETRAIN_MODEL_PATH"
echo ""

# 步骤 1: 合并数据集
echo "[步骤 1] 合并 Dataset001 和 Dataset003..."
python merge_datasets.py
echo ""

# 步骤 2: 生成伪标签
echo "[步骤 2] 使用 Dataset001 模型为 Dataset003 生成伪标签..."
python generate_pseudo_labels.py --use_smallROI --fold all
echo ""

# 步骤 3: 预处理 Dataset004
echo "[步骤 3] 预处理 Dataset004..."
nnUNetv2_plan_and_preprocess -d 004 --verify_dataset_integrity -np 8
echo ""

# 步骤 4: 使用预训练模型训练 Dataset004
echo "[步骤 4] 使用预训练权重训练 Dataset004..."
echo ""

# 设置预训练模型路径环境变量
export nnUNet_pretrained_model_path="$PRETRAIN_MODEL_PATH"

# 训练所有 fold
for FOLD in 0 1 2 3 4; do
    echo "=========================================="
    echo "训练 Fold $FOLD (使用预训练权重)"
    echo "=========================================="
    
    nnUNetv2_train 004 2d $FOLD -tr nnUNetTrainer_smallROI_pretrained
    
    if [ $? -ne 0 ]; then
        echo "错误: Fold $FOLD 训练失败!"
        exit 1
    fi
    
    echo "Fold $FOLD 完成!"
    echo ""
done

echo "=========================================="
echo "增量训练完成!"
echo "=========================================="
echo ""
echo "数据集统计:"
echo "  - Dataset001: 约 115 个有标签病例"
echo "  - Dataset003: 约 117 个病例（通过伪标签获得）"
echo "  - Dataset004: 约 232 个病例（合并后）"
echo ""
echo "下一步: 评估模型性能"
echo "  nnUNetv2_find_best_configuration 004 -tr nnUNetTrainer_smallROI_pretrained -c 2d"
