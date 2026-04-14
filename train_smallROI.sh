#!/bin/bash
# 针对小目标 ROI 优化的训练脚本

# 设置 nnUNet 环境变量
export nnUNet_raw="/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_preprocessed"
export nnUNet_results="/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_results"

# 数据集名称
DATASET="Dataset001_HCC"

# 使用自定义 Trainer 进行训练
echo "开始训练 - 使用 nnUNetTrainer_smallROI (小目标优化版)"
echo "改进点:"
echo "  1. 前景过采样比例: 0.5"
echo "  2. Focal Loss + Dice Loss 组合"
echo "  3. batch_dice=False (对小目标更敏感)"
echo "  4. 减小旋转角度范围"
echo ""

# 训练所有 5 个 fold
for FOLD in 0 1 2 3 4; do
    echo "=========================================="
    echo "Training Fold $FOLD"
    echo "=========================================="
    
    nnUNetv2_train $DATASET 2d $FOLD -tr nnUNetTrainer_smallROI
    
    if [ $? -ne 0 ]; then
        echo "Error: Fold $FOLD training failed!"
        exit 1
    fi
    
    echo "Fold $FOLD completed!"
    echo ""
done

echo "所有 fold 训练完成!"
echo ""
echo "接下来可以运行交叉验证:"
echo "  nnUNetv2_find_best_configuration $DATASET -tr nnUNetTrainer_smallROI -c 2d"
