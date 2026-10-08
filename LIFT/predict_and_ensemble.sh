#!/bin/bash
# ============================================================
# 多折模型预测 + 集成
# 
# 用法:
#   bash predict_and_ensemble.sh 1 2          # 仅预测fold1和fold2并集成
#   bash predict_and_ensemble.sh 1 2 3 4 5    # 5折全部预测并集成
#   bash predict_and_ensemble.sh               # 默认5折
#
# 前提: 每个fold的模型已训练完成，checkpoint在:
#   ./ckpts/Model1_LR5_8_optimized_{fold}/uniformer_small_IL_features/best_f1_checkpoint-*.pth.tar
# ============================================================

cd /mnt/data/KASR/Dengsiyi/LR-HCC/LIFT

PYTHON=/mnt/data/KASR/.conda/envs/nnunetv2/bin/python
CKPT_BASE=./ckpts
MODEL_DIR=uniformer_small_IL_features
DATA_DIR=./data/images
TEST_ANNO=./data/labels/test.txt
PRED_BASE=./pred_results/optimized

# 要预测的folds（默认1~5）
FOLDS=${@:-1 2 3 4 5}
echo "=============================================="
echo "预测folds: $FOLDS"
echo "=============================================="

SCORE_FILES=""
FAILED=0

for FOLD in $FOLDS; do
    echo ""
    echo "--- Fold $FOLD ---"
    
    # 自动查找 best_f1 checkpoint
    CKPT_DIR="${CKPT_BASE}/Model1_LR5_8_optimized_${FOLD}/${MODEL_DIR}"
    if [ ! -d "$CKPT_DIR" ]; then
        # fold1 的目录名不带后缀
        if [ "$FOLD" = "1" ]; then
            CKPT_DIR="${CKPT_BASE}/Model1_LR5_8_optimized/${MODEL_DIR}"
        fi
    fi
    
    CKPT=$(ls ${CKPT_DIR}/best_f1_checkpoint-*.pth.tar 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls ${CKPT_DIR}/model_best.pth.tar 2>/dev/null | head -1)
    fi
    
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found for fold $FOLD in $CKPT_DIR"
        FAILED=1
        continue
    fi
    echo "Checkpoint: $CKPT"
    
    # 预测输出目录
    RESULTS_DIR="${PRED_BASE}/fold${FOLD}"
    mkdir -p "$RESULTS_DIR"
    
    # 如果 score.json 已存在则跳过
    if [ -f "${RESULTS_DIR}/val/score.json" ] || [ -f "${RESULTS_DIR}/score.json" ]; then
        echo "score.json already exists, skipping prediction."
        if [ -f "${RESULTS_DIR}/val/score.json" ]; then
            SCORE_FILES="${SCORE_FILES} ${RESULTS_DIR}/val/score.json"
        else
            SCORE_FILES="${SCORE_FILES} ${RESULTS_DIR}/score.json"
        fi
        continue
    fi
    
    $PYTHON ./main/predict.py \
        --model uniformer_small_IL_features \
        --num-classes 5 \
        --label-mode lr \
        --feature-fusion hierarchical_simple \
        --selected-features 1 3 6 8 9 10 11 \
        --img_size 20 96 96 \
        --crop_size 10 80 80 \
        --data_dir "$DATA_DIR" \
        --val_anno_file "$TEST_ANNO" \
        --checkpoint "$CKPT" \
        --results-dir "$RESULTS_DIR" \
        --score-dir "$RESULTS_DIR" \
        --threshold-optimize \
        -j 8 -b 4 \
        2>&1 | tee "${RESULTS_DIR}/predict.log"
    
    # predict.py 把主预测存在 val/ 子目录（因 test_anno_file 默认触发二级预测）
    SCORE_PATH="${RESULTS_DIR}/val/score.json"
    if [ -f "$SCORE_PATH" ]; then
        SCORE_FILES="${SCORE_FILES} ${SCORE_PATH}"
        echo "Fold $FOLD prediction done."
    elif [ -f "${RESULTS_DIR}/score.json" ]; then
        # fallback: 如果 val/ 不存在，检查根目录
        SCORE_FILES="${SCORE_FILES} ${RESULTS_DIR}/score.json"
        echo "Fold $FOLD prediction done."
    else
        echo "ERROR: Fold $FOLD prediction failed (no score.json)"
        FAILED=1
    fi
done

if [ $FAILED -eq 1 ] && [ -z "$SCORE_FILES" ]; then
    echo "All folds failed, exiting."
    exit 1
fi

# 统计fold数量
NUM_FOLDS=$(echo $SCORE_FILES | wc -w)
echo ""
echo "=============================================="
echo "集成 ${NUM_FOLDS} 个fold的预测结果"
echo "=============================================="

# 运行集成
ENSEMBLE_DIR="${PRED_BASE}/ensemble_${NUM_FOLDS}folds"
mkdir -p "$ENSEMBLE_DIR"

$PYTHON ./main/ensemble_scores.py \
    --score-files $SCORE_FILES \
    --anno-file "$TEST_ANNO" \
    --output-dir "$ENSEMBLE_DIR" \
    --threshold-optimize

echo ""
echo "=============================================="
echo "集成结果保存在: $ENSEMBLE_DIR"
echo "=============================================="
