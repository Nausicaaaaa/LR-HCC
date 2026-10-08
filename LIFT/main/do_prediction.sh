#!/bin/bash
# LIFT 预测脚本（已适配4模态数据）
# 注意：测试集图像目前为空，需先准备测试数据

###### 单模型推理示例（在验证集上推理）
# python ./main/predict.py \
# --model uniformer_small_IL --num-classes 3 \
# --data_dir ../data/classification_dataset/images \
# --val_anno_file ../data/classification_dataset/labels/ALL \
# --case-mapping-file ../../case_name_mapping2.txt \
# --checkpoint ./ckpts/test_run/best_f1_checkpoint-xx.pth.tar \
# --results-dir ./pred_results/test_run --score-dir ./pred_results/test_run \
# -j 0 -b 1

###### 以下为原始5折交叉验证推理示例（需有对应的checkpoint和交叉验证标签）
# ###### Step 1. Model Groups A Inference (uniformer-B with mixup)
# # fold1
# python ./main/predict.py --model uniformer_base_IL --num-classes 3 \
# --data_dir data/test_set/images --val_anno_file data/test_set/labels_test_inaccessible.txt \
# --checkpoint ./ckpts/output_uniformerB_mixup_bs4/fold1_best_f1_checkpoint-54.pth.tar -j 0 \
# --results-dir ./pred_results/output_uniformerB_mixup_bs4/fold1 --score-dir ./pred_results/output_uniformerB_mixup_bs4/fold1 -b 1
# # ... (fold2~fold5 类似，已省略)
#
# ###### Step 2. Model Groups B Inference (uniformer-B with channel cutout)
# # ... (fold1~fold5 类似，已省略)
