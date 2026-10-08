#!/bin/bash
# LIFT 训练脚本（已适配4模态数据）
# 先运行: python split_labels.py 生成交叉验证标签文件

###### 5折交叉验证训练（uniformer_small_IL，适合快速验证）

# fold1
python ./main/train.py --workers 8 --data_dir ./data/classification_dataset/images \
--train_anno_file ./data/classification_dataset/labels/train_fold1.txt --val_anno_file ./data/classification_dataset/labels/val_fold1.txt \
--model uniformer_small_IL --num-classes 3 --lr 1e-4 --warmup-epochs 5 --batch-size=4 --epochs 100 \
--output ./ckpts/output_uniformerS_fold1

# fold2
python ./main/train.py --workers 8 --data_dir ../data/classification_dataset/images \
--train_anno_file ../data/classification_dataset/labels/train_fold2.txt --val_anno_file ../data/classification_dataset/labels/val_fold2.txt \
--model uniformer_small_IL --num-classes 3 --lr 1e-4 --warmup-epochs 5 --batch-size=4 --epochs 100 \
--output ./ckpts/output_uniformerS_fold2

# fold3
python ./main/train.py --workers 8 --data_dir ../data/classification_dataset/images \
--train_anno_file ../data/classification_dataset/labels/train_fold3.txt --val_anno_file ../data/classification_dataset/labels/val_fold3.txt \
--model uniformer_small_IL --num-classes 3 --lr 1e-4 --warmup-epochs 5 --batch-size=4 --epochs 100 \
--output ./ckpts/output_uniformerS_fold3

# fold4
python ./main/train.py --workers 8 --data_dir ../data/classification_dataset/images \
--train_anno_file ../data/classification_dataset/labels/train_fold4.txt --val_anno_file ../data/classification_dataset/labels/val_fold4.txt \
--model uniformer_small_IL --num-classes 3 --lr 1e-4 --warmup-epochs 5 --batch-size=4 --epochs 100 \
--output ./ckpts/output_uniformerS_fold4

# fold5
python ./main/train.py --workers 8 --data_dir ../data/classification_dataset/images \
--train_anno_file ../data/classification_dataset/labels/train_fold5.txt --val_anno_file ../data/classification_dataset/labels/val_fold5.txt \
--model uniformer_small_IL --num-classes 3 --lr 1e-4 --warmup-epochs 5 --batch-size=4 --epochs 100 \
--output ./ckpts/output_uniformerS_fold5