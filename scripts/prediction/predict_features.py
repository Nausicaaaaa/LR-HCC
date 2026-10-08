#!/usr/bin/env python3
"""
推理脚本：对指定数据集跑模型推理，保存每个样本的25个特征预测值和病理类别。

用法：
    python predict_features.py --anno_file LIFT/data/labels/val_fold1.txt --output val_features.csv
    python predict_features.py --anno_file LIFT/data/labels/test.txt --output test_features.csv
"""

import os
import sys
import argparse
import json
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

# 添加 LIFT/main 到 Python 路径
sys.path.insert(0, 'LIFT/main')

from timm.models import create_model, load_checkpoint
from torch.utils.data.dataloader import DataLoader
import models  # noqa: F401 - register model
from datasets.mp_liver_dataset import MultiPhaseLiverDataset, FEATURE_COL_NAMES

# 25个征象英文名（与predict.py中ALL_SIGN_NAMES_25保持一致，对应 data1.xlsx col 17~41）
FEATURE_NAMES_EN = [
    'Arterial phase hyperenhancement',
    'Nonrim arterial phase hyperenhancement',
    'Rim APHE',
    'Nonperipheral washout',
    'Peripheral "Washout"',
    'Corona Enhancement',
    'Enhancing capsule',
    'Nonenhancing capsule',
    'Peripheral Discontinuous Nodular Enhancement',
    'Progressive Enhancement',
    'Centripetal Enhancement',
    'Parallels blood pool enhancement',
    'Uniform AP Enhancement',
    'Uniform PVP Enhancement',
    'Uniform DP Enhancement',
    'Necrosis or severe ischemia',
    'Blood Products in Mass',
    'Nodule-in-nodule architecture',
    'Mosaic Architecture',
    'Delayed Central Enhancement',
    'Infiltrative appearance',
    'Portal venous phase peritumoral hypoenhancement',
    'Fat in Mass, more than Liver',
    'Fat Sparing in Solid Mass',
    'Intratumoral artery',
]

# 25个征象中文名
FEATURE_NAMES_CN = [
    '动脉期高强化',
    '非边缘动脉期高强化',
    '环形动脉期高强化',
    '非边缘廓清',
    '周边"廓清"',
    '晕状强化',
    '强化包膜',
    '非强化包膜',
    '周边不连续结节状强化',
    '渐进性强化',
    '向心性增强',
    '平行血池强化',
    '均匀动脉期强化',
    '均匀门脉期强化',
    '均匀延迟期强化',
    '坏死或严重缺血',
    '瘤内出血',
    '结中结 结构',
    '马赛克结构 / 镶嵌样结构',
    '延迟期中央强化',
    '浸润性外观',
    '门脉期周围低强化',
    '病灶内脂肪（含量多于肝脏）',
    '实性病灶内脂肪缺失',
    '瘤内动脉',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Predict and save 25 feature values')
    parser.add_argument('--checkpoint', default='LIFT/ckpts/uniformer_small_IL_features/model_best.pth.tar', type=str)
    parser.add_argument('--val_anno_file', default='LIFT/data/labels/val_fold1.txt', type=str, dest='val_anno_file')
    parser.add_argument('--data_dir', default='LIFT/data/images/', type=str)
    parser.add_argument('--output', default='results/features_pred.csv', type=str)
    parser.add_argument('--img_size', default=(20, 96, 96), type=int, nargs='+')
    parser.add_argument('--crop_size', default=(16, 80, 80), type=int, nargs='+')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--model', default='uniformer_small_IL_features', type=str)
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--num_feature_classes', default=25, type=int)
    parser.add_argument('--feature_fusion', default='concat', type=str)
    parser.add_argument('--label_mode', default='lr', type=str)
    parser.add_argument('--case_mapping_file', default='', type=str)
    parser.add_argument('--skip_cases_file', default='LIFT/data/skip_cases.txt', type=str)
    parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+', type=str)
    parser.add_argument('--include-clinical', action='store_true', default=False)
    parser.add_argument('--clinical-dim', default=10, type=int)
    parser.add_argument('--data1-file', default='LIFT/data/data1.xlsx', type=str)
    parser.add_argument('--normalize-clinical', action='store_true', default=False)
    return parser.parse_args()


def extract_pathology_label_from_anno(anno_path):
    """从标注文件提取每个case的病理类别（label1）
    返回: {case_name: label1_str}
    """
    df = pd.read_csv(anno_path, sep='\t', header=None)
    # 第1列是case名，第2列是label1（病理类别）
    case_to_pathology = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
    return case_to_pathology


def main():
    args = parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model_kwargs = dict(
        pretrained=False,
        num_classes=args.num_classes,
        num_feature_classes=args.num_feature_classes,
        feature_fusion=args.feature_fusion,
    )
    if args.include_clinical:
        model_kwargs['clinical_dim'] = args.clinical_dim
    model = create_model(args.model, **model_kwargs)
    load_checkpoint(model, args.checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")
    
    # 创建数据集
    dataset = MultiPhaseLiverDataset(args, is_training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=False)
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # 提取病理类别标签
    case_to_pathology = extract_pathology_label_from_anno(args.val_anno_file)
    
    # 推理并收集结果
    all_case_names = []
    all_true_labels_lr = []   # LR分级（5分类）
    all_true_features = []    # 25个特征真实值
    all_pred_features = []    # 25个特征预测值（sigmoid输出）
    all_pred_logits = []      # 主分类logits
    all_pathology = []        # 病理类别
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            extra_features = None
            features_true = None
            if len(batch) == 5:
                inputs, targets, features_true, lr_label, extra_features = batch
            elif len(batch) == 4:
                inputs, targets, features_true, fourth = batch
                if not (isinstance(fourth, torch.Tensor) and fourth.dim() == 1):
                    extra_features = fourth
            elif len(batch) == 3:
                inputs, targets, features_true = batch
            else:
                inputs, targets = batch

            inputs = inputs.to(device)
            if extra_features is not None:
                extra_features = extra_features.to(device)

            if extra_features is not None:
                output = model(inputs, extra_features=extra_features)
            else:
                output = model(inputs)
            if isinstance(output, (tuple, list)):
                main_logits = output[0]
                feature_pred = output[1] if len(output) >= 2 and output[1] is not None and output[1].dim() == 2 else None
            else:
                main_logits = output
                feature_pred = None
            
            all_pred_logits.append(main_logits.cpu().numpy())
            
            if features_true is not None:
                all_true_features.append(features_true.cpu().numpy())
            
            if feature_pred is not None:
                all_pred_features.append(feature_pred.cpu().numpy())
            
            all_true_labels_lr.append(targets.cpu().numpy())
    
    # 合并结果
    all_pred_logits = np.concatenate(all_pred_logits, axis=0)
    all_true_labels_lr = np.concatenate(all_true_labels_lr, axis=0)
    
    if all_true_features:
        all_true_features = np.concatenate(all_true_features, axis=0)
    else:
        all_true_features = np.zeros((len(all_true_labels_lr), 25))
    
    if all_pred_features:
        all_pred_features = np.concatenate(all_pred_features, axis=0)
    else:
        all_pred_features = np.zeros((len(all_true_labels_lr), 25))
    
    # 从数据集中获取case名和病理类别
    # 遍历数据集获取每个样本的信息
    case_names = []
    pathology_labels = []
    
    for i in range(len(dataset)):
        # 获取图像路径，从中提取case名
        img_paths = dataset.img_list[i]
        case_name = os.path.basename(os.path.dirname(img_paths[0]))
        case_names.append(case_name)
        
        # 获取病理类别
        pathology = case_to_pathology.get(case_name, 'Unknown')
        pathology_labels.append(pathology)
    
    # 保存结果，列名仅使用英文
    results = {
        'case_name': case_names,
        'pathology': pathology_labels,
        'true_label_lr': all_true_labels_lr.tolist(),
    }
    
    # 添加25个特征的真实值和预测值
    for i in range(25):
        results[f'true_{FEATURE_NAMES_EN[i]}'] = all_true_features[:, i].tolist()
        results[f'pred_{FEATURE_NAMES_EN[i]}'] = all_pred_features[:, i].tolist()
    
    # 添加主分类预测概率（softmax）
    pred_probs = torch.softmax(torch.from_numpy(all_pred_logits), dim=1).numpy()
    for i in range(args.num_classes):
        results[f'pred_prob_class_{i}'] = pred_probs[:, i].tolist()
    
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"Results saved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Pathology distribution:\n{df['pathology'].value_counts()}")


if __name__ == '__main__':
    main()
