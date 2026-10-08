#!/usr/bin/env python3
"""
3分类模型推理脚本：提取25个征象预测值 + 3分类概率

用法:
    python predict_features_3class.py --checkpoint PATH --anno_file PATH --output CSV
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, 'LIFT/main')

from timm.models import create_model, load_checkpoint
from torch.utils.data.dataloader import DataLoader
import models  # noqa: F401 - register model
from datasets.mp_liver_dataset import MultiPhaseLiverDataset

# 25个征象英文名
FEATURE_NAMES_EN = [
    'arterial phase hyperenhancement', 'Nonrim arterial phase hyperenhancement',
    'Rim APHE', 'Nonperipheral washout', 'Peripheral "Washout"',
    'Corona Enhancement', 'Enhancing capsule', 'Nonenhancing capsule',
    'Peripheral Discontinuous Nodular Enhancement', 'Progressive Enhancement',
    'Centripetal Enhancement', 'Parallels blood pool enhancement',
    'Uniform AP Enhancement', 'Uniform PVP Enhancement', 'Uniform DP Enhancement',
    'Necrosis or severe ischemia', 'Blood Products in Mass',
    'Nodule-in-nodule architecture', 'Mosaic Architecture',
    'Delayed Central Enhancement', 'Infiltrative appearance',
    'portal venous phase peritumoral hypoenhancement is present;',
    'Fat in Mass, more than Liver', 'Fat Sparing in Solid Mass',
    'Intratumoral artery',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Predict 25 features + 3-class probs')
    parser.add_argument('--checkpoint', default='', type=str, required=True)
    parser.add_argument('--val_anno_file', default='LIFT/data/labels/val_fold1.txt', type=str)
    parser.add_argument('--data_dir', default='LIFT/data/images/', type=str)
    parser.add_argument('--output', default='results/features_pred_3class.csv', type=str)
    parser.add_argument('--img_size', default=(20, 96, 96), type=int, nargs='+')
    parser.add_argument('--crop_size', default=(16, 80, 80), type=int, nargs='+')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--model', default='uniformer_small_IL_features', type=str)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--num_feature_classes', default=25, type=int)
    parser.add_argument('--feature_fusion', default='concat', type=str)
    parser.add_argument('--label_mode', default='original', type=str)
    parser.add_argument('--case_mapping_file', default='', type=str)
    parser.add_argument('--skip_cases_file', default='LIFT/data/skip_cases.txt', type=str)
    parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+', type=str)
    return parser.parse_args()


def extract_pathology_label_from_anno(anno_path):
    """从标注文件提取每个case的病理类别（第2列字符串）"""
    df = pd.read_csv(anno_path, sep='\t', header=None)
    case_to_pathology = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
    return case_to_pathology


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        num_feature_classes=args.num_feature_classes,
        feature_fusion=args.feature_fusion,
    )
    load_checkpoint(model, args.checkpoint, strict=False)
    model = model.to(device).eval()
    print(f"Loaded model from {args.checkpoint}")

    # 创建数据集
    dataset = MultiPhaseLiverDataset(args, is_training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=False)
    print(f"Dataset loaded: {len(dataset)} samples")

    # 提取病理类别
    case_to_pathology = extract_pathology_label_from_anno(args.val_anno_file)

    # 收集结果
    all_pred_logits = []
    all_pred_features = []
    all_true_labels = []
    all_case_names = []
    all_pathology = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs = inputs.to(device)

            output = model(inputs)
            if isinstance(output, (tuple, list)):
                main_logits = output[0]
                feature_pred = output[1] if len(output) >= 2 else None
            else:
                main_logits = output
                feature_pred = None

            all_pred_logits.append(main_logits.cpu().numpy())
            if feature_pred is not None:
                all_pred_features.append(feature_pred.cpu().numpy())
            all_true_labels.append(targets.cpu().numpy())

    # 合并结果
    all_pred_logits = np.concatenate(all_pred_logits, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    if all_pred_features:
        all_pred_features = np.concatenate(all_pred_features, axis=0)
    else:
        all_pred_features = np.zeros((len(all_true_labels), 25))

    # 获取case名和病理类别
    case_names = []
    pathology_labels = []
    for i in range(len(dataset)):
        img_paths = dataset.img_list[i]
        case_name = os.path.basename(os.path.dirname(img_paths[0]))
        case_names.append(case_name)
        pathology = case_to_pathology.get(case_name, 'Unknown')
        pathology_labels.append(pathology)

    # 构建结果
    results = {
        'case_name': case_names,
        'pathology': pathology_labels,
        'true_label': all_true_labels.tolist(),
    }

    # 25个征象预测值
    for i in range(25):
        results[f'pred_{FEATURE_NAMES_EN[i]}'] = all_pred_features[:, i].tolist()

    # 3分类预测概率
    pred_probs = torch.softmax(torch.from_numpy(all_pred_logits), dim=1).numpy()
    for i in range(args.num_classes):
        results[f'pred_prob_class_{i}'] = pred_probs[:, i].tolist()

    # 保存
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"Results saved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Pathology distribution:\n{df['pathology'].value_counts()}")


if __name__ == '__main__':
    main()
