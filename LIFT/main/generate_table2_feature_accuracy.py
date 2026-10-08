#!/usr/bin/env python3
"""
生成表2：基于成像特征分类模型在训练集5重交叉验证中预测25个影像学特征的准确性

表2列：
- 成像特征
- Accuracy (5折平均准确率)
- SD (%) (5折准确率的标准差，以百分比表示)
- No. of Lesions without Feature Present (训练集中阴性病灶数)
- No. of Lesions with Feature Present (训练集中阳性病灶数)

用法：
    python generate_table2_feature_accuracy.py
"""

import os
import sys
import re
import glob
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# 10个征象英文名（与 mp_liver_dataset.py / predict.py 一致）
# 原始索引(从25个征象中筛选): 1, 3, 6, 8, 9, 10, 11, 14, 24
# 第10个为肿瘤大小（连续变量，从 data1.xlsx 第16列提取）
FEATURE_COL_NAMES = [
    'Nonrim arterial phase hyperenhancement',   # 原 idx=1, data1.xlsx col 18
    'Nonperipheral washout',                     # 原 idx=3, data1.xlsx col 20
    'Enhancing capsule',                         # 原 idx=6, data1.xlsx col 23
    'Peripheral Discontinuous Nodular Enhancement', # 原 idx=8, data1.xlsx col 25
    'Progressive Enhancement',                   # 原 idx=9, data1.xlsx col 26
    'Centripetal Enhancement',                   # 原 idx=10, data1.xlsx col 27
    'Parallels blood pool enhancement',          # 原 idx=11, data1.xlsx col 28
    'Uniform DP Enhancement',                    # 原 idx=14, data1.xlsx col 31
    'Intratumoral artery',                       # 原 idx=24, data1.xlsx col 41
    'Tumor size (normalized)',                   # data1.xlsx col 16, 连续变量
]

# 9个二分类征象在 data1.xlsx 中的列索引（0-indexed）
FEATURE_DATA1_COL_INDICES = [18, 20, 23, 25, 26, 27, 28, 31, 41]
# 肿瘤大小在 data1.xlsx 中的列索引
TUMOR_SIZE_COL_INDEX = 16

NUM_FEATURES = len(FEATURE_COL_NAMES)  # 10


def load_data1_features(data1_path, mapping_path):
    """从 data1.xlsx 加载 10 维征象标签（9个二分类 + 肿瘤大小）

    Returns:
        dict: {converted_case_name: [10维特征值列表]}
    """
    df = pd.read_excel(data1_path, header=None)
    # 第0,1行为header，第2行开始为数据
    # 第0列是ID(original_case_name)
    # 二分类征象列由 FEATURE_DATA1_COL_INDICES 指定
    # 肿瘤大小由 TUMOR_SIZE_COL_INDEX 指定

    # 读取映射表
    map_df = pd.read_csv(mapping_path, sep='\t')
    orig_to_case = {}
    for _, row in map_df.iterrows():
        orig = str(row.iloc[0]).strip()
        case = str(row.iloc[1]).strip()
        orig_to_case[orig] = case

    features = {}
    for idx in range(2, len(df)):
        val = df.iloc[idx, 0]
        if pd.isna(val):
            continue
        orig_id = str(val).strip()
        case_name = orig_to_case.get(orig_id)
        if case_name is None:
            continue

        feat_vals = []
        valid = True
        # 加载9个二分类征象
        for col_idx in FEATURE_DATA1_COL_INDICES:
            v = df.iloc[idx, col_idx]
            if pd.isna(v):
                valid = False
                break
            feat_vals.append(float(v))

        if not valid or len(feat_vals) != len(FEATURE_DATA1_COL_INDICES):
            continue

        # 加载肿瘤大小（第10个征象，归一化到0~1）
        tumor_val = df.iloc[idx, TUMOR_SIZE_COL_INDEX]
        if pd.isna(tumor_val):
            continue
        if isinstance(tumor_val, str):
            match = re.search(r'[-+]?[0-9]*\.?[0-9]+', tumor_val)
            if match:
                tumor_val = float(match.group())
            else:
                continue
        else:
            tumor_val = float(tumor_val)
        feat_vals.append(tumor_val / 100.0)  # 归一化

        features[case_name] = feat_vals

    return features


def load_all_feature_labels(all_txt_path):
    """从 All.txt 加载所有 case 的 9 维征象标签

    Args:
        all_txt_path: All.txt 文件路径

    Returns:
        dict: {case_name: [9维特征值列表]}
        list: 9个特征的中文名称列表
    """
    with open(all_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 第一行是header
    header = lines[0].strip().split('\t')
    # 前3列是 casename, label1, label2，后面是征象
    feature_names = header[3:3+NUM_FEATURES]

    features = {}
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 3 + NUM_FEATURES:
            continue
        case_name = parts[0]
        try:
            feat_vals = [float(parts[j]) for j in range(3, 3 + NUM_FEATURES)]
        except ValueError:
            continue
        features[case_name] = feat_vals

    return features, feature_names


def count_feature_distribution(features_dict):
    """统计所有样本中每个特征的阴性和阳性数量

    对于二分类征象(0~8): 统计0和1的数量
    对于肿瘤大小(第9个): 统计<0.5(50mm)和≥0.5(50mm)的数量

    Args:
        features_dict: {case_name: [NUM_FEATURES维特征值]}

    Returns:
        tuple: (negative_counts, positive_counts) 各为NUM_FEATURES维列表
    """
    negative_counts = [0] * NUM_FEATURES
    positive_counts = [0] * NUM_FEATURES

    for case_name, feat_vals in features_dict.items():
        for i in range(NUM_FEATURES):
            if i < len(FEATURE_DATA1_COL_INDICES):
                # 二分类征象
                if feat_vals[i] == 1.0:
                    positive_counts[i] += 1
                else:
                    negative_counts[i] += 1
            else:
                # 肿瘤大小（连续变量）：以0.5(50mm)为阈值
                if feat_vals[i] >= 0.5:
                    positive_counts[i] += 1
                else:
                    negative_counts[i] += 1

    return negative_counts, positive_counts


def extract_feature_accuracies_from_summary(summary_path, num_features=None):
    """从 summary.csv 提取每个特征的验证集准确率

    策略：取最后一行（最终epoch）的 eval_feature_{i}_acc 值

    Returns:
        list: NUM_FEATURES维准确率列表，如果文件不存在或没有feature列则返回None
    """
    if num_features is None:
        num_features = NUM_FEATURES
    if not os.path.exists(summary_path):
        return None

    df = pd.read_csv(summary_path)
    if len(df) == 0:
        return None

    # 检查是否包含 feature accuracy 列
    if 'eval_feature_0_acc' not in df.columns:
        return None

    # 取最后一行
    last_row = df.iloc[-1]
    accuracies = []
    for i in range(num_features):
        col_name = f'eval_feature_{i}_acc'
        if col_name in last_row:
            accuracies.append(float(last_row[col_name]))
        else:
            accuracies.append(np.nan)
    return accuracies


def generate_table2(data1_path, mapping_path, all_txt_path, ckpt_dirs, output_path, num_folds=5):
    """生成表2：10个征象的预测准确性"""

    # 1. 加载所有case的征象标签
    print("Loading data1.xlsx features...")
    features_dict = load_data1_features(data1_path, mapping_path)
    print(f"  Loaded {len(features_dict)} cases with feature labels")

    # 2. 从 All.txt 加载征象标签并统计阴性和阳性数量
    print("Loading All.txt for feature distribution...")
    all_features, feature_names_cn = load_all_feature_labels(all_txt_path)
    print(f"  Loaded {len(all_features)} cases from All.txt")
    negative_counts, positive_counts = count_feature_distribution(all_features)
    print(f"  Feature distribution computed")

    # 4. 从每个fold的summary.csv提取准确率
    print("Extracting feature accuracies from checkpoints...")
    fold_accuracies = []  # list of list，每个fold的25维准确率
    for ckpt_dir in ckpt_dirs:
        summary_path = os.path.join(ckpt_dir, 'summary.csv')
        accs = extract_feature_accuracies_from_summary(summary_path)
        if accs is not None:
            fold_accuracies.append(accs)
            print(f"  {ckpt_dir}: extracted accuracies")
        else:
            print(f"  {ckpt_dir}: summary.csv not found or empty")

    # 5. 计算均值和标准差
    if len(fold_accuracies) > 0:
        acc_matrix = np.array(fold_accuracies)  # shape: (num_folds, NUM_FEATURES)
        mean_accs = acc_matrix.mean(axis=0)
        std_accs = acc_matrix.std(axis=0, ddof=1) if len(fold_accuracies) > 1 else np.zeros(NUM_FEATURES)
        print(f"  Computed mean and std across {len(fold_accuracies)} fold(s)")
    else:
        mean_accs = np.zeros(NUM_FEATURES)
        std_accs = np.zeros(NUM_FEATURES)
        print("  Warning: no checkpoint data found, accuracies will be 0")

    # 6. 生成表格
    rows = []
    for i in range(NUM_FEATURES):
        rows.append({
            'Imaging Feature': FEATURE_COL_NAMES[i],
            'Accuracy (%)': f"{mean_accs[i] * 100:.2f}",
            'SD (%)': f"{std_accs[i] * 100:.2f}",
            'No. of Lesions without Feature Present': negative_counts[i],
            'No. of Lesions with Feature Present': positive_counts[i],
        })

    df = pd.DataFrame(rows)

    # 7. 输出
    # CSV
    csv_path = output_path.replace('.md', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nCSV saved to: {csv_path}")

    # Markdown table
    md_lines = []
    md_lines.append("# Table 2: Imaging Feature Prediction Accuracy from Cross-Validation\n")
    md_lines.append("| Imaging Feature | Accuracy (%) | SD (%) | No. of Lesions without Feature Present | No. of Lesions with Feature Present |")
    md_lines.append("|---|---|---|---|---|")
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['Imaging Feature']} | {row['Accuracy (%)']} | {row['SD (%)']} | "
            f"{row['No. of Lesions without Feature Present']} | {row['No. of Lesions with Feature Present']} |"
        )

    md_path = output_path.replace('.csv', '.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"Markdown saved to: {md_path}")

    # 同时打印到控制台
    print("\n" + "=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return df


def main():
    parser = argparse.ArgumentParser(description='Generate Table 2: Feature prediction accuracy from 5-fold cross-validation')
    parser.add_argument('--data1', default='LIFT/data/data1.xlsx', help='Path to data1.xlsx')
    parser.add_argument('--mapping', default='case_name_mapping.txt', help='Path to case_name_mapping.txt')
    parser.add_argument('--all-txt', default='LIFT/data/labels/All.txt', help='Path to All.txt containing all feature labels')
    parser.add_argument('--ckpt-dirs', nargs='+', default=None, help='Checkpoint directories containing summary.csv')
    parser.add_argument('--output', default='LIFT/表2_feature_accuracy.md', help='Output file path')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of cross-validation folds')
    args = parser.parse_args()

    # 如果没有指定ckpt-dirs，自动搜索（只保留包含 feature accuracy 的目录）
    ckpt_dirs = args.ckpt_dirs
    if ckpt_dirs is None:
        ckpt_base = 'LIFT/ckpts'
        if os.path.exists(ckpt_base):
            # 查找包含 summary.csv 且有 eval_feature_0_acc 列的子目录
            ckpt_dirs = []
            for d in sorted(os.listdir(ckpt_base)):
                full_path = os.path.join(ckpt_base, d)
                summary_path = os.path.join(full_path, 'summary.csv')
                if os.path.isdir(full_path) and os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        header = f.readline()
                        if 'eval_feature_0_acc' in header:
                            ckpt_dirs.append(full_path)
            print(f"Auto-discovered {len(ckpt_dirs)} checkpoint directories with feature accuracy")
        else:
            ckpt_dirs = []
            print(f"Warning: checkpoint base directory {ckpt_base} not found")

    generate_table2(
        data1_path=args.data1,
        mapping_path=args.mapping,
        all_txt_path=args.all_txt,
        ckpt_dirs=ckpt_dirs,
        output_path=args.output,
        num_folds=args.num_folds
    )


if __name__ == '__main__':
    main()
