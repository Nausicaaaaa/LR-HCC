#!/usr/bin/env python3
"""
按 HCC 3分类（HCC/noHCC/liangxing）进行分层 5-fold 划分。
固定随机种子 seed=42，确保可复现。

输入：LIFT/data/labels/All.txt
输出：LIFT/data/labels/train_fold{i}.txt, val_fold{i}.txt (i=1..5)
"""

import os
import random
from collections import defaultdict

SEED = 42

# 配置路径
labels_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'labels')
all_file = os.path.join(labels_dir, 'All.txt')
output_dir = labels_dir

# 固定随机种子
random.seed(SEED)

# 读取 ALL 文件并按 HCC 3分类分组（第2列 label1）
stratified_data = defaultdict(list)
first_line = True

with open(all_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if first_line:
            first_line = False
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        case_name = parts[0]
        label1 = parts[1].strip()  # 第二列：HCC/noHCC/liangxing
        stratified_data[label1].append(line)

print(f"发现 {len(stratified_data)} 个分类类别")
for cls, cases in sorted(stratified_data.items()):
    print(f"  {cls}: {len(cases)} 个样本")

# 对每个类别进行 5-fold 划分
folds_train = {i: [] for i in range(1, 6)}
folds_val = {i: [] for i in range(1, 6)}

for cls, cases in sorted(stratified_data.items()):
    random.shuffle(cases)
    n_cases = len(cases)
    fold_size = n_cases // 5
    remainder = n_cases % 5

    start_idx = 0
    for fold_id in range(1, 6):
        current_fold_size = fold_size + (1 if fold_id <= remainder else 0)
        end_idx = start_idx + current_fold_size

        val_cases = cases[start_idx:end_idx]
        train_cases = cases[:start_idx] + cases[end_idx:]

        folds_val[fold_id].extend(val_cases)
        folds_train[fold_id].extend(train_cases)

        start_idx = end_idx

# 输出结果
print("\n=== 5-fold 划分结果 ===")
for fold_id in range(1, 6):
    random.shuffle(folds_train[fold_id])
    random.shuffle(folds_val[fold_id])

    train_file = os.path.join(output_dir, f'train_fold{fold_id}.txt')
    val_file = os.path.join(output_dir, f'val_fold{fold_id}.txt')

    with open(train_file, 'w') as f:
        f.write('\n'.join(folds_train[fold_id]) + '\n')
    with open(val_file, 'w') as f:
        f.write('\n'.join(folds_val[fold_id]) + '\n')

    # 统计 HCC 3分类分布
    train_dist = defaultdict(int)
    val_dist = defaultdict(int)
    for line in folds_train[fold_id]:
        label = line.split('\t')[1].strip()
        train_dist[label] += 1
    for line in folds_val[fold_id]:
        label = line.split('\t')[1].strip()
        val_dist[label] += 1

    # 统计 LR 分布
    train_lr = defaultdict(int)
    val_lr = defaultdict(int)
    for line in folds_train[fold_id]:
        lr = line.split('\t')[2].strip()
        train_lr[lr] += 1
    for line in folds_val[fold_id]:
        lr = line.split('\t')[2].strip()
        val_lr[lr] += 1

    print(f"\nFold {fold_id}: Train={len(folds_train[fold_id])}, Val={len(folds_val[fold_id])}")
    print(f"  Train 诊断: HCC={train_dist.get('HCC',0)}, noHCC={train_dist.get('noHCC',0)}, liangxing={train_dist.get('liangxing',0)}")
    print(f"  Val   诊断: HCC={val_dist.get('HCC',0)}, noHCC={val_dist.get('noHCC',0)}, liangxing={val_dist.get('liangxing',0)}")
    print(f"  Train LR: {dict(sorted(train_lr.items()))}")
    print(f"  Val   LR: {dict(sorted(val_lr.items()))}")

print(f"\n5-fold 交叉验证划分完成！(seed={SEED}, 按HCC 3分类分层)")
