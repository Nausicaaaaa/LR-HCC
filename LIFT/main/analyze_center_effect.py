#!/usr/bin/env python3
"""分析开发集(All.txt)和测试集(test.txt)之间的中心差异"""
import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy import stats

# ========== 配置路径 ==========
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
labels_dir = os.path.join(base_dir, 'data', 'labels')
project_root = os.path.dirname(base_dir)

all_file = os.path.join(labels_dir, 'All.txt')
test_file = os.path.join(labels_dir, 'test.txt')
mapping_dev = os.path.join(project_root, 'case_name_mapping.txt')
mapping_test = os.path.join(project_root, 'case_name_mapping_test.txt')


def get_center_from_original_name(orig_name):
    """根据原始 case name 推断所属中心"""
    orig_upper = orig_name.strip().upper()
    
    # ZLCT 前缀 -> 中心3
    if orig_upper.startswith('ZLCT'):
        return 'Center_3 (ZLCT)'
    # JCT 前缀 -> 中心2
    if orig_upper.startswith('JCT'):
        return 'Center_2 (JCT)'
    # CT 前缀 -> 中心1
    if orig_upper.startswith('CT'):
        return 'Center_1 (CT)'
    # P00 前缀 -> 中心5 (测试集新中心)
    if orig_upper.startswith('P00'):
        return 'Center_5 (P00x)'
    # 字母+数字 (A-Z开头, 非CT/JCT/ZLCT/P00)
    if re.match(r'^[A-Z]\d', orig_upper) and not orig_upper.startswith('CT') and not orig_upper.startswith('JCT'):
        return 'Center_6 (Letter+Num)'
    # T00 前缀
    if orig_upper.startswith('T00'):
        return 'Center_6 (Letter+Num)'
    # Q 前缀
    if orig_upper.startswith('Q00'):
        return 'Center_6 (Letter+Num)'
    # 纯数字
    if re.match(r'^\d+$', orig_name.strip()):
        num = orig_name.strip()
        if len(num) == 10 and num.startswith('800'):
            return 'Center_4 (800x)'
        else:
            return 'Center_4 (Numeric)'
    return 'Unknown'


def load_label_file(filepath):
    """加载标签文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first_line:
                header = line.split('\t')
                first_line = False
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                data.append(parts)
    return header, data


def load_mapping(filepath):
    """加载case name映射"""
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                orig = parts[0]
                converted = parts[1]
                mapping[converted] = orig
    return mapping


# ========== 主分析 ==========
print("=" * 80)
print("中心差异分析报告")
print("=" * 80)

# 1. 加载数据
header_dev, dev_data = load_label_file(all_file)
header_test, test_data = load_label_file(test_file)

dev_mapping = load_mapping(mapping_dev)
test_mapping = load_mapping(mapping_test)

print(f"\n开发集 (All.txt): {len(dev_data)} 样本")
print(f"测试集 (test.txt): {len(test_data)} 样本")

# 2. 为每个样本分配中心
dev_centers = []
dev_center_detail = []
for row in dev_data:
    case_name = row[0]
    orig_name = dev_mapping.get(case_name, case_name)
    center = get_center_from_original_name(orig_name)
    dev_centers.append(center)
    dev_center_detail.append((case_name, orig_name, center))

test_centers = []
test_center_detail = []
for row in test_data:
    case_name = row[0]
    orig_name = test_mapping.get(case_name, case_name)
    center = get_center_from_original_name(orig_name)
    test_centers.append(center)
    test_center_detail.append((case_name, orig_name, center))

# 3. 统计中心分布
dev_center_counts = Counter(dev_centers)
test_center_counts = Counter(test_centers)

all_centers = sorted(set(list(dev_center_counts.keys()) + list(test_center_counts.keys())))

print("\n" + "=" * 80)
print("1. 中心分布统计")
print("=" * 80)

print(f"\n{'中心':<25} {'开发集':>10} {'占比':>10} {'测试集':>10} {'占比':>10}")
print("-" * 70)
for center in all_centers:
    dev_n = dev_center_counts.get(center, 0)
    test_n = test_center_counts.get(center, 0)
    dev_pct = dev_n / len(dev_data) * 100
    test_pct = test_n / len(test_data) * 100
    print(f"{center:<25} {dev_n:>10} {dev_pct:>9.1f}% {test_n:>10} {test_pct:>9.1f}%")

print("-" * 70)
print(f"{'合计':<25} {len(dev_data):>10} {'100.0%':>10} {len(test_data):>10} {'100.0%':>10}")

# 4. 查看测试集的原始命名模式
print("\n" + "=" * 80)
print("2. 测试集原始命名模式分析")
print("=" * 80)

test_orig_patterns = Counter()
for case_name, orig_name, center in test_center_detail:
    # 提取前缀模式
    if re.match(r'^[A-Z]', orig_name):
        prefix = orig_name[0]
        test_orig_patterns[f"Letter_{prefix}+digits"] += 1
    elif re.match(r'^\d+$', orig_name):
        test_orig_patterns[f"Numeric ({len(orig_name)} digits)"] += 1
    else:
        test_orig_patterns[f"Other: {orig_name[:10]}"] += 1

print("\n测试集原始名称模式:")
for pattern, count in sorted(test_orig_patterns.items(), key=lambda x: -x[1]):
    print(f"  {pattern}: {count}")

# 开发集也看一下
dev_orig_patterns = Counter()
for case_name, orig_name, center in dev_center_detail:
    if orig_name.upper().startswith('ZLCT'):
        dev_orig_patterns['ZLCT+num'] += 1
    elif orig_name.upper().startswith('JCT'):
        dev_orig_patterns['JCT+num'] += 1
    elif orig_name.upper().startswith('CT'):
        dev_orig_patterns['CT+num'] += 1
    elif re.match(r'^\d+$', orig_name.strip()):
        dev_orig_patterns[f'Numeric'] += 1
    else:
        dev_orig_patterns[f'Other: {orig_name[:15]}'] += 1

print("\n开发集原始名称模式:")
for pattern, count in sorted(dev_orig_patterns.items(), key=lambda x: -x[1]):
    print(f"  {pattern}: {count}")

# 5. 标签分布对比（按中心）
print("\n" + "=" * 80)
print("3. 各中心的诊断标签分布 (label1)")
print("=" * 80)

# 构建 DataFrame
dev_df = pd.DataFrame(dev_data, columns=header_dev)
dev_df['center'] = dev_centers
test_df = pd.DataFrame(test_data, columns=header_test)
test_df['center'] = test_centers

# 开发集各中心的标签分布
print("\n--- 开发集 (All.txt) 各中心诊断分布 ---")
dev_label_center = pd.crosstab(dev_df['label1'], dev_df['center'])
print(dev_label_center)
print("\n百分比:")
dev_label_center_pct = pd.crosstab(dev_df['label1'], dev_df['center'], normalize='columns') * 100
print(dev_label_center_pct.round(1))

# 测试集各中心的标签分布
print("\n--- 测试集 (test.txt) 各中心诊断分布 ---")
test_label_center = pd.crosstab(test_df['label1'], test_df['center'])
print(test_label_center)
print("\n百分比:")
test_label_center_pct = pd.crosstab(test_df['label1'], test_df['center'], normalize='columns') * 100
print(test_label_center_pct.round(1))

# 6. LR等级分布对比（按中心）
print("\n" + "=" * 80)
print("4. 各中心的 LI-RADS 等级分布 (label2)")
print("=" * 80)

print("\n--- 开发集各中心 LR 分布 ---")
dev_lr_center = pd.crosstab(dev_df['label2'], dev_df['center'])
print(dev_lr_center)
print("\n百分比:")
dev_lr_center_pct = pd.crosstab(dev_df['label2'], dev_df['center'], normalize='columns') * 100
print(dev_lr_center_pct.round(1))

print("\n--- 测试集各中心 LR 分布 ---")
test_lr_center = pd.crosstab(test_df['label2'], test_df['center'])
print(test_lr_center)
print("\n百分比:")
test_lr_center_pct = pd.crosstab(test_df['label2'], test_df['center'], normalize='columns') * 100
print(test_lr_center_pct.round(1))

# 7. 统计检验
print("\n" + "=" * 80)
print("5. 统计检验")
print("=" * 80)

# 7a. 开发集内部: 不同中心之间的诊断分布差异（卡方检验）
print("\n--- 5a. 开发集内部各中心之间的诊断分布差异 (Chi-square) ---")
contingency_dev = pd.crosstab(dev_df['center'], dev_df['label1'])
chi2_dev, p_dev, dof_dev, expected_dev = stats.chi2_contingency(contingency_dev)
print(f"卡方值 = {chi2_dev:.4f}, 自由度 = {dof_dev}, p值 = {p_dev:.6f}")
if p_dev < 0.05:
    print(">>> 开发集内部各中心的诊断分布存在显著差异 (p < 0.05)")
else:
    print(">>> 开发集内部各中心的诊断分布无显著差异 (p >= 0.05)")

# 7b. 开发集内部: LR等级分布差异
print("\n--- 5b. 开发集内部各中心之间的 LR 等级分布差异 (Chi-square) ---")
contingency_lr_dev = pd.crosstab(dev_df['center'], dev_df['label2'])
chi2_lr_dev, p_lr_dev, dof_lr_dev, _ = stats.chi2_contingency(contingency_lr_dev)
print(f"卡方值 = {chi2_lr_dev:.4f}, 自由度 = {dof_lr_dev}, p值 = {p_lr_dev:.6f}")
if p_lr_dev < 0.05:
    print(">>> 开发集内部各中心的 LR 等级分布存在显著差异 (p < 0.05)")
else:
    print(">>> 开发集内部各中心的 LR 等级分布无显著差异 (p >= 0.05)")

# 7c. 开发集 vs 测试集: 整体诊断分布比较
# 需要先统一中心：这里比较"开发集整体" vs "测试集整体"
print("\n--- 5c. 开发集整体 vs 测试集整体 诊断分布比较 ---")
dev_labels = dev_df['label1'].value_counts()
test_labels = test_df['label1'].value_counts()
all_labels = sorted(set(list(dev_labels.index) + list(test_labels.index)))
dev_arr = [dev_labels.get(l, 0) for l in all_labels]
test_arr = [test_labels.get(l, 0) for l in all_labels]
contingency_overall = np.array([dev_arr, test_arr])
chi2_ov, p_ov, dof_ov, _ = stats.chi2_contingency(contingency_overall)
print(f"卡方值 = {chi2_ov:.4f}, 自由度 = {dof_ov}, p值 = {p_ov:.6f}")
print(f"开发集: {dict(zip(all_labels, dev_arr))}")
print(f"测试集: {dict(zip(all_labels, test_arr))}")

# 7d. 开发集 vs 测试集: LR等级分布比较
print("\n--- 5d. 开发集整体 vs 测试集整体 LR 等级分布比较 ---")
dev_lr = dev_df['label2'].value_counts()
test_lr = test_df['label2'].value_counts()
all_lr = sorted(set(list(dev_lr.index) + list(test_lr.index)))
dev_lr_arr = [dev_lr.get(l, 0) for l in all_lr]
test_lr_arr = [test_lr.get(l, 0) for l in all_lr]
contingency_lr_ov = np.array([dev_lr_arr, test_lr_arr])
chi2_lr_ov, p_lr_ov, dof_lr_ov, _ = stats.chi2_contingency(contingency_lr_ov)
print(f"卡方值 = {chi2_lr_ov:.4f}, 自由度 = {dof_lr_ov}, p值 = {p_lr_ov:.6f}")
print(f"开发集: {dict(zip(all_lr, dev_lr_arr))}")
print(f"测试集: {dict(zip(all_lr, test_lr_arr))}")

# 8. 特征分布差异分析 (各中心之间的影像特征差异)
print("\n" + "=" * 80)
print("6. 影像特征分布的中心差异分析")
print("=" * 80)

feature_cols = header_dev[3:]  # 从第4列开始是特征
print(f"\n分析 {len(feature_cols)} 个影像特征...")

# 对每个特征，检验不同中心之间的分布差异
significant_features_dev = []
print(f"\n--- 开发集内部各中心间的特征差异 (Fisher/Chi-square) ---")
print(f"{'特征名':<50} {'p值':>12} {'显著?':>8}")
print("-" * 75)

for col in feature_cols:
    try:
        dev_df[col] = pd.to_numeric(dev_df[col], errors='coerce')
        ct_dev = pd.crosstab(dev_df['center'], dev_df[col])
        if ct_dev.shape[0] > 1 and ct_dev.shape[1] > 1:
            # 如果有期望频数<5，用 Fisher；否则用卡方
            chi2, p, dof, expected = stats.chi2_contingency(ct_dev)
            min_expected = expected.min()
            if min_expected < 5:
                # Fisher only for 2x2
                if ct_dev.shape == (2, 2):
                    _, p = stats.fisher_exact(ct_dev)
                # else keep chi2 p-value but note it
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if sig:
                significant_features_dev.append((col, p, sig))
            print(f"{col[:50]:<50} {p:>12.6f} {sig:>8}")
    except Exception as e:
        print(f"{col[:50]:<50} {'ERROR':>12} {'':>8}")

print(f"\n开发集内 有显著中心差异的特征数: {len(significant_features_dev)}/{len(feature_cols)}")

# 9. 总结
print("\n" + "=" * 80)
print("7. 综合分析与建议")
print("=" * 80)

# 判断测试集是否来自完全不同的中心
dev_center_set = set(dev_centers)
test_center_set = set(test_centers)
shared_centers = dev_center_set & test_center_set
dev_only = dev_center_set - test_center_set
test_only = test_center_set - dev_center_set

print(f"\n开发集包含的中心: {sorted(dev_center_set)}")
print(f"测试集包含的中心: {sorted(test_center_set)}")
print(f"共有的中心: {sorted(shared_centers) if shared_centers else '无'}")
print(f"仅开发集的中心: {sorted(dev_only) if dev_only else '无'}")
print(f"仅测试集的中心: {sorted(test_only) if test_only else '无'}")

if not shared_centers:
    print("\n*** 重要发现: 开发集和测试集没有共享的中心！***")
    print("    测试集完全来自外部不同中心/机构。")

print("\n" + "=" * 80)
print("中心差异分析完成！")
print("=" * 80)
