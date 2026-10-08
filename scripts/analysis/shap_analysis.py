#!/usr/bin/env python3
"""
SHAP分析脚本：基于推理得到的25个特征预测值，计算SHAP值并生成图表。

用法：
    python shap_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UKai CN', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 25个征象的英文列名及中文名
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

FEATURE_NAMES_CN = [
    '动脉期高强化', '非边缘动脉期高强化', '环形动脉期高强化', '非边缘廓清',
    '周边“廓清”', '晕状强化', '强化包膜', '非强化包膜',
    '周边不连续结节状强化', '渐进性强化', '向心性增强', '平行血池强化',
    '均匀动脉期强化', '均匀门脉期强化', '均匀延迟期强化', '坏死或严重缺血',
    '瘤内出血', '结中结 结构', '马赛克结构 / 镶嵌样结构', '延迟期中央强化',
    '浸润性外观', '门脉期周围低强化', '病灶内脂肪（含量多于肝脏）',
    '实性病灶内脂肪缺失', '瘤内动脉',
]


def load_feature_filter(train_accuracy_csv='shap_outputs/Model1/train_5fold_feature_accuracy.csv',
                        threshold=80.0):
    """从训练集5折交叉验证的准确率表加载并过滤征象（保留 accuracy > threshold%）"""
    import os
    selected_indices = []
    excluded_indices = []
    feat_acc = {}

    if os.path.exists(train_accuracy_csv):
        df = pd.read_csv(train_accuracy_csv, encoding='utf-8-sig')
        name_col = df.columns[0]
        acc_col = df.columns[1]
        for i, en_name in enumerate(FEATURE_NAMES_EN):
            match = df[df[name_col].str.strip().str.lower() == en_name.strip().lower()]
            if len(match) == 0:
                for kw in ['portal', 'fat in mass', 'mosaic', 'blood products']:
                    if kw in en_name.lower():
                        match = df[df[name_col].str.contains(kw, case=False, na=False)]
                        if len(match) > 0:
                            break
            if len(match) > 0:
                acc = float(match.iloc[0][acc_col])
                feat_acc[en_name] = acc
                if acc > threshold:
                    selected_indices.append(i)
                else:
                    excluded_indices.append((i, acc))
            else:
                print(f"  [WARNING] Feature not found: {en_name}")
                excluded_indices.append((i, -1))
        print(f"Loaded feature accuracy from 5-fold CV: {train_accuracy_csv}")
    else:
        print(f"[WARNING] 5-fold CV accuracy not found: {train_accuracy_csv}")
        print(f"  Using all 25 features (no filtering).")
        selected_indices = list(range(len(FEATURE_NAMES_EN)))

    return selected_indices, excluded_indices, feat_acc



def load_data(val_csv='results/features_pred.csv', test_csv='results/features_pred_test.csv'):
    """加载推理结果，合并测试集和外部验证集"""
    df_val = pd.read_csv(val_csv, encoding='utf-8-sig')
    df_test = pd.read_csv(test_csv, encoding='utf-8-sig')
    df_val['source'] = 'val'
    df_test['source'] = 'test'
    df_all = pd.concat([df_val, df_test], ignore_index=True)
    print(f"Loaded {len(df_val)} val + {len(df_test)} test = {len(df_all)} total samples")
    return df_all


def prepare_features(df, selected_indices):
    """提取保留征象的预测值作为SHAP输入"""
    pred_cols = [f'pred_{FEATURE_NAMES_EN[i]}' for i in selected_indices]
    X = df[pred_cols].values
    # 病理类别编码
    pathology_map = {'liangxing': 0, 'noHCC': 1, 'HCC': 2}
    y = df['pathology'].map(pathology_map).values
    labels = df['pathology'].values
    return X, y, labels


def compute_shap_values(X, y):
    """使用sklearn RandomForest + shap计算SHAP值"""
    from sklearn.ensemble import RandomForestClassifier
    import shap

    # 训练一个随机森林分类器
    print("Training RandomForest classifier...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    print("Training complete.")

    # 使用TreeExplainer计算SHAP值
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    print("SHAP computation complete.")

    return rf, explainer, shap_values


def plot_shap_bar(df, pathology_class, shap_values, X, feature_names, output_dir, source=''):
    """生成单个病理类别的SHAP条形图"""
    # 筛选指定病理类别的样本
    mask = df['pathology'] == pathology_class
    if mask.sum() == 0:
        print(f"Warning: No samples found for {pathology_class}")
        return

    class_idx_map = {'liangxing': 0, 'noHCC': 1, 'HCC': 2}
    class_idx = class_idx_map.get(pathology_class, 0)

    # 获取该类别的SHAP值
    if isinstance(shap_values, list):
        sv = shap_values[class_idx][mask]
    else:
        # shap_values shape: (n_samples, n_features, n_classes)
        sv = shap_values[mask, :, class_idx]
    X_class = X[mask]

    # 计算平均SHAP值
    mean_shap = np.mean(np.abs(sv), axis=0)

    # 创建DataFrame用于排序
    df_shap = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_shap
    })
    df_shap = df_shap.sort_values('mean_abs_shap', ascending=True)

    # 画图
    plt.figure(figsize=(10, 12))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df_shap)))
    bars = plt.barh(df_shap['feature'], df_shap['mean_abs_shap'], color=colors)
    plt.xlabel('Mean |SHAP value|', fontsize=12)
    plt.title(f'SHAP - {pathology_class} (n={mask.sum()})', fontsize=14)
    plt.tight_layout()

    suffix = f"_{source}" if source else ""
    save_path = os.path.join(output_dir, f'shap_bar_{pathology_class}{suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_heatmap(df, shap_values, X, feature_names, output_dir, source=''):
    """生成SHAP汇总热图（所有样本）"""
    import shap

    # 创建SHAP解释器对象
    class_idx_map = {'liangxing': 0, 'noHCC': 1, 'HCC': 2}

    # 对每种病理类别画summary plot
    for pathology_class in ['liangxing', 'noHCC', 'HCC']:
        mask = df['pathology'] == pathology_class
        if mask.sum() == 0:
            continue

        class_idx = class_idx_map[pathology_class]
        if isinstance(shap_values, list):
            sv = shap_values[class_idx][mask]
        else:
            sv = shap_values[mask, :, class_idx]
        X_class = X[mask]

        plt.figure(figsize=(16, 10))
        shap.summary_plot(sv, X_class, feature_names=feature_names,
                          show=False, max_display=20, plot_size=(16, 10))
        plt.title(f'SHAP Summary - {pathology_class} (n={mask.sum()})', fontsize=14)
        plt.tight_layout()

        suffix = f"_{source}" if source else ""
        save_path = os.path.join(output_dir, f'shap_summary_{pathology_class}{suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def main():
    output_dir = 'shap_outputs'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载征象过滤（基于训练集5折交叉验证准确率）
    selected_indices, excluded_indices, feat_acc = load_feature_filter()
    sel_names = [FEATURE_NAMES_EN[i] for i in selected_indices]
    print(f"\nSHAP分析将使用 {len(selected_indices)} 个征象（训练集5折CV准确率 > 80%）")
    for i in selected_indices:
        print(f"  - {FEATURE_NAMES_CN[i]} ({FEATURE_NAMES_EN[i]}): {feat_acc.get(FEATURE_NAMES_EN[i], '?'):.2f}%")
    for i, acc in excluded_indices:
        print(f"  [已过滤] {FEATURE_NAMES_CN[i]} ({FEATURE_NAMES_EN[i]}): {acc*100 if acc > 0 else '?'}%")

    # 2. 加载数据
    df = load_data()

    # 3. 提取特征
    X, y, labels = prepare_features(df, selected_indices)
    print(f"Feature matrix shape: {X.shape}")

    # 4. 计算SHAP值
    rf, explainer, shap_values = compute_shap_values(X, y)

    # 5. 生成3张SHAP条形图（按病理类别）
    print("\nGenerating SHAP bar plots...")
    for pathology_class in ['liangxing', 'noHCC', 'HCC']:
        plot_shap_bar(df, pathology_class, shap_values, X, sel_names, output_dir)

    # 6. 生成汇总热图（测试集+外部验证集）
    print("\nGenerating summary heatmaps...")
    plot_summary_heatmap(df, shap_values, X, sel_names, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
