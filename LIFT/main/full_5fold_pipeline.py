#!/usr/bin/env python3
"""5折模型完整推理流水线：OOF + 测试集集成

功能：
1. 每折运行 predict.py（val=val_fold{N}, test=test.txt）
2. OOF集成：拼接5折OOF预测 → 全量评估
3. 测试集集成：5折softmax平均 → 评估
4. 两者均输出：混淆矩阵、ROC、evaluation_metrics CSV、feature_accuracy、阈值优化

用法：
    python full_5fold_pipeline.py --folds 1 2 3 4 5
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve,
    roc_auc_score as roc_auc_fn
)
from sklearn.metrics import auc as auc_fn

LIFT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = '/mnt/data/KASR/.conda/envs/nnunetv2/bin/python'
CLASS_NAMES = ['LR-1/2', 'LR-3', 'LR-4', 'LR-5', 'LR-M']
NUM_CLASSES = 5
LR_MAPPING = {'1/2': 0, '3': 1, '4': 2, '5': 3, 'M': 4}

FEATURE_NAMES = [
    'Nonperipheral washout',
    'Peripheral Discontinuous Nodular Enhancement',
    'Parallels blood pool enhancement',
    'Intratumoral artery',
    'Tumor size (normalized)',
    'Feature 10',
    'Feature 11',
    'Uniform DP Enhancement',
]


def load_gt(anno_path):
    """加载标注文件的 case_id → label"""
    try:
        df = pd.read_csv(anno_path, sep='\t', header=0)
        if df.shape[1] < 3:
            df = pd.read_csv(anno_path, sep='\t', header=None)
    except:
        df = pd.read_csv(anno_path, sep='\t', header=None)
    mapping = {}
    for _, row in df.iterrows():
        cid = str(row.iloc[0]).strip()
        if df.shape[1] >= 3:
            lr_str = str(row.iloc[2]).strip().split()[-1]
            label = LR_MAPPING.get(lr_str, -1)
        else:
            continue
        if label >= 0:
            mapping[cid] = label
    return mapping


def load_score_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    ids = [d['image_id'] for d in data]
    scores = np.array([d['score'] for d in data], dtype=np.float64)
    preds = np.array([d['prediction'] for d in data], dtype=int)
    return ids, scores, preds


def compute_metrics(pred_labels, true_labels, pred_scores):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES)))
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        bt = (true_labels == i).astype(int)
        bp = (pred_labels == i).astype(int)
        per_class[name] = {
            'ACC': round(float(accuracy_score(bt, bp)), 4),
            'F1': round(float(f1_score(bt, bp, zero_division=0)), 4),
            'Recall': round(float(recall_score(bt, bp, zero_division=0)), 4),
            'Precision': round(float(precision_score(bt, bp, zero_division=0)), 4),
            'Kappa': round(float(cohen_kappa_score(bt, bp)), 4),
        }
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    acc = accuracy_score(true_labels, pred_labels)
    kappa = cohen_kappa_score(true_labels, pred_labels)
    # AUC (one-vs-rest)
    try:
        auc_val = roc_auc_fn(true_labels, pred_scores, multi_class='ovr', average='macro')
    except:
        auc_val = None
    return {
        'accuracy': round(float(acc), 4),
        'f1_weighted': round(float(weighted_f1), 4),
        'f1_macro': round(float(macro_f1), 4),
        'kappa': round(float(kappa), 4),
        'auc': round(float(auc_val), 4) if auc_val else None,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(true_labels),
    }


def plot_confusion_matrix(cm, output_dir, title_suffix=''):
    cm = np.array(cm)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(cm_sum > 0, cm / cm_sum * 100, 0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=.5, linecolor='gray',
                cbar_kws={'label': 'Percentage (%)'}, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix (Row-normalized %) {title_suffix}')
    axes[0].set_ylabel('True Label'); axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=.5, linecolor='gray', ax=axes[1])
    axes[1].set_title(f'Confusion Matrix (Raw Counts) {title_suffix}')
    axes[1].set_ylabel('True Label'); axes[1].set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 单独保存
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=.5, linecolor='gray', cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Confusion Matrix (Row-normalized %) {title_suffix}')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_percent.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=.5, linecolor='gray')
    plt.title(f'Confusion Matrix (Raw Counts) {title_suffix}')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_raw.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc(pred_scores, true_labels, output_dir, title_suffix=''):
    plt.figure(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        binary_labels = (true_labels == i).astype(int)
        scores = pred_scores[:, i]
        valid = np.isfinite(scores)
        if not valid.any():
            continue
        fpr, tpr, _ = roc_curve(binary_labels[valid], scores[valid])
        a = auc_fn(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{CLASS_NAMES[i]} (AUC = {a:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'Multi-class ROC (One-vs-Rest) {title_suffix}')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve_multiclass.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_csv(metrics, output_dir):
    """保存 evaluation_metrics.csv"""
    path = os.path.join(output_dir, 'evaluation_metrics.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Class', 'ACC', 'F1', 'Recall', 'Precision', 'Kappa'])
        for cn in CLASS_NAMES:
            c = metrics['per_class'][cn]
            w.writerow([cn, c['ACC'], c['F1'], c['Recall'], c['Precision'], c['Kappa']])
        w.writerow(['ALL', metrics['accuracy'], metrics['f1_macro'], '', '', metrics['kappa']])
    print(f"  Saved: {path}")


def save_feature_accuracy(feature_acc_paths, output_dir, title_suffix=''):
    """平均多折 feature_accuracy.csv 并保存"""
    if not feature_acc_paths:
        print("  WARNING: No feature_accuracy.csv files found")
        return

    all_dfs = []
    for p in feature_acc_paths:
        if os.path.exists(p):
            all_dfs.append(pd.read_csv(p))

    if not all_dfs:
        return

    # 取第一个df的特征名
    feat_names = all_dfs[0]['Imaging Feature'].tolist()
    accs = np.array([df['Accuracy (%)'].astype(float).values for df in all_dfs])
    mean_accs = accs.mean(axis=0)
    neg_counts = all_dfs[0]['No. of Lesions without Feature Present'].astype(int).values
    pos_counts = all_dfs[0]['No. of Lesions with Feature Present'].astype(int).values

    # 对于OOF，每折的neg/pos计数不同，这里取总和
    # 但实际上每折的样本不同，总数应该是所有fold之和
    if len(all_dfs) > 1:
        neg_counts = np.array([df['No. of Lesions without Feature Present'].astype(int).values for df in all_dfs]).sum(axis=0)
        pos_counts = np.array([df['No. of Lesions with Feature Present'].astype(int).values for df in all_dfs]).sum(axis=0)

    rows = []
    for i, name in enumerate(feat_names):
        rows.append({
            'Imaging Feature': name,
            'Accuracy (%)': f"{mean_accs[i]:.2f}",
            'No. of Lesions without Feature Present': int(neg_counts[i]),
            'No. of Lesions with Feature Present': int(pos_counts[i]),
        })
    df_out = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, 'feature_accuracy.csv')
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')

    md_lines = [f"# Feature Prediction Accuracy {title_suffix}\n",
                "| Imaging Feature | Accuracy (%) | No. of Lesions without Feature Present | No. of Lesions with Feature Present |",
                "|---|---|---|---|"]
    for _, row in df_out.iterrows():
        md_lines.append(f"| {row['Imaging Feature']} | {row['Accuracy (%)']} | "
                        f"{row['No. of Lesions without Feature Present']} | {row['No. of Lesions with Feature Present']} |")
    md_path = os.path.join(output_dir, 'feature_accuracy.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {md_path}")


def run_prediction(fold, ckpt_dir, output_dir):
    """运行单折预测"""
    val_anno = os.path.join(LIFT_DIR, f'data/labels/val_fold{fold}.txt')
    test_anno = os.path.join(LIFT_DIR, 'data/labels/test.txt')
    val_score = os.path.join(output_dir, 'val', 'score.json')
    test_score = os.path.join(output_dir, 'test', 'score.json')

    if os.path.exists(val_score) and os.path.exists(test_score):
        print(f"  Fold {fold}: predictions already exist, skipping.")
        return val_score, test_score

    ckpts = glob.glob(os.path.join(ckpt_dir, 'uniformer_small_IL_features', 'best_f1_checkpoint-*.pth.tar'))
    if not ckpts:
        print(f"  ERROR: No checkpoint for fold {fold}")
        return None, None

    cmd = [
        PYTHON, os.path.join(LIFT_DIR, 'main/predict.py'),
        '--model', 'uniformer_small_IL_features',
        '--num-classes', '5', '--label-mode', 'lr',
        '--feature-fusion', 'hierarchical_simple',
        '--selected-features', '1', '3', '6', '8', '9', '10', '11',
        '--img_size', '20', '96', '96',
        '--crop_size', '10', '80', '80',
        '--data_dir', os.path.join(LIFT_DIR, 'data/images'),
        '--val_anno_file', val_anno,
        '--test_anno_file', test_anno,
        '--checkpoint', ckpts[0],
        '--results-dir', output_dir,
        '--score-dir', output_dir,
        '-j', '8', '-b', '4',
    ]
    print(f"  Fold {fold}: running predict.py...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR fold {fold}:")
        print(r.stderr[-800:])
        return None, None

    v_exists = os.path.exists(val_score)
    t_exists = os.path.exists(test_score)
    print(f"  Fold {fold}: val={v_exists}, test={t_exists}")
    return (val_score if v_exists else None, test_score if t_exists else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[1,2,3,4,5])
    parser.add_argument('--skip-predict', action='store_true')
    parser.add_argument('--output-base', default=None)
    args = parser.parse_args()

    if args.output_base is None:
        args.output_base = os.path.join(LIFT_DIR, 'pred_results/optimized')

    os.makedirs(args.output_base, exist_ok=True)
    folds = args.folds

    # =========================================================
    # STEP 1: 运行预测
    # =========================================================
    if not args.skip_predict:
        print(f"{'='*60}\nSTEP 1: Running predictions for {len(folds)} folds\n{'='*60}")
        for fold in folds:
            if fold == 1:
                ckpt_dir = os.path.join(LIFT_DIR, 'ckpts/Model1_LR5_8_optimized')
            else:
                ckpt_dir = os.path.join(LIFT_DIR, f'ckpts/Model1_LR5_8_optimized_{fold}')
            pred_dir = os.path.join(args.output_base, f'fold{fold}')
            os.makedirs(pred_dir, exist_ok=True)
            run_prediction(fold, ckpt_dir, pred_dir)
    else:
        print("Skipping prediction step (--skip-predict)")

    # =========================================================
    # STEP 2: OOF 集成（拼接）
    # =========================================================
    print(f"\n{'='*60}\nSTEP 2: OOF Ensemble\n{'='*60}")
    oof_dir = os.path.join(args.output_base, 'oof_ensemble')
    os.makedirs(oof_dir, exist_ok=True)

    oof_all = {}  # image_id -> (score, pred, gt)
    oof_feature_acc_paths = []

    for fold in folds:
        pred_dir = os.path.join(args.output_base, f'fold{fold}')
        val_score_path = os.path.join(pred_dir, 'val', 'score.json')
        if not os.path.exists(val_score_path):
            print(f"  WARNING: Fold {fold} val score.json not found, skipping")
            continue
        # feature accuracy
        fa_path = os.path.join(pred_dir, 'val', 'feature_accuracy.csv')
        if os.path.exists(fa_path):
            oof_feature_acc_paths.append(fa_path)
        # load predictions
        ids, scores, preds = load_score_json(val_score_path)
        val_anno = os.path.join(LIFT_DIR, f'data/labels/val_fold{fold}.txt')
        gt_map = load_gt(val_anno)
        for i, cid in enumerate(ids):
            if cid in gt_map:
                oof_all[cid] = (scores[i], preds[i], gt_map[cid])

    common_ids = sorted(oof_all.keys())
    oof_scores = np.array([oof_all[cid][0] for cid in common_ids])
    oof_preds = np.array([oof_all[cid][1] for cid in common_ids])
    oof_gts = np.array([oof_all[cid][2] for cid in common_ids])
    print(f"  OOF samples: {len(common_ids)}")

    oof_metrics = compute_metrics(oof_preds, oof_gts, oof_scores)
    print(f"  OOF Accuracy: {oof_metrics['accuracy']:.4f}")
    print(f"  OOF Macro F1: {oof_metrics['f1_macro']:.4f}")
    print(f"  OOF Kappa:    {oof_metrics['kappa']:.4f}")
    if oof_metrics['auc']:
        print(f"  OOF AUC:      {oof_metrics['auc']:.4f}")
    for cn in CLASS_NAMES:
        c = oof_metrics['per_class'][cn]
        print(f"    {cn:8s}: F1={c['F1']:.4f}  R={c['Recall']:.4f}  P={c['Precision']:.4f}")

    # 保存OOF结果
    with open(os.path.join(oof_dir, 'evaluation_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(oof_metrics, f, indent=4, ensure_ascii=False)
    save_metrics_csv(oof_metrics, oof_dir)
    plot_confusion_matrix(oof_metrics['confusion_matrix'], oof_dir, '(OOF)')
    plot_roc(oof_scores, oof_gts, oof_dir, '(OOF)')
    save_feature_accuracy(oof_feature_acc_paths, oof_dir, '(OOF)')

    # OOF score.json
    oof_score_list = [{'image_id': cid, 'prediction': int(oof_preds[i]),
                       'score': oof_scores[i].tolist(), 'ground_truth': int(oof_gts[i])}
                      for i, cid in enumerate(common_ids)]
    with open(os.path.join(oof_dir, 'score.json'), 'w') as f:
        json.dump(oof_score_list, f, indent=4)

    # =========================================================
    # STEP 3: 测试集集成（平均softmax）
    # =========================================================
    print(f"\n{'='*60}\nSTEP 3: Test Set Ensemble\n{'='*60}")
    test_dir = os.path.join(args.output_base, 'test_ensemble')
    os.makedirs(test_dir, exist_ok=True)

    test_anno = os.path.join(LIFT_DIR, 'data/labels/test.txt')
    test_gt = load_gt(test_anno)

    # 收集每个fold的test预测
    test_fold_scores = {}  # fold -> {image_id: score}
    test_feature_acc_paths = []

    for fold in folds:
        pred_dir = os.path.join(args.output_base, f'fold{fold}')
        test_score_path = os.path.join(pred_dir, 'test', 'score.json')
        if not os.path.exists(test_score_path):
            print(f"  WARNING: Fold {fold} test score.json not found, skipping")
            continue
        fa_path = os.path.join(pred_dir, 'test', 'feature_accuracy.csv')
        if os.path.exists(fa_path):
            test_feature_acc_paths.append(fa_path)
        ids, scores, _ = load_score_json(test_score_path)
        test_fold_scores[fold] = {cid: sc for cid, sc in zip(ids, scores)}

    # 找到所有fold共有的case
    all_test_ids = set.intersection(*[set(v.keys()) for v in test_fold_scores.values()])
    all_test_ids = sorted(all_test_ids & set(test_gt.keys()))
    n_test = len(all_test_ids)
    print(f"  Test samples: {n_test}, folds used: {list(test_fold_scores.keys())}")

    # 平均softmax
    avg_scores = np.zeros((n_test, NUM_CLASSES))
    for fold, score_map in test_fold_scores.items():
        for i, cid in enumerate(all_test_ids):
            avg_scores[i] += score_map[cid]
    avg_scores /= len(test_fold_scores)

    test_preds = np.argmax(avg_scores, axis=1)
    test_gts = np.array([test_gt[cid] for cid in all_test_ids])

    test_metrics = compute_metrics(test_preds, test_gts, avg_scores)
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro F1: {test_metrics['f1_macro']:.4f}")
    print(f"  Test Kappa:    {test_metrics['kappa']:.4f}")
    if test_metrics['auc']:
        print(f"  Test AUC:      {test_metrics['auc']:.4f}")
    for cn in CLASS_NAMES:
        c = test_metrics['per_class'][cn]
        print(f"    {cn:8s}: F1={c['F1']:.4f}  R={c['Recall']:.4f}  P={c['Precision']:.4f}")

    with open(os.path.join(test_dir, 'evaluation_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)
    save_metrics_csv(test_metrics, test_dir)
    plot_confusion_matrix(test_metrics['confusion_matrix'], test_dir, '(Test)')
    plot_roc(avg_scores, test_gts, test_dir, '(Test)')
    save_feature_accuracy(test_feature_acc_paths, test_dir, '(Test)')

    test_score_list = [{'image_id': cid, 'prediction': int(test_preds[i]),
                        'score': avg_scores[i].tolist(), 'ground_truth': int(test_gts[i])}
                       for i, cid in enumerate(all_test_ids)]
    with open(os.path.join(test_dir, 'score.json'), 'w') as f:
        json.dump(test_score_list, f, indent=4)

    # =========================================================
    # STEP 4: 阈值优化
    # =========================================================
    print(f"\n{'='*60}\nSTEP 4: Threshold Optimization\n{'='*60}")

    for label, score_path, anno_path, out_dir in [
        ('OOF', os.path.join(oof_dir, 'score.json'), test_anno, oof_dir),
        ('Test', os.path.join(test_dir, 'score.json'), test_anno, test_dir),
    ]:
        if not os.path.exists(score_path):
            continue
        print(f"\n--- {label} threshold optimization ---")
        cmd = [
            PYTHON, os.path.join(LIFT_DIR, 'main/optimize_thresholds.py'),
            '--score-file', score_path,
            '--anno-file', anno_path,
            '--output-dir', out_dir,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ERROR: {r.stderr[-500:]}")
        else:
            # 只打印关键行
            for line in r.stdout.split('\n'):
                if any(kw in line for kw in ['macro_F1', 'accuracy', 'kappa', 'Best method', 'Saved']):
                    print(f"  {line.strip()}")

    # =========================================================
    # STEP 5: 汇总对比
    # =========================================================
    print(f"\n{'='*60}\nSUMMARY: 5-fold Optimized Model Ensemble\n{'='*60}")
    print(f"{'Metric':<16} {'OOF':>10} {'Test':>10}")
    print("-" * 40)
    for key in ['accuracy', 'f1_macro', 'f1_weighted', 'kappa', 'auc']:
        o = oof_metrics.get(key, '-')
        t = test_metrics.get(key, '-')
        o_s = f"{o:.4f}" if isinstance(o, float) else str(o)
        t_s = f"{t:.4f}" if isinstance(t, float) else str(t)
        print(f"{key:<16} {o_s:>10} {t_s:>10}")
    print(f"\nPer-class F1:")
    for cn in CLASS_NAMES:
        o_f1 = oof_metrics['per_class'][cn]['F1']
        t_f1 = test_metrics['per_class'][cn]['F1']
        print(f"  {cn:8s}: OOF={o_f1:.4f}  Test={t_f1:.4f}")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  OOF:  {oof_dir}/")
    print(f"  Test: {test_dir}/")


if __name__ == '__main__':
    main()
