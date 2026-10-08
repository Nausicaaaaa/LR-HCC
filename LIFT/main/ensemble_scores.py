#!/usr/bin/env python3
"""多折集成预测：平均多个模型的 softmax 概率，重新计算评估指标。

用法:
  # 2折集成
  python ensemble_scores.py \
      --score-files fold1/score.json fold2/score.json \
      --anno-file ../data/labels/test.txt \
      --output-dir ./ensemble_2folds

  # 5折集成
  python ensemble_scores.py \
      --score-files fold1/score.json fold2/score.json fold3/score.json fold4/score.json fold5/score.json \
      --anno-file ../data/labels/test.txt \
      --output-dir ./ensemble_5folds
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc
from collections import OrderedDict

# LR 标签映射（与 mp_liver_dataset.py 一致）
LR_MAPPING = {'1/2': 0, '3': 1, '4': 2, '5': 3, 'M': 4}
CLASS_NAMES = ['LR-1/2', 'LR-3', 'LR-4', 'LR-5', 'LR-M']


def load_ground_truth(anno_file, label_mode='lr'):
    """从标注文件加载真实标签"""
    first_line = open(anno_file, 'r', encoding='utf-8').readline().strip()
    skip_header = 1 if first_line.lower().startswith('casename') else 0
    df = pd.read_csv(anno_file, sep='\t', header=None, skiprows=skip_header)

    case_ids = df.iloc[:, 0].astype(str).tolist()
    if label_mode == 'lr':
        raw_labels = df.iloc[:, 2].astype(str)
        labels = []
        for raw in raw_labels:
            lr_val = raw.strip().split()[-1]
            cls_id = LR_MAPPING.get(lr_val, -1)
            if cls_id == -1:
                raise ValueError(f"Unknown LR label: '{raw}'")
            labels.append(cls_id)
    else:
        labels = df.iloc[:, 1].astype(int).tolist()

    return case_ids, np.array(labels, dtype=np.int64)


def load_score_json(path):
    """加载 score.json -> dict[image_id -> score_list]"""
    with open(path, 'r') as f:
        data = json.load(f)
    return {item['image_id']: item['score'] for item in data}


def compute_metrics(pred_labels, true_labels, pred_scores, num_classes=5):
    """计算完整评估指标"""
    from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                                 precision_score, cohen_kappa_score,
                                 confusion_matrix, roc_auc_score)

    acc = accuracy_score(true_labels, pred_labels)
    f1_w = f1_score(true_labels, pred_labels, average='weighted')
    f1_m = f1_score(true_labels, pred_labels, average='macro')
    recall_w = recall_score(true_labels, pred_labels, average='weighted')
    precision_w = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))

    # Per-class metrics
    per_class = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec_i = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_i = 2 * prec_i * recall_i / (prec_i + recall_i) if (prec_i + recall_i) > 0 else 0
        acc_i = (tp + tn) / cm.sum()
        per_class[CLASS_NAMES[i]] = {
            'ACC': round(acc_i, 4),
            'F1': round(f1_i, 4),
            'Recall': round(recall_i, 4),
            'Precision': round(prec_i, 4),
        }

    # AUC (需要至少2个类有样本)
    try:
        auc = roc_auc_score(true_labels, pred_scores, multi_class='ovr', average='weighted')
    except Exception:
        auc = None

    return {
        'accuracy': round(acc, 4),
        'f1_weighted': round(f1_w, 4),
        'f1_macro': round(f1_m, 4),
        'recall_weighted': round(recall_w, 4),
        'precision_weighted': round(precision_w, 4),
        'kappa': round(kappa, 4),
        'auc': round(auc, 4) if auc else None,
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'num_samples': len(true_labels),
    }


def plot_confusion_matrix(cm, num_classes, results_dir):
    """绘制混淆矩阵热力图（百分比 + 原始计数）"""
    labels = CLASS_NAMES[:num_classes]

    # 百分比版
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_percent = np.where(cm_sum > 0, cm / cm_sum * 100, 0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=.5, linecolor='gray',
                cbar_kws={'label': 'Percentage (%)'}, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Row-normalized %)', fontsize=13)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=.5, linecolor='gray', ax=axes[1])
    axes[1].set_title('Confusion Matrix (Raw Counts)', fontsize=13)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 单独保存百分比版和原始版（与 predict.py 保持一致）
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=.5, linecolor='gray',
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Confusion Matrix (Row-normalized %)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_percent.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=.5, linecolor='gray')
    plt.title('Confusion Matrix (Raw Counts)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_raw.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiclass_roc(pred_scores, true_labels, num_classes, results_dir):
    """绘制多分类ROC曲线（One-vs-Rest）"""
    plt.figure(figsize=(10, 8))
    labels = CLASS_NAMES[:num_classes]

    for i in range(num_classes):
        binary_labels = (true_labels == i).astype(int)
        scores = pred_scores[:, i]
        valid_mask = np.isfinite(scores)
        if not valid_mask.any():
            continue
        fpr, tpr, _ = roc_curve(binary_labels[valid_mask], scores[valid_mask])
        roc_auc_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{labels[i]} (AUC = {roc_auc_value:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curve (One-vs-Rest)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curve_multiclass.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Multi-fold ensemble: average softmax scores')
    parser.add_argument('--score-files', required=True, nargs='+', type=str,
                        help='Path(s) to score.json from each fold')
    parser.add_argument('--anno-file', required=True, type=str,
                        help='Ground truth annotation file (test.txt or val_foldX.txt)')
    parser.add_argument('--output-dir', default='./ensemble_results', type=str,
                        help='Output directory for ensemble results')
    parser.add_argument('--label-mode', default='lr', choices=['original', 'lr'])
    parser.add_argument('--threshold-optimize', action='store_true', default=False,
                        help='Also run threshold optimization on ensemble scores')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载 ground truth
    case_ids_gt, true_labels = load_ground_truth(args.anno_file, args.label_mode)
    num_classes = len(CLASS_NAMES)
    print(f"Ground truth: {len(case_ids_gt)} cases from {args.anno_file}")

    # 2. 加载所有 score.json
    all_scores = {}  # image_id -> list of score vectors
    for sf in args.score_files:
        scores = load_score_json(sf)
        print(f"  Loaded {len(scores)} predictions from {sf}")
        for cid, score in scores.items():
            if cid not in all_scores:
                all_scores[cid] = []
            all_scores[cid].append(score)

    # 3. 找到所有 score 文件和 GT 都有的 case
    common_ids = [cid for cid in case_ids_gt if cid in all_scores]
    if len(common_ids) < len(case_ids_gt):
        missing = set(case_ids_gt) - set(common_ids)
        print(f"WARNING: {len(missing)} cases in GT but not in scores: {list(missing)[:5]}...")

    print(f"\nEnsemble: {len(args.score_files)} models, {len(common_ids)} common cases")

    # 4. 平均 softmax 概率
    gt_dict = dict(zip(case_ids_gt, true_labels))
    avg_scores = []
    avg_labels = []
    gt_matched = []
    for cid in common_ids:
        scores_list = all_scores[cid]
        avg = np.mean(scores_list, axis=0)
        pred = int(np.argmax(avg))
        avg_scores.append(avg)
        avg_labels.append(pred)
        gt_matched.append(gt_dict[cid])

    avg_scores = np.array(avg_scores)
    avg_labels = np.array(avg_labels)
    gt_matched = np.array(gt_matched)

    # 5. 计算指标
    metrics = compute_metrics(avg_labels, gt_matched, avg_scores, num_classes)

    # 6. 打印结果
    print(f"\n{'='*60}")
    print(f"Ensemble Results ({len(args.score_files)}-fold)")
    print(f"{'='*60}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.4f}")
    print(f"F1 macro:    {metrics['f1_macro']:.4f}")
    print(f"Kappa:       {metrics['kappa']:.4f}")
    if metrics['auc']:
        print(f"AUC:         {metrics['auc']:.4f}")
    print(f"\nPer-class:")
    for cls_name, cls_metrics in metrics['per_class'].items():
        print(f"  {cls_name:8s}: F1={cls_metrics['F1']:.4f}  Recall={cls_metrics['Recall']:.4f}  "
              f"Precision={cls_metrics['Precision']:.4f}")
    print(f"\nConfusion Matrix:")
    header = "          " + "  ".join(f"{n:>6s}" for n in CLASS_NAMES)
    print(f"{'':>10s}{header}")
    for i, row in enumerate(metrics['confusion_matrix']):
        print(f"{CLASS_NAMES[i]:>10s}" + "  ".join(f"{v:6d}" for v in row))

    # 7. 保存结果
    # 保存 ensemble score.json
    ensemble_score = []
    for idx, cid in enumerate(common_ids):
        ensemble_score.append({
            'image_id': cid,
            'prediction': int(avg_labels[idx]),
            'score': avg_scores[idx].tolist(),
        })
    score_path = os.path.join(args.output_dir, 'score.json')
    with open(score_path, 'w') as f:
        json.dump(ensemble_score, f, indent=4)

    # 保存 metrics.json
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # 绘制混淆矩阵和ROC曲线
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), num_classes, args.output_dir)
    plot_multiclass_roc(avg_scores, gt_matched, num_classes, args.output_dir)

    print(f"\nResults saved to {args.output_dir}/")

    # 8. 阈值优化（可选）
    if args.threshold_optimize:
        run_threshold_optimization(avg_scores, gt_matched, num_classes, args.output_dir)


def run_threshold_optimization(pred_scores, true_labels, num_classes, output_dir):
    """简单的逐类阈值搜索"""
    from sklearn.metrics import f1_score

    best_thresholds = [0.5] * num_classes
    best_macro_f1 = 0

    # 网格搜索
    thresholds_range = np.arange(0.05, 0.96, 0.05)
    for _ in range(3):  # 多轮迭代优化
        for c in range(num_classes):
            best_t = best_thresholds[c]
            best_f1_c = 0
            for t in thresholds_range:
                test_thresholds = best_thresholds.copy()
                test_thresholds[c] = t
                # 应用阈值
                preds = np.zeros(len(pred_scores), dtype=np.int64)
                for i, scores in enumerate(pred_scores):
                    above = scores >= np.array(test_thresholds)
                    if above.any():
                        # 在超过阈值的类别中选概率最高的
                        masked = scores * above
                        preds[i] = np.argmax(masked)
                    else:
                        preds[i] = np.argmax(scores)
                macro_f1 = f1_score(true_labels, preds, average='macro')
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_t = t
            best_thresholds[c] = best_t

    # 用最优阈值预测
    final_preds = np.zeros(len(pred_scores), dtype=np.int64)
    for i, scores in enumerate(pred_scores):
        above = scores >= np.array(best_thresholds)
        if above.any():
            masked = scores * above
            final_preds[i] = np.argmax(masked)
        else:
            final_preds[i] = np.argmax(scores)

    opt_metrics = compute_metrics(final_preds, true_labels, pred_scores, num_classes)
    print(f"\n{'='*60}")
    print(f"Threshold-Optimized Results")
    print(f"{'='*60}")
    print(f"Thresholds:  {dict(zip(CLASS_NAMES, [f'{t:.2f}' for t in best_thresholds]))}")
    print(f"F1 macro:    {opt_metrics['f1_macro']:.4f}")
    for cls_name, cls_m in opt_metrics['per_class'].items():
        print(f"  {cls_name:8s}: F1={cls_m['F1']:.4f}  Recall={cls_m['Recall']:.4f}")

    opt_path = os.path.join(output_dir, 'evaluation_metrics_optimized.json')
    opt_metrics['thresholds'] = dict(zip(CLASS_NAMES, [round(t, 2) for t in best_thresholds]))
    with open(opt_path, 'w', encoding='utf-8') as f:
        json.dump(opt_metrics, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
