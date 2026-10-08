#!/usr/bin/env python3
"""
5折交叉验证 OOF (Out-of-Fold) 指标聚合脚本

流程:
  1. 训练5折模型:  train.py --fold 1~5
  2. 每折推理对应 val set:  predict.py --val_anno_file val_fold{i}.txt --checkpoint fold{i}的ckpt
  3. 本脚本聚合5折预测，计算整体 OOF 指标

用法示例:
  python main/compute_oof.py \
      --pred-dirs ckpts/fold1/pred_results ckpts/fold2/pred_results \
                  ckpts/fold3/pred_results ckpts/fold4/pred_results \
                  ckpts/fold5/pred_results
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, cohen_kappa_score,
                             roc_auc_score, confusion_matrix, classification_report)


def load_score_json(score_path):
    """加载 predict.py 输出的 score.json，返回 {case_id: score_array}"""
    with open(score_path, 'r') as f:
        data = json.load(f)
    results = {}
    for item in data:
        case_id = item['image_id']
        scores = np.array(item['score'], dtype=np.float64)
        results[case_id] = scores
    return results


def load_ground_truth(anno_files, label_mode='lr'):
    """从标注文件加载 ground truth 标签"""
    lr_mapping = {'1/2': 0, '3': 1, '4': 2, '5': 3, 'M': 4}
    pathology_mapping = {'liangxing': 0, 'noHCC': 1, 'HCC': 2}

    case_to_label = {}
    for anno_file in anno_files:
        if not os.path.exists(anno_file):
            print(f"  WARNING: annotation file not found: {anno_file}")
            continue
        with open(anno_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        skip_header = 1 if first_line.startswith('casename') else 0
        df = pd.read_csv(anno_file, sep='\t', header=None, skiprows=skip_header)

        for _, row in df.iterrows():
            case_name = str(row.iloc[0]).strip()
            if label_mode == 'lr':
                lr_val = str(row.iloc[2]).strip().split()[-1]
                label = lr_mapping.get(lr_val, -1)
            else:
                label_str = str(row.iloc[1]).strip()
                label = pathology_mapping.get(label_str, -1)
            if label >= 0:
                case_to_label[case_name] = label
    return case_to_label


def find_score_json(pred_dir):
    """在预测目录中查找 score.json（兼容 val/ 子目录和根目录）"""
    candidates = [
        os.path.join(pred_dir, 'val', 'score.json'),
        os.path.join(pred_dir, 'score.json'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description='Compute OOF metrics from 5-fold CV predictions')
    parser.add_argument('--pred-dirs', nargs='+', required=True,
                        help='Prediction result directories (one per fold), '
                             'each containing val/score.json or score.json')
    parser.add_argument('--val-anno-files', nargs='+', default=None,
                        help='Val annotation files for each fold '
                             '(default: auto-inferred as data/labels/val_fold{1-5}.txt)')
    parser.add_argument('--label-mode', default='lr', choices=['original', 'lr'],
                        help='Label mode: "lr" (5-class) or "original" (3-class)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (default: auto from label-mode)')
    parser.add_argument('--output-dir', default='pred_results/oof', type=str,
                        help='Output directory for OOF metrics')
    args = parser.parse_args()

    if args.num_classes is None:
        args.num_classes = 5 if args.label_mode == 'lr' else 3

    # ── 1. 收集所有折的预测 ──
    all_predictions = {}
    for i, pred_dir in enumerate(args.pred_dirs):
        score_path = find_score_json(pred_dir)
        if score_path is None:
            print(f"WARNING: score.json not found in {pred_dir}, skipping fold {i+1}")
            continue
        fold_preds = load_score_json(score_path)
        all_predictions.update(fold_preds)
        print(f"Fold {i+1}: loaded {len(fold_preds)} predictions from {score_path}")

    print(f"\nTotal OOF predictions: {len(all_predictions)}")

    # ── 2. 加载 ground truth ──
    if args.val_anno_files:
        anno_files = args.val_anno_files
    else:
        anno_files = [f'data/labels/val_fold{i}.txt' for i in range(1, 6)]

    gt = load_ground_truth(anno_files, args.label_mode)

    # ── 3. 对齐预测和标签 ──
    common_cases = sorted(set(all_predictions.keys()) & set(gt.keys()))
    missing_pred = set(gt.keys()) - set(all_predictions.keys())
    missing_gt = set(all_predictions.keys()) - set(gt.keys())

    if missing_pred:
        print(f"WARNING: {len(missing_pred)} cases in GT but missing predictions")
    if missing_gt:
        print(f"WARNING: {len(missing_gt)} cases in predictions but missing GT")

    if not common_cases:
        print("ERROR: no aligned samples found, check case_id format")
        return

    pred_scores = np.array([all_predictions[c] for c in common_cases])
    true_labels = np.array([gt[c] for c in common_cases])
    pred_labels = pred_scores.argmax(axis=1)

    print(f"Aligned samples: {len(common_cases)}")

    # ── 4. 计算指标 ──
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(true_labels, pred_labels)

    try:
        auc = roc_auc_score(true_labels, pred_scores, multi_class='ovr', average='weighted')
    except Exception:
        auc = None

    cm = confusion_matrix(true_labels, pred_labels)

    # ── 5. 输出 ──
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("OOF (Out-of-Fold) CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {len(common_cases)}")
    print(f"Accuracy:  {acc * 100:.2f}%")
    if auc:
        print(f"AUC:       {auc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Kappa:     {kappa:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(true_labels, pred_labels, zero_division=0)}")
    print("=" * 60)

    # 保存 JSON
    result = {
        'num_samples': len(common_cases),
        'accuracy': float(acc),
        'f1_score': float(f1),
        'recall': float(recall),
        'precision': float(precision),
        'kappa': float(kappa),
        'confusion_matrix': cm.tolist(),
    }
    if auc:
        result['auc'] = float(auc)

    json_path = os.path.join(args.output_dir, 'oof_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {json_path}")

    # 保存每个 case 的详细预测
    detail_rows = []
    for case_id, scores, true_lbl, pred_lbl in zip(common_cases, pred_scores, true_labels, pred_labels):
        row = {'case_id': case_id, 'true_label': int(true_lbl), 'pred_label': int(pred_lbl),
               'correct': int(true_lbl == pred_lbl)}
        for ci in range(pred_scores.shape[1]):
            row[f'score_{ci}'] = float(scores[ci])
        detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    csv_path = os.path.join(args.output_dir, 'oof_detail_predictions.csv')
    detail_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Detail predictions saved to {csv_path}")


if __name__ == '__main__':
    main()
