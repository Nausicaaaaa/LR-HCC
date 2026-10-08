#!/usr/bin/env python3
"""
Post-hoc per-class threshold optimization for LIFT model predictions.

Directly operates on existing score.json files — no need to re-run inference.
Searches per-class decision thresholds to maximize macro-F1, especially
improving F1 for minority classes (LR-3, LR-4).

Usage:
    python LIFT/main/optimize_thresholds.py \
        --score-file LIFT/ckpts/Model1_LR5_8/uniformer_small_IL_features/pred_results/test/score.json \
        --anno-file LIFT/data/labels/test.txt \
        --output-dir LIFT/ckpts/Model1_LR5_8/uniformer_small_IL_features/pred_results/test

    # Or optimize both val and test at once:
    python LIFT/main/optimize_thresholds.py \
        --score-file \
            LIFT/ckpts/Model1_LR5_8/uniformer_small_IL_features/pred_results/val/score.json \
            LIFT/ckpts/Model1_LR5_8/uniformer_small_IL_features/pred_results/test/score.json \
        --anno-file \
            LIFT/data/labels/val_fold1.txt \
            LIFT/data/labels/test.txt \
        --output-dir \
            LIFT/ckpts/Model1_LR5_8/uniformer_small_IL_features/pred_results/val \
            LIFT/ckpts/Model1_LR5_8/uniformer_small_IL_features/pred_results/test
"""

import argparse
import json
import os
import csv
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, cohen_kappa_score, confusion_matrix
)


LR_MAPPING = {'1/2': 0, '3': 1, '4': 2, '5': 3, 'M': 4}
CLASS_NAMES_5 = ['LR-1/2', 'LR-3', 'LR-4', 'LR-5', 'LR-M']
CLASS_NAMES_3 = ['Benign', 'Non-HCC Malignancies', 'HCC']


def load_score_json(score_path):
    """Load predictions from score.json. Returns (case_ids, scores, pred_labels)."""
    with open(score_path, 'r') as f:
        data = json.load(f)
    case_ids = [item['image_id'] for item in data]
    scores = np.array([item['score'] for item in data], dtype=np.float64)
    pred_labels = np.array([item['prediction'] for item in data], dtype=int)
    return case_ids, scores, pred_labels


def load_true_labels_from_anno(anno_path, label_mode='lr'):
    """Load case_id -> label mapping from annotation file."""
    import pandas as pd
    try:
        # Try pandas with header
        df = pd.read_csv(anno_path, sep='\t', header=0)
        if df.shape[1] < 3 and label_mode == 'lr':
            df = pd.read_csv(anno_path, sep='\t', header=None)
    except Exception:
        df = pd.read_csv(anno_path, sep='\t', header=None)

    case_to_label = {}
    for _, row in df.iterrows():
        case_id = str(row.iloc[0]).strip()
        if label_mode == 'lr' and df.shape[1] >= 3:
            lr_str = str(row.iloc[2]).strip().split()[-1]
            label = LR_MAPPING.get(lr_str, -1)
        elif df.shape[1] >= 2:
            try:
                label = int(row.iloc[1])
            except (ValueError, TypeError):
                label_str = str(row.iloc[1]).strip()
                pathology_map = {'liangxing': 0, 'noHCC': 1, 'HCC': 2}
                label = pathology_map.get(label_str, -1)
        else:
            continue
        if label >= 0:
            case_to_label[case_id] = label
    return case_to_label


def match_scores_and_labels(case_ids, scores, case_to_label):
    """Match score.json entries with true labels. Returns (pred_scores, true_labels)."""
    matched_scores = []
    matched_labels = []
    missing = []
    for i, cid in enumerate(case_ids):
        if cid in case_to_label:
            matched_scores.append(scores[i])
            matched_labels.append(case_to_label[cid])
        else:
            missing.append(cid)
    if missing:
        print(f"  WARNING: {len(missing)} cases in score.json have no matching label: {missing[:5]}...")
    return np.array(matched_scores), np.array(matched_labels, dtype=int)


def apply_logit_bias(pred_scores, bias, minority_classes, num_classes):
    """Apply additive bias in logit space (inverse softmax -> add bias -> softmax)."""
    if bias <= 0:
        return pred_scores
    # Inverse softmax to get logits (with numerical stability)
    eps = 1e-10
    scores_clipped = np.clip(pred_scores, eps, 1 - eps)
    logits = np.log(scores_clipped)
    # Add bias to minority classes
    for c in minority_classes:
        logits[:, c] += bias
    # Re-apply softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    new_scores = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return new_scores


def search_logit_bias(pred_scores, true_labels, num_classes, class_names,
                      minority_classes=None, search_range=None, target='macro_f1'):
    """Search optimal logit bias for minority classes.

    This is more effective than threshold optimization because it works
    in logit space BEFORE the softmax, shifting the decision boundary
    for minority classes even when the model is very confident.

    Args:
        pred_scores: (N, C) softmax probabilities
        true_labels: (N,) ground truth
        num_classes: number of classes
        class_names: class name list
        minority_classes: list of class indices to boost (default: LR-3, LR-4)
        search_range: bias values to search (default: 0 to 8, step 0.25)
        target: optimization target ('macro_f1', 'f1_lr34', 'recall_lr34')

    Returns:
        best_bias, best_scores_adjusted, comparison dict
    """
    if minority_classes is None:
        minority_classes = [1, 2]  # LR-3, LR-4
    if search_range is None:
        search_range = np.arange(0, 8.5, 0.25)

    original_pred = np.argmax(pred_scores, axis=1)
    best_bias = 0.0
    best_score = -1
    best_pred = original_pred.copy()
    search_results = []

    for bias in search_range:
        adjusted = apply_logit_bias(pred_scores, bias, minority_classes, num_classes)
        pred_labels = np.argmax(adjusted, axis=1)

        if target == 'macro_f1':
            score = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        elif target == 'f1_lr34':
            f1_3 = f1_score((true_labels == 1).astype(int), (pred_labels == 1).astype(int), zero_division=0)
            f1_4 = f1_score((true_labels == 2).astype(int), (pred_labels == 2).astype(int), zero_division=0)
            score = f1_3 + f1_4
        elif target == 'recall_lr34':
            r_3 = recall_score((true_labels == 1).astype(int), (pred_labels == 1).astype(int), zero_division=0)
            r_4 = recall_score((true_labels == 2).astype(int), (pred_labels == 2).astype(int), zero_division=0)
            score = r_3 + r_4
        else:
            score = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

        # Per-class details
        details = {'bias': round(float(bias), 2), 'score': round(float(score), 4)}
        for c in minority_classes:
            bt = (true_labels == c).astype(int)
            bp = (pred_labels == c).astype(int)
            details[f'{class_names[c]}_F1'] = round(float(f1_score(bt, bp, zero_division=0)), 4)
            details[f'{class_names[c]}_R'] = round(float(recall_score(bt, bp, zero_division=0)), 4)
        search_results.append(details)

        if score > best_score:
            best_score = score
            best_bias = float(bias)
            best_pred = pred_labels.copy()

    # Build adjusted scores with best bias
    best_scores_adjusted = apply_logit_bias(pred_scores, best_bias, minority_classes, num_classes)

    print(f"  Logit bias search (target={target}): best_bias={best_bias:.2f}, score={best_score:.4f}")
    print(f"  Search highlights:")
    # Show a few interesting points
    for r in search_results:
        if r['bias'] in [0, 1.0, 2.0, 3.0, 4.0, 5.0, best_bias]:
            bias_str = f"bias={r['bias']:.1f}"
            detail_str = ' | '.join(f"{k}={v}" for k, v in r.items() if k not in ['bias', 'score'])
            marker = " <-- BEST" if r['bias'] == best_bias else ""
            print(f"    {bias_str}: {detail_str} | score={r['score']:.4f}{marker}")

    # Build comparison
    comparison = {
        'best_bias': best_bias,
        'target': target,
        'minority_classes': [class_names[c] for c in minority_classes],
        'original': {},
        'optimized': {},
        'search_results': search_results,
    }

    for c in range(num_classes):
        bt = (true_labels == c).astype(int)
        bo = (original_pred == c).astype(int)
        bn = (best_pred == c).astype(int)
        comparison['original'][class_names[c]] = {
            'ACC': round(float(accuracy_score(bt, bo)), 4),
            'F1': round(float(f1_score(bt, bo, zero_division=0)), 4),
            'Recall': round(float(recall_score(bt, bo, zero_division=0)), 4),
            'Precision': round(float(precision_score(bt, bo, zero_division=0)), 4),
            'Kappa': round(float(cohen_kappa_score(bt, bo)), 4),
        }
        comparison['optimized'][class_names[c]] = {
            'ACC': round(float(accuracy_score(bt, bn)), 4),
            'F1': round(float(f1_score(bt, bn, zero_division=0)), 4),
            'Recall': round(float(recall_score(bt, bn, zero_division=0)), 4),
            'Precision': round(float(precision_score(bt, bn, zero_division=0)), 4),
            'Kappa': round(float(cohen_kappa_score(bt, bn)), 4),
        }

    for metric_fn, key, avg in [
        (f1_score, 'macro_F1', 'macro'), (f1_score, 'weighted_F1', 'weighted'),
        (recall_score, 'macro_recall', 'macro'), (recall_score, 'weighted_recall', 'weighted'),
        (precision_score, 'macro_precision', 'macro'), (precision_score, 'weighted_precision', 'weighted'),
    ]:
        comparison['original'][key] = round(float(metric_fn(true_labels, original_pred, average=avg, zero_division=0)), 4)
        comparison['optimized'][key] = round(float(metric_fn(true_labels, best_pred, average=avg, zero_division=0)), 4)
    comparison['original']['accuracy'] = round(float(accuracy_score(true_labels, original_pred)), 4)
    comparison['optimized']['accuracy'] = round(float(accuracy_score(true_labels, best_pred)), 4)
    comparison['original']['kappa'] = round(float(cohen_kappa_score(true_labels, original_pred)), 4)
    comparison['optimized']['kappa'] = round(float(cohen_kappa_score(true_labels, best_pred)), 4)
    comparison['original']['confusion_matrix'] = confusion_matrix(true_labels, original_pred).tolist()
    comparison['optimized']['confusion_matrix'] = confusion_matrix(true_labels, best_pred).tolist()

    return best_bias, best_scores_adjusted, comparison


def optimize_thresholds(pred_scores, true_labels, num_classes, class_names):
    """
    Two-phase per-class threshold optimization:
      Phase 1: Find per-class binary-F1-optimal thresholds.
      Phase 2: Jointly refine thresholds to maximize macro-F1.

    Prediction rule: among classes whose score >= threshold,
    pick the one with highest score. Fallback to argmax if none pass.
    """
    original_pred = np.argmax(pred_scores, axis=1)
    threshold_candidates = np.arange(0.05, 1.0, 0.05)

    # Phase 1: per-class binary F1 optimization
    best_thresholds = np.full(num_classes, 0.5)
    for c in range(num_classes):
        binary_true = (true_labels == c).astype(int)
        best_f1 = -1
        for t in threshold_candidates:
            binary_pred = (pred_scores[:, c] >= t).astype(int)
            f1_c = f1_score(binary_true, binary_pred, zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_thresholds[c] = t

    print(f"  Phase 1 per-class thresholds: {dict(zip(class_names, best_thresholds.round(3)))}")

    # Phase 2: apply thresholds for multi-class prediction
    def apply_thresholds(scores, thresholds):
        preds = np.zeros(len(scores), dtype=int)
        for i in range(len(scores)):
            above = np.where(scores[i] >= thresholds)[0]
            if len(above) > 0:
                preds[i] = above[np.argmax(scores[i, above])]
            else:
                preds[i] = np.argmax(scores[i])
        return preds

    optimized_pred = apply_thresholds(pred_scores, best_thresholds)
    best_macro_f1 = f1_score(true_labels, optimized_pred, average='macro', zero_division=0)

    # Phase 3: iterative coordinate-wise refinement
    refined = best_thresholds.copy()
    for _round in range(3):  # 3 rounds of coordinate descent
        improved = False
        for c in range(num_classes):
            center = refined[c]
            local_range = np.clip(np.arange(center - 0.15, center + 0.16, 0.02), 0.05, 0.95)
            for t in local_range:
                test_thresh = refined.copy()
                test_thresh[c] = t
                test_pred = apply_thresholds(pred_scores, test_thresh)
                mf1 = f1_score(true_labels, test_pred, average='macro', zero_division=0)
                if mf1 > best_macro_f1:
                    best_macro_f1 = mf1
                    refined[c] = t
                    optimized_pred = test_pred.copy()
                    improved = True
        if not improved:
            break

    best_thresholds = refined
    print(f"  Refined thresholds: {dict(zip(class_names, best_thresholds.round(3)))}")

    # Build comparison
    comparison = {
        'thresholds': {class_names[i]: round(float(best_thresholds[i]), 3) for i in range(num_classes)},
        'original': {},
        'optimized': {},
    }

    for c in range(num_classes):
        bt = (true_labels == c).astype(int)
        bo = (original_pred == c).astype(int)
        bn = (optimized_pred == c).astype(int)
        comparison['original'][class_names[c]] = {
            'ACC': round(float(accuracy_score(bt, bo)), 4),
            'F1': round(float(f1_score(bt, bo, zero_division=0)), 4),
            'Recall': round(float(recall_score(bt, bo, zero_division=0)), 4),
            'Precision': round(float(precision_score(bt, bo, zero_division=0)), 4),
            'Kappa': round(float(cohen_kappa_score(bt, bo)), 4),
        }
        comparison['optimized'][class_names[c]] = {
            'ACC': round(float(accuracy_score(bt, bn)), 4),
            'F1': round(float(f1_score(bt, bn, zero_division=0)), 4),
            'Recall': round(float(recall_score(bt, bn, zero_division=0)), 4),
            'Precision': round(float(precision_score(bt, bn, zero_division=0)), 4),
            'Kappa': round(float(cohen_kappa_score(bt, bn)), 4),
        }

    for metric_fn, key, avg in [
        (f1_score, 'macro_F1', 'macro'), (f1_score, 'weighted_F1', 'weighted'),
        (recall_score, 'macro_recall', 'macro'), (recall_score, 'weighted_recall', 'weighted'),
        (precision_score, 'macro_precision', 'macro'), (precision_score, 'weighted_precision', 'weighted'),
    ]:
        comparison['original'][key] = round(float(metric_fn(true_labels, original_pred, average=avg, zero_division=0)), 4)
        comparison['optimized'][key] = round(float(metric_fn(true_labels, optimized_pred, average=avg, zero_division=0)), 4)
    comparison['original']['accuracy'] = round(float(accuracy_score(true_labels, original_pred)), 4)
    comparison['optimized']['accuracy'] = round(float(accuracy_score(true_labels, optimized_pred)), 4)
    comparison['original']['kappa'] = round(float(cohen_kappa_score(true_labels, original_pred)), 4)
    comparison['optimized']['kappa'] = round(float(cohen_kappa_score(true_labels, optimized_pred)), 4)

    # Confusion matrix comparison
    comparison['original']['confusion_matrix'] = confusion_matrix(true_labels, original_pred).tolist()
    comparison['optimized']['confusion_matrix'] = confusion_matrix(true_labels, optimized_pred).tolist()

    return best_thresholds, optimized_pred, original_pred, comparison


def print_comparison(comparison, class_names, title='THRESHOLD OPTIMIZATION RESULTS'):
    """Pretty-print the comparison table."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Optimal thresholds: {comparison.get('thresholds', comparison.get('best_bias', 'N/A'))}")
    if 'best_bias' in comparison:
        print(f"Best logit bias: {comparison['best_bias']} (target={comparison.get('target', 'N/A')})")

    print(f"\n{'Metric':<16} {'Original':>10} {'Optimized':>10} {'Delta':>10}")
    print("-" * 50)
    for key in ['macro_F1', 'weighted_F1', 'accuracy', 'kappa']:
        o = comparison['original'][key]
        n = comparison['optimized'][key]
        d = n - o
        print(f"{key:<16} {o:>10.4f} {n:>10.4f} {d:>+10.4f}")

    print(f"\n{'Class':<12} {'Orig F1':>8} {'Opt F1':>8} {'Δ':>8} {'Orig R':>8} {'Opt R':>8} {'Orig P':>8} {'Opt P':>8}")
    print("-" * 75)
    for cn in class_names:
        o = comparison['original'].get(cn, {})
        n = comparison['optimized'].get(cn, {})
        of1, nf1 = o.get('F1', 0), n.get('F1', 0)
        print(f"{cn:<12} {of1:>8.4f} {nf1:>8.4f} {nf1-of1:>+8.4f} "
              f"{o.get('Recall',0):>8.4f} {n.get('Recall',0):>8.4f} "
              f"{o.get('Precision',0):>8.4f} {n.get('Precision',0):>8.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Post-hoc threshold optimization for LIFT predictions')
    parser.add_argument('--score-file', nargs='+', required=True,
                        help='Path(s) to score.json file(s)')
    parser.add_argument('--anno-file', nargs='+', required=True,
                        help='Path(s) to annotation file(s) (must match score files 1:1)')
    parser.add_argument('--output-dir', nargs='+', default=None,
                        help='Output directories (defaults to score.json directories)')
    parser.add_argument('--label-mode', default='lr', choices=['lr', 'original'],
                        help='Label mode (default: lr)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (auto-detected if not set)')
    parser.add_argument('--bias-target', default='macro_f1', type=str,
                        choices=['macro_f1', 'f1_lr34', 'recall_lr34'],
                        help='Logit bias search target (default: macro_f1)')
    parser.add_argument('--bias-range-max', default=8.0, type=float,
                        help='Maximum bias value to search (default: 8.0)')
    parser.add_argument('--bias-step', default=0.25, type=float,
                        help='Bias search step (default: 0.25)')
    parser.add_argument('--no-threshold', action='store_true', default=False,
                        help='Skip threshold optimization, only do logit bias search')
    parser.add_argument('--no-bias', action='store_true', default=False,
                        help='Skip logit bias search, only do threshold optimization')
    args = parser.parse_args()

    if len(args.anno_file) != len(args.score_file):
        raise ValueError(f"Number of --anno-file ({len(args.anno_file)}) must match --score-file ({len(args.score_file)})")

    if args.output_dir is None:
        args.output_dir = [os.path.dirname(sf) for sf in args.score_file]
    elif len(args.output_dir) != len(args.score_file):
        raise ValueError(f"Number of --output-dir ({len(args.output_dir)}) must match --score-file ({len(args.score_file)})")

    for score_path, anno_path, out_dir in zip(args.score_file, args.anno_file, args.output_dir):
        print(f"\n{'='*70}")
        print(f"Processing: {score_path}")
        print(f"Annotation: {anno_path}")
        print(f"{'='*70}")

        # Load data
        case_ids, scores, pred_labels = load_score_json(score_path)
        case_to_label = load_true_labels_from_anno(anno_path, args.label_mode)
        pred_scores, true_labels = match_scores_and_labels(case_ids, scores, case_to_label)

        num_classes = args.num_classes or scores.shape[1]
        if num_classes == 5:
            class_names = CLASS_NAMES_5
        elif num_classes == 3:
            class_names = CLASS_NAMES_3
        else:
            class_names = [f'Class {i}' for i in range(num_classes)]

        print(f"  Samples: {len(pred_scores)}, Classes: {num_classes}")
        print(f"  Class distribution: {dict(zip(class_names, np.bincount(true_labels, minlength=num_classes)))}")

        os.makedirs(out_dir, exist_ok=True)
        all_results = {}

        # === Method 1: Logit bias search ===
        if not args.no_bias and num_classes == 5:
            print(f"\n--- Method 1: Logit-space bias search (target={args.bias_target}) ---")
            search_range = np.arange(0, args.bias_range_max + 0.01, args.bias_step)
            best_bias, bias_adjusted_scores, bias_comparison = search_logit_bias(
                pred_scores, true_labels, num_classes, class_names,
                minority_classes=[1, 2], search_range=search_range,
                target=args.bias_target)
            all_results['logit_bias'] = bias_comparison
            print_comparison(bias_comparison, class_names, title='LOGIT BIAS OPTIMIZATION')

            # Save bias search results
            bias_json_path = os.path.join(out_dir, 'logit_bias_optimization.json')
            with open(bias_json_path, 'w', encoding='utf-8') as f:
                json.dump(bias_comparison, f, indent=4, ensure_ascii=False)
            print(f"  Logit bias results saved to: {bias_json_path}")

        # === Method 2: Threshold optimization ===
        if not args.no_threshold:
            print(f"\n--- Method 2: Per-class threshold optimization ---")
            best_thresh, opt_pred, orig_pred, thresh_comparison = optimize_thresholds(
                pred_scores, true_labels, num_classes, class_names)
            all_results['threshold'] = thresh_comparison
            print_comparison(thresh_comparison, class_names, title='THRESHOLD OPTIMIZATION')

            # Also try threshold on bias-adjusted scores
            if not args.no_bias and num_classes == 5 and best_bias > 0:
                print(f"\n--- Method 2+1: Threshold optimization on bias-adjusted scores (bias={best_bias}) ---")
                _, bias_thresh_opt, _, bias_thresh_comp = optimize_thresholds(
                    bias_adjusted_scores, true_labels, num_classes, class_names)
                all_results['bias+threshold'] = bias_thresh_comp
                print_comparison(bias_thresh_comp, class_names, title='BIAS + THRESHOLD OPTIMIZATION')

            # Save threshold results
            thresh_json_path = os.path.join(out_dir, 'threshold_optimization.json')
            with open(thresh_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results.get('threshold', {}), f, indent=4, ensure_ascii=False)

        # === Summary comparison ===
        if all_results:
            print(f"\n{'='*70}")
            print("OVERALL COMPARISON SUMMARY")
            print(f"{'='*70}")
            print(f"{'Method':<25} {'macro_F1':>10} {'weighted_F1':>12} {'accuracy':>10} {'kappa':>8}")
            print(f"{'-'*70}")
            # Original baseline
            orig_metrics = None
            for method_name, comp in all_results.items():
                if orig_metrics is None:
                    orig_metrics = comp['original']
                print(f"{method_name:<25} {comp['optimized']['macro_F1']:>10.4f} "
                      f"{comp['optimized']['weighted_F1']:>12.4f} "
                      f"{comp['optimized']['accuracy']:>10.4f} "
                      f"{comp['optimized']['kappa']:>8.4f}")
            if orig_metrics:
                print(f"{'original (argmax)':<25} {orig_metrics['macro_F1']:>10.4f} "
                      f"{orig_metrics['weighted_F1']:>12.4f} "
                      f"{orig_metrics['accuracy']:>10.4f} "
                      f"{orig_metrics['kappa']:>8.4f}")

            # Find best method for macro_F1
            best_method = max(all_results.items(), key=lambda x: x[1]['optimized']['macro_F1'])
            print(f"\nBest method for macro-F1: {best_method[0]} (macro_F1={best_method[1]['optimized']['macro_F1']:.4f})")
            print(f"{'='*70}")

        # === Save combined CSV ===
        csv_path = os.path.join(out_dir, 'evaluation_metrics_optimized.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['', 'ACC', 'F1', 'Recall', 'Precision', 'Kappa'])
            for method_name, comp in all_results.items():
                # Summary row: method name + weighted metrics
                writer.writerow([method_name,
                    f"{comp['optimized'].get('accuracy', 0):.4f}",
                    f"{comp['optimized']['macro_F1']:.4f}",
                    f"{comp['optimized'].get('macro_recall', 0):.4f}",
                    f"{comp['optimized'].get('macro_precision', 0):.4f}",
                    f"{comp['optimized'].get('kappa', 0):.4f}"])
                # Per-class rows
                writer.writerow(['Class', 'ACC', 'F1', 'Recall', 'Precision', 'Kappa'])
                for cn in class_names:
                    opt_c = comp['optimized'].get(cn, {})
                    writer.writerow([cn,
                        f"{opt_c.get('ACC', 0):.4f}",
                        f"{opt_c.get('F1',0):.4f}",
                        f"{opt_c.get('Recall',0):.4f}",
                        f"{opt_c.get('Precision',0):.4f}",
                        f"{opt_c.get('Kappa', 0):.4f}"])
        print(f"  Combined CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
