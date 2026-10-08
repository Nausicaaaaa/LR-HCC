#!/usr/bin/env python3
"""OOF (Out-of-Fold) 集成预测：每个fold模型预测自己的验证集，拼接成全量OOF评估。

原理：
  - fold1 模型 → 预测 val_fold1.txt
  - fold2 模型 → 预测 val_fold2.txt
  - ...
  - fold5 模型 → 预测 val_fold5.txt
  - 拼接所有预测 → 覆盖全量数据（~1897个样本），得到无偏OOF精度

用法:
  python oof_ensemble.py --folds 1 2 3 4 5
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import subprocess

# 复用 ensemble_scores.py 中的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_scores import (
    load_ground_truth, compute_metrics, CLASS_NAMES,
    plot_confusion_matrix, plot_multiclass_roc
)

PYTHON = '/mnt/data/KASR/.conda/envs/nnunetv2/bin/python'
LIFT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_prediction(fold, ckpt_dir, output_dir):
    """对单个fold运行预测"""
    val_anno = os.path.join(LIFT_DIR, f'data/labels/val_fold{fold}.txt')
    results_dir = os.path.join(output_dir, f'fold{fold}')
    os.makedirs(results_dir, exist_ok=True)

    # 如果已有score.json则跳过
    score_path = os.path.join(results_dir, 'val', 'score.json')
    if os.path.exists(score_path):
        print(f"  Fold {fold}: score.json already exists, skipping.")
        return score_path

    score_path_root = os.path.join(results_dir, 'score.json')
    if os.path.exists(score_path_root):
        return score_path_root

    # 查找 best_f1 checkpoint
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, 'uniformer_small_IL_features', 'best_f1_checkpoint-*.pth.tar'))
    if not ckpts:
        ckpts = glob.glob(os.path.join(ckpt_dir, 'uniformer_small_IL_features', 'model_best.pth.tar'))
    if not ckpts:
        print(f"  ERROR: No checkpoint found for fold {fold} in {ckpt_dir}")
        return None
    checkpoint = ckpts[0]
    print(f"  Checkpoint: {checkpoint}")

    cmd = [
        PYTHON, os.path.join(LIFT_DIR, 'main/predict.py'),
        '--model', 'uniformer_small_IL_features',
        '--num-classes', '5',
        '--label-mode', 'lr',
        '--feature-fusion', 'hierarchical_simple',
        '--selected-features', '1', '3', '6', '8', '9', '10', '11',
        '--img_size', '20', '96', '96',
        '--crop_size', '10', '80', '80',
        '--data_dir', os.path.join(LIFT_DIR, 'data/images'),
        '--val_anno_file', val_anno,
        '--test_anno_file', '',  # 不需要test集预测
        '--checkpoint', checkpoint,
        '--results-dir', results_dir,
        '--score-dir', results_dir,
        '-j', '8', '-b', '4',
    ]

    print(f"  Running prediction for fold {fold}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Prediction failed for fold {fold}")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return None

    # 返回实际score.json路径
    if os.path.exists(score_path):
        return score_path
    elif os.path.exists(score_path_root):
        return score_path_root
    else:
        print(f"  ERROR: No score.json found after prediction for fold {fold}")
        return None


def main():
    parser = argparse.ArgumentParser(description='OOF Ensemble: each fold predicts its own val set')
    parser.add_argument('--folds', default=[1, 2, 3, 4, 5], nargs='+', type=int,
                        help='Fold numbers to include (default: 1 2 3 4 5)')
    parser.add_argument('--ckpt-base', default=None, type=str,
                        help='Base directory for checkpoints (default: auto-detect)')
    parser.add_argument('--output-dir', default=None, type=str,
                        help='Output directory for OOF results')
    parser.add_argument('--skip-predict', action='store_true',
                        help='Skip prediction, only ensemble existing results')
    args = parser.parse_args()

    # 默认路径
    if args.ckpt_base is None:
        args.ckpt_base = os.path.join(LIFT_DIR, 'ckpts')
    if args.output_dir is None:
        args.output_dir = os.path.join(LIFT_DIR, 'pred_results/optimized/oof_ensemble')
    os.makedirs(args.output_dir, exist_ok=True)

    pred_dir = os.path.join(LIFT_DIR, 'pred_results/optimized')
    num_classes = len(CLASS_NAMES)

    print(f"{'='*60}")
    print(f"OOF Ensemble: folds {args.folds}")
    print(f"{'='*60}")

    # 1. 运行每个fold的预测（或加载已有结果）
    all_scores = {}  # image_id -> score vector
    all_preds = {}   # image_id -> predicted class
    all_gts = {}     # image_id -> ground truth
    fold_counts = {}

    for fold in args.folds:
        # 确定 checkpoint 目录
        if fold == 1:
            ckpt_dir = os.path.join(args.ckpt_base, 'Model1_LR5_8_optimized')
        else:
            ckpt_dir = os.path.join(args.ckpt_base, f'Model1_LR5_8_optimized_{fold}')

        fold_pred_dir = os.path.join(pred_dir, f'fold{fold}')

        if not args.skip_predict:
            score_path = run_prediction(fold, ckpt_dir, fold_pred_dir)
        else:
            # 查找已有结果
            score_path = os.path.join(fold_pred_dir, 'val', 'score.json')
            if not os.path.exists(score_path):
                score_path = os.path.join(fold_pred_dir, 'score.json')
            if not os.path.exists(score_path):
                print(f"  WARNING: No score.json for fold {fold}, skipping.")
                score_path = None

        if score_path is None:
            continue

        # 加载预测结果
        with open(score_path, 'r') as f:
            data = json.load(f)
        fold_counts[fold] = len(data)
        print(f"  Fold {fold}: {len(data)} predictions")

        for item in data:
            cid = item['image_id']
            all_scores[cid] = item['score']
            all_preds[cid] = item['prediction']

        # 加载 ground truth
        val_anno = os.path.join(LIFT_DIR, f'data/labels/val_fold{fold}.txt')
        case_ids, labels = load_ground_truth(val_anno, 'lr')
        for cid, lab in zip(case_ids, labels):
            all_gts[cid] = int(lab)

    # 2. 合并所有 OOF 预测
    common_ids = sorted(set(all_scores.keys()) & set(all_gts.keys()))
    print(f"\nTotal OOF samples: {len(common_ids)}")
    print(f"Per-fold counts: {fold_counts}")

    if len(common_ids) == 0:
        print("ERROR: No common samples found!")
        return

    # 3. 构建数组
    oof_scores = np.array([all_scores[cid] for cid in common_ids])
    oof_preds = np.array([all_preds[cid] for cid in common_ids])
    oof_gts = np.array([all_gts[cid] for cid in common_ids])

    # 4. 计算指标
    metrics = compute_metrics(oof_preds, oof_gts, oof_scores, num_classes)

    # 5. 打印结果
    print(f"\n{'='*60}")
    print(f"OOF Ensemble Results ({len(args.folds)} folds, {len(common_ids)} samples)")
    print(f"{'='*60}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.4f}")
    print(f"F1 macro:    {metrics['f1_macro']:.4f}")
    print(f"Kappa:       {metrics['kappa']:.4f}")
    if metrics['auc']:
        print(f"AUC:         {metrics['auc']:.4f}")
    print(f"\nPer-class:")
    for cls_name, cls_m in metrics['per_class'].items():
        print(f"  {cls_name:8s}: F1={cls_m['F1']:.4f}  Recall={cls_m['Recall']:.4f}  "
              f"Precision={cls_m['Precision']:.4f}  ACC={cls_m['ACC']:.4f}")
    print(f"\nConfusion Matrix:")
    header = "          " + "  ".join(f"{n:>6s}" for n in CLASS_NAMES)
    print(f"{'':>10s}{header}")
    for i, row in enumerate(metrics['confusion_matrix']):
        print(f"{CLASS_NAMES[i]:>10s}" + "  ".join(f"{v:6d}" for v in row))

    # 6. 保存结果
    # score.json
    oof_score_list = []
    for idx, cid in enumerate(common_ids):
        oof_score_list.append({
            'image_id': cid,
            'prediction': int(oof_preds[idx]),
            'score': oof_scores[idx].tolist(),
            'ground_truth': int(oof_gts[idx]),
        })
    with open(os.path.join(args.output_dir, 'score.json'), 'w') as f:
        json.dump(oof_score_list, f, indent=4)

    with open(os.path.join(args.output_dir, 'evaluation_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # 绘图
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), num_classes, args.output_dir)
    plot_multiclass_roc(oof_scores, oof_gts, num_classes, args.output_dir)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
