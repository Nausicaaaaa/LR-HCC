#!/usr/bin/env python3
"""
实时评估 nnUNet 预测的 Dice 指标，并与 Model2 分类结果交叉对比。

每次运行会扫描 LIFT/data/masks_predicted/val/ 中已完成的预测，
与 ground truth 对比计算 Dice，并报告每个类别中 Dice 最高的正确预测案例。

用法:
    python LIFT/evaluate_masks_realtime.py
    python LIFT/evaluate_masks_realtime.py --watch  # 持续监控模式
"""

import os
import sys
import re
import glob
import csv
import json
import numpy as np
import nibabel as nib
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LIFT_DIR = PROJECT_ROOT / 'LIFT'
PRED_DIR = LIFT_DIR / 'data' / 'masks_predicted'
GT_DIR = PROJECT_ROOT / 'data' / 'nnUNet_raw' / 'Dataset001_HCC' / 'labelsTr'
MODEL2_DIR = LIFT_DIR / 'ckpts' / 'Model2' / 'uniformer_small_IL_features' / 'pred_results' / 'tables'

# 分类名称
CLASS_NAMES_3 = ['Benign', 'Non-HCC Malignancies', 'HCC']
CLASS_NAME_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES_3)}


def parse_class(val):
    """Parse class name or index to int index"""
    if isinstance(val, int):
        return val
    val = str(val).strip()
    if val.isdigit():
        return int(val)
    return CLASS_NAME_TO_IDX.get(val, -1)


def compute_dice(pred_mask, gt_mask):
    """计算二值 Dice 系数"""
    pred_bin = (pred_mask > 0).astype(np.float32)
    gt_bin = (gt_mask > 0).astype(np.float32)

    intersection = np.sum(pred_bin * gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)

    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / total


def load_classification_results():
    """加载 Model2 分类预测结果，返回 case -> (pred_class, true_class) 映射"""
    results = {}

    for split in ['val', 'test']:
        pred_file = MODEL2_DIR / f'{split}_predictions.csv'
        gt_file = MODEL2_DIR / f'{split}_ground_truth.csv'
        if not pred_file.exists() or not gt_file.exists():
            continue

        with open(pred_file, 'r', encoding='utf-8-sig') as f:
            preds = list(csv.DictReader(f))
        with open(gt_file, 'r', encoding='utf-8-sig') as f:
            gts = list(csv.DictReader(f))

        for p, g in zip(preds, gts):
            case_id = p.get('Case ID', '')
            pred_class = parse_class(p.get('Predicted Class', -1))
            true_class = parse_class(g.get('True Class', -1))
            if case_id:
                results[case_id] = {
                    'pred_class': pred_class,
                    'true_class': true_class,
                    'split': split,
                    'correct': pred_class == true_class,
                }

    return results


def evaluate_current_predictions():
    """评估当前已完成的预测"""
    # 加载分类结果
    cls_results = load_classification_results()

    # 扫描已预测的 val mask
    pred_files = sorted(glob.glob(str(PRED_DIR / 'val' / 'case_*.nii.gz')))

    if not pred_files:
        print("暂无已完成的预测。")
        return

    # 计算每个 case 的 Dice
    dice_results = []
    for pred_path in pred_files:
        case_id = os.path.basename(pred_path).replace('.nii.gz', '')
        gt_path = GT_DIR / f'{case_id}.nii.gz'

        if not gt_path.exists():
            continue

        pred_img = nib.load(pred_path)
        gt_img = nib.load(str(gt_path))

        pred_data = pred_img.get_fdata()
        gt_data = gt_img.get_fdata()

        # 确保 shape 一致
        if pred_data.shape != gt_data.shape:
            continue

        dice = compute_dice(pred_data, gt_data)

        # 获取分类结果
        cls_info = cls_results.get(case_id, None)

        dice_results.append({
            'case_id': case_id,
            'dice': dice,
            'cls_info': cls_info,
        })

    if not dice_results:
        print("没有可评估的 case。")
        return

    # 排序：按 Dice 从高到低
    dice_results.sort(key=lambda x: x['dice'], reverse=True)

    # 统计
    dices = [r['dice'] for r in dice_results]
    print(f"\n{'='*75}")
    print(f"nnUNet 3-fold Ensemble 实时 Dice 评估")
    print(f"已评估: {len(dice_results)} / {len(pred_files)} 个 val cases")
    print(f"Dice 统计: mean={np.mean(dices):.4f}, median={np.median(dices):.4f}, "
          f"min={np.min(dices):.4f}, max={np.max(dices):.4f}")
    print(f"{'='*75}")

    # 分类交叉分析
    correct_cls = [r for r in dice_results if r['cls_info'] and r['cls_info']['correct']]
    incorrect_cls = [r for r in dice_results if r['cls_info'] and not r['cls_info']['correct']]
    no_cls = [r for r in dice_results if r['cls_info'] is None]

    print(f"\n分类正确且有 Dice: {len(correct_cls)} cases")
    print(f"分类错误且有 Dice: {len(incorrect_cls)} cases")
    print(f"无分类信息: {len(no_cls)} cases")

    # 每个类别中 Dice 最高的正确预测案例
    print(f"\n{'='*75}")
    print(f"各类别 Dice 最高的正确预测案例 (实时更新)")
    print(f"{'='*75}")

    for cls_idx, cls_name in enumerate(CLASS_NAMES_3):
        # 找分类正确且属于该类别的 case
        candidates = [
            r for r in dice_results
            if r['cls_info']
            and r['cls_info']['correct']
            and r['cls_info']['true_class'] == cls_idx
        ]

        if candidates:
            best = candidates[0]  # 已按 Dice 降序排列
            print(f"\n  [{cls_name}] (共 {len(candidates)} 个正确预测)")
            print(f"    Case: {best['case_id']}")
            print(f"    Dice: {best['dice']:.4f}")
            print(f"    分类: {CLASS_NAMES_3[best['cls_info']['pred_class']]} (正确)")
        else:
            print(f"\n  [{cls_name}] 暂无正确预测的 case")

    # Top 10 Dice 表格
    print(f"\n{'='*75}")
    print(f"Top 10 Dice Cases")
    print(f"{'Case':<15} {'Dice':>8} {'分类预测':<25} {'正确?':<6}")
    print(f"{'-'*55}")
    for r in dice_results[:10]:
        if r['cls_info']:
            pred_name = CLASS_NAMES_3[r['cls_info']['pred_class']]
            true_name = CLASS_NAMES_3[r['cls_info']['true_class']]
            cls_str = f"{pred_name} (GT:{true_name})"
            correct_str = "Yes" if r['cls_info']['correct'] else "No"
        else:
            cls_str = "N/A"
            correct_str = "N/A"
        print(f"{r['case_id']:<15} {r['dice']:>8.4f} {cls_str:<25} {correct_str:<6}")

    print()
    return dice_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true', help='持续监控模式')
    parser.add_argument('--interval', type=int, default=120, help='监控间隔秒数')
    args = parser.parse_args()

    if args.watch:
        print(f"持续监控模式，每 {args.interval} 秒刷新...")
        last_count = -1
        while True:
            pred_files = glob.glob(str(PRED_DIR / 'val' / 'case_*.nii.gz'))
            if len(pred_files) != last_count:
                os.system('clear')
                results = evaluate_current_predictions()
                last_count = len(pred_files)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 无新预测 (当前 {len(pred_files)} cases)")
            time.sleep(args.interval)
    else:
        evaluate_current_predictions()


if __name__ == '__main__':
    main()
