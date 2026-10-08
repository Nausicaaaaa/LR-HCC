#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新统计CSV格式的验证结果精度指标
支持排除dice=0、计算检测准确率、排除指定case等功能
输出结果也为CSV格式
# 排除Dice<0.3的case
python recompute_metrics.py validation_details_folds_0_1_2.csv --exclude-dice-lt 0.3

# 排除Dice=0的case（旧方式，兼容）
python recompute_metrics.py validation_details_folds_0_1_2.csv --exclude-dice-zero

# 排除Dice=0的case（新方式）
python recompute_metrics.py validation_details_folds_0_1_2.csv --exclude-dice-eq 0

# 组合使用
python recompute_metrics.py validation_details_folds_0_1_2.csv --exclude-dice-lt 0.3 --detection-acc

"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def load_csv(csv_path):
    """加载CSV文件"""
    if Path(csv_path).stat().st_size == 0:
        print(f"错误：CSV文件为空 {csv_path}")
        exit(1)
    df = pd.read_csv(csv_path)
    print(f"加载CSV文件: {csv_path}")
    print(f"总共加载 {len(df)} 个case\n")
    return df


def compute_metrics_summary(df, label=""):
    """计算平均指标并返回DataFrame"""
    metrics = ['dice', 'iou', 'recall', 'precision']
    if 'kappa' in df.columns:
        metrics.append('kappa')
    
    summary = {
        'metric': metrics,
        'mean': [df[m].mean() for m in metrics],
        'std': [df[m].std() for m in metrics],
        'median': [df[m].median() for m in metrics],
        'min': [df[m].min() for m in metrics],
        'max': [df[m].max() for m in metrics],
        'num_cases': [len(df)] * len(metrics)
    }
    return pd.DataFrame(summary)


def print_metrics(df, title=""):
    """打印指标到终端"""
    print("="*60)
    print(title)
    print("="*60)
    print(f"Case数量: {len(df)}")
    print(f"Dice:    {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
    print(f"IoU:     {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
    print(f"Recall:  {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
    print(f"Precision: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
    if 'kappa' in df.columns:
        print(f"Kappa:   {df['kappa'].mean():.4f} ± {df['kappa'].std():.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='重新统计CSV格式的验证结果精度指标')
    parser.add_argument('csv_path', help='输入的CSV文件路径')
    parser.add_argument('--exclude-dice-zero', action='store_true', 
                        help='排除dice=0的case并重新统计(兼容旧参数)')
    parser.add_argument('--exclude-dice-lt', type=float, default=None,
                        help='排除dice小于指定阈值的case(例如: --exclude-dice-lt 0.3 排除dice<0.3)')
    parser.add_argument('--exclude-dice-eq', type=float, default=None,
                        help='排除dice等于指定值的case(例如: --exclude-dice-eq 0 排除dice=0)')
    parser.add_argument('--detection-acc', action='store_true',
                        help='计算IoU大于阈值的case的检测准确率')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='检测准确率的IoU阈值(默认0.3)')
    parser.add_argument('--exclude-file', type=str, default=None,
                        help='排除的case列表文件(每行一个case名称)')
    parser.add_argument('--output', type=str, default='recomputed_metrics.csv',
                        help='输出结果的CSV文件路径(默认: recomputed_metrics.csv)')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_csv(args.csv_path)
    
    results_list = []
    
    # 原始统计
    print_metrics(df, "【原始统计 - 所有case】")
    summary_original = compute_metrics_summary(df, "original")
    summary_original['scenario'] = 'all_cases'
    results_list.append(df.copy())
    
    # 1. 排除dice满足条件的case
    if args.exclude_dice_zero or args.exclude_dice_lt is not None or args.exclude_dice_eq is not None:
        df_filtered = df.copy()
        scenario_name = "exclude_dice"
        
        if args.exclude_dice_zero or args.exclude_dice_eq is not None:
            threshold = 0 if args.exclude_dice_zero else args.exclude_dice_eq
            df_filtered = df_filtered[df_filtered['dice'] != threshold]
            scenario_name = f"exclude_dice_eq_{threshold}"
            print_metrics(df_filtered, f"【统计1 - 排除Dice={threshold}的case】")
        
        if args.exclude_dice_lt is not None:
            threshold = args.exclude_dice_lt
            df_filtered = df_filtered[df_filtered['dice'] >= threshold]
            scenario_name = f"exclude_dice_lt_{threshold}"
            print_metrics(df_filtered, f"【统计1 - 排除Dice<{threshold}的case】")
        
        results_list.append(df_filtered)
        print(f"排除了 {len(df) - len(df_filtered)} 个case\n")
    
    # 2. 检测准确率
    if args.detection_acc:
        print("="*60)
        print(f"【统计2 - 检测准确率(IoU>{args.iou_threshold})】")
        print("="*60)
        detected = df[df['iou'] > args.iou_threshold]
        det_acc = len(detected) / len(df) if len(df) > 0 else 0
        print(f"总case数: {len(df)}")
        print(f"检测到(IoU>{args.iou_threshold}): {len(detected)}")
        print(f"检测准确率: {det_acc:.4f} ({det_acc*100:.2f}%)\n")
        
        # 在结果CSV中添加标记列
        df_det = df.copy()
        df_det['is_detected'] = df_det['iou'] > args.iou_threshold
        results_list.append(df_det)
    
    # 3. 排除指定case
    if args.exclude_file:
        print("="*60)
        print(f"【统计3 - 排除指定case】")
        print("="*60)
        with open(args.exclude_file, 'r') as f:
            exclude_names = set([line.strip() for line in f if line.strip()])
        
        df_remaining = df[~df['case_name'].isin(exclude_names)].copy()
        
        print(f"排除前: {len(df)} cases")
        print(f"排除了: {len(exclude_names)} cases")
        print(f"剩余: {len(df_remaining)} cases\n")
        print_metrics(df_remaining, "")
        
        results_list.append(df_remaining)
    
    # 合并所有case详情并输出CSV
    print(f"\n结果已保存到: {args.output}")
    print("包含以下场景的case详情和汇总统计:")
    print("- original: 所有原始case")
    if args.exclude_dice_zero:
        print("- exclude_dice_zero: 排除Dice=0的case")
    if args.detection_acc:
        print(f"- detection_iou_{args.iou_threshold}: 包含检测标记的case")
    if args.exclude_file:
        print("- excluded: 排除指定case后的剩余case")
    
    # 将汇总统计写入单独的CSV
    summary_path = args.output.replace('.csv', '_summary.csv')
    summary_original.to_csv(summary_path, index=False)
    print(f"汇总统计已保存到: {summary_path}")


if __name__ == '__main__':
    main()
