#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新统计summary.json中的精度指标
支持两种格式:
1. internal validation格式: case_details数组
2. nnUNet预测格式: metric_per_case数组

功能:
1. 排除dice=0的case并重新统计
2. 统计iou>0.3的case的检测准确率(检测到数量/总数量)
3. 支持排除指定case后重新统计
"""

import json
import argparse
import numpy as np
from pathlib import Path


def load_summary(json_path):
    """加载summary.json并统一转换为标准格式"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cases = []
    
    # 格式1: internal validation格式
    if 'case_details' in data:
        for case in data['case_details']:
            cases.append({
                'case_name': case['case_name'],
                'dice': case['dice'],
                'iou': case['iou'],
                'recall': case['recall'],
                'precision': case['precision'],
                'kappa': case.get('kappa', 0),
                'tp': case['tp'],
                'fp': case['fp'],
                'tn': case['tn'],
                'fn': case['fn']
            })
    
    # 格式2: nnUNet预测格式
    elif 'metric_per_case' in data:
        for case in data['metric_per_case']:
            metrics = case['metrics']['1']
            # 从prediction_file提取case名称
            pred_file = case['prediction_file']
            case_name = Path(pred_file).stem
            
            tp = metrics['TP']
            fp = metrics['FP']
            tn = metrics['TN']
            fn = metrics['FN']
            
            cases.append({
                'case_name': case_name,
                'dice': metrics['Dice'],
                'iou': metrics['IoU'],
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
    
    else:
        raise ValueError("无法识别的summary.json格式")
    
    return cases


def compute_metrics(cases):
    """计算一组case的平均指标"""
    if len(cases) == 0:
        return None
    
    dice_values = [c['dice'] for c in cases]
    iou_values = [c['iou'] for c in cases]
    recall_values = [c['recall'] for c in cases]
    precision_values = [c['precision'] for c in cases]
    
    metrics = {
        'num_cases': len(cases),
        'dice': {
            'mean': float(np.mean(dice_values)),
            'std': float(np.std(dice_values)),
            'median': float(np.median(dice_values)),
            'min': float(np.min(dice_values)),
            'max': float(np.max(dice_values))
        },
        'iou': {
            'mean': float(np.mean(iou_values)),
            'std': float(np.std(iou_values)),
            'median': float(np.median(iou_values)),
            'min': float(np.min(iou_values)),
            'max': float(np.max(iou_values))
        },
        'recall': {
            'mean': float(np.mean(recall_values)),
            'std': float(np.std(recall_values)),
            'median': float(np.median(recall_values)),
        },
        'precision': {
            'mean': float(np.mean(precision_values)),
            'std': float(np.std(precision_values)),
            'median': float(np.median(precision_values)),
        }
    }
    
    return metrics


def exclude_dice_zero(cases):
    """排除dice=0的case"""
    return [c for c in cases if c['dice'] > 0]


def detection_accuracy(cases, iou_threshold=0.3):
    """计算IoU大于阈值的case的检测准确率"""
    detected = [c for c in cases if c['iou'] > iou_threshold]
    return {
        'total_cases': len(cases),
        'detected_cases': len(detected),
        'detection_accuracy': len(detected) / len(cases) if len(cases) > 0 else 0,
        'iou_threshold': iou_threshold
    }


def exclude_specific_cases(cases, exclude_file):
    """根据排除文件排除指定的case"""
    with open(exclude_file, 'r') as f:
        exclude_names = set([line.strip() for line in f if line.strip()])
    
    remaining = [c for c in cases if c['case_name'] not in exclude_names]
    return remaining, exclude_names


def main():
    parser = argparse.ArgumentParser(description='重新统计summary.json中的精度指标')
    parser.add_argument('summary_path', help='summary.json文件路径')
    parser.add_argument('--exclude-dice-zero', action='store_true', 
                        help='排除dice=0的case并重新统计')
    parser.add_argument('--detection-acc', action='store_true',
                        help='计算IoU大于阈值的case的检测准确率')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='检测准确率的IoU阈值(默认0.3)')
    parser.add_argument('--exclude-file', type=str, default=None,
                        help='排除的case列表文件(每行一个case名称)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果到文件')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载summary文件: {args.summary_path}")
    cases = load_summary(args.summary_path)
    print(f"总共加载 {len(cases)} 个case\n")
    
    results = {}
    
    # 原始统计
    print("="*60)
    print("【原始统计 - 所有case】")
    print("="*60)
    original_metrics = compute_metrics(cases)
    if original_metrics:
        print(f"Case数量: {original_metrics['num_cases']}")
        print(f"Dice: {original_metrics['dice']['mean']:.4f} ± {original_metrics['dice']['std']:.4f}")
        print(f"IoU: {original_metrics['iou']['mean']:.4f} ± {original_metrics['iou']['std']:.4f}")
        print(f"Recall: {original_metrics['recall']['mean']:.4f} ± {original_metrics['recall']['std']:.4f}")
        print(f"Precision: {original_metrics['precision']['mean']:.4f} ± {original_metrics['precision']['std']:.4f}")
    results['original'] = original_metrics
    
    # 1. 排除dice=0的case
    if args.exclude_dice_zero:
        print("\n" + "="*60)
        print("【统计1 - 排除Dice=0的case】")
        print("="*60)
        cases_no_zero = exclude_dice_zero(cases)
        print(f"排除前: {len(cases)} cases")
        print(f"排除后: {len(cases_no_zero)} cases (排除了 {len(cases) - len(cases_no_zero)} 个Dice=0的case)")
        
        metrics_no_zero = compute_metrics(cases_no_zero)
        if metrics_no_zero:
            print(f"Dice: {metrics_no_zero['dice']['mean']:.4f} ± {metrics_no_zero['dice']['std']:.4f}")
            print(f"IoU: {metrics_no_zero['iou']['mean']:.4f} ± {metrics_no_zero['iou']['std']:.4f}")
            print(f"Recall: {metrics_no_zero['recall']['mean']:.4f} ± {metrics_no_zero['recall']['std']:.4f}")
            print(f"Precision: {metrics_no_zero['precision']['mean']:.4f} ± {metrics_no_zero['precision']['std']:.4f}")
        
        results['exclude_dice_zero'] = {
            'metrics': metrics_no_zero,
            'excluded_count': len(cases) - len(cases_no_zero)
        }
    
    # 2. 检测准确率
    if args.detection_acc:
        print("\n" + "="*60)
        print(f"【统计2 - 检测准确率(IoU>{args.iou_threshold})】")
        print("="*60)
        det_acc = detection_accuracy(cases, args.iou_threshold)
        print(f"总case数: {det_acc['total_cases']}")
        print(f"检测到(IoU>{args.iou_threshold}): {det_acc['detected_cases']}")
        print(f"检测准确率: {det_acc['detection_accuracy']:.4f} ({det_acc['detection_accuracy']*100:.2f}%)")
        results['detection_accuracy'] = det_acc
    
    # 3. 排除指定case
    if args.exclude_file:
        print("\n" + "="*60)
        print(f"【统计3 - 排除指定case】")
        print("="*60)
        cases_remaining, excluded_names = exclude_specific_cases(cases, args.exclude_file)
        print(f"排除前: {len(cases)} cases")
        print(f"排除了: {len(excluded_names)} cases")
        print(f"剩余: {len(cases_remaining)} cases")
        
        metrics_remaining = compute_metrics(cases_remaining)
        if metrics_remaining:
            print(f"Dice: {metrics_remaining['dice']['mean']:.4f} ± {metrics_remaining['dice']['std']:.4f}")
            print(f"IoU: {metrics_remaining['iou']['mean']:.4f} ± {metrics_remaining['iou']['std']:.4f}")
            print(f"Recall: {metrics_remaining['recall']['mean']:.4f} ± {metrics_remaining['recall']['std']:.4f}")
            print(f"Precision: {metrics_remaining['precision']['mean']:.4f} ± {metrics_remaining['precision']['std']:.4f}")
        
        results['exclude_specific'] = {
            'metrics': metrics_remaining,
            'excluded_cases': list(excluded_names),
            'excluded_count': len(excluded_names)
        }
    
    # 输出到文件
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
