#!/usr/bin/env python
"""
验证集检测结果统计分析脚本
基于 IoU 阈值判断 case 是否被检测到，统计检测层面的敏感度、精确度、特异度

用法:
    python validate_detection_metrics.py --dataset_id 001 --fold 0 --iou_threshold 0.4
    
示例:
    python validate_detection_metrics.py
    python validate_detection_metrics.py --dataset_id 001 --fold 0 --iou_threshold 0.4
    python validate_detection_metrics.py --dataset_id 002 --fold 0 --configuration 2d --iou_threshold 0.5
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_summary_json(summary_path: Path) -> Dict:
    """加载 summary.json 文件"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_detection_metrics_from_iou(
    metric_per_case: List[Dict],
    iou_threshold: float = 0.4,
    label_key: str = "1"
) -> Dict:
    """
    基于 IoU 阈值计算检测层面的指标
    
    逻辑:
        - 如果 case 的 IoU > threshold: 表示该 case 被正确检测到 (True Positive)
        - 如果 case 的 IoU <= threshold 但 n_ref > 0 (有真实标签): 表示漏检 (False Negative)
        - 如果 case 的 IoU <= threshold 且 n_ref == 0 (无真实标签): 表示真阴性 (True Negative)
        - 如果 case 的 IoU > threshold 但 n_ref == 0 (无真实标签): 表示误检 (False Positive)
    
    参数:
        metric_per_case: summary.json 中的 metric_per_case 列表
        iou_threshold: IoU 阈值，超过此值认为 case 被检测到
        label_key: 标签键 (默认 "1")
        
    返回:
        包含检测指标的字典
    """
    tp = 0  # True Positive: IoU > threshold 且 n_ref > 0 (正确检测到病灶)
    fp = 0  # False Positive: IoU > threshold 但 n_ref == 0 (误检)
    tn = 0  # True Negative: IoU <= threshold 且 n_ref == 0 (正确判断无病灶)
    fn = 0  # False Negative: IoU <= threshold 但 n_ref > 0 (漏检)
    
    case_details = []
    
    for case_data in metric_per_case:
        metrics = case_data.get("metrics", {}).get(label_key, {})
        prediction_file = case_data.get("prediction_file", "")
        reference_file = case_data.get("reference_file", "")
        
        # 提取 case_id
        case_id = Path(prediction_file).stem.replace('.nii', '') if prediction_file else "unknown"
        
        # 获取 IoU 和 n_ref
        iou = metrics.get("IoU", 0.0)
        n_ref = metrics.get("n_ref", 0)
        n_pred = metrics.get("n_pred", 0)
        dice = metrics.get("Dice", 0.0)
        
        # 判断是否有真实病灶 (n_ref > 0)
        has_lesion = n_ref > 0
        
        # 判断是否被检测到 (IoU > threshold)
        is_detected = iou > iou_threshold
        
        # 分类
        if has_lesion and is_detected:
            category = "TP"
            tp += 1
        elif has_lesion and not is_detected:
            category = "FN"
            fn += 1
        elif not has_lesion and is_detected:
            category = "FP"
            fp += 1
        else:  # not has_lesion and not is_detected
            category = "TN"
            tn += 1
        
        case_details.append({
            'case_id': case_id,
            'iou': iou,
            'dice': dice,
            'n_ref': n_ref,
            'n_pred': n_pred,
            'has_lesion': has_lesion,
            'is_detected': is_detected,
            'category': category
        })
    
    # 计算指标
    total = tp + fp + tn + fn
    
    # 敏感度 (Sensitivity) = Recall = TP / (TP + FN) = 正确检测到的病灶数 / 所有真实病灶数
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 精确度 (Precision) = TP / (TP + FP) = 正确检测到的病灶数 / 所有被检测为病灶的数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # 特异度 (Specificity) = TN / (TN + FP) = 正确判断无病灶的数 / 所有真实无病灶的数
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 准确率 (Accuracy)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # F1 Score
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    # 阴性预测值 (NPV) = TN / (TN + FN)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # 计算检测到病灶的 case (TP) 的平均 Dice 和 IoU
    tp_cases = [c for c in case_details if c['category'] == 'TP']
    if tp_cases:
        avg_dice_tp = np.mean([c['dice'] for c in tp_cases])
        avg_iou_tp = np.mean([c['iou'] for c in tp_cases])
        std_dice_tp = np.std([c['dice'] for c in tp_cases])
        std_iou_tp = np.std([c['iou'] for c in tp_cases])
    else:
        avg_dice_tp = 0.0
        avg_iou_tp = 0.0
        std_dice_tp = 0.0
        std_iou_tp = 0.0
    
    # 计算所有有病灶的 case (TP + FN) 的平均 Dice 和 IoU
    lesion_cases = [c for c in case_details if c['has_lesion']]
    if lesion_cases:
        avg_dice_all_lesions = np.mean([c['dice'] for c in lesion_cases])
        avg_iou_all_lesions = np.mean([c['iou'] for c in lesion_cases])
        std_dice_all_lesions = np.std([c['dice'] for c in lesion_cases])
        std_iou_all_lesions = np.std([c['iou'] for c in lesion_cases])
    else:
        avg_dice_all_lesions = 0.0
        avg_iou_all_lesions = 0.0
        std_dice_all_lesions = 0.0
        std_iou_all_lesions = 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'total': total,
        'sensitivity': sensitivity,
        'recall': sensitivity,
        'precision': precision,
        'specificity': specificity,
        'accuracy': accuracy,
        'f1': f1,
        'npv': npv,
        'tp_metrics': {
            'avg_dice': avg_dice_tp,
            'avg_iou': avg_iou_tp,
            'std_dice': std_dice_tp,
            'std_iou': std_iou_tp,
            'count': len(tp_cases)
        },
        'all_lesions_metrics': {
            'avg_dice': avg_dice_all_lesions,
            'avg_iou': avg_iou_all_lesions,
            'std_dice': std_dice_all_lesions,
            'std_iou': std_iou_all_lesions,
            'count': len(lesion_cases)
        },
        'case_details': case_details
    }


def analyze_detection_metrics(
    dataset_id: str = '001',
    fold: int = 0,
    configuration: str = '2d',
    trainer: str = 'nnUNetTrainer_smallROI',
    data_base: Path = Path('/mnt/data/KASR/Dengsiyi/LR-HCC/data'),
    output_dir: Path = Path('./validation_detection_results'),
    iou_threshold: float = 0.4,
    label_key: str = "1"
) -> Dict:
    """
    分析验证集的检测结果指标
    
    参数:
        dataset_id: 数据集 ID
        fold: 交叉验证折数
        configuration: 网络配置 (2d, 3d_fullres, 3d_lowres)
        trainer: 训练器名称
        data_base: 数据根目录
        output_dir: 输出目录
        iou_threshold: IoU 阈值
        label_key: 标签键
        
    返回:
        包含检测结果指标的字典
    """
    # 构建 summary.json 路径
    summary_path = data_base / f'nnUNet_results/Dataset{dataset_id}_HCC/{trainer}__nnUNetPlans__{configuration}/fold_{fold}/validation/summary.json'
    
    if not summary_path.exists():
        raise FileNotFoundError(f'summary.json 不存在: {summary_path}')
    
    print(f'加载 summary.json: {summary_path}')
    
    # 加载 summary.json
    summary_data = load_summary_json(summary_path)
    metric_per_case = summary_data.get("metric_per_case", [])
    
    if not metric_per_case:
        raise ValueError('summary.json 中没有 metric_per_case 数据')
    
    print(f'找到 {len(metric_per_case)} 个 case')
    print(f'IoU 阈值: {iou_threshold}')
    print()
    
    # 计算检测指标
    detection_metrics = compute_detection_metrics_from_iou(
        metric_per_case,
        iou_threshold=iou_threshold,
        label_key=label_key
    )
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个 case 的详细结果到 CSV
    df_details = pd.DataFrame(detection_metrics['case_details'])
    csv_path = output_dir / f'detection_metrics_details_fold{fold}_iou{iou_threshold}.csv'
    df_details.to_csv(csv_path, index=False)
    print(f'详细结果已保存到: {csv_path}')
    
    # 准备输出结果
    results = {
        'dataset_id': dataset_id,
        'fold': fold,
        'configuration': configuration,
        'trainer': trainer,
        'iou_threshold': iou_threshold,
        'num_cases': len(metric_per_case),
        'confusion_matrix': {
            'TP': detection_metrics['tp'],
            'FP': detection_metrics['fp'],
            'TN': detection_metrics['tn'],
            'FN': detection_metrics['fn']
        },
        'metrics': {
            'sensitivity': detection_metrics['sensitivity'],
            'precision': detection_metrics['precision'],
            'specificity': detection_metrics['specificity'],
            'accuracy': detection_metrics['accuracy'],
            'f1': detection_metrics['f1'],
            'npv': detection_metrics['npv']
        },
        'tp_metrics': detection_metrics['tp_metrics'],
        'all_lesions_metrics': detection_metrics['all_lesions_metrics'],
        'case_details': detection_metrics['case_details']
    }
    
    # 保存统计结果到 JSON
    json_path = output_dir / f'detection_metrics_summary_fold{fold}_iou{iou_threshold}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'统计结果已保存到: {json_path}')
    
    # 打印汇总结果
    print('\n' + '='*70)
    print('验证集检测结果统计分析 (基于 IoU 阈值)')
    print('='*70)
    print(f'数据集: Dataset{dataset_id}_HCC')
    print(f'Fold: {fold}')
    print(f'配置: {configuration}')
    print(f'IoU 阈值: {iou_threshold}')
    print(f'Case 数量: {len(metric_per_case)}')
    print('-'*70)
    print('混淆矩阵:')
    print(f'  True Positive (TP):  {detection_metrics["tp"]:>4d}  (正确检测到病灶)')
    print(f'  False Positive (FP): {detection_metrics["fp"]:>4d}  (误检 - 无病灶但检测到)')
    print(f'  True Negative (TN):  {detection_metrics["tn"]:>4d}  (正确判断无病灶)')
    print(f'  False Negative (FN): {detection_metrics["fn"]:>4d}  (漏检 - 有病灶但未检测到)')
    print('-'*70)
    print('检测指标:')
    print(f'  敏感度 (Sensitivity/Recall): {detection_metrics["sensitivity"]:>8.4f}  ({detection_metrics["sensitivity"]*100:>6.2f}%)')
    print(f'  精确度 (Precision):          {detection_metrics["precision"]:>8.4f}  ({detection_metrics["precision"]*100:>6.2f}%)')
    print(f'  特异度 (Specificity):        {detection_metrics["specificity"]:>8.4f}  ({detection_metrics["specificity"]*100:>6.2f}%)')
    print(f'  准确率 (Accuracy):           {detection_metrics["accuracy"]:>8.4f}  ({detection_metrics["accuracy"]*100:>6.2f}%)')
    print(f'  F1 Score:                    {detection_metrics["f1"]:>8.4f}  ({detection_metrics["f1"]*100:>6.2f}%)')
    print(f'  阴性预测值 (NPV):            {detection_metrics["npv"]:>8.4f}  ({detection_metrics["npv"]*100:>6.2f}%)')
    print('-'*70)
    print('检测到病灶的 Case (TP) 分割指标:')
    print(f'  平均 Dice: {detection_metrics["tp_metrics"]["avg_dice"]:.4f} ± {detection_metrics["tp_metrics"]["std_dice"]:.4f}')
    print(f'  平均 IoU:  {detection_metrics["tp_metrics"]["avg_iou"]:.4f} ± {detection_metrics["tp_metrics"]["std_iou"]:.4f}')
    print(f'  Case 数量: {detection_metrics["tp_metrics"]["count"]}')
    print('-'*70)
    print('所有有病灶的 Case (TP+FN) 分割指标:')
    print(f'  平均 Dice: {detection_metrics["all_lesions_metrics"]["avg_dice"]:.4f} ± {detection_metrics["all_lesions_metrics"]["std_dice"]:.4f}')
    print(f'  平均 IoU:  {detection_metrics["all_lesions_metrics"]["avg_iou"]:.4f} ± {detection_metrics["all_lesions_metrics"]["std_iou"]:.4f}')
    print(f'  Case 数量: {detection_metrics["all_lesions_metrics"]["count"]}')
    print('='*70)
    
    # 打印各类别的 case 列表
    print('\n各类别 Case 详情:')
    print('-'*70)
    
    tp_cases = [c['case_id'] for c in detection_metrics['case_details'] if c['category'] == 'TP']
    fp_cases = [c['case_id'] for c in detection_metrics['case_details'] if c['category'] == 'FP']
    tn_cases = [c['case_id'] for c in detection_metrics['case_details'] if c['category'] == 'TN']
    fn_cases = [c['case_id'] for c in detection_metrics['case_details'] if c['category'] == 'FN']
    
    if tp_cases:
        print(f'TP Cases ({len(tp_cases)}): {", ".join(tp_cases[:10])}{"..." if len(tp_cases) > 10 else ""}')
    if fp_cases:
        print(f'FP Cases ({len(fp_cases)}): {", ".join(fp_cases[:10])}{"..." if len(fp_cases) > 10 else ""}')
    if tn_cases:
        print(f'TN Cases ({len(tn_cases)}): {", ".join(tn_cases[:10])}{"..." if len(tn_cases) > 10 else ""}')
    if fn_cases:
        print(f'FN Cases ({len(fn_cases)}): {", ".join(fn_cases[:10])}{"..." if len(fn_cases) > 10 else ""}')
    
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='验证集检测结果统计分析脚本 - 基于 IoU 阈值统计敏感度、精确度、特异度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python validate_detection_metrics.py
  python validate_detection_metrics.py --dataset_id 001 --fold 0 --iou_threshold 0.4
  python validate_detection_metrics.py --dataset_id 002 --fold 0 --configuration 2d --iou_threshold 0.5
        """
    )
    
    parser.add_argument(
        '--dataset_id',
        type=str,
        default='001',
        help='数据集ID (默认: 001)'
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help='交叉验证折数 (默认: 0)'
    )
    
    parser.add_argument(
        '--configuration',
        type=str,
        default='2d',
        choices=['2d', '3d_fullres', '3d_lowres'],
        help='网络配置 (默认: 2d)'
    )
    
    parser.add_argument(
        '--trainer',
        type=str,
        default='nnUNetTrainer_smallROI',
        help='训练器名称 (默认: nnUNetTrainer_smallROI)'
    )
    
    parser.add_argument(
        '--data_base',
        type=Path,
        default=Path('/mnt/data/KASR/Dengsiyi/LR-HCC/data'),
        help='数据根目录'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('./validation_detection_results'),
        help='输出目录 (默认: ./validation_detection_results)'
    )
    
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.4,
        help='IoU 阈值，超过此值认为 case 被检测到 (默认: 0.4)'
    )
    
    parser.add_argument(
        '--label_key',
        type=str,
        default='1',
        help='标签键 (默认: 1)'
    )
    
    args = parser.parse_args()
    
    # 运行分析
    try:
        result = analyze_detection_metrics(
            dataset_id=args.dataset_id,
            fold=args.fold,
            configuration=args.configuration,
            trainer=args.trainer,
            data_base=args.data_base,
            output_dir=args.output_dir,
            iou_threshold=args.iou_threshold,
            label_key=args.label_key
        )
        
        print(f'分析完成! 结果已保存到 {args.output_dir}')
        
    except Exception as e:
        print(f'错误: {e}')
        raise


if __name__ == '__main__':
    main()
