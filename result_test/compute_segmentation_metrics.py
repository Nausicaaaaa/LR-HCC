#!/usr/bin/env python
"""
验证集分割性能指标计算脚本
基于 validation/summary.json 中的结果计算以下指标：
- mIoU (mean Intersection over Union)
- Recall Rate (召回率)
- Accuracy (准确率)
- Kappa Coefficient (Cohen's Kappa)
- Dice Coefficient (Dice系数)

用法:
    python compute_segmentation_metrics.py --dataset_id 001 --fold 0
    
示例:
    python compute_segmentation_metrics.py
    python compute_segmentation_metrics.py --dataset_id 001 --fold 0
    python compute_segmentation_metrics.py --dataset_id 002 --fold 0 --configuration 2d
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_summary_json(summary_path: Path) -> Dict:
    """加载 summary.json 文件"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_metrics_per_case(metric_per_case: List[Dict], label_key: str = "1") -> pd.DataFrame:
    """
    为每个 case 计算分割指标
    
    参数:
        metric_per_case: summary.json 中的 metric_per_case 列表
        label_key: 标签键 (默认 "1")
        
    返回:
        包含每个 case 指标的 DataFrame
    """
    results = []
    
    for case_data in metric_per_case:
        metrics = case_data.get("metrics", {}).get(label_key, {})
        prediction_file = case_data.get("prediction_file", "")
        reference_file = case_data.get("reference_file", "")
        
        # 提取 case_id
        case_id = Path(prediction_file).stem.replace('.nii', '') if prediction_file else "unknown"
        
        # 获取混淆矩阵元素
        tp = metrics.get("TP", 0)
        fp = metrics.get("FP", 0)
        tn = metrics.get("TN", 0)
        fn = metrics.get("FN", 0)
        
        # 获取已有的 Dice 和 IoU
        dice = metrics.get("Dice", 0.0)
        iou = metrics.get("IoU", 0.0)
        
        # 计算 Recall (召回率) = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 计算 Precision (精确率) = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # 计算 Kappa Coefficient
        # Kappa = (p_o - p_e) / (1 - p_e)
        # p_o = observed agreement = (TP + TN) / total
        # p_e = expected agreement by chance
        # p_e = [(TP+FP)*(TP+FN) + (FN+TN)*(FP+TN)] / total^2
        if total > 0:
            p_o = (tp + tn) / total
            pos_pred = tp + fp
            pos_actual = tp + fn
            neg_pred = fn + tn
            neg_actual = fp + tn
            
            p_e = (pos_pred * pos_actual + neg_pred * neg_actual) / (total * total)
            
            if p_e < 1.0:
                kappa = (p_o - p_e) / (1 - p_e)
            else:
                kappa = 1.0 if p_o == 1.0 else 0.0
        else:
            kappa = 0.0
        
        results.append({
            'case_id': case_id,
            'dice': dice,
            'iou': iou,
            'recall': recall,
            'precision': precision,
            'kappa': kappa,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    return pd.DataFrame(results)


def compute_aggregate_metrics(df: pd.DataFrame) -> Dict:
    """
    计算聚合指标（平均值和标准差）
    
    参数:
        df: 包含每个 case 指标的 DataFrame
        
    返回:
        包含聚合指标的字典
    """
    aggregate = {
        'mIoU': {
            'mean': df['iou'].mean(),
            'std': df['iou'].std(),
            'median': df['iou'].median()
        },
        'recall_rate': {
            'mean': df['recall'].mean(),
            'std': df['recall'].std(),
            'median': df['recall'].median()
        },
        'precision': {
            'mean': df['precision'].mean(),
            'std': df['precision'].std(),
            'median': df['precision'].median()
        },
        'kappa_coefficient': {
            'mean': df['kappa'].mean(),
            'std': df['kappa'].std(),
            'median': df['kappa'].median()
        },
        'dice_coefficient': {
            'mean': df['dice'].mean(),
            'std': df['dice'].std(),
            'median': df['dice'].median()
        }
    }
    
    return aggregate


def analyze_segmentation_metrics(
    dataset_id: str = '001',
    fold: int = 0,
    configuration: str = '2d',
    trainer: str = 'nnUNetTrainer_smallROI',
    data_base: Path = Path('/mnt/data/KASR/Dengsiyi/LR-HCC/data'),
    output_dir: Path = Path('./segmentation_metrics_results'),
    label_key: str = "1"
) -> Dict:
    """
    分析验证集的分割性能指标
    
    参数:
        dataset_id: 数据集 ID
        fold: 交叉验证折数
        configuration: 网络配置 (2d, 3d_fullres, 3d_lowres)
        trainer: 训练器名称
        data_base: 数据根目录
        output_dir: 输出目录
        label_key: 标签键
        
    返回:
        包含分割指标的字典
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
    print()
    
    # 计算每个 case 的指标
    df_cases = compute_metrics_per_case(metric_per_case, label_key=label_key)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个 case 的详细结果到 CSV
    csv_path = output_dir / f'segmentation_metrics_details_fold{fold}.csv'
    df_cases.to_csv(csv_path, index=False)
    print(f'详细结果已保存到: {csv_path}')
    
    # 计算聚合指标
    aggregate_metrics = compute_aggregate_metrics(df_cases)
    
    # 准备输出结果
    results = {
        'dataset_id': dataset_id,
        'fold': fold,
        'configuration': configuration,
        'trainer': trainer,
        'num_cases': len(metric_per_case),
        'aggregate_metrics': aggregate_metrics,
        'case_details': df_cases.to_dict('records')
    }
    
    # 保存统计结果到 JSON
    json_path = output_dir / f'segmentation_metrics_summary_fold{fold}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'统计结果已保存到: {json_path}')
    
    # 打印汇总结果
    print('\n' + '='*70)
    print('验证集分割性能指标分析')
    print('='*70)
    print(f'数据集: Dataset{dataset_id}_HCC')
    print(f'Fold: {fold}')
    print(f'配置: {configuration}')
    print(f'Case 数量: {len(metric_per_case)}')
    print('-'*70)
    print('聚合指标 (Mean ± Std):')
    print(f'  mIoU (平均交并比):           {aggregate_metrics["mIoU"]["mean"]:>8.4f} ± {aggregate_metrics["mIoU"]["std"]:>6.4f}')
    print(f'  Recall Rate (召回率):        {aggregate_metrics["recall_rate"]["mean"]:>8.4f} ± {aggregate_metrics["recall_rate"]["std"]:>6.4f}')
    print(f'  Precision (精确率):          {aggregate_metrics["precision"]["mean"]:>8.4f} ± {aggregate_metrics["precision"]["std"]:>6.4f}')
    print(f'  Kappa Coefficient:           {aggregate_metrics["kappa_coefficient"]["mean"]:>8.4f} ± {aggregate_metrics["kappa_coefficient"]["std"]:>6.4f}')
    print(f'  Dice Coefficient:            {aggregate_metrics["dice_coefficient"]["mean"]:>8.4f} ± {aggregate_metrics["dice_coefficient"]["std"]:>6.4f}')
    print('-'*70)
    print('中位数指标:')
    print(f'  mIoU:                        {aggregate_metrics["mIoU"]["median"]:>8.4f}')
    print(f'  Recall Rate:                 {aggregate_metrics["recall_rate"]["median"]:>8.4f}')
    print(f'  Precision:                   {aggregate_metrics["precision"]["median"]:>8.4f}')
    print(f'  Kappa Coefficient:           {aggregate_metrics["kappa_coefficient"]["median"]:>8.4f}')
    print(f'  Dice Coefficient:            {aggregate_metrics["dice_coefficient"]["median"]:>8.4f}')
    print('='*70)
    
    # 打印指标分布统计
    print('\n指标分布统计:')
    print('-'*70)
    for metric_name in ['iou', 'recall', 'precision', 'kappa', 'dice']:
        values = df_cases[metric_name]
        print(f'{metric_name.upper():15s}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}, std={values.std():.4f}')
    print('='*70)
    
    # 打印表现最好和最差的 case
    print('\n表现最好的 5 个 Case (按 Dice 排序):')
    print('-'*70)
    top_5 = df_cases.nlargest(5, 'dice')
    for _, row in top_5.iterrows():
        print(f'  {row["case_id"]:15s} | Dice: {row["dice"]:.4f} | IoU: {row["iou"]:.4f} | Recall: {row["recall"]:.4f} | Precision: {row["precision"]:.4f} | Kappa: {row["kappa"]:.4f}')
    
    print('\n表现最差的 5 个 Case (按 Dice 排序):')
    print('-'*70)
    bottom_5 = df_cases.nsmallest(5, 'dice')
    for _, row in bottom_5.iterrows():
        print(f'  {row["case_id"]:15s} | Dice: {row["dice"]:.4f} | IoU: {row["iou"]:.4f} | Recall: {row["recall"]:.4f} | Precision: {row["precision"]:.4f} | Kappa: {row["kappa"]:.4f}')
    
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='验证集分割性能指标计算脚本 - 计算 mIoU、Recall、Accuracy、Kappa、Dice',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python compute_segmentation_metrics.py
  python compute_segmentation_metrics.py --dataset_id 001 --fold 0
  python compute_segmentation_metrics.py --dataset_id 002 --fold 0 --configuration 2d
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
        default=Path('./segmentation_metrics_results'),
        help='输出目录 (默认: ./segmentation_metrics_results)'
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
        result = analyze_segmentation_metrics(
            dataset_id=args.dataset_id,
            fold=args.fold,
            configuration=args.configuration,
            trainer=args.trainer,
            data_base=args.data_base,
            output_dir=args.output_dir,
            label_key=args.label_key
        )
        
        print(f'分析完成! 结果已保存到 {args.output_dir}')
        
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
