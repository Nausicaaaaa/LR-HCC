#!/usr/bin/env python
"""
内部验证集精度测试脚本
基于已训练好的 nnUNet 模型，对任意内部验证数据进行联合预测并计算分割精度指标。

用法示例:
    # 使用 fold 0 和 fold 1 的模型联合预测 nnUNetPlans_2d_val 中的数据
    python validate_internal_set.py \
        --dataset_id 001 \
        --input_folder /mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_preprocessed/Dataset001_HCC/nnUNetPlans_2d_val \
        --folds 0 1 \
        --configuration 2d \
        --trainer nnUNetTrainer_smallROI \
        --gpu_ids 4

    # 只使用 fold 0 的模型
    python validate_internal_set.py \
        --dataset_id 001 \
        --input_folder /mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_preprocessed/Dataset001_HCC/nnUNetPlans_2d_val \
        --folds 0 \
        --configuration 2d \
        --trainer nnUNetTrainer_smallROI \
        --gpu_ids 4
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import torch

# 设置环境变量
os.environ['nnUNet_raw'] = str(Path('/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_raw'))
os.environ['nnUNet_preprocessed'] = str(Path('/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_preprocessed'))
os.environ['nnUNet_results'] = str(Path('/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_results'))

# 导入 nnUNet 相关模块
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.label_handling.label_handling import LabelManager

# Patch PlansManager 以兼容旧版 plans.json
def _patched_get_label_manager(self, dataset_json_or_dict):
    if isinstance(dataset_json_or_dict, str):
        from batchgenerators.utilities.file_and_folder_operations import load_json
        dataset_json = load_json(dataset_json_or_dict)
    else:
        dataset_json = dataset_json_or_dict
    
    # 手动解析 label_manager_class，绕过 property 限制
    lm_name = self.plans.get('label_manager', 'LabelManager')
    if isinstance(lm_name, str):
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        import nnunetv2
        from batchgenerators.utilities.file_and_folder_operations import join
        
        label_manager_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "utilities", "label_handling"),
            lm_name,
            current_module="nnunetv2.utilities.label_handling"
        )
        if label_manager_class is None:
            from nnunetv2.utilities.label_handling.label_handling import LabelManager
            label_manager_class = LabelManager
    else:
        label_manager_class = self.label_manager_class
    
    return label_manager_class(
        label_dict=dataset_json['labels'],
        regions_class_order=dataset_json.get('regions_class_order')
    )

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
PlansManager.get_label_manager = _patched_get_label_manager


def load_case_data(input_folder: Path, case_name: str) -> tuple:
    """
    加载单个病例的图像和标签数据
    
    Returns:
        (image_array, seg_array, properties_dict)
    """
    image_path = input_folder / f"{case_name}.npy"
    seg_path = input_folder / f"{case_name}_seg.npy"
    pkl_path = input_folder / f"{case_name}.pkl"
    
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {seg_path}")
    if not pkl_path.exists():
        raise FileNotFoundError(f"属性文件不存在: {pkl_path}")
    
    image = np.load(image_path)
    seg = np.load(seg_path)
    
    import pickle
    with open(pkl_path, 'rb') as f:
        properties = pickle.load(f)
    
    return image, seg, properties


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, label: int = 1) -> Dict:
    """
    计算分割指标
    
    Args:
        pred_mask: 预测的二值掩码 (0或label)
        gt_mask: 真实的二值掩码 (0或label)
        label: 目标标签值
        
    Returns:
        包含各项指标的字典
    """
    # 转换为二值
    pred_binary = (pred_mask == label).astype(np.uint8)
    gt_binary = (gt_mask == label).astype(np.uint8)
    
    # 计算混淆矩阵
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    
    total = tp + fp + tn + fn
    
    # Dice
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    # IoU
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Kappa
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
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'recall': float(recall),
        'precision': float(precision),
        'kappa': float(kappa),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def predict_and_evaluate(
    dataset_id: str,
    input_folder: Path,
    folds: List[int],
    configuration: str = '2d',
    trainer: str = 'nnUNetTrainer_smallROI',
    gpu_ids: str = None,
    output_dir: Path = None,
    label_key: int = 1
) -> Dict:
    """
    对内部验证集进行预测并评估精度
    
    Args:
        dataset_id: 数据集ID
        input_folder: 输入文件夹（包含 .npy, _seg.npy, .pkl 文件）
        folds: 使用的模型折数列表
        configuration: 网络配置
        trainer: 训练器名称
        gpu_ids: GPU ID
        output_dir: 输出目录
        label_key: 目标标签值
        
    Returns:
        包含所有结果的字典
    """
    if output_dir is None:
        output_dir = Path('/mnt/data/KASR/Dengsiyi/LR-HCC/result_test/internal_validation_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置GPU
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"使用GPU: {gpu_ids}")
    
    # 获取所有病例名称
    case_files = list(input_folder.glob('*_seg.npy'))
    case_names = sorted([f.stem.replace('_seg', '') for f in case_files])
    
    if not case_names:
        raise ValueError(f"在 {input_folder} 中未找到任何病例")
    
    print(f"\n找到 {len(case_names)} 个病例: {case_names[:5]}..." if len(case_names) > 5 else f"\n找到 {len(case_names)} 个病例: {case_names}")
    
    # 初始化预测器
    print("\n初始化预测器...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # 加载多折模型
    model_folder = Path('/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_results') / f'Dataset{dataset_id}_HCC' / f'{trainer}__nnUNetPlans__{configuration}'
    
    if not model_folder.exists():
        raise FileNotFoundError(f"模型文件夹不存在: {model_folder}")
    
    print(f"从 {len(folds)} 个fold加载模型: {folds}")
    checkpoint_paths = [model_folder / f'fold_{f}' / 'checkpoint_final.pth' for f in folds]
    
    for ckpt_path in checkpoint_paths:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"检查点不存在: {ckpt_path}")
    
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=folds,
        checkpoint_name='checkpoint_final.pth'
    )
    
    print("✅ 模型加载完成\n")
    
    # 逐病例预测和评估
    results = []
    
    # 创建预测mask保存目录
    mask_save_dir = output_dir / 'predicted_masks'
    mask_save_dir.mkdir(parents=True, exist_ok=True)
    
    for case_name in tqdm(case_names, desc="预测与评估"):
        try:
            # 加载数据
            image, gt_seg, properties = load_case_data(input_folder, case_name)
            
            # 预测（返回概率图）
            # 当 save_or_return_probabilities=True 时，返回 (segmentation, probabilities)
            # segmentation 已经是 argmax 后的结果，不需要再处理
            result = predictor.predict_single_npy_array(
                input_image=image,
                image_properties=properties,
                save_or_return_probabilities=True
            )
            
            # 解包返回值
            if isinstance(result, tuple):
                pred_label, predicted_probabilities = result
            else:
                pred_label = result
                predicted_probabilities = None
            
            # 保存预测的mask
            mask_path = mask_save_dir / f"{case_name}_pred.npy"
            np.save(mask_path, pred_label)
            
            print(f"\n{case_name}:")
            if predicted_probabilities is not None:
                print(f"  predicted_probabilities shape: {predicted_probabilities.shape}")
                print(f"  predicted_probabilities min/max: {predicted_probabilities.min():.4f} / {predicted_probabilities.max():.4f}")
            
            print(f"  pred_label shape: {pred_label.shape}")
            print(f"  pred_label unique values: {np.unique(pred_label)}")
            print(f"  pred_label value counts: {dict(zip(*np.unique(pred_label, return_counts=True)))}")
            print(f"  ✅ 预测mask已保存: {mask_path}")
            
            # 真实标签已经是正确的格式，不需要去掉batch维度
            # seg.npy 的shape是 (1, H, W) 对于2D，直接取 [0]
            gt_label = gt_seg[0] if gt_seg.ndim == 3 else gt_seg
            print(f"  gt_label shape: {gt_label.shape}")
            print(f"  gt_label unique values: {np.unique(gt_label)}")
            print(f"  gt_label value counts: {dict(zip(*np.unique(gt_label, return_counts=True)))}")
            
            # 计算指标
            metrics = compute_metrics(pred_label, gt_label, label=label_key)
            metrics['case_name'] = case_name
            metrics['mask_path'] = str(mask_path)
            
            results.append(metrics)
            
        except Exception as e:
            print(f"\n❌ 处理 {case_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        raise ValueError("没有成功处理的病例")
    
    # 保存详细结果
    csv_path = output_dir / f'validation_details_folds_{"_".join(map(str, folds))}.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n📊 详细结果已保存到: {csv_path}")
    
    # 计算聚合指标
    aggregate_metrics = {
        metric: {
            'mean': df_results[metric].mean(),
            'std': df_results[metric].std(),
            'median': df_results[metric].median(),
            'min': df_results[metric].min(),
            'max': df_results[metric].max()
        }
        for metric in ['dice', 'iou', 'recall', 'precision', 'kappa']
    }
    
    # 保存汇总结果
    summary = {
        'dataset_id': dataset_id,
        'folds': folds,
        'configuration': configuration,
        'trainer': trainer,
        'num_cases': len(df_results),
        'aggregate_metrics': aggregate_metrics,
        'case_details': df_results.to_dict('records')
    }
    
    json_path = output_dir / f'validation_summary_folds_{"_".join(map(str, folds))}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📊 汇总结果已保存到: {json_path}")
    
    # 打印结果
    print('\n' + '='*70)
    print('内部验证集分割性能指标分析')
    print('='*70)
    print(f'数据集: Dataset{dataset_id}_HCC')
    print(f'Folds: {folds}')
    print(f'配置: {configuration}')
    print(f'Trainer: {trainer}')
    print(f'Case 数量: {len(df_results)}')
    print('-'*70)
    print('聚合指标 (Mean ± Std):')
    print(f'  Dice:     {aggregate_metrics["dice"]["mean"]:>8.4f} ± {aggregate_metrics["dice"]["std"]:>6.4f}')
    print(f'  IoU:      {aggregate_metrics["iou"]["mean"]:>8.4f} ± {aggregate_metrics["iou"]["std"]:>6.4f}')
    print(f'  Recall:   {aggregate_metrics["recall"]["mean"]:>8.4f} ± {aggregate_metrics["recall"]["std"]:>6.4f}')
    print(f'  Precision:{aggregate_metrics["precision"]["mean"]:>8.4f} ± {aggregate_metrics["precision"]["std"]:>6.4f}')
    print(f'  Kappa:    {aggregate_metrics["kappa"]["mean"]:>8.4f} ± {aggregate_metrics["kappa"]["std"]:>6.4f}')
    print('-'*70)
    
    # Top/Bottom cases
    print('\n表现最好的 5 个 Case (按 Dice):')
    top_5 = df_results.nlargest(5, 'dice')
    for _, row in top_5.iterrows():
        print(f'  {row["case_name"]:15s} | Dice: {row["dice"]:.4f} | IoU: {row["iou"]:.4f}')
    
    print('\n表现最差的 5 个 Case (按 Dice):')
    bottom_5 = df_results.nsmallest(5, 'dice')
    for _, row in bottom_5.iterrows():
        print(f'  {row["case_name"]:15s} | Dice: {row["dice"]:.4f} | IoU: {row["iou"]:.4f}')
    
    print('='*70)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='内部验证集精度测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset_id', type=str, default='001',
                       help='数据集ID (默认: 001)')
    
    parser.add_argument('--input_folder', type=str, required=True,
                       help='输入文件夹路径（包含预处理后的 .npy, _seg.npy, .pkl 文件）')
    
    parser.add_argument('--folds', type=int, nargs='+', default=[0],
                       help='使用的模型折数列表 (默认: [0])，例如: --folds 0 1')
    
    parser.add_argument('--configuration', type=str, default='2d',
                       choices=['2d', '3d_fullres', '3d_lowres'],
                       help='网络配置 (默认: 2d)')
    
    parser.add_argument('--trainer', type=str, default='nnUNetTrainer_smallROI',
                       help='训练器名称 (默认: nnUNetTrainer_smallROI)')
    
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='GPU ID，例如: "4" (默认: 自动选择)')
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: ./internal_validation_results)')
    
    parser.add_argument('--label_key', type=int, default=1,
                       help='目标标签值 (默认: 1)')
    
    args = parser.parse_args()
    
    try:
        result = predict_and_evaluate(
            dataset_id=args.dataset_id,
            input_folder=Path(args.input_folder),
            folds=args.folds,
            configuration=args.configuration,
            trainer=args.trainer,
            gpu_ids=args.gpu_ids,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            label_key=args.label_key
        )
        
        print(f'\n✅ 验证完成! 结果已保存到 {args.output_dir or "./internal_validation_results"}')
        
    except Exception as e:
        print(f'\n❌ 错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
