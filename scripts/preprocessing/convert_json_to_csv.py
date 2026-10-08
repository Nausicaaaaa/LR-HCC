#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 nnUNet 的 summary.json 评估结果转换为 CSV 格式。

功能：
    读取 nnUNet 生成的 summary.json，提取每个 case 的 Dice、IoU、TP、FP、TN、FN 等指标，
    计算 recall 和 precision，输出为结构化的 CSV 文件。

使用示例：
    python convert_json_to_csv.py

也可以在其他脚本中调用：
    from convert_json_to_csv import convert_json_to_csv
    df = convert_json_to_csv(json_path='/path/to/summary.json', output_csv='/path/to/output.csv')
"""

import json
import pandas as pd
from pathlib import Path
import sys

def convert_json_to_csv(json_path, output_csv=None):
    """将summary.json转换为CSV"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cases_data = []
    
    for case in data['metric_per_case']:
        metrics = case['metrics']['1']
        pred_file = case['prediction_file']
        ref_file = case['reference_file']
        
        case_name = Path(pred_file).stem
        tp = metrics['TP']
        fp = metrics['FP']
        tn = metrics['TN']
        fn = metrics['FN']
        
        # 计算recall和precision
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        cases_data.append({
            'dice': metrics['Dice'],
            'iou': metrics['IoU'],
            'recall': recall,
            'precision': precision,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'case_name': case_name,
            'prediction_file': pred_file,
            'reference_file': ref_file
        })
    
    df = pd.DataFrame(cases_data)
    
    # 确定输出路径
    if output_csv is None:
        output_csv = str(Path(json_path).parent / 'validation_details.csv')
    
    df.to_csv(output_csv, index=False)
    print(f"转换完成: {output_csv}")
    print(f"共 {len(df)} 个case")
    
    return df

if __name__ == '__main__':
    json_path = '/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_results/Dataset001_HCC/predicted_2d/summary.json'
    df = convert_json_to_csv(json_path)
