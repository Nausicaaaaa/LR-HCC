#!/usr/bin/env python3
"""
检查并修正 ALL 标签文件中的 label 值，确保与 case_name_mapping.txt 中的分组一致。

功能：
    读取 ALL 标签文件和 case_name_mapping.txt，根据分组信息自动修正 label：
    - HCC -> 0
    - noHCC -> 1
    - liangxing -> 2
    修正后按自然顺序排序并保存。

使用示例：
    python check_and_fix_labels.py
"""

import pandas as pd
import re

def natural_sort_key(s):
    """自然排序键函数"""
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def main():
    # 读取ALL文件
    print("读取ALL文件...")
    all_file = '/mnt/data/KASR/Dengsiyi/LR-HCC/LIFT/data/labels/ALL'
    all_df = pd.read_csv(all_file, sep='\t', dtype=str)
    print(f"ALL文件包含 {len(all_df)} 条记录")
    
    # 读取case_name_mapping.txt
    print("\n读取case_name_mapping.txt...")
    mapping_file = '/mnt/data/KASR/Dengsiyi/LR-HCC/case_name_mapping.txt'
    mapping_df = pd.read_csv(mapping_file, sep='\t', dtype=str)
    print(f"映射文件包含 {len(mapping_df)} 条记录")
    
    # 创建patient_id到group的映射
    patient_to_group = {}
    for idx, row in mapping_df.iterrows():
        patient_id = str(row['converted_case_name']).strip()
        group = str(row['group']).strip()
        if patient_id:
            patient_to_group[patient_id] = group
    
    # group到label的映射
    group_to_label = {
        'HCC': '0',
        'noHCC': '1',
        'liangxing': '2'
    }
    
    # 检查并修正label
    print("\n检查label值...")
    corrected_count = 0
    error_details = []
    
    for idx, row in all_df.iterrows():
        patient_id = row['patient_id']
        current_label = str(row['label']).strip()
        
        if patient_id in patient_to_group:
            group = patient_to_group[patient_id]
            expected_label = group_to_label.get(group, None)
            
            if expected_label and current_label != expected_label:
                error_details.append({
                    'patient_id': patient_id,
                    'current_label': current_label,
                    'group': group,
                    'expected_label': expected_label
                })
                all_df.at[idx, 'label'] = expected_label
                corrected_count += 1
    
    print(f"发现并修正 {corrected_count} 个错误的label值")
    
    if error_details:
        print("\n错误详情（前10个）:")
        for err in error_details[:10]:
            print(f"  {err['patient_id']}: label={err['current_label']} -> {err['expected_label']} (group={err['group']})")
    
    # 按自然顺序排序
    print("\n按自然顺序排序...")
    all_df['sort_key'] = all_df['patient_id'].apply(natural_sort_key)
    all_df = all_df.sort_values('sort_key').drop('sort_key', axis=1)
    all_df = all_df.reset_index(drop=True)
    
    # 保存结果
    all_df.to_csv(all_file, sep='\t', index=False)
    print(f"\n结果已保存到: {all_file}")
    
    # 统计信息
    print("\n=== 统计信息 ===")
    print(f"总记录数: {len(all_df)}")
    print("\nlabel分布:")
    print(all_df['label'].value_counts().sort_index())

if __name__ == '__main__':
    main()
