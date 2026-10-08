#!/usr/bin/env python3
"""
用 nnUNet 3-fold (fold_0, fold_1, fold_2) 集成模型预测分类 val 和 test 集的分割 mask。

输出:
- LIFT/data/masks_predicted/val/  : val 集每个 case 的预测 mask (case_xxx.nii.gz)
- LIFT/data/masks_predicted/test/ : test 集每个 case 的预测 mask (case_xxx.nii.gz)

用法:
    python LIFT/predict_masks_ensemble.py
    python LIFT/predict_masks_ensemble.py --skip-symlink  # 跳过symlink创建，直接预测
    python LIFT/predict_masks_ensemble.py --val-only      # 只预测val集
    python LIFT/predict_masks_ensemble.py --test-only     # 只预测test集
"""

import os
import sys
import re
import argparse
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LIFT_DIR = PROJECT_ROOT / 'LIFT'
NNUNET_RAW_DIR = PROJECT_ROOT / 'data' / 'nnUNet_raw'
NNUNET_RESULTS_DIR = PROJECT_ROOT / 'data' / 'nnUNet_results'
NNUNET_PREPROCESSED_DIR = PROJECT_ROOT / 'data' / 'nnUNet_preprocessed'
CONDA_BIN = '/mnt/data/KASR/.conda/envs/nnunetv2/bin'
PYTHON = os.path.join(CONDA_BIN, 'python')

# 输出目录
OUTPUT_BASE = LIFT_DIR / 'data' / 'masks_predicted'


def parse_val_cases(anno_file):
    """从 val_fold1.txt 解析所有唯一的 base case ID"""
    cases = set()
    with open(anno_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cid = parts[0]
            # 移除多病灶后缀 (_1, _2 等)
            m = re.match(r'(case_\d+)(?:_\d+)?$', cid)
            if m:
                cases.add(m.group(1))
            else:
                cases.add(cid)
    return sorted(cases)


def parse_test_cases(anno_file):
    """从 test.txt 解析所有 case ID"""
    cases = set()
    skip_header = 0
    with open(anno_file, 'r') as f:
        first_line = f.readline().strip()
    if first_line.startswith('casename'):
        skip_header = 1
    with open(anno_file, 'r') as f:
        for i, line in enumerate(f):
            if i < skip_header:
                continue
            parts = line.strip().split('\t')
            if parts:
                cases.add(parts[0])
    return sorted(cases)


def create_symlink_dir(src_dir, case_ids, dst_dir):
    """为指定 case 创建包含 symlink 的输入目录"""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧 symlink
    for f in dst_dir.iterdir():
        if f.is_symlink() or f.is_file():
            f.unlink()

    count = 0
    for cid in case_ids:
        for mod in range(4):
            src_file = src_dir / f'{cid}_{mod:04d}.nii.gz'
            dst_file = dst_dir / f'{cid}_{mod:04d}.nii.gz'
            if src_file.exists():
                dst_file.symlink_to(src_file.resolve())
                count += 1
            else:
                print(f"  WARNING: {src_file} not found")
    return count


def run_nnunet_predict(input_dir, output_dir, folds='0 1 2', num_processes=3):
    """运行 nnUNetv2_predict"""
    env = os.environ.copy()
    env['nnUNet_raw'] = str(NNUNET_RAW_DIR)
    env['nnUNet_preprocessed'] = str(NNUNET_PREPROCESSED_DIR)
    env['nnUNet_results'] = str(NNUNET_RESULTS_DIR)
    env['PATH'] = CONDA_BIN + ':' + env.get('PATH', '')

    cmd = [
        os.path.join(CONDA_BIN, 'nnUNetv2_predict'),
        '-i', str(input_dir),
        '-o', str(output_dir),
        '-d', 'Dataset001_HCC',
        '-c', '2d',
        '-f', *folds.split(),
        '-tr', 'nnUNetTrainer_smallROI',
        '-npp', str(num_processes),
        '-nps', str(num_processes),
        '--save_probabilities',
    ]

    print(f"\n运行命令:")
    print(f"  {' '.join(cmd)}")
    print(f"  nnUNet_raw={env['nnUNet_raw']}")
    print(f"  nnUNet_results={env['nnUNet_results']}")
    print()

    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='nnUNet 3-fold ensemble 预测 val/test mask')
    parser.add_argument('--skip-symlink', action='store_true',
                        help='跳过 symlink 创建，直接使用已有输入目录')
    parser.add_argument('--val-only', action='store_true', help='只预测 val 集')
    parser.add_argument('--test-only', action='store_true', help='只预测 test 集')
    parser.add_argument('--folds', default='0 1 2', help='使用的 fold (default: 0 1 2)')
    parser.add_argument('--num-processes', type=int, default=3, help='预处理/导出进程数')
    args = parser.parse_args()

    val_anno = LIFT_DIR / 'data' / 'labels' / 'val_fold1.txt'
    test_anno = LIFT_DIR / 'data' / 'labels' / 'test.txt'

    predict_val = not args.test_only
    predict_test = not args.val_only

    # ====== Step 1: 解析 case ID ======
    val_cases = parse_val_cases(val_anno) if predict_val else []
    test_cases = parse_test_cases(test_anno) if predict_test else []

    print(f"{'='*60}")
    print(f"nnUNet 3-fold Ensemble 预测")
    print(f"Folds: {args.folds}")
    print(f"Val cases: {len(val_cases)}")
    print(f"Test cases: {len(test_cases)}")
    print(f"{'='*60}")

    # ====== Step 2: 创建 symlink 输入目录 ======
    if not args.skip_symlink:
        if predict_val:
            val_input = OUTPUT_BASE / '_input_val'
            print(f"\n创建 val 输入目录: {val_input}")
            n = create_symlink_dir(NNUNET_RAW_DIR / 'Dataset001_HCC' / 'imagesTr', val_cases, val_input)
            print(f"  创建了 {n} 个 symlink ({len(val_cases)} cases × 4 modalities)")

        if predict_test:
            test_input = OUTPUT_BASE / '_input_test'
            print(f"\n创建 test 输入目录: {test_input}")
            n = create_symlink_dir(NNUNET_RAW_DIR / 'Dataset001_HCC' / 'imagesTs', test_cases, test_input)
            print(f"  创建了 {n} 个 symlink ({len(test_cases)} cases × 4 modalities)")

    # ====== Step 3: 运行 nnUNet 预测 ======
    if predict_val:
        val_input = OUTPUT_BASE / '_input_val'
        val_output = OUTPUT_BASE / 'val'
        val_output.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"预测 Val 集 ({len(val_cases)} cases)")
        print(f"{'='*60}")

        rc = run_nnunet_predict(val_input, val_output, args.folds, args.num_processes)
        if rc != 0:
            print(f"ERROR: Val 预测失败 (return code: {rc})")
            sys.exit(1)
        print(f"Val 预测完成! 结果在: {val_output}")

    if predict_test:
        test_input = OUTPUT_BASE / '_input_test'
        test_output = OUTPUT_BASE / 'test'
        test_output.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"预测 Test 集 ({len(test_cases)} cases)")
        print(f"{'='*60}")

        rc = run_nnunet_predict(test_input, test_output, args.folds, args.num_processes)
        if rc != 0:
            print(f"ERROR: Test 预测失败 (return code: {rc})")
            sys.exit(1)
        print(f"Test 预测完成! 结果在: {test_output}")

    # ====== Step 4: 验证结果 ======
    print(f"\n{'='*60}")
    print(f"验证结果")
    print(f"{'='*60}")

    for split_name, expected_cases in [('val', val_cases), ('test', test_cases)]:
        if not expected_cases:
            continue
        output_dir = OUTPUT_BASE / split_name
        if not output_dir.exists():
            print(f"  {split_name}: 输出目录不存在!")
            continue
        pred_files = sorted([f.stem.replace('.nii', '') for f in output_dir.glob('case_*.nii.gz')])
        pred_cases = set(pred_files)
        expected_set = set(expected_cases)
        found = expected_set & pred_cases
        missing = expected_set - pred_cases
        print(f"  {split_name}: 期望 {len(expected_set)} cases, 预测到 {len(found)} cases")
        if missing:
            print(f"    缺失: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}")

    print(f"\n预测结果保存在: {OUTPUT_BASE}")
    print("完成!")


if __name__ == '__main__':
    main()
