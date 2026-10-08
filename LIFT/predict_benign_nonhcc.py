#!/usr/bin/env python3
"""只预测 val 集中 Benign + Non-HCC Malignancies 的 case (nnUNet 3-fold ensemble)"""
import os, sys, csv, glob, subprocess, re, shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LIFT_DIR = PROJECT_ROOT / 'LIFT'
NNUNET_RAW = PROJECT_ROOT / 'data' / 'nnUNet_raw' / 'Dataset001_HCC'
NNUNET_RESULTS = PROJECT_ROOT / 'data' / 'nnUNet_results'
NNUNET_PREPROCESSED = PROJECT_ROOT / 'data' / 'nnUNet_preprocessed'
CONDA_BIN = '/mnt/data/KASR/.conda/envs/nnunetv2/bin'
MODEL2_DIR = LIFT_DIR / 'ckpts' / 'Model2' / 'uniformer_small_IL_features' / 'pred_results' / 'tables'

CN = ['Benign', 'Non-HCC Malignancies', 'HCC']
CI = {n: i for i, n in enumerate(CN)}


def pc(v):
    v = str(v).strip()
    return int(v) if v.isdigit() else CI.get(v, -1)


# 读取 val ground truth，筛选 Benign(0) 和 Non-HCC(1)
target_cases = []
with open(MODEL2_DIR / 'val_ground_truth.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        tc = pc(row.get('True Class', -1))
        if tc in [0, 1]:
            target_cases.append(row['Case ID'])

print(f"目标: {len(target_cases)} cases (Benign + Non-HCC Malignancies)")

# 创建专用输入目录
input_dir = LIFT_DIR / 'data' / 'masks_predicted' / '_input_benign_nonhcc'
if input_dir.exists():
    shutil.rmtree(input_dir)
input_dir.mkdir(parents=True)

imagesTr = NNUNET_RAW / 'imagesTr'
modality_suffixes = ['_0000.nii.gz', '_0001.nii.gz', '_0002.nii.gz', '_0003.nii.gz']

linked = 0
for cid in target_cases:
    m = re.match(r'(case_\d+)(?:_\d+)?$', cid)
    if not m:
        continue
    base_id = m.group(1)
    for suf in modality_suffixes:
        src = imagesTr / f'{base_id}{suf}'
        dst = input_dir / f'{base_id}{suf}'
        if src.exists() and not dst.exists():
            os.symlink(str(src), str(dst))
            linked += 1

unique_bases = set(re.match(r'(case_\d+)(?:_\d+)?$', c).group(1) for c in target_cases)
print(f"Unique base cases: {len(unique_bases)}, symlinks created: {linked}")

# 输出目录
output_dir = LIFT_DIR / 'data' / 'masks_predicted' / 'val'
output_dir.mkdir(parents=True, exist_ok=True)

# 运行 nnUNet 3-fold ensemble
env = os.environ.copy()
env['nnUNet_raw'] = str(NNUNET_RAW.parent)
env['nnUNet_preprocessed'] = str(NNUNET_PREPROCESSED)
env['nnUNet_results'] = str(NNUNET_RESULTS)
env['PATH'] = CONDA_BIN + ':' + env.get('PATH', '')

cmd = [
    os.path.join(CONDA_BIN, 'nnUNetv2_predict'),
    '-i', str(input_dir),
    '-o', str(output_dir),
    '-d', 'Dataset001_HCC',
    '-c', '2d',
    '-f', '0', '1', '2',
    '-tr', 'nnUNetTrainer_smallROI',
    '-npp', '3', '-nps', '3',
    '--save_probabilities',
]

print(f"\n开始预测 {len(unique_bases)} base cases (3-fold ensemble)...")
print(f"预计耗时: ~{len(unique_bases) * 50 // 60} 分钟")
print(f"命令: {' '.join(cmd)}\n")
sys.stdout.flush()

result = subprocess.run(cmd, env=env)

# 完成后统计
pred_files = glob.glob(str(output_dir / 'case_*.nii.gz'))
print(f"\n预测完成! 共 {len(pred_files)} 个 mask 文件")

if result.returncode != 0:
    print(f"WARNING: nnUNet exited with code {result.returncode}")
