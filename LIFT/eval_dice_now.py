#!/usr/bin/env python3
"""实时评估 nnUNet ensemble Dice，可与分类结果交叉对比"""
import os, sys, glob, csv, time
import numpy as np
import SimpleITK as sitk

PRED_DIR = 'LIFT/data/masks_predicted'
GT_DIR = 'data/nnUNet_raw/Dataset001_HCC/labelsTr'
MODEL2_DIR = 'LIFT/ckpts/Model2/uniformer_small_IL_features/pred_results/tables'
CN = ['Benign', 'Non-HCC Malignancies', 'HCC']
CI = {n: i for i, n in enumerate(CN)}

def pc(v):
    v = str(v).strip()
    return int(v) if v.isdigit() else CI.get(v, -1)

def dice_sitk(pred_path, gt_path):
    """用 SimpleITK 快速计算 Dice"""
    p = sitk.ReadImage(pred_path)
    g = sitk.ReadImage(gt_path)
    pa = sitk.GetArrayFromImage(p)
    ga = sitk.GetArrayFromImage(g)
    pb = (pa > 0).astype(np.uint8)
    gb = (ga > 0).astype(np.uint8)
    inter = np.sum(pb & gb)
    total = np.sum(pb) + np.sum(gb)
    return 2.0 * inter / total if total > 0 else (1.0 if inter == 0 else 0.0)

# Load classification
cls_map = {}
for s in ['val', 'test']:
    pf = os.path.join(MODEL2_DIR, f'{s}_predictions.csv')
    gf = os.path.join(MODEL2_DIR, f'{s}_ground_truth.csv')
    if not os.path.exists(pf):
        continue
    with open(pf, encoding='utf-8-sig') as f:
        preds = list(csv.DictReader(f))
    with open(gf, encoding='utf-8-sig') as f:
        gts = list(csv.DictReader(f))
    for p, g in zip(preds, gts):
        c = p.get('Case ID', '')
        if c:
            cls_map[c] = {'p': pc(p.get('Predicted Class', -1)), 't': pc(g.get('True Class', -1))}
            cls_map[c]['ok'] = cls_map[c]['p'] == cls_map[c]['t']

sys.stdout.write(f'Loaded {len(cls_map)} classification results\n')
sys.stdout.flush()

# Compute Dice for val
pred_files = sorted(glob.glob(os.path.join(PRED_DIR, 'val', 'case_*.nii.gz')))
sys.stdout.write(f'Found {len(pred_files)} predicted val masks\n\n')
sys.stdout.flush()

results = []
for pf in pred_files:
    cid = os.path.basename(pf).replace('.nii.gz', '')
    gt_path = os.path.join(GT_DIR, f'{cid}.nii.gz')
    if not os.path.exists(gt_path):
        sys.stdout.write(f'  {cid}: no GT, skip\n')
        sys.stdout.flush()
        continue
    try:
        d = dice_sitk(pf, gt_path)
    except Exception as e:
        sys.stdout.write(f'  {cid}: ERROR {e}\n')
        sys.stdout.flush()
        continue
    ci = cls_map.get(cid)
    results.append({'c': cid, 'd': d, 'cls': ci})
    sys.stdout.write(f'  {cid}: Dice={d:.4f}')
    if ci:
        sys.stdout.write(f'  cls={CN[ci["p"]]}(GT:{CN[ci["t"]]}) {"OK" if ci["ok"] else "WRONG"}')
    sys.stdout.write('\n')
    sys.stdout.flush()

results.sort(key=lambda x: x['d'], reverse=True)
ds = [r['d'] for r in results]

print(f'\n{"="*72}')
print(f'nnUNet 3-fold Ensemble | Val set | {len(results)} cases evaluated')
print(f'Dice: mean={np.mean(ds):.4f} median={np.median(ds):.4f} max={np.max(ds):.4f} min={np.min(ds):.4f}')
print(f'{"="*72}')

# Per-class best
print(f'\n>>> Per-class best correct prediction:')
for ci_idx, cn in enumerate(CN):
    cs = [r for r in results if r['cls'] and r['cls']['ok'] and r['cls']['t'] == ci_idx]
    if cs:
        b = cs[0]
        print(f'  [{cn:25s}] {b["c"]:>10s}  Dice={b["d"]:.4f}  (n={len(cs)} correct)')
    else:
        print(f'  [{cn:25s}] no correct case yet')

# Top 10
print(f'\n>>> Top 10 by Dice:')
print(f'{"Case":<12} {"Dice":>7} {"Predicted":<24} {"Ground Truth":<24} {"OK":>4}')
print('-' * 72)
for r in results[:10]:
    c = r['cls']
    if c:
        print(f'{r["c"]:<12} {r["d"]:>7.4f} {CN[c["p"]]:<24} {CN[c["t"]]:<24} {"Y" if c["ok"] else "N":>4}')
    else:
        print(f'{r["c"]:<12} {r["d"]:>7.4f} {"N/A":<24} {"N/A":<24} {"N/A":>4}')
