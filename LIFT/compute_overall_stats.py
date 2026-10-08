#!/usr/bin/env python3
"""Val/Test 整体精度统计"""
import csv, numpy as np

CN = ['Benign', 'Non-HCC Malignancies', 'HCC']
CI = {n: i for i, n in enumerate(CN)}

def pc(v):
    v = str(v).strip()
    return int(v) if v.isdigit() else CI.get(v, -1)

def load_cls(split):
    d = 'LIFT/ckpts/Model2/uniformer_small_IL_features/pred_results/tables'
    with open(f'{d}/{split}_predictions.csv', encoding='utf-8-sig') as f:
        preds = list(csv.DictReader(f))
    with open(f'{d}/{split}_ground_truth.csv', encoding='utf-8-sig') as f:
        gts = list(csv.DictReader(f))
    out = []
    for p, g in zip(preds, gts):
        ti = pc(g.get('True Class', -1))
        pi = pc(p.get('Predicted Class', -1))
        out.append({'t': ti, 'p': pi, 'ok': ti == pi})
    return out

def load_val_dice():
    results = []
    with open('/tmp/dice_eval_full.txt') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('case_') or 'Dice=' not in line:
                continue
            parts = line.split()
            cid = parts[0].rstrip(':')
            dv = float(parts[1].split('=')[1])
            cls_info = None
            if 'cls=' in line:
                cp = line.split('cls=')[1].split(')')[0]
                ps = cp.split('(GT:')[0]
                gs = cp.split('(GT:')[1]
                pi = CI.get(ps, -1)
                ti = CI.get(gs, -1)
                ok = 'OK' in line
                cls_info = {'t': ti, 'p': pi, 'ok': ok}
            results.append({'c': cid, 'd': dv, 'cls': cls_info})
    return results

def load_test_dice():
    results = []
    with open('LIFT/data/masks_predicted/test_dice_results.csv') as f:
        for row in csv.DictReader(f):
            d = float(row['dice'])
            pi = CI.get(row['predicted_class'], -1)
            ti = CI.get(row['true_class'], -1)
            ok = row['correct'] == 'Y'
            results.append({'c': row['case_id'], 'd': d, 'cls': {'t': ti, 'p': pi, 'ok': ok}})
    return results

HDR = f"{'Category':<25s} {'N':>5} {'Detect':>6} {'mean':>7} {'median':>7} {'std':>7} {'>0.5':>5} {'>0.8':>5}"
SEP = '-' * 75

def print_dice_table(dice_list, label):
    print(f"\n--- {label} 分割精度 (Dice) ---")
    print(HDR)
    print(SEP)
    # ALL
    ds = [v['d'] for v in dice_list]
    det = [d for d in ds if d > 0]
    print(f"{'ALL':<25s} {len(ds):>5} {len(det):>6} {np.mean(ds):>7.4f} {np.median(ds):>7.4f} {np.std(ds):>7.4f} {sum(1 for d in ds if d>0.5):>5} {sum(1 for d in ds if d>0.8):>5}")
    for ci, cn in enumerate(CN):
        cs = [v for v in dice_list if v['cls'] and v['cls']['t'] == ci]
        ds = [v['d'] for v in cs]
        det = [d for d in ds if d > 0]
        if ds:
            print(f"{cn:<25s} {len(ds):>5} {len(det):>6} {np.mean(ds):>7.4f} {np.median(ds):>7.4f} {np.std(ds):>7.4f} {sum(1 for d in ds if d>0.5):>5} {sum(1 for d in ds if d>0.8):>5}")

def print_cls_table(cls_list, label):
    print(f"\n--- {label} 分类精度 (Model2) ---")
    hdr = f"{'Category':<25s} {'N':>5} {'Correct':>8} {'ACC':>8}"
    print(hdr)
    print('-' * 50)
    total = len(cls_list)
    ok = sum(1 for v in cls_list if v['ok'])
    print(f"{'ALL':<25s} {total:>5} {ok:>8} {ok/total:>8.4f} ({ok/total*100:.1f}%)")
    for ci, cn in enumerate(CN):
        cs = [v for v in cls_list if v['t'] == ci]
        cok = sum(1 for v in cs if v['ok'])
        if cs:
            print(f"{cn:<25s} {len(cs):>5} {cok:>8} {cok/len(cs):>8.4f} ({cok/len(cs)*100:.1f}%)")

def print_joint(dice_list, label):
    print(f"\n--- {label} 分割+分类联合 ---")
    joint = [v for v in dice_list if v['cls'] and v['cls']['ok'] and v['d'] > 0]
    print(f"  分类正确 且 Dice>0: {len(joint)}/{len(dice_list)} ({len(joint)/len(dice_list)*100:.1f}%)")
    if joint:
        jds = [v['d'] for v in joint]
        print(f"  Dice: mean={np.mean(jds):.4f}, median={np.median(jds):.4f}")
    for ci, cn in enumerate(CN):
        cj = [v for v in joint if v['cls']['t'] == ci]
        if cj:
            print(f"  {cn}: {len(cj)} cases, mean_dice={np.mean([v['d'] for v in cj]):.4f}")

# ========== VAL ==========
print('=' * 75)
print('  VAL 集整体精度统计')
print('=' * 75)
val_dice = load_val_dice()
val_cls = load_cls('val')
print_dice_table(val_dice, 'VAL')
print_cls_table(val_cls, 'VAL')
print_joint(val_dice, 'VAL')

# ========== TEST ==========
print(f"\n{'=' * 75}")
print('  TEST 集整体精度统计')
print('=' * 75)
test_dice = load_test_dice()
test_cls = load_cls('test')
print_dice_table(test_dice, 'TEST')
print_cls_table(test_cls, 'TEST')
print_joint(test_dice, 'TEST')
