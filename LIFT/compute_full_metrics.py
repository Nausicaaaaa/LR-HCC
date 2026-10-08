#!/usr/bin/env python3
"""Val/Test 完整分割指标统计: Dice, Precision, Recall, Kappa, mIoU"""
import csv, numpy as np, glob, os, SimpleITK as sitk

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
    out = {}
    for p, g in zip(preds, gts):
        cid = p.get('Case ID', '')
        ti = pc(g.get('True Class', -1))
        pi = pc(p.get('Predicted Class', -1))
        out[cid] = {'t': ti, 'p': pi, 'ok': ti == pi}
    return out

def compute_all_metrics(pred_path, gt_path):
    """计算完整分割指标"""
    p = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
    g = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
    if p.shape != g.shape:
        return None
    pb = (p > 0).astype(np.uint8)
    gb = (g > 0).astype(np.uint8)
    tp = int(np.sum(pb & gb))
    fp = int(np.sum(pb & (~gb.astype(bool))))
    fn = int(np.sum((~pb.astype(bool)) & gb))
    tn = int(np.sum((~pb.astype(bool)) & (~gb.astype(bool))))
    total = tp + fp + fn + tn

    # Dice
    dice = 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if (tp + fp + fn) == 0 else 0.0)
    # IoU
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if (tp + fp + fn) == 0 else 0.0)
    # Precision
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
    # Recall
    rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
    # Cohen's Kappa (binary)
    if total > 0:
        po = (tp + tn) / total
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total ** 2)
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    else:
        kappa = 0.0

    return {'dice': dice, 'iou': iou, 'prec': prec, 'rec': rec, 'kappa': kappa,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

def evaluate_split(pred_dir, gt_dir, cls_map, label):
    """评估一个 split 的全部指标"""
    pred_files = sorted(glob.glob(os.path.join(pred_dir, 'case_*.nii.gz')))
    results = []
    for pf in pred_files:
        cid = os.path.basename(pf).replace('.nii.gz', '')
        gt_path = os.path.join(gt_dir, f'{cid}.nii.gz')
        if not os.path.exists(gt_path):
            continue
        m = compute_all_metrics(pf, gt_path)
        if m is None:
            continue
        ci = cls_map.get(cid)
        results.append({'c': cid, **m, 'cls': ci})

    # ===== 打印分割指标表 =====
    print(f"\n{'='*85}")
    print(f"  {label} 分割指标 (nnUNet 3-fold Ensemble)")
    print(f"{'='*85}")

    metrics_keys = ['dice', 'prec', 'rec', 'iou', 'kappa']
    metric_names = ['Dice', 'Precision', 'Recall', 'mIoU', 'Kappa']

    # 全量统计（包含 Dice=0 的 case）
    for subset_name, subset_filter in [("全部 cases", lambda r: True),
                                        ("仅检出 cases (Dice>0)", lambda r: r['dice'] > 0)]:
        print(f"\n  --- {subset_name} ---")
        hdr = f"  {'Category':<25s} {'N':>5}"
        for mn in metric_names:
            hdr += f" {mn:>10}"
        print(hdr)
        print(f"  {'-'*80}")

        filtered = [r for r in results if subset_filter(r)]
        for ci_idx in [None] + list(range(3)):
            if ci_idx is None:
                cn = 'ALL'
                cs = filtered
            else:
                cn = CN[ci_idx]
                cs = [r for r in filtered if r['cls'] and r['cls']['t'] == ci_idx]
            if not cs:
                continue
            row = f"  {cn:<25s} {len(cs):>5}"
            for mk in metrics_keys:
                vals = [r[mk] for r in cs]
                row += f" {np.mean(vals):>10.4f}"
            print(row)

    # ===== 分类精度 =====
    all_cls = list(cls_map.values())
    print(f"\n  --- 分类精度 (Model2) ---")
    hdr = f"  {'Category':<25s} {'N':>5} {'Correct':>8} {'ACC':>8}"
    print(hdr)
    print(f"  {'-'*50}")
    total = len(all_cls)
    ok = sum(1 for v in all_cls if v['ok'])
    print(f"  {'ALL':<25s} {total:>5} {ok:>8} {ok/total:>8.4f} ({ok/total*100:.1f}%)")
    for ci_idx, cn in enumerate(CN):
        cs = [v for v in all_cls if v['t'] == ci_idx]
        cok = sum(1 for v in cs if v['ok'])
        if cs:
            print(f"  {cn:<25s} {len(cs):>5} {cok:>8} {cok/len(cs):>8.4f} ({cok/len(cs)*100:.1f}%)")

    # ===== 联合 =====
    joint = [r for r in results if r['cls'] and r['cls']['ok'] and r['dice'] > 0]
    print(f"\n  --- 分割+分类联合 (分类正确 且 Dice>0) ---")
    print(f"  联合: {len(joint)}/{len(results)} ({len(joint)/len(results)*100:.1f}%)")
    if joint:
        hdr = f"  {'Category':<25s} {'N':>5}"
        for mn in metric_names:
            hdr += f" {mn:>10}"
        print(hdr)
        print(f"  {'-'*80}")
        for ci_idx in [None] + list(range(3)):
            if ci_idx is None:
                cn = 'ALL'
                cs = joint
            else:
                cn = CN[ci_idx]
                cs = [r for r in joint if r['cls']['t'] == ci_idx]
            if not cs:
                continue
            row = f"  {cn:<25s} {len(cs):>5}"
            for mk in metrics_keys:
                vals = [r[mk] for r in cs]
                row += f" {np.mean(vals):>10.4f}"
            print(row)

    return results

# ===== VAL =====
print("正在计算 VAL 集指标...")
val_cls = load_cls('val')
val_results = evaluate_split(
    'LIFT/data/masks_predicted/val',
    'data/nnUNet_raw/Dataset001_HCC/labelsTr',
    val_cls, 'VAL'
)

# ===== TEST =====
print("\n正在计算 TEST 集指标...")
test_cls = load_cls('test')
test_results = evaluate_split(
    'LIFT/data/masks_predicted/test',
    'data/nnUNet_raw/Dataset001_HCC/labelsTs',
    test_cls, 'TEST'
)
