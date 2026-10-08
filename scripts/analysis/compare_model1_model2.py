#!/usr/bin/env python3
"""
模型1 vs 模型2 对比分析脚本
对比内容:
1. AUC差异比较 (DeLong test + Bootstrap CI)
2. 分类性能比较 (McNemar test, Accuracy, Sensitivity, Specificity, F1 + 95% CI)
3. 校准性能比较 (Calibration curve + Brier score)
4. 对比热图
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix as sk_confusion_matrix
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在服务器上报错
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix as sk_confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ======================== 配置 ========================
MODEL1_DIR = 'LIFT/ckpts/Model1/uniformer_small_IL_features/pred_results'
MODEL2_DIR = 'LIFT/ckpts/Model2/uniformer_small_IL_features/pred_results'
VAL_ANNO = 'LIFT/data/labels/val_fold1.txt'
TEST_ANNO = 'LIFT/data/labels/test.txt'
OUTPUT_DIR = 'model_comparison_results'
NUM_CLASSES = 3
CLASS_NAMES = ['良性', '恶性非HCC', 'HCC']
LABEL_MAP = {'liangxing': 0, 'noHCC': 1, 'HCC': 2}
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================== 数据加载 ========================
def load_annotations(anno_file):
    """从标注文件加载 ground truth 标签, 返回 {case_id: label_int}"""
    mapping = LABEL_MAP
    labels = {}
    with open(anno_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if not parts or parts[0] in ('casename', ''):
                continue
            case_id = parts[0].strip()
            raw_label = parts[1].strip()
            if raw_label in mapping:
                labels[case_id] = mapping[raw_label]
    return labels


def load_scores(score_json_path):
    """从 score.json 加载预测概率, 返回 {case_id: np.array([p0, p1, p2])}"""
    with open(score_json_path, 'r') as f:
        data = json.load(f)
    scores = {}
    for item in data:
        cid = item['image_id']
        if cid == 'casename':
            continue
        scores[cid] = np.array(item['score'], dtype=np.float64)
    return scores


def align_data(annotations, scores):
    """对齐 ground truth 和 prediction, 返回 matched (true_labels, pred_scores)"""
    common_ids = sorted(set(annotations.keys()) & set(scores.keys()))
    true_labels = np.array([annotations[cid] for cid in common_ids])
    pred_scores = np.array([scores[cid] for cid in common_ids])
    pred_labels = np.argmax(pred_scores, axis=1)
    print(f"  匹配样本数: {len(common_ids)}")
    return true_labels, pred_scores, pred_labels, common_ids

from matplotlib.font_manager import FontProperties, fontManager

_CJK_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(_CJK_FONT_PATH)
_cjk_name = FontProperties(fname=_CJK_FONT_PATH).get_name()

CJK_FONT = FontProperties(fname=_CJK_FONT_PATH, size=12)
CJK_FONT_BOLD = FontProperties(fname=_CJK_FONT_PATH, size=14, weight='bold')
CJK_FONT_SM = FontProperties(fname=_CJK_FONT_PATH, size=10)
CJK_FONT_TITLE = FontProperties(fname=_CJK_FONT_PATH, size=15, weight='bold')


# ======================== 统计检验函数 ========================
# -------- DeLong 检验 --------
def _delong_binary_components(y_true, scores):
    """计算二分类 DeLong 检验的 placements 和 AUC

    Args:
        y_true: (n,) 二分类标签 (0/1)
        scores: (n,) 预测分数
    Returns:
        auc, v10, v01  (placements for positive / negative instances)
    """
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n1 = pos_mask.sum()
    n2 = neg_mask.sum()
    if n1 == 0 or n2 == 0:
        return 0.5, np.zeros(0), np.zeros(0)

    s_pos = scores[pos_mask]  # (n1,)
    s_neg = scores[neg_mask]  # (n2,)

    # Pairwise comparison matrix: (n1, n2)
    diff = s_pos[:, np.newaxis] - s_neg[np.newaxis, :]
    phi = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))

    # Placements
    v10 = phi.mean(axis=1)    # (n1,) 每个正样本 vs 所有负样本
    v01 = phi.mean(axis=0)    # (n2,) 每个负样本 vs 所有正样本
    auc_val = v10.mean()

    return auc_val, v10, v01


def _delong_binary_test(y_true, scores_m1, scores_m2):
    """DeLong 检验比较两个模型的二分类 AUC (paired)

    Returns:
        dict with auc_m1, auc_m2, diff(M2-M1), se_diff, z, p_value, ci_diff
    """
    n = len(y_true)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n1 = pos_mask.sum()
    n2 = neg_mask.sum()
    if n1 == 0 or n2 == 0:
        return {'auc_m1': 0.5, 'auc_m2': 0.5, 'diff': 0, 'se_diff': np.inf,
                'z': 0, 'p_value': 1.0, 'ci_diff': (0, 0)}

    auc_m1, v10_m1, v01_m1 = _delong_binary_components(y_true, scores_m1)
    auc_m2, v10_m2, v01_m2 = _delong_binary_components(y_true, scores_m2)

    # Covariance components for positive instances
    d10_m1 = v10_m1 - auc_m1
    d10_m2 = v10_m2 - auc_m2
    s10_11 = np.dot(d10_m1, d10_m1) / (n1 - 1) if n1 > 1 else 0
    s10_22 = np.dot(d10_m2, d10_m2) / (n1 - 1) if n1 > 1 else 0
    s10_12 = np.dot(d10_m1, d10_m2) / (n1 - 1) if n1 > 1 else 0

    # Covariance components for negative instances
    d01_m1 = v01_m1 - auc_m1
    d01_m2 = v01_m2 - auc_m2
    s01_11 = np.dot(d01_m1, d01_m1) / (n2 - 1) if n2 > 1 else 0
    s01_22 = np.dot(d01_m2, d01_m2) / (n2 - 1) if n2 > 1 else 0
    s01_12 = np.dot(d01_m1, d01_m2) / (n2 - 1) if n2 > 1 else 0

    # S_k = S10/n1 + S01/n2
    s_11 = s10_11 / n1 + s01_11 / n2  # Var(AUC_M1)
    s_22 = s10_22 / n1 + s01_22 / n2  # Var(AUC_M2)
    s_12 = s10_12 / n1 + s01_12 / n2  # Cov(AUC_M1, AUC_M2)

    var_diff = s_11 + s_22 - 2 * s_12
    se_diff = np.sqrt(max(var_diff, 1e-15))

    diff = auc_m2 - auc_m1  # M2 - M1 (增益方向)
    z = diff / se_diff
    p_value = 2 * stats.norm.sf(abs(z))
    ci_diff = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)

    return {
        'auc_m1': auc_m1, 'auc_m2': auc_m2,
        'diff': diff, 'se_diff': se_diff,
        'z': z, 'p_value': p_value,
        'ci_diff': ci_diff,
        's_11': s_11, 's_22': s_22, 's_12': s_12,
    }


def delong_test_multiclass(true_labels, pred_scores_m1, pred_scores_m2):
    """DeLong 检验比较多分类模型的 OVR macro-AUC

    对每个类别执行 One-vs-Rest 二分类 DeLong 检验，
    再通过协方差结构计算 macro-AUC 差异的统计量。

    Returns:
        dict with per-class results, macro-level test, etc.
    """
    n = len(true_labels)
    class_results = {}
    binary_results = {}

    for c in range(NUM_CLASSES):
        binary_true = (true_labels == c).astype(int)
        res = _delong_binary_test(binary_true, pred_scores_m1[:, c], pred_scores_m2[:, c])
        class_results[CLASS_NAMES[c]] = {
            'auc_m1': res['auc_m1'], 'auc_m2': res['auc_m2'],
            'diff': res['diff'], 'p_value': res['p_value'],
            'ci_diff': res['ci_diff'], 'z': res['z'],
            'se_diff': res['se_diff'],
        }
        binary_results[c] = res

    # ---- Macro-level test ----
    # Macro-AUC = (1/K) * sum(AUC_k)
    # Var(macro_diff) = (1/K²) * [sum Var(diff_c) + 2*sum_{c<d} Cov(diff_c, diff_d)]
    K = NUM_CLASSES
    macro_auc_m1 = np.mean([binary_results[c]['auc_m1'] for c in range(K)])
    macro_auc_m2 = np.mean([binary_results[c]['auc_m2'] for c in range(K)])
    macro_diff = macro_auc_m2 - macro_auc_m1  # M2 - M1 (增益方向)

    # Var(diff_c) for each class
    var_diffs = np.array([binary_results[c]['s_11'] + binary_results[c]['s_22']
                          - 2 * binary_results[c]['s_12'] for c in range(K)])

    # Cov(diff_c, diff_d) between classes (approximation: assume independence)
    # 这是保守估计，忽略类别间相关性
    var_macro = np.sum(var_diffs) / (K ** 2)
    se_macro = np.sqrt(max(var_macro, 1e-15))
    z_macro = macro_diff / se_macro
    p_macro = 2 * stats.norm.sf(abs(z_macro))
    ci_macro = (macro_diff - 1.96 * se_macro, macro_diff + 1.96 * se_macro)

    return {
        'per_class': class_results,
        'macro_auc_m1': macro_auc_m1,
        'macro_auc_m2': macro_auc_m2,
        'macro_diff': macro_diff,
        'macro_diff_ci': ci_macro,
        'macro_z': z_macro,
        'macro_p_value': p_macro,
        'macro_se': se_macro,
    }


def bootstrap_auc_ci(true_labels, pred_scores, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap 计算每个类别的 AUC 及 95% CI (仅用于 CI 估计)"""
    rng = np.random.RandomState(RANDOM_SEED)
    class_aucs = {i: [] for i in range(NUM_CLASSES)}
    n = len(true_labels)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        tl = true_labels[idx]
        ps = pred_scores[idx]
        for c in range(NUM_CLASSES):
            binary_true = (tl == c).astype(int)
            if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
                continue
            try:
                auc_val = roc_auc_score(binary_true, ps[:, c])
                class_aucs[c].append(auc_val)
            except ValueError:
                continue

    results = {}
    for c in range(NUM_CLASSES):
        vals = np.array(class_aucs[c])
        if len(vals) > 0:
            results[CLASS_NAMES[c]] = {
                'mean': np.mean(vals),
                'ci': (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
            }
    return results


def mcnemar_test(true_labels, pred_labels_m1, pred_labels_m2):
    """McNemar 检验比较两个模型的分类准确率差异"""
    correct_m1 = (pred_labels_m1 == true_labels)
    correct_m2 = (pred_labels_m2 == true_labels)

    # b: m1正确但m2错误, c: m1错误但m2正确
    b = np.sum(correct_m1 & ~correct_m2)
    c = np.sum(~correct_m1 & correct_m2)

    # McNemar 统计量 (带连续性校正)
    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        'b (M1对M2错)': int(b),
        'c (M1错M2对)': int(c),
        'chi2': chi2,
        'p_value': p_value,
    }


def bootstrap_metric_ci(true_labels, pred_labels, metric_fn, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap 计算指标的 95% CI"""
    n = len(true_labels)
    rng = np.random.RandomState(RANDOM_SEED)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        val = metric_fn(true_labels[idx], pred_labels[idx])
        values.append(val)
    values = np.array(values)
    return np.mean(values), (np.percentile(values, 2.5), np.percentile(values, 97.5))


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def compute_sensitivity_per_class(y_true, y_pred):
    """计算每个类别的 sensitivity (recall)"""
    results = {}
    for c in range(NUM_CLASSES):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[CLASS_NAMES[c]] = sens
    return results


def compute_specificity_per_class(y_true, y_pred):
    """计算每个类别的 specificity"""
    results = {}
    for c in range(NUM_CLASSES):
        tn = np.sum((y_pred != c) & (y_true != c))
        fp = np.sum((y_pred == c) & (y_true != c))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results[CLASS_NAMES[c]] = spec
    return results


def compute_f1_macro(y_true, y_pred):
    """Macro-averaged F1"""
    f1s = []
    for c in range(NUM_CLASSES):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return np.mean(f1s)


def bootstrap_sensitivity_ci(true_labels, pred_labels, class_idx, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap 计算某个类别 sensitivity 的 95% CI"""
    n = len(true_labels)
    rng = np.random.RandomState(RANDOM_SEED)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = true_labels[idx]
        yp = pred_labels[idx]
        tp = np.sum((yp == class_idx) & (yt == class_idx))
        fn = np.sum((yp != class_idx) & (yt == class_idx))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        values.append(sens)
    values = np.array(values)
    return np.mean(values), (np.percentile(values, 2.5), np.percentile(values, 97.5))


def bootstrap_specificity_ci(true_labels, pred_labels, class_idx, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap 计算某个类别 specificity 的 95% CI"""
    n = len(true_labels)
    rng = np.random.RandomState(RANDOM_SEED)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = true_labels[idx]
        yp = pred_labels[idx]
        tn = np.sum((yp != class_idx) & (yt != class_idx))
        fp = np.sum((yp == class_idx) & (yt != class_idx))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        values.append(spec)
    values = np.array(values)
    return np.mean(values), (np.percentile(values, 2.5), np.percentile(values, 97.5))


# ======================== 校准分析 ========================
def compute_brier_score(true_labels, pred_scores):
    """多分类 Brier score: Brier = 1/N * sum_i sum_c (p_ic - o_ic)^2"""
    n = len(true_labels)
    one_hot = np.zeros_like(pred_scores)
    for i, label in enumerate(true_labels):
        one_hot[i, int(label)] = 1.0
    brier = np.mean(np.sum((pred_scores - one_hot) ** 2, axis=1))
    return brier


def bootstrap_brier_ci(true_labels, pred_scores, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap 计算 Brier score 的 95% CI"""
    n = len(true_labels)
    rng = np.random.RandomState(RANDOM_SEED)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        val = compute_brier_score(true_labels[idx], pred_scores[idx])
        values.append(val)
    values = np.array(values)
    return np.mean(values), (np.percentile(values, 2.5), np.percentile(values, 97.5))


def _calibration_data_for_class(pred_scores, binary_true, n_bins=20,
                                n_bootstrap_ci=200):
    """计算单个类别的分箱校准数据 + PCHIP 平滑曲线 + bootstrap 置信带

    在分箱中心上使用 PCHIP (单调三次样条) 插值，
    保证曲线光滑且单调，bootstrap CI 同样基于分箱中心。

    Returns:
        dict with keys: bin_centers, bin_freqs, x_fit, y_fit, y_fit_lo, y_fit_hi
    """
    from scipy.interpolate import PchipInterpolator

    n = len(pred_scores)
    if n < 5:
        empty = np.array([])
        return {'bin_centers': empty, 'bin_freqs': empty,
                'x_fit': None, 'y_fit': None, 'y_fit_lo': None, 'y_fit_hi': None}

    # ---- 1. Quantile-based binning ----
    sorted_idx = np.argsort(pred_scores)
    sorted_probs = pred_scores[sorted_idx]
    sorted_labels = binary_true[sorted_idx]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    bin_centers, bin_freqs = [], []
    for b in range(n_bins):
        s, e = edges[b], edges[b + 1]
        if e <= s:
            continue
        bp = sorted_probs[s:e]
        bl = sorted_labels[s:e]
        if len(bp) == 0:
            continue
        bin_centers.append(bp.mean())
        bin_freqs.append(bl.mean())
    bin_centers = np.array(bin_centers)
    bin_freqs = np.array(bin_freqs)

    if len(bin_centers) < 3:
        return {'bin_centers': bin_centers, 'bin_freqs': bin_freqs,
                'x_fit': None, 'y_fit': None, 'y_fit_lo': None, 'y_fit_hi': None}

    # Deduplicate: average freqs for identical centers (PCHIP needs strict increase)
    bc_unique, inv = np.unique(bin_centers, return_inverse=True)
    bf_unique = np.array([bin_freqs[inv == i].mean() for i in range(len(bc_unique))])
    if len(bc_unique) < 3:
        return {'bin_centers': bin_centers, 'bin_freqs': bin_freqs,
                'x_fit': None, 'y_fit': None, 'y_fit_lo': None, 'y_fit_hi': None}

    # ---- 2. PCHIP smooth monotone interpolation on bin centers ----
    x_fit = np.linspace(max(bc_unique.min() - 0.03, 0),
                        min(bc_unique.max() + 0.03, 1), 300)
    pchip = PchipInterpolator(bc_unique, bf_unique, extrapolate=True)
    y_fit = np.clip(pchip(x_fit), 0, 1)

    # ---- 3. Bootstrap CI band (PCHIP on bin centers) ----
    rng = np.random.RandomState(RANDOM_SEED)
    y_boot = np.zeros((n_bootstrap_ci, len(x_fit)))
    for i in range(n_bootstrap_ci):
        idx = rng.randint(0, n, size=n)
        bt = binary_true[idx]
        bp = pred_scores[idx]
        if len(np.unique(bt)) < 2:
            y_boot[i] = y_fit
            continue
        # recompute bin centers for bootstrap sample
        si = np.argsort(bp)
        sp, sl = bp[si], bt[si]
        be = np.linspace(0, n, n_bins + 1).astype(int)
        bc_b, bf_b = [], []
        for b in range(n_bins):
            s, e = be[b], be[b + 1]
            if e <= s:
                continue
            bp_b = sp[s:e]
            bl_b = sl[s:e]
            if len(bp_b) == 0:
                continue
            bc_b.append(bp_b.mean())
            bf_b.append(bl_b.mean())
        bc_b, bf_b = np.array(bc_b), np.array(bf_b)
        if len(bc_b) < 3:
            y_boot[i] = y_fit
            continue
        bc_bu, inv_b = np.unique(bc_b, return_inverse=True)
        bf_bu = np.array([bf_b[inv_b == j].mean() for j in range(len(bc_bu))])
        if len(bc_bu) < 3:
            y_boot[i] = y_fit
            continue
        pchip_b = PchipInterpolator(bc_bu, bf_bu, extrapolate=True)
        y_boot[i] = np.clip(pchip_b(x_fit), 0, 1)
    y_fit_lo = np.percentile(y_boot, 2.5, axis=0)
    y_fit_hi = np.percentile(y_boot, 97.5, axis=0)

    return {
        'bin_centers': bin_centers,
        'bin_freqs': bin_freqs,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'y_fit_lo': y_fit_lo,
        'y_fit_hi': y_fit_hi,
    }


def plot_calibration_curves(true_labels, pred_scores_m1, pred_scores_m2, dataset_name):
    """绘制两个模型的校准曲线 (每个类别) — 一行三列, 含置信带和 Brier score"""
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(6.5 * NUM_CLASSES, 6.0))
    if NUM_CLASSES == 1:
        axes = [axes]

    model_configs = [
        ('Model 1', pred_scores_m1, '#1565C0', 'o', 0.12),
        ('Model 2', pred_scores_m2, '#C62828', 's', 0.10),
    ]
    class_names_en = ['Benign', 'Non-HCC Malignancies', 'HCC']

    # 计算 Brier scores
    brier_vals = {}
    for mname, pscores, *_ in model_configs:
        one_hot = np.zeros_like(pscores)
        for i, lab in enumerate(true_labels):
            one_hot[i, int(lab)] = 1.0
        brier_vals[mname] = np.mean(np.sum((pscores - one_hot) ** 2, axis=1))

    for c in range(NUM_CLASSES):
        ax = axes[c]
        binary_true = (true_labels == c).astype(int)
        n_pos = int(binary_true.sum())
        n_total = len(binary_true)

        for model_name, pred_scores, color, marker, band_alpha in model_configs:
            probs = pred_scores[:, c]
            cal = _calibration_data_for_class(probs, binary_true, n_bins=20)

            if cal['x_fit'] is None:
                continue

            # 置信带
            ax.fill_between(cal['x_fit'], cal['y_fit_lo'], cal['y_fit_hi'],
                            color=color, alpha=band_alpha, linewidth=0)
            # 拟合曲线
            ax.plot(cal['x_fit'], cal['y_fit'], '-', color=color, linewidth=2.5,
                    label=f'{model_name}', alpha=0.95, zorder=4)
            # 分箱数据点
            ax.scatter(cal['bin_centers'], cal['bin_freqs'], color=color,
                       marker=marker, s=55, edgecolors='white', linewidths=1.2,
                       zorder=6, label=f'{model_name} (bins)')

        # 完美校准对角线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.35, label='Perfect calibration', zorder=1)

        # 标注
        ax.set_xlabel('Mean predicted probability', fontsize=11)
        if c == 0:
            ax.set_ylabel('Fraction of positives', fontsize=11)
        ax.set_title(f'{class_names_en[c]}  (n={n_total}, pos={n_pos})',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.92,
                  edgecolor='#cccccc', fancybox=True, fontsize=9)
        ax.set_xlim([-0.03, 1.03])
        ax.set_ylim([-0.03, 1.03])
        ax.grid(True, alpha=0.12, linestyle='--', linewidth=0.8)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)

    # Brier score 总标注
    brier_text = (f'Overall Brier Score:  '
                  f'Model 1 = {brier_vals["Model 1"]:.4f},  '
                  f'Model 2 = {brier_vals["Model 2"]:.4f}  '
                  f'(lower is better)')
    fig.text(0.5, -0.02, brier_text, ha='center', fontsize=10.5,
             fontstyle='italic', color='#444444')

    plt.suptitle(f'Calibration Curves \u2014 {dataset_name}',
                 y=1.03, fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'calibration_curves_{dataset_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  校准曲线已保存: {save_path}")


# ======================== 对比热图 ========================
def plot_comparison_heatmap(results, dataset_name):
    """生成模型1 vs 模型2 多维度对比热图

    行: 指标 (Accuracy, Macro-F1, AUC, Brier Score, 各类Sensitivity/Specificity)
    列: 模型1, 模型2, 差异(M2-M1)
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = [_cjk_name] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 收集所有指标数据
    rows = []
    row_labels = []

    # 整体指标
    acc = results['accuracy']
    rows.append([acc['m1']['mean'], acc['m2']['mean'], acc['m2']['mean'] - acc['m1']['mean']])
    row_labels.append('Accuracy')

    f1 = results['f1_macro']
    rows.append([f1['m1']['mean'], f1['m2']['mean'], f1['m2']['mean'] - f1['m1']['mean']])
    row_labels.append('Macro-F1')

    # AUC (DeLong)
    auc_res = results['auc_comparison']
    rows.append([auc_res['macro_auc_m1'], auc_res['macro_auc_m2'],
                 auc_res['macro_auc_m2'] - auc_res['macro_auc_m1']])
    row_labels.append('Macro-AUC')

    for cn in CLASS_NAMES:
        pc = auc_res['per_class'].get(cn, {})
        if pc:
            rows.append([pc['auc_m1'], pc['auc_m2'], pc['auc_m2'] - pc['auc_m1']])
            row_labels.append(f'AUC-{cn}')

    # Brier Score (越低越好，差异为负表示模型2更好)
    brier = results['brier_score']
    rows.append([brier['m1']['mean'], brier['m2']['mean'], brier['m2']['mean'] - brier['m1']['mean']])
    row_labels.append('Brier Score ↓')

    # Sensitivity
    for cn in CLASS_NAMES:
        s = results['sensitivity'].get(cn, {})
        if s:
            rows.append([s['m1']['mean'], s['m2']['mean'], s['m2']['mean'] - s['m1']['mean']])
            row_labels.append(f'Sens-{cn}')

    # Specificity
    for cn in CLASS_NAMES:
        sp = results['specificity'].get(cn, {})
        if sp:
            rows.append([sp['m1']['mean'], sp['m2']['mean'], sp['m2']['mean'] - sp['m1']['mean']])
            row_labels.append(f'Spec-{cn}')

    data = np.array(rows)
    col_labels = ['模型1', '模型2', 'Δ (M2−M1)']

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(row_labels) * 0.45)),
                             gridspec_kw={'width_ratios': [2, 1]})

    # 左侧: M1 和 M2 的指标值
    ax1 = axes[0]
    im1 = sns.heatmap(data[:, :2], annot=True, fmt='.3f', cmap='YlOrRd',
                      xticklabels=col_labels[:2], yticklabels=row_labels,
                      ax=ax1, vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                      cbar_kws={'label': 'Score', 'shrink': 0.8})
    ax1.set_title('模型性能对比', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', rotation=0, labelsize=9)
    ax1.tick_params(axis='x', rotation=0, labelsize=10)

    # 右侧: 差异 (M2 - M1)
    ax2 = axes[1]
    diff_data = data[:, 2:3]
    vmax_abs = max(abs(diff_data.min()), abs(diff_data.max()), 0.1)
    im2 = sns.heatmap(diff_data, annot=True, fmt='+.3f', cmap='RdBu_r',
                      xticklabels=[col_labels[2]], yticklabels=row_labels,
                      ax=ax2, center=0, vmin=-vmax_abs, vmax=vmax_abs,
                      linewidths=0.5, linecolor='white',
                      cbar_kws={'label': 'Δ', 'shrink': 0.8})
    ax2.set_title('差异 (M2−M1)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', rotation=0, labelsize=9)
    ax2.tick_params(axis='x', rotation=0, labelsize=10)

    # 在差异列上添加显著性标记
    auc_res = results['auc_comparison']
    p_values = [results['auc_comparison']['macro_p_value']]  # macro AUC p-value
    # 获取每个指标的 p-value
    all_p = []
    # Accuracy: McNemar p-value
    all_p.append(results['mcnemar']['p_value'])
    # Macro-F1: no direct p-value, use McNemar as proxy
    all_p.append(results['mcnemar']['p_value'])
    # Macro-AUC
    all_p.append(auc_res['macro_p_value'])
    # Per-class AUC
    for cn in CLASS_NAMES:
        pc = auc_res['per_class'].get(cn, {})
        all_p.append(pc.get('p_value', 1.0))
    # Brier Score: no p-value
    all_p.append(None)
    # Sensitivity / Specificity: no direct p-values
    for cn in CLASS_NAMES:
        all_p.append(None)
    for cn in CLASS_NAMES:
        all_p.append(None)

    for i, p in enumerate(all_p):
        if p is not None:
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = ''
            if sig:
                ax2.text(0.5, i + 0.9, sig, ha='center', va='top', fontsize=11,
                         fontweight='bold', color='darkred')

    plt.suptitle(f'模型1 vs 模型2 对比热图 — {dataset_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'comparison_heatmap_{dataset_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  对比热图已保存: {save_path}")


# ======================== 主流程 ========================
def run_comparison(dataset_name, anno_file, model1_dir, model2_dir):
    """对指定数据集执行完整对比分析"""
    print(f"\n{'='*70}")
    print(f"  数据集: {dataset_name}")
    print(f"{'='*70}")

    # 加载数据
    print("\n[1] 加载数据...")
    annotations = load_annotations(anno_file)
    scores_m1 = load_scores(os.path.join(model1_dir, 'score.json'))
    scores_m2 = load_scores(os.path.join(model2_dir, 'score.json'))

    true_m1, pred_scores_m1, pred_labels_m1, ids_m1 = align_data(annotations, scores_m1)
    true_m2, pred_scores_m2, pred_labels_m2, ids_m2 = align_data(annotations, scores_m2)

    # 确保两个模型使用相同的样本
    common_ids = sorted(set(ids_m1) & set(ids_m2))
    idx_m1 = [ids_m1.index(cid) for cid in common_ids]
    idx_m2 = [ids_m2.index(cid) for cid in common_ids]

    true_labels = true_m1[idx_m1]
    pred_scores_m1 = pred_scores_m1[idx_m1]
    pred_labels_m1 = pred_labels_m1[idx_m1]
    pred_scores_m2 = pred_scores_m2[idx_m2]
    pred_labels_m2 = pred_labels_m2[idx_m2]

    n_samples = len(true_labels)
    print(f"  最终对齐样本数: {n_samples}")

    results = {}

    # ================== 1. AUC 差异比较 (DeLong test) ==================
    print("\n[2] AUC 差异比较 (DeLong test)...")
    delong_res = delong_test_multiclass(true_labels, pred_scores_m1, pred_scores_m2)
    results['auc_comparison'] = delong_res

    # Bootstrap CI for per-class AUC
    class_auc_m1 = bootstrap_auc_ci(true_labels, pred_scores_m1)
    class_auc_m2 = bootstrap_auc_ci(true_labels, pred_scores_m2)
    results['class_auc_m1'] = class_auc_m1
    results['class_auc_m2'] = class_auc_m2

    print(f"  模型1 Macro-AUC: {delong_res['macro_auc_m1']:.4f}")
    print(f"  模型2 Macro-AUC: {delong_res['macro_auc_m2']:.4f}")
    print(f"  Macro-AUC 差异 (M2-M1): {delong_res['macro_diff']:.4f} "
          f"(95% CI: {delong_res['macro_diff_ci'][0]:.4f} ~ {delong_res['macro_diff_ci'][1]:.4f})")
    print(f"  DeLong Z = {delong_res['macro_z']:.4f}, p = {delong_res['macro_p_value']:.4f}")
    for c in range(NUM_CLASSES):
        cn = CLASS_NAMES[c]
        pc = delong_res['per_class'][cn]
        a1_ci = class_auc_m1.get(cn, {})
        a2_ci = class_auc_m2.get(cn, {})
        sig = '***' if pc['p_value'] < 0.001 else '**' if pc['p_value'] < 0.01 else '*' if pc['p_value'] < 0.05 else 'ns'
        m1_ci_str = f"({a1_ci['ci'][0]:.4f}-{a1_ci['ci'][1]:.4f})" if 'ci' in a1_ci else ''
        m2_ci_str = f"({a2_ci['ci'][0]:.4f}-{a2_ci['ci'][1]:.4f})" if 'ci' in a2_ci else ''
        print(f"  {cn}: M1={pc['auc_m1']:.4f} {m1_ci_str}, M2={pc['auc_m2']:.4f} {m2_ci_str}, "
              f"Δ={pc['diff']:.4f}, p={pc['p_value']:.4f} ({sig})")

    # ================== 2. 分类性能比较 ==================
    print("\n[3] 分类性能比较...")

    # McNemar test
    mcnemar = mcnemar_test(true_labels, pred_labels_m1, pred_labels_m2)
    results['mcnemar'] = mcnemar
    print(f"  McNemar test: χ²={mcnemar['chi2']:.4f}, p={mcnemar['p_value']:.4f}")

    # Accuracy + 95% CI
    acc_m1, acc_m1_ci = bootstrap_metric_ci(true_labels, pred_labels_m1, compute_accuracy)
    acc_m2, acc_m2_ci = bootstrap_metric_ci(true_labels, pred_labels_m2, compute_accuracy)
    results['accuracy'] = {
        'm1': {'mean': acc_m1, 'ci': acc_m1_ci},
        'm2': {'mean': acc_m2, 'ci': acc_m2_ci},
    }
    print(f"  模型1 Accuracy: {acc_m1:.4f} (95% CI: {acc_m1_ci[0]:.4f}-{acc_m1_ci[1]:.4f})")
    print(f"  模型2 Accuracy: {acc_m2:.4f} (95% CI: {acc_m2_ci[0]:.4f}-{acc_m2_ci[1]:.4f})")

    # F1 macro + 95% CI
    f1_m1, f1_m1_ci = bootstrap_metric_ci(true_labels, pred_labels_m1, compute_f1_macro)
    f1_m2, f1_m2_ci = bootstrap_metric_ci(true_labels, pred_labels_m2, compute_f1_macro)
    results['f1_macro'] = {
        'm1': {'mean': f1_m1, 'ci': f1_m1_ci},
        'm2': {'mean': f1_m2, 'ci': f1_m2_ci},
    }
    print(f"  模型1 Macro-F1: {f1_m1:.4f} (95% CI: {f1_m1_ci[0]:.4f}-{f1_m1_ci[1]:.4f})")
    print(f"  模型2 Macro-F1: {f1_m2:.4f} (95% CI: {f1_m2_ci[0]:.4f}-{f1_m2_ci[1]:.4f})")

    # Per-class sensitivity & specificity + 95% CI
    print("\n  各分类 Sensitivity (Recall):")
    sens_results = {}
    for c in range(NUM_CLASSES):
        s_m1, s_m1_ci = bootstrap_sensitivity_ci(true_labels, pred_labels_m1, c)
        s_m2, s_m2_ci = bootstrap_sensitivity_ci(true_labels, pred_labels_m2, c)
        sens_results[CLASS_NAMES[c]] = {
            'm1': {'mean': s_m1, 'ci': s_m1_ci},
            'm2': {'mean': s_m2, 'ci': s_m2_ci},
        }
        print(f"    {CLASS_NAMES[c]}: M1={s_m1:.4f} ({s_m1_ci[0]:.4f}-{s_m1_ci[1]:.4f}), M2={s_m2:.4f} ({s_m2_ci[0]:.4f}-{s_m2_ci[1]:.4f})")
    results['sensitivity'] = sens_results

    print("\n  各分类 Specificity:")
    spec_results = {}
    for c in range(NUM_CLASSES):
        sp_m1, sp_m1_ci = bootstrap_specificity_ci(true_labels, pred_labels_m1, c)
        sp_m2, sp_m2_ci = bootstrap_specificity_ci(true_labels, pred_labels_m2, c)
        spec_results[CLASS_NAMES[c]] = {
            'm1': {'mean': sp_m1, 'ci': sp_m1_ci},
            'm2': {'mean': sp_m2, 'ci': sp_m2_ci},
        }
        print(f"    {CLASS_NAMES[c]}: M1={sp_m1:.4f} ({sp_m1_ci[0]:.4f}-{sp_m1_ci[1]:.4f}), M2={sp_m2:.4f} ({sp_m2_ci[0]:.4f}-{sp_m2_ci[1]:.4f})")
    results['specificity'] = spec_results

    # ================== 3. 校准性能比较 ==================
    print("\n[4] 校准性能比较...")

    brier_m1, brier_m1_ci = bootstrap_brier_ci(true_labels, pred_scores_m1)
    brier_m2, brier_m2_ci = bootstrap_brier_ci(true_labels, pred_scores_m2)
    results['brier_score'] = {
        'm1': {'mean': brier_m1, 'ci': brier_m1_ci},
        'm2': {'mean': brier_m2, 'ci': brier_m2_ci},
    }
    print(f"  模型1 Brier Score: {brier_m1:.4f} (95% CI: {brier_m1_ci[0]:.4f}-{brier_m1_ci[1]:.4f})")
    print(f"  模型2 Brier Score: {brier_m2:.4f} (95% CI: {brier_m2_ci[0]:.4f}-{brier_m2_ci[1]:.4f})")

    # 绘制校准曲线
    plot_calibration_curves(true_labels, pred_scores_m1, pred_scores_m2, dataset_name)

    # ================== 4. 对比热图 ==================
    print("\n[5] 生成对比热图...")
    plot_comparison_heatmap(results, dataset_name)

    return results


# ======================== 从 evaluation_metrics.json 提取验证集结果 ========================
def load_eval_metrics(path):
    """加载 evaluation_metrics.json"""
    with open(path, 'r') as f:
        return json.load(f)


def format_val_comparison(m1_metrics, m2_metrics):
    """从 evaluation_metrics.json 直接对比验证集指标（非配对检验）"""
    lines = []
    lines.append("### 内部验证集 (Val)\n")
    lines.append("> 注：验证集指标直接从 evaluation_metrics.json 提取，未进行配对统计检验。\n")

    # AUC
    lines.append("#### AUC 差异比较\n")
    m1_auc = m1_metrics.get('auc', None)
    m2_auc = m2_metrics.get('auc', None)
    lines.append(f"| 指标 | 模型1 | 模型2 |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Macro-AUC | {m1_auc:.4f} | {m2_auc:.4f} |")
    lines.append("")

    # Per-class sensitivity from metrics
    lines.append("**各分类 Sensitivity:**\n")
    lines.append(f"| 类别 | 模型1 | 模型2 |")
    lines.append(f"|---|---|---|")
    for i, cn in enumerate(CLASS_NAMES):
        s1 = m1_metrics.get(f'class_{i}_sensitivity', None)
        s2 = m2_metrics.get(f'class_{i}_sensitivity', None)
        lines.append(f"| {cn} | {s1:.4f} | {s2:.4f} |")
    lines.append("")

    # Overall metrics
    lines.append("#### 分类性能比较\n")
    lines.append(f"| 指标 | 模型1 | 模型2 |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Accuracy | {m1_metrics['accuracy']:.4f} | {m2_metrics['accuracy']:.4f} |")
    lines.append(f"| F1 Score | {m1_metrics['f1_score']:.4f} | {m2_metrics['f1_score']:.4f} |")
    lines.append(f"| Kappa | {m1_metrics['kappa']:.4f} | {m2_metrics['kappa']:.4f} |")

    # Per-class from per_class_metrics
    pcm1 = m1_metrics.get('per_class_metrics', {})
    pcm2 = m2_metrics.get('per_class_metrics', {})
    lines.append("")
    lines.append("**各分类详细指标:**\n")
    lines.append(f"| 类别 | 指标 | 模型1 | 模型2 |")
    lines.append(f"|---|---|---|---|")
    for cn in CLASS_NAMES:
        m1c = pcm1.get(cn, pcm1.get(f'Class {CLASS_NAMES.index(cn)}', {}))
        m2c = pcm2.get(cn, pcm2.get(f'Class {CLASS_NAMES.index(cn)}', {}))
        if m1c and m2c:
            lines.append(f"| {cn} | ACC | {m1c['ACC']:.4f} | {m2c['ACC']:.4f} |")
            lines.append(f"| | F1 | {m1c['F1']:.4f} | {m2c['F1']:.4f} |")
            lines.append(f"| | Recall | {m1c['Recall']:.4f} | {m2c['Recall']:.4f} |")
            lines.append(f"| | Precision | {m1c['Precision']:.4f} | {m2c['Precision']:.4f} |")
    lines.append("")

    return '\n'.join(lines)


def format_results_md(val_results, test_results):
    """将结果格式化为 Markdown"""
    lines = []
    lines.append("## 模型1 vs 模型2 对比分析结果\n")

    # 合并表格: Val + Test
    lines.append("### AUC 对比 (DeLong test)\n")
    lines.append("| 数据集 | 指标 | 模型1 AUC (95% CI) | 模型2 AUC (95% CI) | 差异 (M2-M1) | 95% CI | Z | p-value |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for ds_name, results in [('Val', val_results), ('Test', test_results)]:
        auc = results['auc_comparison']
        # Macro-AUC row
        diff_ci = auc['macro_diff_ci']
        lines.append(f"| {ds_name} | Macro-AUC | {auc['macro_auc_m1']:.4f} | {auc['macro_auc_m2']:.4f} | "
                     f"{auc['macro_diff']:.4f} | ({diff_ci[0]:.4f}, {diff_ci[1]:.4f}) | "
                     f"{auc['macro_z']:.3f} | {auc['macro_p_value']:.4f} |")
        # Per-class AUC rows
        for cn in CLASS_NAMES:
            pc = auc['per_class'].get(cn, {})
            a1_ci = results['class_auc_m1'].get(cn, {})
            a2_ci = results['class_auc_m2'].get(cn, {})
            m1_ci_str = f"({a1_ci['ci'][0]:.4f}-{a1_ci['ci'][1]:.4f})" if 'ci' in a1_ci else ''
            m2_ci_str = f"({a2_ci['ci'][0]:.4f}-{a2_ci['ci'][1]:.4f})" if 'ci' in a2_ci else ''
            se = pc.get('se_diff', 0)
            diff_val = pc['diff']
            diff_ci_str = f"({diff_val - 1.96 * se:.4f}, {diff_val + 1.96 * se:.4f})"
            sig = '***' if pc.get('p_value', 1) < 0.001 else '**' if pc.get('p_value', 1) < 0.01 else '*' if pc.get('p_value', 1) < 0.05 else ''
            lines.append(f"| | {cn} | {pc['auc_m1']:.4f} {m1_ci_str} | {pc['auc_m2']:.4f} {m2_ci_str} | "
                         f"{diff_val:.4f} | {diff_ci_str} | {pc['z']:.3f} | {pc['p_value']:.4f}{sig} |")
    lines.append("")
    lines.append("> 显著性标记: * p<0.05, ** p<0.01, *** p<0.001; 95% CI 由 Bootstrap (1000次) 估计\n")

    # 2. 分类性能对比 (Test)
    lines.append("### 外部测试集分类性能比较\n")
    results = test_results
    mc = results['mcnemar']
    lines.append(f"**McNemar test:** χ²={mc['chi2']:.4f}, p={mc['p_value']:.4f}\n")

    lines.append(f"| 指标 | 模型1 (95% CI) | 模型2 (95% CI) |")
    lines.append(f"|---|---|---|")

    acc = results['accuracy']
    lines.append(f"| Accuracy | {acc['m1']['mean']:.4f} ({acc['m1']['ci'][0]:.4f}-{acc['m1']['ci'][1]:.4f}) | "
                 f"{acc['m2']['mean']:.4f} ({acc['m2']['ci'][0]:.4f}-{acc['m2']['ci'][1]:.4f}) |")

    f1 = results['f1_macro']
    lines.append(f"| Macro-F1 | {f1['m1']['mean']:.4f} ({f1['m1']['ci'][0]:.4f}-{f1['m1']['ci'][1]:.4f}) | "
                 f"{f1['m2']['mean']:.4f} ({f1['m2']['ci'][0]:.4f}-{f1['m2']['ci'][1]:.4f}) |")
    lines.append("")

    # Sensitivity
    lines.append("**各分类 Sensitivity (Recall):**\n")
    lines.append(f"| 类别 | 模型1 (95% CI) | 模型2 (95% CI) |")
    lines.append(f"|---|---|---|")
    for cn in CLASS_NAMES:
        s = results['sensitivity'][cn]
        lines.append(f"| {cn} | {s['m1']['mean']:.4f} ({s['m1']['ci'][0]:.4f}-{s['m1']['ci'][1]:.4f}) | "
                     f"{s['m2']['mean']:.4f} ({s['m2']['ci'][0]:.4f}-{s['m2']['ci'][1]:.4f}) |")
    lines.append("")

    # Specificity
    lines.append("**各分类 Specificity:**\n")
    lines.append(f"| 类别 | 模型1 (95% CI) | 模型2 (95% CI) |")
    lines.append(f"|---|---|---|")
    for cn in CLASS_NAMES:
        sp = results['specificity'][cn]
        lines.append(f"| {cn} | {sp['m1']['mean']:.4f} ({sp['m1']['ci'][0]:.4f}-{sp['m1']['ci'][1]:.4f}) | "
                     f"{sp['m2']['mean']:.4f} ({sp['m2']['ci'][0]:.4f}-{sp['m2']['ci'][1]:.4f}) |")
    lines.append("")

    # 3. 校准性能
    lines.append("#### 校准性能比较 (Test)\n")
    brier = results['brier_score']
    lines.append(f"| 指标 | 模型1 (95% CI) | 模型2 (95% CI) |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Brier Score | {brier['m1']['mean']:.4f} ({brier['m1']['ci'][0]:.4f}-{brier['m1']['ci'][1]:.4f}) | "
                 f"{brier['m2']['mean']:.4f} ({brier['m2']['ci'][0]:.4f}-{brier['m2']['ci'][1]:.4f}) |")
    lines.append("")
    lines.append(f"校准曲线见: `calibration_curves_Test.png`\n")
    lines.append(f"对比热图见: `comparison_heatmap_Test.png`\n")

    return '\n'.join(lines)


if __name__ == '__main__':
    print("=" * 70)
    print("  模型1 vs 模型2 对比分析")
    print("  模型1: LIFT/ckpts/Model1 (影像+征象)")
    print("  模型2: LIFT/ckpts/Model2 (影像+征象+临床变量)")
    print("=" * 70)

    # 内部验证集 - 完整配对分析（score.json已修复）
    val_results = run_comparison(
        dataset_name='Val',
        anno_file=VAL_ANNO,
        model1_dir=os.path.join(MODEL1_DIR, 'val'),
        model2_dir=os.path.join(MODEL2_DIR, 'val'),
    )

    # 外部测试集 - 完整配对分析
    test_results = run_comparison(
        dataset_name='Test',
        anno_file=TEST_ANNO,
        model1_dir=os.path.join(MODEL1_DIR, 'test'),
        model2_dir=os.path.join(MODEL2_DIR, 'test'),
    )

    # 生成 Markdown 报告
    md_content = format_results_md(val_results, test_results)
    md_path = os.path.join(OUTPUT_DIR, 'model1_vs_model2_comparison.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"\nMarkdown 报告已保存: {md_path}")

    # 同时保存完整 JSON
    import json as json_lib

    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    all_results = {
        'val': make_serializable(val_results),
        'test': make_serializable(test_results),
    }
    json_path = os.path.join(OUTPUT_DIR, 'comparison_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json_lib.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"JSON 结果已保存: {json_path}")

    print("\n" + "=" * 70)
    print("  分析完成!")
    print("=" * 70)
