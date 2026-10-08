#!/usr/bin/env python3
"""
Model2_0 (纯DL基线) vs features_2 (多模态融合) 对比分析
- Model2_0: uniformer_small_IL, 无征象头, 纯图像
- features_2: uniformer_small_IL_features, hierarchical_simple融合, 25征象
"""
import json
import numpy as np
import os
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, recall_score, precision_score
from statsmodels.stats.contingency_tables import mcnemar

# ============ 路径配置 ============
MODEL_A_DIR = 'LIFT/ckpts/Model2_0/uniformer_small_IL'       # 纯DL基线
MODEL_B_DIR = 'LIFT/ckpts/uniformer_small_IL_features_2/uniformer_small_IL_features'  # 多模态融合
OUTPUT_DIR = 'model_comparison_results'

MODEL_A_NAME = 'Model2_0 (纯DL基线)'
MODEL_B_NAME = 'features_2 (多模态融合)'

CLASS_NAMES = ['良性', '恶性非HCC', 'HCC']

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_scores(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    case_ids = [d['image_id'] for d in data]
    preds = [d['prediction'] for d in data]
    scores = np.array([d['score'] for d in data])
    return case_ids, preds, scores


def print_metrics_table(metrics_a, metrics_b, label):
    """打印两个模型的指标对比表"""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"{'指标':<25} {MODEL_A_NAME:>25} {MODEL_B_NAME:>25} {'Delta':>10}")
    print('-' * 88)

    keys = [
        ('accuracy', 'Accuracy'),
        ('auc', 'AUC'),
        ('f1_score', 'F1 (Weighted)'),
        ('recall_weighted', 'Recall (Weighted)'),
        ('precision_weighted', 'Precision (Weighted)'),
        ('kappa', 'Cohen Kappa'),
    ]
    for key, name in keys:
        va = metrics_a.get(key, 0)
        vb = metrics_b.get(key, 0)
        delta = vb - va
        sign = '+' if delta > 0 else ''
        print(f"{name:<25} {va:>24.4f}  {vb:>24.4f}  {sign}{delta:>8.4f}")


def print_per_class_comparison(metrics_a, metrics_b, label):
    """打印分类别指标对比"""
    print(f"\n{'='*70}")
    print(f"  分类别指标对比 - {label}")
    print(f"{'='*70}")

    pc_a = metrics_a.get('per_class_metrics', {})
    pc_b = metrics_b.get('per_class_metrics', {})

    for cls_name in CLASS_NAMES:
        a_cls = pc_a.get(cls_name, pc_a.get(cls_name.replace('良性', 'Benign').replace('恶性非HCC', 'Non-HCC Malignancies'), {}))
        b_cls = pc_b.get(cls_name, pc_b.get(cls_name.replace('良性', 'Benign').replace('恶性非HCC', 'Non-HCC Malignancies'), {}))
        if not a_cls or not b_cls:
            continue
        print(f"\n  [{cls_name}]")
        print(f"  {'指标':<15} {MODEL_A_NAME:>25} {MODEL_B_NAME:>25} {'Delta':>10}")
        print(f"  {'-'*78}")
        for metric in ['ACC', 'F1', 'Recall', 'Precision', 'Kappa']:
            va = a_cls.get(metric, 0)
            vb = b_cls.get(metric, 0)
            delta = vb - va
            sign = '+' if delta > 0 else ''
            print(f"  {metric:<15} {va:>24.4f}  {vb:>24.4f}  {sign}{delta:>8.4f}")


def run_mcnemar_test(scores_a, scores_b, labels, case_ids_a, case_ids_b):
    """在测试集上执行McNemar配对检验"""
    # 对齐样本
    id_to_idx_a = {cid: i for i, cid in enumerate(case_ids_a)}
    id_to_idx_b = {cid: i for i, cid in enumerate(case_ids_b)}
    common_ids = sorted(set(id_to_idx_a.keys()) & set(id_to_idx_b.keys()))

    if len(common_ids) == 0:
        print("WARNING: 两个模型没有共同的测试样本，无法进行配对检验")
        return None

    preds_a = np.array([scores_a[id_to_idx_a[cid]] for cid in common_ids])
    preds_b = np.array([scores_b[id_to_idx_b[cid]] for cid in common_ids])
    true_labels = np.array([labels[id_to_idx_a[cid]] for cid in common_ids])

    print(f"\n{'='*70}")
    print(f"  McNemar 配对检验 (Test Set, N={len(common_ids)})")
    print(f"{'='*70}")

    results = {}
    # 总体 McNemar (正确/错误)
    correct_a = (preds_a == true_labels).astype(int)
    correct_b = (preds_b == true_labels).astype(int)

    # 2x2 contingency table:
    #            B correct   B wrong
    # A correct    n11        n10
    # A wrong      n01        n00
    n11 = np.sum((correct_a == 1) & (correct_b == 1))
    n10 = np.sum((correct_a == 1) & (correct_b == 0))
    n01 = np.sum((correct_a == 0) & (correct_b == 1))
    n00 = np.sum((correct_a == 0) & (correct_b == 0))

    table = np.array([[n11, n10], [n01, n00]])
    print(f"\n  总体 McNemar 列联表:")
    print(f"                    {MODEL_B_NAME}正确  {MODEL_B_NAME}错误")
    print(f"  {MODEL_A_NAME}正确    {n11:>8}       {n10:>8}")
    print(f"  {MODEL_A_NAME}错误    {n01:>8}       {n00:>8}")

    try:
        result = mcnemar(table, exact=False, correction=True)
        print(f"\n  McNemar statistic: {result.statistic:.4f}")
        print(f"  p-value: {result.pvalue:.6f}")
        if result.pvalue < 0.05:
            print(f"  => p < 0.05, 两个模型性能差异具有统计学显著性")
        else:
            print(f"  => p >= 0.05, 两个模型性能差异不具有统计学显著性")
        results['overall'] = {'statistic': result.statistic, 'pvalue': result.pvalue,
                              'table': table.tolist()}
    except Exception as e:
        print(f"  McNemar test failed: {e}")

    # 分类别 McNemar
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        binary_true = (true_labels == cls_idx).astype(int)
        binary_pred_a = (preds_a == cls_idx).astype(int)
        binary_pred_b = (preds_b == cls_idx).astype(int)

        corr_a = (binary_pred_a == binary_true).astype(int)
        corr_b = (binary_pred_b == binary_true).astype(int)

        n11_c = np.sum((corr_a == 1) & (corr_b == 1))
        n10_c = np.sum((corr_a == 1) & (corr_b == 0))
        n01_c = np.sum((corr_a == 0) & (corr_b == 1))
        n00_c = np.sum((corr_a == 0) & (corr_b == 0))

        table_c = np.array([[n11_c, n10_c], [n01_c, n00_c]])
        print(f"\n  [{cls_name}] McNemar 列联表:")
        print(f"                    {MODEL_B_NAME}正确  {MODEL_B_NAME}错误")
        print(f"  {MODEL_A_NAME}正确    {n11_c:>8}       {n10_c:>8}")
        print(f"  {MODEL_A_NAME}错误    {n01_c:>8}       {n00_c:>8}")

        try:
            result_c = mcnemar(table_c, exact=False, correction=True)
            print(f"  McNemar statistic: {result_c.statistic:.4f}")
            print(f"  p-value: {result_c.pvalue:.6f}")
            results[cls_name] = {'statistic': result_c.statistic, 'pvalue': result_c.pvalue,
                                 'table': table_c.tolist()}
        except Exception as e:
            print(f"  McNemar test failed: {e}")

    return results


def compare_confusion_matrices(metrics_a, metrics_b, label):
    """对比混淆矩阵"""
    print(f"\n{'='*70}")
    print(f"  混淆矩阵对比 - {label}")
    print(f"{'='*70}")

    for name, metrics in [(MODEL_A_NAME, metrics_a), (MODEL_B_NAME, metrics_b)]:
        cm = np.array(metrics['confusion_matrix'])
        cm_pct = np.where(cm.sum(axis=1, keepdims=True) > 0,
                          cm / cm.sum(axis=1, keepdims=True) * 100, 0)
        print(f"\n  {name}:")
        print(f"  {'':>15} {'预测:良性':>10} {'预测:恶性非HCC':>15} {'预测:HCC':>10}")
        for i, cls in enumerate(CLASS_NAMES):
            print(f"  {cls:>12}  {cm[i,0]:>5}({cm_pct[i,0]:>5.1f}%)  {cm[i,1]:>8}({cm_pct[i,1]:>5.1f}%)  {cm[i,2]:>8}({cm_pct[i,2]:>5.1f}%)")


def generate_comparison_report(val_metrics_a, val_metrics_b, test_metrics_a, test_metrics_b,
                                mcnemar_results, scores_info_a, scores_info_b):
    """生成JSON格式的对比报告"""
    report = {
        'models': {
            'A': {'name': MODEL_A_NAME, 'dir': MODEL_A_DIR,
                   'type': 'uniformer_small_IL', 'fusion': 'none'},
            'B': {'name': MODEL_B_NAME, 'dir': MODEL_B_DIR,
                   'type': 'uniformer_small_IL_features', 'fusion': 'hierarchical_simple'},
        },
        'validation': {
            'A': {k: val_metrics_a[k] for k in ['accuracy', 'auc', 'f1_score', 'kappa'] if k in val_metrics_a},
            'B': {k: val_metrics_b[k] for k in ['accuracy', 'auc', 'f1_score', 'kappa'] if k in val_metrics_b},
        },
        'test': {
            'A': {k: test_metrics_a[k] for k in ['accuracy', 'auc', 'f1_score', 'kappa'] if k in test_metrics_a},
            'B': {k: test_metrics_b[k] for k in ['accuracy', 'auc', 'f1_score', 'kappa'] if k in test_metrics_b},
        },
    }

    if mcnemar_results:
        report['mcnemar_test'] = {}
        for key, val in mcnemar_results.items():
            report['mcnemar_test'][key] = {
                'statistic': val['statistic'],
                'pvalue': val['pvalue'],
            }

    report_path = os.path.join(OUTPUT_DIR, 'model2_0_vs_features_2_comparison.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n对比报告已保存至: {report_path}")


def main():
    # 1. 加载验证集指标
    val_metrics_a = load_json(os.path.join(MODEL_A_DIR, 'pred_results/val/evaluation_metrics.json'))
    val_metrics_b = load_json(os.path.join(MODEL_B_DIR, 'pred_results/val/evaluation_metrics.json'))

    # 2. 加载测试集指标
    test_metrics_a = load_json(os.path.join(MODEL_A_DIR, 'pred_results/test/evaluation_metrics.json'))
    test_metrics_b = load_json(os.path.join(MODEL_B_DIR, 'pred_results/test/evaluation_metrics.json'))

    # 3. 加载测试集逐样本预测分数
    case_ids_a, preds_a, scores_a = load_scores(os.path.join(MODEL_A_DIR, 'pred_results/test/score.json'))
    case_ids_b, preds_b, scores_b = load_scores(os.path.join(MODEL_B_DIR, 'pred_results/test/score.json'))

    # 获取测试集真实标签 (从test.txt)
    import pandas as pd
    test_anno = 'LIFT/data/labels/test.txt'
    df = pd.read_csv(test_anno, sep='\t', header=0)
    pathology_mapping = {'liangxing': 0, 'noHCC': 1, 'HCC': 2}
    labels_str = df.iloc[:, 1].astype(str).str.strip()
    true_labels = np.array([pathology_mapping.get(l, -1) for l in labels_str])

    print("=" * 70)
    print(f"  {MODEL_A_NAME} vs {MODEL_B_NAME} 对比分析")
    print("=" * 70)
    print(f"  A: {MODEL_A_NAME} (model=uniformer_small_IL, fusion=none)")
    print(f"  B: {MODEL_B_NAME} (model=uniformer_small_IL_features, fusion=hierarchical_simple)")

    # 4. 验证集对比
    print_metrics_table(val_metrics_a, val_metrics_b, '验证集 (Validation Set)')
    print_per_class_comparison(val_metrics_a, val_metrics_b, '验证集')
    compare_confusion_matrices(val_metrics_a, val_metrics_b, '验证集')

    # 5. 测试集对比
    print_metrics_table(test_metrics_a, test_metrics_b, '外部测试集 (External Test Set)')
    print_per_class_comparison(test_metrics_a, test_metrics_b, '外部测试集')
    compare_confusion_matrices(test_metrics_a, test_metrics_b, '外部测试集')

    # 6. McNemar配对检验
    mcnemar_results = run_mcnemar_test(preds_a, preds_b, true_labels, case_ids_a, case_ids_b)

    # 7. 生成报告
    generate_comparison_report(val_metrics_a, val_metrics_b, test_metrics_a, test_metrics_b,
                                mcnemar_results, scores_a, scores_b)

    # 8. 总结
    print(f"\n{'='*70}")
    print(f"  总结")
    print(f"{'='*70}")
    acc_diff_val = val_metrics_b['accuracy'] - val_metrics_a['accuracy']
    acc_diff_test = test_metrics_b['accuracy'] - test_metrics_a['accuracy']
    f1_diff_test = test_metrics_b['f1_score'] - test_metrics_a['f1_score']
    auc_diff_test = test_metrics_b.get('auc', 0) - test_metrics_a.get('auc', 0)

    print(f"  验证集: features_2 Accuracy {'+' if acc_diff_val>0 else ''}{acc_diff_val*100:.2f}% "
          f"({val_metrics_a['accuracy']*100:.2f}% -> {val_metrics_b['accuracy']*100:.2f}%)")
    print(f"  测试集: features_2 Accuracy {'+' if acc_diff_test>0 else ''}{acc_diff_test*100:.2f}% "
          f"({test_metrics_a['accuracy']*100:.2f}% -> {test_metrics_b['accuracy']*100:.2f}%)")
    print(f"  测试集: features_2 F1 {'+' if f1_diff_test>0 else ''}{f1_diff_test:.4f} "
          f"({test_metrics_a['f1_score']:.4f} -> {test_metrics_b['f1_score']:.4f})")
    print(f"  测试集: features_2 AUC {'+' if auc_diff_test>0 else ''}{auc_diff_test:.4f} "
          f"({test_metrics_a.get('auc',0):.4f} -> {test_metrics_b.get('auc',0):.4f})")

    if mcnemar_results and 'overall' in mcnemar_results:
        p = mcnemar_results['overall']['pvalue']
        if p < 0.05:
            winner = MODEL_B_NAME if acc_diff_test > 0 else MODEL_A_NAME
            print(f"\n  McNemar p={p:.6f} < 0.05 => 差异显著, {winner} 更优")
        else:
            print(f"\n  McNemar p={p:.6f} >= 0.05 => 差异不显著, 两个模型性能相当")


if __name__ == '__main__':
    main()
