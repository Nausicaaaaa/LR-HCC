#!/usr/bin/env python3
"""对比 features_3, features_3b, features_3c 三个模型的配置与推理结果差异"""

import json
import os
import yaml

# ============================================================
# 1. 配置对比
# ============================================================
config_paths = {
    "features_3":  "LIFT/ckpts/uniformer_small_IL_features_3/uniformer_small_IL_features/args.yaml",
    "features_3b": "LIFT/ckpts/uniformer_small_IL_features_3b/uniformer_small_IL_features/args.yaml",
    "features_3c": "LIFT/ckpts/uniformer_small_IL_features_3c/uniformer_small_IL_features/args.yaml",
}

configs = {}
for name, path in config_paths.items():
    with open(path, 'r') as f:
        configs[name] = yaml.safe_load(f)

# 找出三个配置中有差异的 key
all_keys = set()
for c in configs.values():
    all_keys.update(c.keys())

diff_keys = []
for key in sorted(all_keys):
    vals = [configs[n].get(key, "<未设置>") for n in config_paths]
    if len(set(str(v) for v in vals)) > 1:
        diff_keys.append(key)

print("=" * 90)
print("一、配置差异（仅列出不同项）")
print("=" * 90)
print(f"{'参数':<25} {'features_3':<20} {'features_3b':<20} {'features_3c':<20}")
print("-" * 90)
for key in diff_keys:
    if key == "output":
        continue
    vals = [str(configs[n].get(key, "<未设置>")) for n in config_paths]
    print(f"{key:<25} {vals[0]:<20} {vals[1]:<20} {vals[2]:<20}")

print("\n共同配置（相同项）: lr=0.0001, opt=adamw, sched=cosine, batch_size=4, seed=42,")
print("  fold=1, num_classes=3, model=uniformer_small_IL_features,")
print("  feature_fusion=hierarchical, num_feature_classes=25, feature_loss_weight=0.05,")
print("  lr_loss_weight=0.1, include_clinical=true, label_mode=original")

# ============================================================
# 2. 推理结果对比
# ============================================================
metric_paths = {}
for model_name, subdir in [
    ("features_3",  "uniformer_small_IL_features_3"),
    ("features_3b", "uniformer_small_IL_features_3b"),
    ("features_3c", "uniformer_small_IL_features_3c"),
]:
    metric_paths[model_name] = {
        "val":  f"LIFT/ckpts/{subdir}/uniformer_small_IL_features/pred_results/val/evaluation_metrics.json",
        "test": f"LIFT/ckpts/{subdir}/uniformer_small_IL_features/pred_results/test/evaluation_metrics.json",
    }

metrics = {}
for name, paths in metric_paths.items():
    metrics[name] = {}
    for split, path in paths.items():
        with open(path, 'r') as f:
            metrics[name][split] = json.load(f)

# ---------- 验证集 ----------
print("\n" + "=" * 90)
print("二、验证集 (Val) 推理结果对比")
print("=" * 90)

# 打印总样本数
for name in metric_paths:
    cm = metrics[name]["val"]["confusion_matrix"]
    total = sum(sum(row) for row in cm)
    supports = [sum(row) for row in cm]
    print(f"  {name}: 总样本={total}  类别分布={supports}")

print(f"\n{'指标':<30} {'features_3':<16} {'features_3b':<16} {'features_3c':<16} {'最佳'}")
print("-" * 90)

overall_keys = [
    ("Accuracy", "accuracy", True),
    ("AUC", "auc", True),
    ("F1 Score", "f1_score", True),
    ("Kappa", "kappa", True),
]

best_model = {k[0]: "" for k in overall_keys}
for label, key, higher_better in overall_keys:
    vals = {n: metrics[n]["val"][key] for n in metric_paths}
    if higher_better:
        best_name = max(vals, key=vals.get)
    else:
        best_name = min(vals, key=vals.get)
    best_val = vals[best_name]
    row = f"{label:<30}"
    for n in metric_paths:
        v = vals[n]
        marker = " ★" if n == best_name and abs(v - best_val) < 1e-9 else ""
        row += f" {v:<14.4f}{marker:<2}"
    row += f" {best_name}"
    print(row)

# 分类别
class_names = ["良性", "恶性非HCC", "HCC"]
class_key_map = {
    "features_3":  ["良性", "恶性非HCC", "HCC"],
    "features_3b": ["良性", "恶性非HCC", "HCC"],
    "features_3c": ["Benign", "Non-HCC Malignancies", "HCC"],
}

print(f"\n{'分类别指标':<30} {'features_3':<16} {'features_3b':<16} {'features_3c':<16}")
print("-" * 90)
for ci, display_name in enumerate(class_names):
    for metric_label, metric_key in [("F1", "F1"), ("Recall", "Recall"), ("Precision", "Precision")]:
        row = f"  {display_name}-{metric_label:<20}"
        for n in metric_paths:
            keys = class_key_map[n]
            pcm = metrics[n]["val"]["per_class_metrics"]
            v = pcm[keys[ci]][metric_key]
            row += f" {v:<16.4f}"
        print(row)
    print()

# ---------- 外部测试集 ----------
print("=" * 90)
print("三、外部测试集 (Test) 推理结果对比")
print("=" * 90)

for name in metric_paths:
    cm = metrics[name]["test"]["confusion_matrix"]
    total = sum(sum(row) for row in cm)
    supports = [sum(row) for row in cm]
    print(f"  {name}: 总样本={total}  类别分布={supports}")

print(f"\n{'指标':<30} {'features_3':<16} {'features_3b':<16} {'features_3c':<16} {'最佳'}")
print("-" * 90)

for label, key, higher_better in overall_keys:
    vals = {n: metrics[n]["test"][key] for n in metric_paths}
    if higher_better:
        best_name = max(vals, key=vals.get)
    else:
        best_name = min(vals, key=vals.get)
    best_val = vals[best_name]
    row = f"{label:<30}"
    for n in metric_paths:
        v = vals[n]
        marker = " ★" if n == best_name and abs(v - best_val) < 1e-9 else ""
        row += f" {v:<14.4f}{marker:<2}"
    row += f" {best_name}"
    print(row)

print(f"\n{'分类别指标':<30} {'features_3':<16} {'features_3b':<16} {'features_3c':<16}")
print("-" * 90)
for ci, display_name in enumerate(class_names):
    for metric_label, metric_key in [("F1", "F1"), ("Recall", "Recall"), ("Precision", "Precision")]:
        row = f"  {display_name}-{metric_label:<20}"
        for n in metric_paths:
            keys = class_key_map[n]
            pcm = metrics[n]["test"]["per_class_metrics"]
            v = pcm[keys[ci]][metric_key]
            row += f" {v:<16.4f}"
        print(row)
    print()

# ============================================================
# 3. 变化趋势分析
# ============================================================
print("=" * 90)
print("四、逐步优化效果分析")
print("=" * 90)

print("\n【3→3b】增加 epochs(100→120) + clinical_scale=3.0 + normalize_clinical")
for split, split_cn in [("val", "验证集"), ("test", "测试集")]:
    print(f"  {split_cn}:")
    for label, key, _ in overall_keys:
        v3 = metrics["features_3"][split][key]
        v3b = metrics["features_3b"][split][key]
        delta = v3b - v3
        sign = "+" if delta >= 0 else ""
        print(f"    {label}: {v3:.4f} → {v3b:.4f} ({sign}{delta:.4f})")

print("\n【3b→3c】增加 dropout(0.0→0.05)")
for split, split_cn in [("val", "验证集"), ("test", "测试集")]:
    print(f"  {split_cn}:")
    for label, key, _ in overall_keys:
        v3b = metrics["features_3b"][split][key]
        v3c = metrics["features_3c"][split][key]
        delta = v3c - v3b
        sign = "+" if delta >= 0 else ""
        print(f"    {label}: {v3b:.4f} → {v3c:.4f} ({sign}{delta:.4f})")

# ============================================================
# 4. 保存JSON报告
# ============================================================
report = {
    "config_differences": {},
    "val_results": {},
    "test_results": {},
    "delta_analysis": {},
}

for key in diff_keys:
    if key == "output":
        continue
    report["config_differences"][key] = {n: str(configs[n].get(key, "<未设置>")) for n in config_paths}

for split in ["val", "test"]:
    for name in metric_paths:
        m = metrics[name][split]
        report[f"{split}_results"][name] = {
            "accuracy": m["accuracy"],
            "auc": m["auc"],
            "f1": m["f1_score"],
            "kappa": m["kappa"],
            "per_class": m["per_class_metrics"],
        }

for split in ["val", "test"]:
    report["delta_analysis"][f"3_to_3b_{split}"] = {}
    report["delta_analysis"][f"3b_to_3c_{split}"] = {}
    for label, key, _ in overall_keys:
        v3 = metrics["features_3"][split][key]
        v3b = metrics["features_3b"][split][key]
        v3c = metrics["features_3c"][split][key]
        report["delta_analysis"][f"3_to_3b_{split}"][label] = round(v3b - v3, 6)
        report["delta_analysis"][f"3b_to_3c_{split}"][label] = round(v3c - v3b, 6)

os.makedirs("model_comparison_results", exist_ok=True)
out_path = "model_comparison_results/features_3_3b_3c_comparison.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n报告已保存: {out_path}")
