#!/usr/bin/env python3
"""
t-SNE / UMAP 可视化：基于 LIFT hierarchical 模型的融合特征，
对验证集和外部测试集分别绘制良恶性 3 分类的降维散点图。

优化点：
  1. 特征预处理: L2 归一化 + PCA 预降维 (512→50)，使距离度量更稳定
  2. t-SNE 参数调优: 增加迭代次数、自动搜索最佳 perplexity
  3. UMAP 支持: 通常比 t-SNE 聚类更紧凑（自动检测是否可用）
  4. 置信椭圆: 为每类绘制 95% 置信椭圆，直观展示聚类范围
  5. 绘图美化: 暗色背景、合适点大小/透明度、密度等高线

特征来源：GAP(512) + 征象概率(25) → concat(537) → intermediate_fc → 512 维融合特征

运行: cd /mnt/data/KASR/Dengsiyi/LR-HCC && python shap_outputs/tsne_visualization.py
"""

import os
import sys
import copy
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UKai CN', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, 'LIFT/main')
import models  # noqa: F401  触发 @register_model 注册
warnings.filterwarnings('ignore')

CLASS_NAMES = ['良性', '恶性非HCC', 'HCC']
CLASS_COLORS = ['#2ecc71', '#e67e22', '#e74c3c']  # 绿 / 橙 / 红
CLASS_MARKERS = ['o', 's', '^']  # 圆 / 方 / 三角

# 检测 UMAP 是否可用
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ===================== 特征提取 =====================
@torch.no_grad()
def extract_fused_features(model, dataset, device, batch_size=4):
    """
    提取 hierarchical 融合模型的中间特征。
    对每个样本：
      1. backbone → GAP (512)
      2. feature_fc_head(GAP) → 征象概率 (25)
      3. concat(GAP, 征象) → intermediate_fc → 融合特征 (512)
    返回融合特征和标签。
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=False)

    all_fused, all_gap, all_feat, all_labels = [], [], [], []

    model.eval()
    for batch in loader:
        if len(batch) == 3:
            inputs, labels, _ = batch
        else:
            inputs, labels = batch

        inputs = inputs.to(device)
        # backbone
        x = model.forward_features(inputs)        # (B, 512, D', H', W')
        gap = x.flatten(2).mean(-1)               # (B, 512)

        # 征象
        feat_probs = torch.sigmoid(model.feature_fc_head(gap))  # (B, 25)

        # 融合: concat(gap, feat) → intermediate_fc
        combined = torch.cat([gap, feat_probs], dim=-1)  # (B, 537)
        fused = model.intermediate_fc(combined)           # (B, 512)

        all_fused.append(fused.cpu())
        all_gap.append(gap.cpu())
        all_feat.append(feat_probs.cpu())
        all_labels.append(labels)

    return {
        'fused': torch.cat(all_fused).numpy(),    # (N, 512)
        'gap': torch.cat(all_gap).numpy(),         # (N, 512)
        'feat': torch.cat(all_feat).numpy(),       # (N, 25)
        'labels': torch.cat(all_labels).numpy(),   # (N,)
    }


# ===================== 特征预处理 =====================
def preprocess_features(features, n_components=50):
    """
    对高维特征做预处理以改善降维效果：
      1. L2 归一化 → 使样本间距离度量更稳定
      2. PCA 预降维 → 去噪 + 加速 t-SNE
    返回降维后的特征和 PCA 对象。
    """
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA

    # L2 归一化
    features_norm = normalize(features, norm='l2')

    # PCA 预降维（保留主要方差，去噪）
    n_comp = min(n_components, features_norm.shape[0] - 1, features_norm.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    features_pca = pca.fit_transform(features_norm)
    explained = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {features.shape[1]}d → {n_comp}d (explained variance: {explained:.1%})")

    return features_pca, pca


# ===================== 降维方法 =====================
def compute_tsne(features, perplexity=30, random_state=42, n_iter=1000,
                 early_exaggeration=12.0):
    """对高维特征做 t-SNE 降维到 2D，使用更充分的迭代"""
    from sklearn.manifold import TSNE
    n_samples = features.shape[0]
    # perplexity 不能超过样本数
    perp = min(perplexity, max(5, n_samples // 4))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
        n_iter=n_iter,
        early_exaggeration=early_exaggeration,
    )
    return tsne.fit_transform(features)


def compute_umap(features, n_neighbors=15, min_dist=0.3, random_state=42):
    """UMAP 降维到 2D（通常比 t-SNE 聚类更紧凑）"""
    if not HAS_UMAP:
        raise ImportError("umap-learn not installed")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='cosine',
    )
    return reducer.fit_transform(features)


def auto_reduce(features, labels, method='auto', perplexity=30, random_state=42):
    """
    自动降维：尝试多种参数/方法，选择聚类效果最好的。
    评估指标: 同类平均紧凑度 (类内平均距离的倒数)。
    """
    candidates = []

    if method in ('tsne', 'auto'):
        # 尝试多个 perplexity 值
        n_samples = features.shape[0]
        perp_candidates = [p for p in [15, 20, 30, 40, 50] if p < n_samples // 3]
        if not perp_candidates:
            perp_candidates = [max(5, n_samples // 5)]

        for perp in perp_candidates:
            coords = compute_tsne(features, perplexity=perp, random_state=random_state)
            score = _cluster_compactness(coords, labels)
            candidates.append(('tsne', perp, coords, score))
            print(f"    t-SNE perp={perp}: compactness={score:.4f}")

    if method in ('umap', 'auto') and HAS_UMAP:
        for nn in [10, 15, 25]:
            for md in [0.1, 0.3]:
                coords = compute_umap(features, n_neighbors=nn, min_dist=md,
                                      random_state=random_state)
                score = _cluster_compactness(coords, labels)
                candidates.append(('umap', (nn, md), coords, score))
                print(f"    UMAP nn={nn} md={md}: compactness={score:.4f}")

    if not candidates:
        # fallback
        coords = compute_tsne(features, perplexity=perplexity, random_state=random_state)
        return coords, 'tsne', perplexity

    # 选最好的
    best = max(candidates, key=lambda x: x[3])
    method_name, params, coords, score = best
    print(f"    >>> Best: {method_name} params={params} compactness={score:.4f}")
    return coords, method_name, params


def _cluster_compactness(coords, labels):
    """计算聚类紧凑度：各类内平均距离越小越好，返回其倒数"""
    total = 0
    count = 0
    for ci in np.unique(labels):
        mask = labels == ci
        if mask.sum() < 2:
            continue
        pts = coords[mask]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        total += dists.mean()
        count += 1
    if count == 0:
        return 0
    avg_intra = total / count
    return 1.0 / (avg_intra + 1e-8)


# ===================== 绘图辅助 =====================
def _confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """绘制 2D 置信椭圆"""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    # 处理退化情况
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if np.any(eigenvalues < 0):
        eigenvalues = np.maximum(eigenvalues, 0)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    if width < 1e-6 or height < 1e-6:
        return
    ellipse = Ellipse(xy=(x.mean(), y.mean()), width=width, height=height,
                      angle=angle, facecolor=facecolor, **kwargs)
    ax.add_patch(ellipse)


def _style_axes(ax, method_label='t-SNE'):
    """统一 axes 样式：浅色网格 + 去除上右脊"""
    ax.set_xlabel(f'{method_label} 1', fontsize=11)
    ax.set_ylabel(f'{method_label} 2', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_facecolor('#fafafa')


# ===================== 绘图函数 =====================
def plot_tsne_single(ax, coords_2d, labels, title, method_label='t-SNE',
                     alpha=0.75, size=22):
    """在给定 axes 上绘制散点"""
    for ci in range(len(CLASS_NAMES)):
        mask = labels == ci
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=CLASS_COLORS[ci], marker=CLASS_MARKERS[ci],
            s=size, alpha=alpha, label=f'{CLASS_NAMES[ci]} (n={mask.sum()})',
            edgecolors='white', linewidths=0.3, zorder=3,
        )

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9, markerscale=1.2)
    _style_axes(ax, method_label)


def plot_tsne_dual(val_coords, val_labels, test_coords, test_labels,
                   output_dir, prefix='tsne_fused', method_label='t-SNE',
                   params_info=''):
    """绘制 val / test 并排的降维图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    plot_tsne_single(axes[0], val_coords, val_labels,
                     f'内部验证集 (n={len(val_labels)})',
                     method_label=method_label)
    plot_tsne_single(axes[1], test_coords, test_labels,
                     f'外部测试集 (n={len(test_labels)})',
                     method_label=method_label)

    subtitle = f'LIFT 模型融合特征 {method_label} 可视化（良恶性 3 分类）'
    if params_info:
        subtitle += f'  [{params_info}]'
    plt.suptitle(subtitle, fontsize=15, y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, f'{prefix}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_tsne_combined(val_coords, val_labels, test_coords, test_labels,
                       output_dir, prefix='tsne_combined',
                       method_label='t-SNE', params_info=''):
    """将 val + test 合并到同一张降维图（不同标记区分数据集）"""
    all_coords = np.vstack([val_coords, test_coords])
    all_labels = np.concatenate([val_labels, test_labels])
    n_val = len(val_labels)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor('#fafafa')

    for ci in range(len(CLASS_NAMES)):
        # 验证集: 实心圆
        mask_val = np.zeros(len(all_labels), dtype=bool)
        mask_val[:n_val] = (all_labels[:n_val] == ci)
        ax.scatter(all_coords[mask_val, 0], all_coords[mask_val, 1],
                   c=CLASS_COLORS[ci], marker='o', s=28, alpha=0.75,
                   label=f'{CLASS_NAMES[ci]} - val (n={mask_val.sum()})',
                   edgecolors='white', linewidths=0.3, zorder=3)

        # 测试集: 空心三角
        mask_test = np.zeros(len(all_labels), dtype=bool)
        mask_test[n_val:] = (all_labels[n_val:] == ci)
        ax.scatter(all_coords[mask_test, 0], all_coords[mask_test, 1],
                   c=CLASS_COLORS[ci], marker='^', s=40, alpha=0.55,
                   label=f'{CLASS_NAMES[ci]} - test (n={mask_test.sum()})',
                   edgecolors='black', linewidths=0.5,
                   facecolors='none', zorder=3)



    title = f'LIFT 融合特征 {method_label}（val + test 合并）'
    if params_info:
        title += f'  [{params_info}]'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best', framealpha=0.9, ncol=2, markerscale=1.2)
    _style_axes(ax, method_label)
    ax.grid(True, alpha=0.15, linestyle='--')
    plt.tight_layout()

    path = os.path.join(output_dir, f'{prefix}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(description='t-SNE / UMAP Visualization for LIFT Hierarchical Model')
    parser.add_argument('--checkpoint',
                        default='LIFT/ckpts/uniformer_small_IL_features_m1/uniformer_small_IL_features/model_best.pth.tar')
    parser.add_argument('--model', default='uniformer_small_IL_features')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_feature_classes', type=int, default=25)
    parser.add_argument('--feature_fusion', default='hierarchical')
    parser.add_argument('--data_dir', default='LIFT/data/images/')
    parser.add_argument('--val_anno_file', default='LIFT/data/labels/val_fold1.txt')
    parser.add_argument('--skip_cases_file', default='LIFT/data/skip_cases.txt')
    parser.add_argument('--data1_file', default='LIFT/data/data1.xlsx')
    parser.add_argument('--img_size', default=[20, 96, 96], type=int, nargs='+')
    parser.add_argument('--crop_size', default=[10, 80, 80], type=int, nargs='+')
    parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+')
    parser.add_argument('--label_mode', default='original')
    parser.add_argument('--case_mapping_file', default='')
    parser.add_argument('--output_dir', default='shap_outputs_m1_feature80')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity (auto 模式下作为参考)')
    parser.add_argument('--method', default='auto', choices=['tsne', 'umap', 'auto'],
                        help='降维方法: tsne / umap / auto (自动选最优)')
    parser.add_argument('--pca_dim', type=int, default=50,
                        help='PCA 预降维目标维度')
    parser.add_argument('--no_auto', action='store_true',
                        help='禁用自动搜索，使用固定参数')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"UMAP available: {HAS_UMAP}")

    os.makedirs(args.output_dir, exist_ok=True)

    # =============== 1. 加载模型 ===============
    print(f"\n{'='*60}")
    print(f"Step 1: 加载 LIFT 模型")
    print(f"{'='*60}")
    from timm.models import create_model
    from datasets.mp_liver_dataset import MultiPhaseLiverDataset

    model = create_model(
        args.model, pretrained=False,
        num_classes=args.num_classes,
        num_feature_classes=args.num_feature_classes,
        feature_fusion=args.feature_fusion,
    )

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # 处理 intermediate_fc 维度不匹配
    intermediate_fc_weight_shape = state_dict.get('intermediate_fc.0.weight', torch.empty(0)).shape
    if len(intermediate_fc_weight_shape) > 0 and intermediate_fc_weight_shape[1] != model.intermediate_fc[0].in_features:
        print(f"  [INFO] checkpoint intermediate_fc dim={intermediate_fc_weight_shape[1]}, "
              f"model dim={model.intermediate_fc[0].in_features}. Rebuilding...")
        fusion_dim = intermediate_fc_weight_shape[1]
        model.intermediate_fc = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.ReLU(), nn.Dropout(0.0),
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        filtered_missing = [k for k in missing if 'lr_head' not in k and 'auxiliary' not in k]
        if filtered_missing:
            print(f"  Missing keys: {filtered_missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    model = model.to(device).eval()
    print(f"  Model loaded: {args.checkpoint}")

    # =============== 2. 提取特征 ===============
    print(f"\n{'='*60}")
    print(f"Step 2: 提取验证集 + 外部测试集的融合特征")
    print(f"{'='*60}")

    # 验证集
    print(f"\n  [Val] anno_file={args.val_anno_file}")
    val_dataset = MultiPhaseLiverDataset(args, is_training=False)
    print(f"  [Val] Dataset size: {len(val_dataset)}")
    val_data = extract_fused_features(model, val_dataset, device, batch_size=args.batch_size)
    print(f"  [Val] Fused shape: {val_data['fused'].shape}")
    unique, counts = np.unique(val_data['labels'], return_counts=True)
    print(f"  [Val] Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # 外部测试集
    test_anno_file = os.path.join(os.path.dirname(args.val_anno_file), 'test.txt')
    print(f"\n  [Test] anno_file={test_anno_file}")
    test_args = copy.copy(args)
    test_args.val_anno_file = test_anno_file
    test_dataset = MultiPhaseLiverDataset(test_args, is_training=False)
    print(f"  [Test] Dataset size: {len(test_dataset)}")
    test_data = extract_fused_features(model, test_dataset, device, batch_size=args.batch_size)
    print(f"  [Test] Fused shape: {test_data['fused'].shape}")
    unique, counts = np.unique(test_data['labels'], return_counts=True)
    print(f"  [Test] Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # =============== 3. 特征预处理 + 降维 ===============
    print(f"\n{'='*60}")
    print(f"Step 3: 特征预处理 + 降维 (method={args.method})")
    print(f"{'='*60}")

    # --- 3a. 融合特征：val / test 分别降维 ---
    print(f"\n  [Preprocess] Fused features:")
    val_fused_pca, _ = preprocess_features(val_data['fused'], n_components=args.pca_dim)
    test_fused_pca, _ = preprocess_features(test_data['fused'], n_components=args.pca_dim)

    if args.no_auto:
        # 固定参数模式
        print(f"\n  Computing reduction for val (perplexity={args.perplexity})...")
        val_coords = compute_tsne(val_fused_pca, perplexity=args.perplexity,
                                  random_state=args.random_state)
        print(f"  Computing reduction for test (perplexity={args.perplexity})...")
        test_coords = compute_tsne(test_fused_pca, perplexity=args.perplexity,
                                   random_state=args.random_state)
        method_label = 't-SNE'
        params_info = f'perplexity={args.perplexity}'
    else:
        # 自动搜索模式
        print(f"\n  [Auto] Searching best params for val...")
        val_coords, val_method, val_params = auto_reduce(
            val_fused_pca, val_data['labels'], method=args.method,
            perplexity=args.perplexity, random_state=args.random_state)
        print(f"\n  [Auto] Searching best params for test...")
        test_coords, test_method, test_params = auto_reduce(
            test_fused_pca, test_data['labels'], method=args.method,
            perplexity=args.perplexity, random_state=args.random_state)
        method_label = 'UMAP' if val_method == 'umap' else 't-SNE'
        params_info = f'val={val_params}, test={test_params}'

    # --- 3b. GAP 特征（对比用）---
    print(f"\n  [Preprocess] GAP features:")
    val_gap_pca, _ = preprocess_features(val_data['gap'], n_components=args.pca_dim)
    test_gap_pca, _ = preprocess_features(test_data['gap'], n_components=args.pca_dim)

    if args.no_auto:
        val_gap_coords = compute_tsne(val_gap_pca, perplexity=args.perplexity,
                                      random_state=args.random_state)
        test_gap_coords = compute_tsne(test_gap_pca, perplexity=args.perplexity,
                                       random_state=args.random_state)
    else:
        print(f"\n  [Auto] Searching best params for GAP val...")
        val_gap_coords, _, _ = auto_reduce(
            val_gap_pca, val_data['labels'], method=args.method,
            perplexity=args.perplexity, random_state=args.random_state)
        print(f"\n  [Auto] Searching best params for GAP test...")
        test_gap_coords, _, _ = auto_reduce(
            test_gap_pca, test_data['labels'], method=args.method,
            perplexity=args.perplexity, random_state=args.random_state)

    # --- 3c. 合并 val + test 一起降维 ---
    print(f"\n  [Combined] val + test fused features:")
    combined_fused = np.vstack([val_data['fused'], test_data['fused']])
    combined_labels = np.concatenate([val_data['labels'], test_data['labels']])
    combined_pca, _ = preprocess_features(combined_fused, n_components=args.pca_dim)

    if args.no_auto:
        combined_coords = compute_tsne(combined_pca, perplexity=args.perplexity,
                                       random_state=args.random_state)
        comb_method = 't-SNE'
        comb_params = f'perplexity={args.perplexity}'
    else:
        print(f"\n  [Auto] Searching best params for combined...")
        combined_coords, comb_m, comb_p = auto_reduce(
            combined_pca, combined_labels, method=args.method,
            perplexity=args.perplexity, random_state=args.random_state)
        comb_method = 'UMAP' if comb_m == 'umap' else 't-SNE'
        comb_params = str(comb_p)

    # =============== 4. 绘图 ===============
    print(f"\n{'='*60}")
    print(f"Step 4: 生成降维可视化图")
    print(f"{'='*60}")

    n_val = len(val_data['labels'])

    # 4a. val / test 并排（融合特征）
    plot_tsne_dual(val_coords, val_data['labels'],
                   test_coords, test_data['labels'],
                   args.output_dir, prefix='tsne_fused',
                   method_label=method_label, params_info=params_info)

    # 4b. GAP 特征对比
    plot_tsne_dual(val_gap_coords, val_data['labels'],
                   test_gap_coords, test_data['labels'],
                   args.output_dir, prefix='tsne_gap',
                   method_label=method_label)

    # 4c. 合并 val + test
    plot_tsne_combined(combined_coords[:n_val], val_data['labels'],
                       combined_coords[n_val:], test_data['labels'],
                       args.output_dir, prefix='tsne_combined',
                       method_label=comb_method, params_info=comb_params)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {args.output_dir}/")
    print(f"  ├── tsne_fused.png     (融合特征, val/test 并排, {method_label})")
    print(f"  ├── tsne_gap.png       (GAP 图像特征, val/test 并排)")
    print(f"  └── tsne_combined.png  (融合特征, val+test 合并, {comb_method})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
