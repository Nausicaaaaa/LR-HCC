#!/usr/bin/env python3
"""
Compute feature accuracy on the training set (train_fold1, 1552 cases).

Load Model1 checkpoint, run inference on all 1552 training cases,
compute per-feature accuracy for 25 imaging features.

Output:
  - train_5fold_feature_accuracy.csv   (used by SHAP scripts for feature filtering)
  - train_5fold_feature_accuracy.md    (supplementary table for paper)

Usage:
  python shap_outputs/compute_5fold_feature_accuracy.py \
      --checkpoint LIFT/ckpts/Model1/uniformer_small_IL_features/model_best.pth.tar \
      --output_dir shap_outputs/Model1
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, 'LIFT/main')
import models  # noqa: F401
warnings.filterwarnings('ignore')

# 25 Imaging Feature Names (matching ALL_SIGN_NAMES_25, data1.xlsx col 17~41)
IMAGING_FEATURE_NAMES_EN = [
    'Arterial phase hyperenhancement',
    'Nonrim arterial phase hyperenhancement',
    'Rim APHE',
    'Nonperipheral washout',
    'Peripheral "Washout"',
    'Corona Enhancement',
    'Enhancing capsule',
    'Nonenhancing capsule',
    'Peripheral Discontinuous Nodular Enhancement',
    'Progressive Enhancement',
    'Centripetal Enhancement',
    'Parallels blood pool enhancement',
    'Uniform AP Enhancement',
    'Uniform PVP Enhancement',
    'Uniform DP Enhancement',
    'Necrosis or severe ischemia',
    'Blood Products in Mass',
    'Nodule-in-nodule architecture',
    'Mosaic Architecture',
    'Delayed Central Enhancement',
    'Infiltrative appearance',
    'Portal venous phase peritumoral hypoenhancement',
    'Fat in Mass, more than Liver',
    'Fat Sparing in Solid Mass',
    'Intratumoral artery',
]

NUM_FEATURES = 25


def load_model(checkpoint, model_name='uniformer_small_IL_features',
               num_classes=3, num_feature_classes=25,
               feature_fusion='hierarchical_simple', clinical_dim=0,
               include_clinical=False, device='cpu'):
    """Load model from checkpoint, handling dimension mismatches."""
    from timm.models import create_model

    model_kwargs = dict(
        pretrained=False,
        num_classes=num_classes,
        num_feature_classes=num_feature_classes,
        feature_fusion=feature_fusion,
    )
    if include_clinical and clinical_dim > 0:
        model_kwargs['clinical_dim'] = clinical_dim

    model = create_model(model_name, **model_kwargs)

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    # Handle intermediate_fc dimension mismatch
    if hasattr(model, 'intermediate_fc'):
        w_shape = state_dict.get('intermediate_fc.0.weight', torch.empty(0)).shape
        if len(w_shape) > 0 and w_shape[1] != model.intermediate_fc[0].in_features:
            print(f"  [INFO] Rebuilding intermediate_fc (ckpt dim={w_shape[1]}, "
                  f"model dim={model.intermediate_fc[0].in_features})")
            fusion_dim = w_shape[1]
            model.intermediate_fc = nn.Sequential(
                nn.Linear(fusion_dim, 512), nn.ReLU(), nn.Dropout(0.0),
            )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        filtered = [k for k in missing if 'lr_head' not in k and 'auxiliary' not in k]
        if filtered:
            print(f"  Missing keys: {filtered}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    model = model.to(device).eval()
    print(f"  Loaded: {checkpoint}")
    return model


def extract_feature_predictions(model, dataset, device, batch_size=4):
    """Extract 25-dim feature probabilities and ground truth from a dataset.

    Returns:
        dict with 'feat' (N, 25), 'labels' (N,)
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=False)

    all_feat, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            x = model.forward_features(inputs)
            gap = x.flatten(2).mean(-1)
            feat_probs = torch.sigmoid(model.feature_fc_head(gap))
            all_feat.append(feat_probs.cpu())
            all_labels.append(batch[1])

    return {
        'feat': torch.cat(all_feat).numpy(),
        'labels': torch.cat(all_labels).numpy(),
    }


def compute_feature_accuracy_table(all_feat_pred, all_feat_true, feature_names):
    """Compute per-feature accuracy across all samples.

    Args:
        all_feat_pred: (N, 25) sigmoid probabilities
        all_feat_true: (N, 25) binary ground truth
        feature_names: list of 25 feature names

    Returns:
        DataFrame with columns: Imaging Feature, Accuracy (%),
            No. of Lesions without Feature Present, No. of Lesions with Feature Present
    """
    pred_binary = (all_feat_pred >= 0.5).astype(int)
    true_binary = all_feat_true.astype(int)

    rows = []
    for i, name in enumerate(feature_names):
        acc = np.mean(pred_binary[:, i] == true_binary[:, i]) * 100
        neg_count = int(np.sum(true_binary[:, i] == 0))
        pos_count = int(np.sum(true_binary[:, i] == 1))
        rows.append({
            'Imaging Feature': name,
            'Accuracy (%)': round(acc, 2),
            'No. of Lesions without Feature Present': neg_count,
            'No. of Lesions with Feature Present': pos_count,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Compute feature accuracy on training set (train_fold1, 1552 cases)')
    parser.add_argument('--checkpoint', type=str,
                        default='LIFT/ckpts/Model1/uniformer_small_IL_features/model_best.pth.tar',
                        help='Model checkpoint path')
    parser.add_argument('--train_anno', type=str,
                        default='LIFT/data/labels/train_fold1.txt',
                        help='Training set anno file (1552 cases)')
    parser.add_argument('--model', default='uniformer_small_IL_features')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_feature_classes', type=int, default=25)
    parser.add_argument('--feature_fusion', default='hierarchical_simple')
    parser.add_argument('--clinical_dim', type=int, default=0)
    parser.add_argument('--include_clinical', action='store_true', default=False)
    parser.add_argument('--data_dir', default='LIFT/data/images/')
    parser.add_argument('--labels_dir', default='LIFT/data/labels')
    parser.add_argument('--skip_cases_file', default='LIFT/data/skip_cases.txt')
    parser.add_argument('--data1_file', default='LIFT/data/data1.xlsx')
    parser.add_argument('--case_mapping_file', default='')
    parser.add_argument('--img_size', default=[20, 96, 96], type=int, nargs='+')
    parser.add_argument('--crop_size', default=[10, 80, 80], type=int, nargs='+')
    parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+')
    parser.add_argument('--label_mode', default='original')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=80.0,
                        help='Feature accuracy threshold (%%) for filtering')
    parser.add_argument('--output_dir', default='shap_outputs/Model1')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    from datasets.mp_liver_dataset import MultiPhaseLiverDataset
    import copy

    # =============== Load model and run inference on training set ===============
    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        print(f"ERROR: Checkpoint not found: {ckpt}")
        return

    print(f"\n{'='*60}")
    print(f"Training set: {args.train_anno}")
    print(f"Checkpoint: {ckpt}")
    print(f"{'='*60}")

    model = load_model(ckpt, args.model, args.num_classes,
                       args.num_feature_classes, args.feature_fusion,
                       args.clinical_dim, args.include_clinical, device)

    train_args = copy.copy(args)
    train_args.val_anno_file = args.train_anno
    dataset = MultiPhaseLiverDataset(train_args, is_training=False)
    total_samples = len(dataset)
    print(f"  Dataset size: {total_samples}")

    data = extract_feature_predictions(model, dataset, device,
                                       batch_size=args.batch_size)
    n = len(data['labels'])
    print(f"  Extracted {n} samples, feat shape: {data['feat'].shape}")

    # Extract feature ground truth from the dataset
    case_names = dataset.get_case_names()
    feat_gt = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        features = dataset.case_to_features.get(case_names[i])
        if features is not None:
            feat_len = min(len(features), NUM_FEATURES)
            feat_gt[i, :feat_len] = features[:feat_len].numpy()

    all_feat_pred = data['feat']
    all_feat_true = feat_gt

    # Free GPU memory
    del model, dataset, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if total_samples == 0:
        print("ERROR: No samples collected!")
        return
    print(f"\nTotal training samples: {total_samples}")
    print(f"Feature predictions shape: {all_feat_pred.shape}")
    print(f"Feature ground truth shape: {all_feat_true.shape}")

    # =============== Compute accuracy table ===============
    df_acc = compute_feature_accuracy_table(
        all_feat_pred, all_feat_true, IMAGING_FEATURE_NAMES_EN)

    # =============== Save CSV ===============
    csv_path = os.path.join(args.output_dir, 'train_5fold_feature_accuracy.csv')
    df_acc.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {csv_path}")

    # =============== Save Markdown table (supplementary table for paper) ===============
    md_path = os.path.join(args.output_dir, 'train_5fold_feature_accuracy.md')
    md_lines = [
        "# 训练集 — 25个成像特征预测准确率\n",
        f"总样本数: {total_samples} 例 (train_fold1)\n",
        "| No. | Imaging Feature | Accuracy (%) | No. of Lesions without Feature Present | No. of Lesions with Feature Present |",
        "|---|---|---|---|---|",
    ]
    for idx, row in df_acc.iterrows():
        mark = '✓' if row['Accuracy (%)'] > args.threshold else '✗'
        md_lines.append(
            f"| {idx+1} | {row['Imaging Feature']} | {row['Accuracy (%)']:.2f} | "
            f"{row['No. of Lesions without Feature Present']} | "
            f"{row['No. of Lesions with Feature Present']} | {mark} |"
        )

    n_selected = int((df_acc['Accuracy (%)'] > args.threshold).sum())
    n_excluded = NUM_FEATURES - n_selected
    md_lines.append(f"\n**保留特征数 (Accuracy > {args.threshold}%):** {n_selected}")
    md_lines.append(f"**过滤特征数:** {n_excluded}")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {md_path}")

    # =============== Print summary ===============
    print(f"\n{'='*60}")
    print(f"Training Set Feature Accuracy Summary (threshold={args.threshold}%)")
    print(f"{'='*60}")
    print(f"  Total training samples: {total_samples}")
    print(f"\n  Features RETAINED (Accuracy > {args.threshold}%):")
    for _, row in df_acc.iterrows():
        if row['Accuracy (%)'] > args.threshold:
            print(f"    {row['Accuracy (%)']:6.2f}%  {row['Imaging Feature']}")
    print(f"\n  Features EXCLUDED (Accuracy ≤ {args.threshold}%):")
    for _, row in df_acc.iterrows():
        if row['Accuracy (%)'] <= args.threshold:
            print(f"    {row['Accuracy (%)']:6.2f}%  {row['Imaging Feature']}")
    print(f"\n  Retained: {n_selected} / {NUM_FEATURES}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
