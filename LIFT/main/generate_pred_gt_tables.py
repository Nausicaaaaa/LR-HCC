#!/usr/bin/env python3
"""
生成两张表格：
1. 预测结果表：每例患者的预测类别、预测征象（val + test）
2. 真实标签表：每例患者的真实类别、真实征象（val + test）

用法:
    conda activate class_young
    cd /mnt/data/KASR/Dengsiyi/LR-HCC
    python LIFT/main/generate_pred_gt_tables.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# 项目根目录
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)

# ---- 25征象完整名称（与 predict.py ALL_SIGN_NAMES_25 一致，对应 data1.xlsx col 17~41）----
FEATURE_NAMES = [
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

CLASS_NAMES_3 = ['Benign', 'Non-HCC Malignancies', 'HCC']

PRED_DIR = 'LIFT/ckpts/Model2/uniformer_small_IL_features/pred_results'
CKPT_DIR = 'LIFT/ckpts/Model2/uniformer_small_IL_features'
ANNO_DIR = 'LIFT/data/labels'


def parse_anno_file(anno_path):
    """解析标注文件，返回每行的信息"""
    with open(anno_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    skip_header = 1 if first_line.startswith('casename') else 0
    df = pd.read_csv(anno_path, sep='\t', header=None, skiprows=skip_header)

    pathology_mapping = {'liangxing': 0, 'noHCC': 1, 'HCC': 2}
    rows = []

    for idx, row in df.iterrows():
        case_name = str(row.iloc[0]).strip()
        pathology_str = str(row.iloc[1]).strip()
        lr_str = str(row.iloc[2]).strip() if df.shape[1] > 2 else ''
        true_cls = pathology_mapping.get(pathology_str, -1)

        # 征象标签（最后25列）
        num_features = 25
        features = []
        if df.shape[1] >= 28:
            for col_idx in range(df.shape[1] - num_features, df.shape[1]):
                val = row.iloc[col_idx]
                features.append(int(float(val)) if pd.notna(val) else 0)
        else:
            features = [0] * num_features

        rows.append({
            'case_name': case_name,
            'true_class_id': true_cls,
            'true_class': pathology_str,
            'lr_grade': lr_str,
            'features': features,
        })
    return rows


def load_score_json(score_path):
    """加载 score.json"""
    with open(score_path, 'r') as f:
        data = json.load(f)
    return {item['image_id']: item for item in data}


def try_model_inference(anno_file, ckpt_dir):
    """尝试加载模型运行推理，返回预测征象。失败则返回 None"""
    try:
        import yaml
        import torch
        from timm.models import create_model, load_checkpoint

        # 导入模型定义（确保 LIFT/main 优先于系统 packages）
        main_dir = os.path.join(ROOT, 'LIFT', 'main')
        if main_dir not in sys.path:
            sys.path.insert(0, main_dir)
        # 清除可能与本地 datasets 冲突的已缓存模块
        for mod_name in list(sys.modules.keys()):
            if mod_name == 'datasets' or mod_name.startswith('datasets.'):
                del sys.modules[mod_name]
        import models as _  # noqa: register models

        # 加载 args.yaml
        args_file = os.path.join(ckpt_dir, 'args.yaml')
        with open(args_file, 'r') as f:
            args_dict = yaml.safe_load(f)

        args = argparse.Namespace(**args_dict)
        args.val_anno_file = anno_file
        args.data_dir = 'LIFT/data/images'
        args.skip_cases_file = 'LIFT/data/skip_cases.txt'

        # 确保所有必要属性存在
        defaults = {
            'selected_features': None,
            'include_clinical': True,
            'clinical_dim': 10,
            'normalize_clinical': False,
            'feature_fusion': 'hierarchical_simple',
            'clinical_scale': 1.0,
            'head_drop_rate': 0.0,
            'fusion_hidden_dim': 512,
        }
        for k, v in defaults.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

        from datasets.mp_liver_dataset import MultiPhaseLiverDataset

        # 创建模型
        model_kwargs = dict(
            pretrained=False,
            num_classes=args.num_classes,
            num_feature_classes=args.num_feature_classes,
            feature_fusion=args.feature_fusion,
        )
        if args.include_clinical:
            model_kwargs['clinical_dim'] = args.clinical_dim
            model_kwargs['clinical_scale_init'] = args.clinical_scale

        model = create_model(args.model, **model_kwargs)

        # 加载 checkpoint
        ckpt_path = os.path.join(ckpt_dir, 'model_best.pth.tar')
        if not os.path.exists(ckpt_path):
            for f in sorted(os.listdir(ckpt_dir)):
                if f.endswith('.pth.tar') and 'checkpoint' in f:
                    ckpt_path = os.path.join(ckpt_dir, f)
                    break
        print(f"  Loading checkpoint: {ckpt_path}")
        load_checkpoint(model, ckpt_path, False)
        model = model.cuda()
        model.eval()

        # 创建数据集
        dataset = MultiPhaseLiverDataset(args, is_training=False)
        case_names = dataset.get_case_names()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, num_workers=4,
            pin_memory=False, shuffle=False
        )

        # 推理
        all_pred_features = []
        all_true_features = []

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 5:
                    input_t, target, features, lr_label, extra_features = batch
                    extra_features = extra_features.cuda()
                elif len(batch) == 4:
                    input_t, target, features, fourth = batch
                    if isinstance(fourth, torch.Tensor) and fourth.dim() == 1:
                        extra_features = None
                    else:
                        extra_features = fourth
                elif len(batch) == 3:
                    input_t, target, features = batch
                    extra_features = None
                else:
                    input_t, target = batch
                    features = None
                    extra_features = None

                input_t = input_t.cuda()

                if extra_features is not None:
                    output = model(input_t, extra_features=extra_features)
                else:
                    output = model(input_t)

                if isinstance(output, (tuple, list)):
                    feature_output = output[1] if len(output) >= 2 and output[1] is not None and output[1].dim() == 2 else None
                else:
                    feature_output = None

                if feature_output is not None:
                    pred_feat = torch.sigmoid(feature_output).cpu().numpy()
                    pred_feat_binary = (pred_feat >= 0.5).astype(int)
                    all_pred_features.extend(pred_feat_binary.tolist())

                if features is not None:
                    all_true_features.extend(features.numpy().astype(int).tolist())

        print(f"  Inference done: {len(all_pred_features)} cases with feature predictions")
        return case_names, all_pred_features, all_true_features

    except Exception as e:
        print(f"  WARNING: Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_tables(anno_file, score_json_path, split_name, output_dir):
    """为一个 split (val/test) 生成预测表和真实表"""

    print(f"\n{'='*60}")
    print(f"Processing {split_name} set...")
    print(f"{'='*60}")

    # 1. 解析标注文件
    anno_data = parse_anno_file(anno_file)
    anno_dict = {r['case_name']: r for r in anno_data}

    # 2. 加载 score.json
    score_data = load_score_json(score_json_path)

    # 3. 尝试模型推理获取征象预测
    print("  Trying model inference for feature predictions...")
    inference_result = try_model_inference(anno_file, CKPT_DIR)

    if inference_result is not None:
        inf_case_names, inf_pred_features, inf_true_features = inference_result
        inf_pred_dict = {cn: inf_pred_features[i] for i, cn in enumerate(inf_case_names)}
        has_feature_pred = True
        print(f"  Feature predictions available for {len(inf_pred_dict)} cases")
    else:
        has_feature_pred = False
        inf_pred_dict = {}
        print("  Feature predictions NOT available (inference failed)")

    # 4. 构建预测结果表
    # 使用 score.json 中的 case 顺序
    case_names_ordered = list(score_data.keys())

    pred_rows = []
    for cn in case_names_ordered:
        score_item = score_data[cn]
        pred_cls_id = score_item['prediction']
        pred_cls_name = CLASS_NAMES_3[pred_cls_id] if pred_cls_id < len(CLASS_NAMES_3) else f'Class_{pred_cls_id}'

        row = {'Patient ID': cn, 'Predicted Class': pred_cls_name}

        if has_feature_pred and cn in inf_pred_dict:
            feat_binary = inf_pred_dict[cn]
            active_features = [FEATURE_NAMES[j] for j in range(len(feat_binary))
                             if feat_binary[j] == 1 and j < len(FEATURE_NAMES)]
            row['Predicted Features'] = '; '.join(active_features) if active_features else 'None'
            for j, fname in enumerate(FEATURE_NAMES):
                if j < len(feat_binary):
                    row[fname] = int(feat_binary[j])
        else:
            row['Predicted Features'] = 'N/A'

        pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    pred_csv_path = os.path.join(output_dir, f'{split_name}_predictions.csv')
    pred_df.to_csv(pred_csv_path, index=False, encoding='utf-8-sig')
    print(f"  Predictions saved to: {pred_csv_path} ({len(pred_rows)} cases)")

    # 5. 构建真实标签表
    true_rows = []
    for cn in case_names_ordered:
        if cn in anno_dict:
            info = anno_dict[cn]
            true_cls_name = info['true_class']
            lr_grade = info['lr_grade']
            feat = info['features']
        else:
            true_cls_name = 'Unknown'
            lr_grade = ''
            feat = [0] * 25

        row = {'Patient ID': cn, 'True Class': true_cls_name, 'LR Grade': lr_grade}

        active_features = [FEATURE_NAMES[j] for j in range(len(feat))
                         if feat[j] == 1 and j < len(FEATURE_NAMES)]
        row['Active Features'] = '; '.join(active_features) if active_features else 'None'
        for j, fname in enumerate(FEATURE_NAMES):
            if j < len(feat):
                row[fname] = int(feat[j])

        true_rows.append(row)

    true_df = pd.DataFrame(true_rows)
    true_csv_path = os.path.join(output_dir, f'{split_name}_ground_truth.csv')
    true_df.to_csv(true_csv_path, index=False, encoding='utf-8-sig')
    print(f"  Ground truth saved to: {true_csv_path} ({len(true_rows)} cases)")

    return pred_csv_path, true_csv_path


def main():
    output_dir = os.path.join(PRED_DIR, 'tables')
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Val set
    val_anno = os.path.join(ANNO_DIR, 'val_fold1.txt')
    val_score = os.path.join(PRED_DIR, 'val', 'score.json')
    if os.path.exists(val_anno) and os.path.exists(val_score):
        results['val'] = generate_tables(val_anno, val_score, 'val', output_dir)
    else:
        print(f"Val files not found: anno={os.path.exists(val_anno)}, score={os.path.exists(val_score)}")

    # Test set
    test_anno = os.path.join(ANNO_DIR, 'test.txt')
    test_score = os.path.join(PRED_DIR, 'test', 'score.json')
    if os.path.exists(test_anno) and os.path.exists(test_score):
        results['test'] = generate_tables(test_anno, test_score, 'test', output_dir)
    else:
        print(f"Test files not found: anno={os.path.exists(test_anno)}, score={os.path.exists(test_score)}")

    print(f"\n{'='*60}")
    print(f"All done! Tables saved to: {output_dir}")
    for split, (pred_path, true_path) in results.items():
        print(f"  {split}: {pred_path}, {true_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
