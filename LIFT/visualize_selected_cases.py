#!/usr/bin/env python3
"""
可视化挑选案例：原始 CT 图像 + GT mask vs 预测 mask 对比
每个 case 输出: 中间切片 4 模态 1x4 拼接图, 叠加 GT(绿色) 和 Pred(红色) 轮廓
以及 GT vs Pred 对比大图

用法:
    python LIFT/visualize_selected_cases.py
"""

import os
import re
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ========== 配置 ==========
PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_TR = PROJECT_ROOT / 'data' / 'nnUNet_raw' / 'Dataset001_HCC' / 'imagesTr'
GT_DIR = PROJECT_ROOT / 'data' / 'nnUNet_raw' / 'Dataset001_HCC' / 'labelsTr'
PRED_DIR = PROJECT_ROOT / 'LIFT' / 'data' / 'masks_predicted' / 'val'
OUTPUT_DIR = PROJECT_ROOT / 'LIFT' / 'data' / 'masks_predicted' / 'visualization'

MODALITY_LABELS = {'0000': 'Arterial', '0001': 'Delayed', '0002': 'Portal', '0003': 'Subtraction'}

# 9 个挑选案例
SELECTED_CASES = {
    'HCC': [
        ('case_019', 0.9424),
        ('case_095', 0.9235),
        ('case_057', 0.8803),
    ],
    'Benign': [
        ('case_1987', 0.9217),
        ('case_473', 0.9035),
        ('case_2267', 0.8279),
    ],
    'Non-HCC Malignancies': [
        ('case_457', 0.8958),
        ('case_454', 0.8954),
        ('case_448', 0.8935),
    ],
}

WL, WW = 60.0, 400.0  # 肝脏窗
CONTOUR_THICKNESS = 2
ROTATE_180 = True  # 顺时针旋转180度


def rotate180(arr):
    """顺时针旋转180度（等同翻转两个轴）"""
    if ROTATE_180:
        return np.rot90(arr, -1, axes=(0, 1))  # 先逆时针90再... 实际180=flip both
    return arr


def rotate180_2d(arr):
    """顺时针旋转 2D 数组 180 度"""
    if ROTATE_180:
        return arr[::-1, ::-1]
    return arr


def hu_window(arr, wl, ww):
    low, high = wl - ww / 2, wl + ww / 2
    return (np.clip(arr, low, high) - low) / (high - low + 1e-8) * 255


def to_rgb(arr_u8):
    return np.stack([arr_u8, arr_u8, arr_u8], axis=-1).astype(np.uint8)


def extract_contour(mask_2d, thickness=2):
    """提取二值掩膜轮廓"""
    m = mask_2d > 0
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    boundary = m & (~inner)
    if thickness > 1:
        for _ in range(thickness - 1):
            expanded = boundary.copy()
            expanded[1:, :] |= boundary[:-1, :]
            expanded[:-1, :] |= boundary[1:, :]
            expanded[:, 1:] |= boundary[:, :-1]
            expanded[:, :-1] |= boundary[:, 1:]
            boundary = expanded & m
    return boundary


def overlay_contour(rgb, mask_2d, color, thickness=2):
    """在 RGB 图上叠加轮廓"""
    contour = extract_contour(mask_2d, thickness)
    rgb[contour] = np.array(color, dtype=np.uint8)
    return rgb


def overlay_filled(rgb, mask_2d, color, alpha=0.25):
    """在 RGB 图上叠加半透明填充"""
    m = mask_2d > 0
    if not np.any(m):
        return rgb
    rgb_f = rgb.astype(np.float32)
    c = np.array(color, dtype=np.float32)
    for ch in range(3):
        rgb_f[:, :, ch] = np.where(m, rgb_f[:, :, ch] * (1 - alpha) + c[ch] * alpha, rgb_f[:, :, ch])
    return rgb_f.astype(np.uint8)


def load_volume(path):
    """用 SimpleITK 加载 NIfTI"""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img)  # (Z, Y, X)


def add_title_bar(img_array, text, bg_color=(40, 40, 40), text_color=(255, 255, 255)):
    """在图像顶部加标题栏"""
    h_bar = 30
    bar = np.full((h_bar, img_array.shape[1], 3), bg_color, dtype=np.uint8)
    result = np.concatenate([bar, img_array], axis=0)
    pil = Image.fromarray(result)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((10, 6), text, fill=text_color, font=font)
    return np.array(pil)


def find_middle_slice_with_mask(mask_vol):
    """找到 mask 非空的中间切片"""
    z_indices = np.where(np.any(mask_vol > 0, axis=(1, 2)))[0]
    if len(z_indices) == 0:
        return mask_vol.shape[0] // 2
    return z_indices[len(z_indices) // 2]


def visualize_case(case_id, dice_score, category, output_dir):
    """可视化单个 case"""
    base_id = re.match(r'(case_\d+)(?:_\d+)?$', case_id).group(1)

    # 加载 4 模态图像
    mod_vols = {}
    for suf in ['0000', '0001', '0002', '0003']:
        fpath = IMAGES_TR / f'{base_id}_{suf}.nii.gz'
        if fpath.exists():
            mod_vols[suf] = load_volume(fpath)

    if not mod_vols:
        print(f'  {case_id}: no images found, skip')
        return

    # 加载 GT mask
    gt_path = GT_DIR / f'{base_id}.nii.gz'
    pred_path = PRED_DIR / f'{base_id}.nii.gz'

    gt_vol = load_volume(str(gt_path)) if gt_path.exists() else None
    pred_vol = load_volume(str(pred_path)) if pred_path.exists() else None

    # 确定显示切片
    ref_mask = gt_vol if gt_vol is not None else pred_vol
    if ref_mask is None:
        slice_idx = list(mod_vols.values())[0].shape[0] // 2
    else:
        slice_idx = find_middle_slice_with_mask(ref_mask)

    # ===== 图1: 4模态 + GT(绿) + Pred(红) 轮廓 1x4 拼接 =====
    panels = []
    for suf in sorted(mod_vols.keys()):
        vol = mod_vols[suf]
        ct_slice = rotate180_2d(vol[slice_idx, :, :])  # (Y, X) rotated 180
        ct_u8 = hu_window(ct_slice, WL, WW).astype(np.uint8)
        rgb = to_rgb(ct_u8)

        # 叠加 GT 轮廓(绿色) 和 Pred 轮廓(红色)
        if gt_vol is not None:
            gt_slice = rotate180_2d(gt_vol[slice_idx, :, :])
            rgb = overlay_contour(rgb, gt_slice, (0, 255, 0), CONTOUR_THICKNESS)
        if pred_vol is not None:
            pred_slice = rotate180_2d(pred_vol[slice_idx, :, :])
            rgb = overlay_contour(rgb, pred_slice, (255, 0, 0), CONTOUR_THICKNESS)

        # 加模态标签
        label = MODALITY_LABELS.get(suf, suf)
        rgb_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(rgb_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        draw.text((5, 5), label, fill=(255, 255, 0), font=font)
        panels.append(np.array(rgb_img))

    # 1x4 拼接
    montage = np.concatenate(panels, axis=1)

    # 加标题
    title = f'{case_id} | {category} | Dice={dice_score:.4f} | Slice {slice_idx}/{ref_mask.shape[0] if ref_mask is not None else "?"}'
    title += '  [Green=GT, Red=Pred]'
    montage_with_title = add_title_bar(montage, title)

    out_path = output_dir / f'{case_id}_overview.png'
    Image.fromarray(montage_with_title).save(str(out_path))

    # ===== 图2: GT vs Pred 对比大图 (Arterial phase) =====
    art_vol = mod_vols.get('0000', list(mod_vols.values())[0])
    ct_slice = rotate180_2d(art_vol[slice_idx, :, :])
    ct_u8 = hu_window(ct_slice, WL, WW).astype(np.uint8)

    # 左: 原图
    orig_rgb = to_rgb(ct_u8)

    # 中: GT overlay
    gt_img = to_rgb(ct_u8)
    if gt_vol is not None:
        gt_slice = rotate180_2d(gt_vol[slice_idx, :, :])
        gt_img = overlay_filled(gt_img, gt_slice, (0, 255, 0), 0.3)
        gt_img = overlay_contour(gt_img, gt_slice, (0, 255, 0), CONTOUR_THICKNESS)

    # 右: Pred overlay
    pred_img = to_rgb(ct_u8)
    if pred_vol is not None:
        pred_slice = rotate180_2d(pred_vol[slice_idx, :, :])
        pred_img = overlay_filled(pred_img, pred_slice, (255, 0, 0), 0.3)
        pred_img = overlay_contour(pred_img, pred_slice, (255, 0, 0), CONTOUR_THICKNESS)

    # 拼接
    gap = np.full((ct_u8.shape[0], 4, 3), 200, dtype=np.uint8)
    comparison = np.concatenate([orig_rgb, gap, gt_img, gap, pred_img], axis=1)

    # 标签
    comp_title = f'{case_id} | {category} | Dice={dice_score:.4f}'
    comparison = add_title_bar(comparison, comp_title)

    # 加子标签
    pil_comp = Image.fromarray(comparison)
    draw = ImageDraw.Draw(pil_comp)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    w = ct_u8.shape[1]
    draw.text((10, 34), 'Original', fill=(255, 255, 255), font=font)
    draw.text((w + 14, 34), 'GT Mask (Green)', fill=(0, 255, 0), font=font)
    draw.text((2 * w + 18, 34), 'Pred Mask (Red)', fill=(255, 0, 0), font=font)

    out_path2 = output_dir / f'{case_id}_comparison.png'
    pil_comp.save(str(out_path2))

    print(f'  {case_id}: saved {out_path.name}, {out_path2.name}')


def create_summary_grid(output_dir):
    """创建 9 个案例的总览图 (3x3)"""
    categories = list(SELECTED_CASES.keys())
    panel_size = (400, 120)  # 缩略图尺寸

    grid_img = Image.new('RGB', (panel_size[0] * 3, panel_size[1] * 3 + 90), (30, 30, 30))
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    for col, cat in enumerate(categories):
        draw.text((col * panel_size[0] + 10, 10), cat, fill=(255, 255, 255), font=font)

    for col, cat in enumerate(categories):
        for row, (case_id, dice) in enumerate(SELECTED_CASES[cat]):
            overview_path = output_dir / f'{case_id}_overview.png'
            if overview_path.exists():
                img = Image.open(str(overview_path))
                img.thumbnail(panel_size)
                x = col * panel_size[0] + (panel_size[0] - img.width) // 2
                y = 40 + row * (panel_size[1] + 10) + (panel_size[1] - img.height) // 2
                grid_img.paste(img, (x, y))

    grid_path = output_dir / 'summary_grid.png'
    grid_img.save(str(grid_path))
    print(f'\nSummary grid saved: {grid_path}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'输出目录: {OUTPUT_DIR}')
    print(f'可视化 {sum(len(v) for v in SELECTED_CASES.values())} 个案例...\n')

    for category, cases in SELECTED_CASES.items():
        print(f'[{category}]')
        for case_id, dice in cases:
            visualize_case(case_id, dice, category, OUTPUT_DIR)

    print('\n生成总览图...')
    create_summary_grid(OUTPUT_DIR)
    print('完成!')


if __name__ == '__main__':
    main()
