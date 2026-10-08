"""
可视化对比脚本：同时显示真实标签 ROI 和预测 ROI
用于比较验证集上的分割结果差异

用法:
    # 对比单个 case 的所有切片
    python visualization_prediction.py --case_id case_001 --slice_n=all
    
    # 对比单个 case 的指定切片
    python visualization_prediction.py --case_id case_001 --slice_n 50
    
    # 指定不同的 fold 或配置
    python visualization_prediction.py --case_id case_001 --slice_n all --fold 0 --configuration 2d

    默认参数：数据集 001 , slice_n=all, fold=0, configuration=2d 
    输出路径：visualization_output
"""

import argparse
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODALITY_LABELS = {
    '0000': 'A',
    '0001': 'D',
    '0002': 'P',
    '0003': 'S/PS',
}
ANNOTATION_FONT_SIZE = 20


def strip_nii_suffix(filename: str) -> str:
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    if filename.endswith('.nii'):
        return filename[:-4]
    return Path(filename).stem


def hu_window(ct_slice: np.ndarray, window_level: float, window_width: float) -> np.ndarray:
    """对 CT 切片应用窗宽窗位，并映射到 8 位灰度图。"""
    low = window_level - window_width / 2.0
    high = window_level + window_width / 2.0
    clipped = np.clip(ct_slice, low, high)
    norm = (clipped - low) / (high - low + 1e-8)
    return (norm * 255).astype(np.uint8)


def inner_pixels(mask: np.ndarray) -> np.ndarray:
    """返回四邻域都位于掩膜内部的像素。"""
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return inner


def dilate4(mask: np.ndarray, iterations: int) -> np.ndarray:
    """使用四邻域对二值掩膜做膨胀。"""
    out = mask.copy()
    for _ in range(max(0, iterations)):
        expanded = out.copy()
        expanded[1:, :] |= out[:-1, :]
        expanded[:-1, :] |= out[1:, :]
        expanded[:, 1:] |= out[:, :-1]
        expanded[:, :-1] |= out[:, 1:]
        out = expanded
    return out


def overlay_contour(
    ct_u8: np.ndarray,
    label_slice: np.ndarray,
    contour_color=(255, 0, 0),
    contour_thickness: int = 1,
) -> np.ndarray:
    """在灰度 CT 切片上叠加 ROI 轮廓。"""
    mask = label_slice > 0
    rgb = np.stack([ct_u8, ct_u8, ct_u8], axis=-1)
    if not np.any(mask):
        return rgb

    boundary = mask & (~inner_pixels(mask))
    if contour_thickness > 1:
        boundary = dilate4(boundary, contour_thickness - 1)

    rgb[boundary] = np.array(contour_color, dtype=np.uint8)
    return rgb


def overlay_filled_region(
    ct_u8: np.ndarray,
    label_slice: np.ndarray,
    fill_color=(255, 0, 0),
    alpha: float = 0.3,
) -> np.ndarray:
    """在 CT 切片上叠加半透明的 ROI 填充区域。"""
    mask = label_slice > 0
    rgb = np.stack([ct_u8, ct_u8, ct_u8], axis=-1).astype(np.float32)
    
    if not np.any(mask):
        return rgb.astype(np.uint8)
    
    # 创建彩色填充
    color_array = np.array(fill_color, dtype=np.float32)
    for c in range(3):
        rgb[:, :, c] = np.where(
            mask,
            rgb[:, :, c] * (1 - alpha) + color_array[c] * alpha,
            rgb[:, :, c]
        )
    
    return rgb.astype(np.uint8)


def load_nii(path: Path) -> np.ndarray:
    """以 float32 数组形式读取 NIfTI 文件。"""
    nii = nib.load(str(path))
    return nii.get_fdata(dtype=np.float32)


def split_case_and_modality(base_name: str):
    """将 case_001_0000 这类名称拆分为 case_id 和模态编号。"""
    if '_' not in base_name:
        return None, None
    case_id, modality = base_name.rsplit('_', 1)
    if not modality.isdigit():
        return None, None
    return case_id, modality


def find_case_groups(image_dir: Path):
    """按 case_id 和模态编号对 imageTr 下的文件分组。"""
    groups = {}
    for p in sorted(image_dir.iterdir()):
        if not p.is_file():
            continue
        if not (p.name.endswith('.nii') or p.name.endswith('.nii.gz')):
            continue

        case_id, modality = split_case_and_modality(strip_nii_suffix(p.name))
        if case_id is None:
            continue

        if case_id not in groups:
            groups[case_id] = {}
        groups[case_id][modality] = p

    return groups


def find_label_path(label_dir: Path, case_id: str):
    """查找与指定 case_id 对应的标签文件。"""
    c1 = label_dir / f'{case_id}.nii.gz'
    c2 = label_dir / f'{case_id}.nii'
    if c1.exists():
        return c1
    if c2.exists():
        return c2
    return None


def find_prediction_path(pred_dir: Path, case_id: str):
    """查找与指定 case_id 对应的预测结果文件。"""
    c1 = pred_dir / f'{case_id}.nii.gz'
    c2 = pred_dir / f'{case_id}.nii'
    if c1.exists():
        return c1
    if c2.exists():
        return c2
    return None


def load_annotation_font(font_size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """加载较大的标注字体，失败时回退到默认字体。"""
    candidate_fonts = [
        'arial.ttf',
        'Arial.ttf',
        'DejaVuSans.ttf',
    ]
    for font_name in candidate_fonts:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_image(
    rgb_slice: np.ndarray,
    title: str,
    color=(255, 255, 255),
) -> np.ndarray:
    """在图像上绘制标题文字。"""
    image = Image.fromarray(rgb_slice)
    draw = ImageDraw.Draw(image)
    font = load_annotation_font(ANNOTATION_FONT_SIZE)
    padding = 4

    # 在左上角绘制标题
    draw.text((padding, padding), title, fill=color, font=font)

    return np.asarray(image)


def create_comparison_image(
    ct_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    window_level: float,
    window_width: float,
    contour_thickness: int,
    show_filled: bool = True,
) -> np.ndarray:
    """
    创建对比图像：左图为真实标签(GT)，右图为预测结果(Pred)
    
    颜色说明:
    - 绿色轮廓/填充: 真实标签 (Ground Truth)
    - 红色轮廓/填充: 预测结果 (Prediction)
    - 黄色区域: GT 和 Pred 重叠部分
    """
    ct_u8 = hu_window(ct_slice, window_level, window_width)
    
    # 左图: 真实标签 (绿色)
    if show_filled:
        left_img = overlay_filled_region(ct_u8, gt_slice, fill_color=(0, 255, 0), alpha=0.3)
    else:
        left_img = overlay_contour(ct_u8, gt_slice, contour_color=(0, 255, 0), contour_thickness=contour_thickness)
    left_img = annotate_image(left_img, "Ground Truth (Green)", color=(0, 255, 0))
    
    # 右图: 预测结果 (红色)
    if show_filled:
        right_img = overlay_filled_region(ct_u8, pred_slice, fill_color=(255, 0, 0), alpha=0.3)
    else:
        right_img = overlay_contour(ct_u8, pred_slice, contour_color=(255, 0, 0), contour_thickness=contour_thickness)
    right_img = annotate_image(right_img, "Prediction (Red)", color=(255, 0, 0))
    
    # 合并为左右对比图
    comparison = np.concatenate([left_img, right_img], axis=1)
    
    return comparison


def create_overlay_image(
    ct_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    window_level: float,
    window_width: float,
    contour_thickness: int,
) -> np.ndarray:
    """
    创建叠加图像：在同一图上显示 GT 和 Pred 的对比
    
    颜色说明:
    - 绿色: 只有真实标签有的区域 (True Positive + False Negative)
    - 红色: 只有预测结果有的区域 (False Positive)
    - 黄色: GT 和 Pred 都有的区域 (True Positive)
    """
    ct_u8 = hu_window(ct_slice, window_level, window_width)
    rgb = np.stack([ct_u8, ct_u8, ct_u8], axis=-1).astype(np.float32)
    
    gt_mask = gt_slice > 0
    pred_mask = pred_slice > 0
    
    # 定义颜色
    GREEN = np.array([0, 255, 0], dtype=np.float32)  # 只有 GT
    RED = np.array([255, 0, 0], dtype=np.float32)    # 只有 Pred
    YELLOW = np.array([255, 255, 0], dtype=np.float32)  # 两者都有
    
    alpha = 0.4
    
    # TP: GT 和 Pred 都有 -> 黄色
    tp_mask = gt_mask & pred_mask
    # FN: 只有 GT 有 -> 绿色
    fn_mask = gt_mask & (~pred_mask)
    # FP: 只有 Pred 有 -> 红色
    fp_mask = (~gt_mask) & pred_mask
    
    for c in range(3):
        rgb[:, :, c] = np.where(
            tp_mask,
            rgb[:, :, c] * (1 - alpha) + YELLOW[c] * alpha,
            rgb[:, :, c]
        )
        rgb[:, :, c] = np.where(
            fn_mask,
            rgb[:, :, c] * (1 - alpha) + GREEN[c] * alpha,
            rgb[:, :, c]
        )
        rgb[:, :, c] = np.where(
            fp_mask,
            rgb[:, :, c] * (1 - alpha) + RED[c] * alpha,
            rgb[:, :, c]
        )
    
    rgb = rgb.astype(np.uint8)
    
    # 添加轮廓线
    gt_boundary = gt_mask & (~inner_pixels(gt_mask))
    pred_boundary = pred_mask & (~inner_pixels(pred_mask))
    
    if contour_thickness > 1:
        gt_boundary = dilate4(gt_boundary, contour_thickness - 1)
        pred_boundary = dilate4(pred_boundary, contour_thickness - 1)
    
    rgb[gt_boundary] = np.array([0, 200, 0], dtype=np.uint8)  # 深绿色轮廓
    rgb[pred_boundary] = np.array([200, 0, 0], dtype=np.uint8)  # 深红色轮廓
    
    return rgb


def render_comparison_slice(
    case_id: str,
    image_vol: np.ndarray,
    gt_vol: np.ndarray,
    pred_vol: np.ndarray,
    slice_idx: int,
    window_level: float,
    window_width: float,
    contour_thickness: int,
    show_filled: bool = True,
) -> np.ndarray:
    """渲染单个切片的对比图，三张图摆成一行。"""
    z_dim = gt_vol.shape[2]

    if slice_idx < 0 or slice_idx >= z_dim:
        raise ValueError(
            f'Slice index out of range for {case_id}: {slice_idx}, valid [0, {z_dim - 1}]'
        )

    gt_slice = gt_vol[:, :, slice_idx]
    pred_slice = pred_vol[:, :, slice_idx]
    ct_slice = image_vol[:, :, slice_idx]
    ct_u8 = hu_window(ct_slice, window_level, window_width)
    
    # 左图: 真实标签 (绿色)
    if show_filled:
        left_img = overlay_filled_region(ct_u8, gt_slice, fill_color=(0, 255, 0), alpha=0.3)
    else:
        left_img = overlay_contour(ct_u8, gt_slice, contour_color=(0, 255, 0), contour_thickness=contour_thickness)

    # 中图: 预测结果 (红色)
    if show_filled:
        middle_img = overlay_filled_region(ct_u8, pred_slice, fill_color=(255, 0, 0), alpha=0.3)
    else:
        middle_img = overlay_contour(ct_u8, pred_slice, contour_color=(255, 0, 0), contour_thickness=contour_thickness)

    # 右图: 叠加对比图
    right_img = create_overlay_image(
        ct_slice, gt_slice, pred_slice,
        window_level, window_width, contour_thickness
    )

    # 对三个子图逆时针旋转90度
    left_img = np.rot90(left_img, k=1)
    middle_img = np.rot90(middle_img, k=1)
    right_img = np.rot90(right_img, k=1)

    # 旋转后添加注释（保持在左上角）
    left_img = annotate_image(left_img, "Ground Truth", color=(0, 255, 0))
    middle_img = annotate_image(middle_img, "Prediction", color=(255, 0, 0))
    right_img = annotate_image(right_img, "Overlay (Y=TP,G=FN,R=FP)", color=(255, 255, 255))

    # 确保三张图高度一致
    h1, w1 = left_img.shape[:2]
    h2, w2 = middle_img.shape[:2]
    h3, w3 = right_img.shape[:2]
    max_h = max(h1, h2, h3)

    if h1 < max_h:
        pad = np.zeros((max_h - h1, w1, 3), dtype=np.uint8)
        left_img = np.concatenate([left_img, pad], axis=0)
    if h2 < max_h:
        pad = np.zeros((max_h - h2, w2, 3), dtype=np.uint8)
        middle_img = np.concatenate([middle_img, pad], axis=0)
    if h3 < max_h:
        pad = np.zeros((max_h - h3, w3, 3), dtype=np.uint8)
        right_img = np.concatenate([right_img, pad], axis=0)

    # 三张图横向拼接
    final_image = np.concatenate([left_img, middle_img, right_img], axis=1)
    
    # 添加切片信息
    image = Image.fromarray(final_image)
    draw = ImageDraw.Draw(image)
    font = load_annotation_font(ANNOTATION_FONT_SIZE)
    
    # 在底部添加切片编号
    slice_text = f'{case_id} - Slice {slice_idx}/{z_dim-1}'
    bbox = draw.textbbox((0, 0), slice_text, font=font)
    text_w = bbox[2] - bbox[0]
    img_w = image.width
    draw.text(((img_w - text_w) // 2, image.height - ANNOTATION_FONT_SIZE - 4), 
              slice_text, fill=(255, 255, 255), font=font)
    
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser(
        description='可视化对比真实标签和预测结果 ROI'
    )
    parser.add_argument(
        '--dataset_id',
        type=str,
        default='001',
        help='数据集ID (默认: 001)',
    )
    parser.add_argument(
        '--case_id',
        type=str,
        required=True,
        help='Case id to visualize, e.g. case_001.',
    )
    parser.add_argument(
        '--slice_n',
        type=str,
        default=None,
        help='Z-axis slice index n, or "all" to save all slices of this case. If not provided, only slices with GT or Pred ROI will be saved.',
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help='使用哪一折的验证结果 (默认: 0)',
    )
    parser.add_argument(
        '--configuration',
        type=str,
        default='2d',
        choices=['2d', '3d_fullres', '3d_lowres'],
        help='网络配置 (默认: 2d)',
    )
    parser.add_argument(
        '--trainer',
        type=str,
        default='nnUNetTrainer_smallROI',
        help='训练器名称 (默认: nnUNetTrainer_smallROI)',
    )
    parser.add_argument(
        '--data_base',
        type=Path,
        default=Path('/mnt/data/KASR/Dengsiyi/LR-HCC/data'),
        help='数据根目录',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('./result_test/seg_comparison'),
        help='输出目录',
    )
    parser.add_argument('--wl', type=float, default=60.0, help='Window level for CT display.')
    parser.add_argument('--ww', type=float, default=400.0, help='Window width for CT display.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Contour thickness in pixels (>=1).',
    )
    parser.add_argument(
        '--contour_only',
        action='store_true',
        help='只显示轮廓，不显示填充区域',
    )
    args = parser.parse_args()

    if args.thickness < 1:
        raise ValueError('--thickness must be >= 1')

    # 构建路径
    raw_dir = args.data_base / f'nnUNet_raw/Dataset{args.dataset_id}_HCC'
    pred_dir = args.data_base / f'nnUNet_results/Dataset{args.dataset_id}_HCC/{args.trainer}__nnUNetPlans__{args.configuration}/fold_{args.fold}/validation'
    
    if not raw_dir.exists():
        raise FileNotFoundError(f'Raw data directory not found: {raw_dir}')
    if not pred_dir.exists():
        raise FileNotFoundError(f'Prediction directory not found: {pred_dir}\n请确保已经运行过验证或预测!')

    image_dir = raw_dir / 'imagesTr'
    label_dir = raw_dir / 'labelsTr'
    
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f'Expected subfolders imagesTr and labelsTr under: {raw_dir}')

    # 查找 case 的图像文件 (使用 0000 模态)
    image_path = image_dir / f'{args.case_id}_0000.nii.gz'
    if not image_path.exists():
        image_path = image_dir / f'{args.case_id}_0000.nii'
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found for case: {args.case_id} (expected {args.case_id}_0000.nii.gz)')

    # 查找标签文件
    gt_path = find_label_path(label_dir, args.case_id)
    if gt_path is None:
        raise FileNotFoundError(f'Ground truth label not found for case: {args.case_id}')

    # 查找预测文件
    pred_path = find_prediction_path(pred_dir, args.case_id)
    if pred_path is None:
        raise FileNotFoundError(f'Prediction not found for case: {args.case_id} in {pred_dir}')

    print(f'Loading image: {image_path}')
    print(f'Loading ground truth: {gt_path}')
    print(f'Loading prediction: {pred_path}')

    # 加载数据
    image_vol = load_nii(image_path)
    gt_vol = load_nii(gt_path)
    pred_vol = load_nii(pred_path)
    
    # 检查形状是否匹配
    if image_vol.shape != gt_vol.shape:
        raise ValueError(f'Shape mismatch: image {image_vol.shape} vs GT {gt_vol.shape}')
    if image_vol.shape != pred_vol.shape:
        raise ValueError(f'Shape mismatch: image {image_vol.shape} vs prediction {pred_vol.shape}')

    # 创建输出目录
    case_output_dir = args.output_dir / args.case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    show_filled = not args.contour_only

    if args.slice_n is None:
        z_dim = gt_vol.shape[2]
        print(f'Processing slices with ROI (GT or Pred)...')

        saved_count = 0
        for idx in range(z_dim):
            gt_slice = gt_vol[:, :, idx]
            pred_slice = pred_vol[:, :, idx]

            # 只保存 GT 或 Pred 至少有一个有内容的层
            if not np.any(gt_slice > 0) and not np.any(pred_slice > 0):
                continue

            montage = render_comparison_slice(
                case_id=args.case_id,
                image_vol=image_vol,
                gt_vol=gt_vol,
                pred_vol=pred_vol,
                slice_idx=idx,
                window_level=args.wl,
                window_width=args.ww,
                contour_thickness=args.thickness,
                show_filled=show_filled,
            )
            out_path = case_output_dir / f'{args.case_id}-slice-{idx:03d}.png'
            Image.fromarray(montage).save(out_path)
            saved_count += 1

            if saved_count % 10 == 0:
                print(f'  Saved {saved_count} slices...')

        print(f'Saved {saved_count} slices for {args.case_id}')
        print(f'Output directory: {case_output_dir.resolve()}')
        return

    if args.slice_n.lower() == 'all':
        z_dim = gt_vol.shape[2]
        print(f'Processing all {z_dim} slices...')

        for idx in range(z_dim):
            montage = render_comparison_slice(
                case_id=args.case_id,
                image_vol=image_vol,
                gt_vol=gt_vol,
                pred_vol=pred_vol,
                slice_idx=idx,
                window_level=args.wl,
                window_width=args.ww,
                contour_thickness=args.thickness,
                show_filled=show_filled,
            )
            out_path = case_output_dir / f'{args.case_id}-slice-{idx:03d}.png'
            Image.fromarray(montage).save(out_path)

            if (idx + 1) % 10 == 0 or idx == z_dim - 1:
                print(f'  Saved {idx + 1}/{z_dim} slices...')

        print(f'Saved all slices for {args.case_id}: {z_dim} images')
        print(f'Output directory: {case_output_dir.resolve()}')
        return

    try:
        slice_idx = int(args.slice_n)
    except ValueError as exc:
        raise ValueError('--slice_n must be an integer or "all"') from exc

    montage = render_comparison_slice(
        case_id=args.case_id,
        image_vol=image_vol,
        gt_vol=gt_vol,
        pred_vol=pred_vol,
        slice_idx=slice_idx,
        window_level=args.wl,
        window_width=args.ww,
        contour_thickness=args.thickness,
        show_filled=show_filled,
    )

    out_path = case_output_dir / f'{args.case_id}-slice-{slice_idx:03d}.png'
    Image.fromarray(montage).save(out_path)
    print(f'Saved: {out_path.resolve()}')


if __name__ == '__main__':
    main()
