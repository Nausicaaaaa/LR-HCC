"""
可视化脚本：将 classification_dataset 中的 masks 叠加绘制在 images 上

用法:
    # 可视化所有 case 的中间切片
    python LIFT/box_mask.py
    
    # 可视化指定 case 的所有切片
    python LIFT/box_mask.py --case_id case_001 --slice_n all
    
    # 可视化指定 case 的指定切片
    python LIFT/box_mask.py --case_id case_005 --slice_n 25
    
    # 自定义窗宽窗位和输出目录
    python LIFT/box_mask.py --wl 50 --ww 300 --output_dir LIFT/mask_visualization
"""

import argparse
from pathlib import Path
from typing import Union, Optional

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

MODALITY_LABELS = {
    '0000': 'A',
    '0001': 'D',
    '0002': 'P',
    '0003': 'S/PS',
}

ANNOTATION_FONT_SIZE = 14
TITLE_FONT_SIZE = 18
BORDER_SIZE = 10
SLICE_GAP = 4

# 不同标签类别的轮廓颜色 (R, G, B)
CONTOUR_COLORS = {
    1: (255, 0, 0),      # 红色
    2: (0, 255, 0),      # 绿色
    3: (0, 0, 255),      # 蓝色
    4: (255, 255, 0),    # 黄色
}


def hu_window(ct_slice: np.ndarray, window_level: float, window_width: float) -> np.ndarray:
    """对 CT 切片应用窗宽窗位，并映射到 8 位灰度图。"""
    low = window_level - window_width / 2.0
    high = window_level + window_width / 2.0
    clipped = np.clip(ct_slice, low, high)
    norm = (clipped - low) / (high - low + 1e-8)
    return (norm * 255).astype(np.uint8)


def load_nii(path: Path) -> np.ndarray:
    """以 float32 数组形式读取 NIfTI 文件。"""
    nii = nib.load(str(path))
    return nii.get_fdata(dtype=np.float32)


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


def extract_contours(mask_slice: np.ndarray) -> dict:
    """从 mask 切片中提取各标签类别的轮廓。

    返回 dict: label_value -> contour_mask (bool array)
    """
    contours = {}
    unique_labels = np.unique(mask_slice)
    for label in unique_labels:
        if label == 0:
            continue
        binary = mask_slice == label
        if not binary.any():
            continue
        # 使用形态学腐蚀找边界
        eroded = ndi.binary_erosion(binary)
        contour = binary & (~eroded)
        if contour.any():
            contours[int(label)] = contour
    return contours


def overlay_contours(rgb: np.ndarray, contours: dict, colors: dict = CONTOUR_COLORS) -> np.ndarray:
    """将多类别轮廓 overlay 到 RGB 图像上。"""
    result = rgb.copy()
    for label, contour_mask in contours.items():
        color = colors.get(label, (255, 0, 0))
        result[contour_mask] = color
    return result


def to_rgb(ct_u8: np.ndarray) -> np.ndarray:
    """将灰度图转为 RGB。"""
    return np.stack([ct_u8, ct_u8, ct_u8], axis=-1)


def add_white_border(image: np.ndarray, border_size: int = BORDER_SIZE) -> np.ndarray:
    """为 RGB 图像四周添加白色边框。"""
    return np.pad(image, ((border_size, border_size), (border_size, border_size), (0, 0)), 
                  mode='constant', constant_values=255)


def make_montage_1x4(images):
    """将四张图横向拼接成 1x4 图像。"""
    if len(images) != 4:
        raise ValueError(f'Expected 4 images, got {len(images)}')
    return np.concatenate(images, axis=1)


def add_title_bar(montage: np.ndarray, case_id: str, slice_idx: int = None) -> np.ndarray:
    """在拼接图顶部添加标题条，显示 case_id 和 slice 信息。"""
    h, w = montage.shape[:2]
    bar_h = 32
    bar = np.full((bar_h, w, 3), 30, dtype=np.uint8)
    image = Image.fromarray(bar)
    draw = ImageDraw.Draw(image)
    font = load_annotation_font(TITLE_FONT_SIZE)

    title = f'{case_id}'
    if slice_idx is not None:
        title += f'  |  Slice: {slice_idx}'
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max((w - text_w) // 2, 0)
    y = max((bar_h - text_h) // 2, 0)
    draw.text((x, y), title, fill=(255, 255, 255), font=font)

    bar_rgb = np.asarray(image)
    return np.concatenate([bar_rgb, montage], axis=0)


def _load_modality_volumes(case_dir: Path) -> dict:
    """读取 case 目录下的 4 个模态体积。"""
    modality_files = {}
    for f in sorted(case_dir.iterdir()):
        if f.is_file() and (f.name.endswith('.nii.gz') or f.name.endswith('.nii')):
            mod_key = f.name.split('.')[0]
            modality_files[mod_key] = f

    if len(modality_files) != 4:
        raise ValueError(
            f'Case does not have exactly 4 modalities, found {len(modality_files)}'
        )

    modality_vols = {}
    for mod_key in sorted(modality_files.keys()):
        modality_vols[mod_key] = load_nii(modality_files[mod_key])
    return modality_vols


def render_modality_slice(image_vol: np.ndarray, slice_idx: int, wl: float, ww: float) -> np.ndarray:
    """渲染单个模态的指定切片为灰度图。"""
    z_dim = image_vol.shape[2]
    if slice_idx < 0 or slice_idx >= z_dim:
        raise ValueError(f'Slice index {slice_idx} out of range [0, {z_dim - 1}]')
    ct_slice = image_vol[:, :, slice_idx]
    return hu_window(ct_slice, wl, ww)


def _render_single_slice_with_mask(
    modality_vols: dict,
    mask_vol: np.ndarray,
    slice_idx: int,
    wl: float,
    ww: float,
) -> np.ndarray:
    """渲染单层 1x4 模态拼接图，并叠加 mask 轮廓（有白边）。"""
    images = []
    
    # 提取当前切片的 mask 轮廓
    contours = None
    if slice_idx < mask_vol.shape[2]:
        contours = extract_contours(mask_vol[:, :, slice_idx])

    for mod_key in sorted(modality_vols.keys()):
        vol = modality_vols[mod_key]
        ct_u8 = render_modality_slice(vol, slice_idx, wl, ww)
        rgb = to_rgb(ct_u8)
        
        # 如果有轮廓，则叠加
        if contours:
            rgb = overlay_contours(rgb, contours)
        
        rgb = add_white_border(rgb)
        images.append(rgb)
    
    return make_montage_1x4(images)


def render_case_slice_with_mask(
    case_id: str,
    case_dir: Path,
    mask_vol: np.ndarray,
    slice_idx: int,
    wl: float,
    ww: float,
) -> np.ndarray:
    """渲染一个 case 的单层切片，4 模态 1x4 拼接，并叠加 mask。"""
    modality_vols = _load_modality_volumes(case_dir)
    montage = _render_single_slice_with_mask(modality_vols, mask_vol, slice_idx, wl, ww)
    montage_with_title = add_title_bar(montage, case_id, slice_idx)
    return montage_with_title


def render_case_all_slices_with_mask(
    case_id: str,
    case_dir: Path,
    mask_vol: np.ndarray,
    wl: float,
    ww: float,
) -> np.ndarray:
    """渲染一个 case 的所有切片到一张大图上，按 slice 顺序纵向排列。"""
    modality_vols = _load_modality_volumes(case_dir)
    z_dim = list(modality_vols.values())[0].shape[2]

    slice_montages = []
    for idx in range(z_dim):
        m = _render_single_slice_with_mask(modality_vols, mask_vol, idx, wl, ww)
        slice_montages.append(m)
        # 层与层之间添加灰色分隔线
        if idx < z_dim - 1:
            gap = np.full((SLICE_GAP, m.shape[1], 3), 200, dtype=np.uint8)
            slice_montages.append(gap)

    full_image = np.concatenate(slice_montages, axis=0)
    full_image_with_title = add_title_bar(full_image, case_id)
    return full_image_with_title


def main():
    parser = argparse.ArgumentParser(
        description='Visualize classification dataset with masks overlaid on images.'
    )
    parser.add_argument(
        '--image_dir',
        type=Path,
        default=Path('LIFT/data/images'),
        help='Directory containing case_xxx subdirectories with modality NIfTI files.',
    )
    parser.add_argument(
        '--mask_dir',
        type=Path,
        default=Path('LIFT/data/masks'),
        help='Directory containing case_xxx subdirectories with mask NIfTI files.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('visualization_output/box_mask'),
        help='Output directory for PNG images.',
    )
    parser.add_argument(
        '--case_id',
        type=str,
        default=None,
        help='Specific case id to visualize, e.g. case_001. If not set, all cases will be processed.',
    )
    parser.add_argument(
        '--slice_n',
        type=str,
        default='middle',
        help='Slice index to visualize, or "all" to save all slices, or "middle" for middle slice.',
    )
    parser.add_argument('--wl', type=float, default=60.0, help='Window level for CT display.')
    parser.add_argument('--ww', type=float, default=400.0, help='Window width for CT display.')
    args = parser.parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f'Image directory not found: {args.image_dir}')
    if not args.mask_dir.exists():
        raise FileNotFoundError(f'Mask directory not found: {args.mask_dir}')

    # 确定要处理的 case 列表
    if args.case_id:
        case_dirs = [args.image_dir / args.case_id]
        if not case_dirs[0].exists():
            raise FileNotFoundError(f'Case directory not found: {case_dirs[0]}')
    else:
        case_dirs = sorted([d for d in args.image_dir.iterdir() if d.is_dir()])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Processing {len(case_dirs)} cases...')
    print(f'Output directory: {output_dir.resolve()}')
    print('-' * 80)

    for case_dir in case_dirs:
        case_id = case_dir.name
        
        # 检查是否有 NIfTI 文件
        mod_files = [
            f for f in case_dir.iterdir()
            if f.is_file() and f.name.endswith(('.nii', '.nii.gz'))
        ]
        if not mod_files:
            print(f'Warning: No NIfTI files found in {case_dir}, skipping')
            continue

        # 加载第一个模态以确定维度
        first_vol = load_nii(sorted(mod_files)[0])
        z_dim = first_vol.shape[2]

        # 加载对应的 mask
        mask_case_dir = args.mask_dir / case_id
        if not mask_case_dir.exists():
            print(f'Warning: Mask directory not found for {case_id}: {mask_case_dir}, skipping')
            continue
        
        mask_files = [
            f for f in mask_case_dir.iterdir()
            if f.is_file() and f.name.endswith(('.nii', '.nii.gz'))
        ]
        if not mask_files:
            print(f'Warning: No mask files found in {mask_case_dir}, skipping')
            continue
        
        # 加载 mask（假设每个 case 只有一个 mask 文件）
        mask_path = sorted(mask_files)[0]
        try:
            mask_vol = load_nii(mask_path)
            
            # 检查 mask 和 image 的空间维度是否匹配（至少前两个维度应该一致）
            if mask_vol.shape[:2] != first_vol.shape[:2]:
                print(f'Warning: Mask shape {mask_vol.shape} mismatch with image {first_vol.shape} for {case_id}, skipping')
                continue
        except Exception as e:
            print(f'Warning: Failed to load mask for {case_id}: {e}, skipping')
            continue

        # 根据 slice_n 参数决定渲染方式
        if args.slice_n.lower() == 'all':
            case_output_dir = output_dir / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)
            full_image = render_case_all_slices_with_mask(
                case_id, case_dir, mask_vol, args.wl, args.ww
            )
            out_path = case_output_dir / f'{case_id}-all-slices.png'
            Image.fromarray(full_image).save(out_path)
            print(f'✓ Saved all {z_dim} slices for {case_id} -> {out_path}')
        else:
            # 确定切片索引
            if args.slice_n.lower() == 'middle':
                slice_idx = z_dim // 2
            else:
                try:
                    slice_idx = int(args.slice_n)
                except ValueError:
                    print(f'Warning: Invalid slice_n value "{args.slice_n}", using middle slice')
                    slice_idx = z_dim // 2
            
            # 检查切片索引是否有效
            if slice_idx < 0 or slice_idx >= z_dim:
                print(f'Warning: Slice index {slice_idx} out of range [0, {z_dim-1}] for {case_id}, skipping')
                continue

            montage = render_case_slice_with_mask(
                case_id, case_dir, mask_vol, slice_idx, args.wl, args.ww
            )
            case_output_dir = output_dir / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)
            out_path = case_output_dir / f'{case_id}-slice-{slice_idx:03d}.png'
            Image.fromarray(montage).save(out_path)
            print(f'✓ Saved {case_id} slice {slice_idx} -> {out_path}')

    print('-' * 80)
    print(f'Done! All visualizations saved to: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
