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


def get_roi_bbox_from_label(label_path: Path):
    """从标签文件中提取ROI的外接框（bbox为闭区间）。"""
    label_img = nib.load(str(label_path))
    label_data = label_img.get_fdata()
    nonzero_coords = np.where(label_data > 0)
    if len(nonzero_coords[0]) == 0:
        raise ValueError(f'标签文件中没有ROI区域: {label_path}')
    x_min = int(np.min(nonzero_coords[0]))
    x_max = int(np.max(nonzero_coords[0]))
    y_min = int(np.min(nonzero_coords[1]))
    y_max = int(np.max(nonzero_coords[1]))
    z_min = int(np.min(nonzero_coords[2]))
    z_max = int(np.max(nonzero_coords[2]))
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def crop_volume_by_bbox(vol_data: np.ndarray, bbox, margin=(15, 15, 15)) -> np.ndarray:
    """基于ROI外接框+Margin裁切体积数据。

    Args:
        vol_data: 3D数组，维度为 (x, y, z)
        bbox: (x_min, x_max, y_min, y_max, z_min, z_max) 闭区间
        margin: (dz, dy, dx) 对应 (z, y, x) 方向
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    dz_margin, dy_margin, dx_margin = margin
    x_start = max(0, x_min - dx_margin)
    x_end = min(vol_data.shape[0] - 1, x_max + dx_margin)
    y_start = max(0, y_min - dy_margin)
    y_end = min(vol_data.shape[1] - 1, y_max + dy_margin)
    z_start = max(0, z_min - dz_margin)
    z_end = min(vol_data.shape[2] - 1, z_max + dz_margin)
    return vol_data[x_start:x_end + 1, y_start:y_end + 1, z_start:z_end + 1]


def load_and_crop_mask(mask_path: Path, margin=(0, 15, 15)) -> np.ndarray:
    """加载完整mask并按ROI外接框+Margin裁切。"""
    bbox = get_roi_bbox_from_label(mask_path)
    mask_data = nib.load(str(mask_path)).get_fdata()
    return crop_volume_by_bbox(mask_data, bbox, margin=margin)


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


def load_all_labels(label_dir: Path) -> dict:
    """读取 labels 目录下所有 txt 文件，合并为 case_id -> label 的映射。"""
    label_map = {}
    for txt_file in sorted(label_dir.glob('*.txt')):
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    case_id = parts[0]
                    try:
                        label = int(parts[1])
                        label_map[case_id] = label
                    except ValueError:
                        continue
    return label_map


def render_modality_slice(image_vol: np.ndarray, slice_idx: int, wl: float, ww: float) -> np.ndarray:
    """渲染单个模态的指定切片为灰度图。"""
    z_dim = image_vol.shape[2]
    if slice_idx < 0 or slice_idx >= z_dim:
        raise ValueError(f'Slice index {slice_idx} out of range [0, {z_dim - 1}]')
    ct_slice = image_vol[:, :, slice_idx]
    return hu_window(ct_slice, wl, ww)


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
    return np.pad(image, ((border_size, border_size), (border_size, border_size), (0, 0)), mode='constant', constant_values=255)


def make_montage_1x4(images):
    """将四张图横向拼接成 1x4 图像。"""
    if len(images) != 4:
        raise ValueError(f'Expected 4 images, got {len(images)}')
    return np.concatenate(images, axis=1)


def add_title_bar(montage: np.ndarray, case_id: str, label: int, slice_idx: int = None) -> np.ndarray:
    """在拼接图顶部添加标题条，显示 case_id、label。"""
    h, w = montage.shape[:2]
    bar_h = 32
    bar = np.full((bar_h, w, 3), 30, dtype=np.uint8)
    image = Image.fromarray(bar)
    draw = ImageDraw.Draw(image)
    font = load_annotation_font(TITLE_FONT_SIZE)

    title = f'{case_id}  |  Label: {label}'
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


def _render_single_slice_montage(
    modality_vols: dict,
    slice_idx: int,
    wl: float,
    ww: float,
    mask_vol: Optional[np.ndarray] = None,
) -> np.ndarray:
    """渲染单层 1x4 模态拼接图（可选 mask 轮廓，有白边）。"""
    images = []
    contours = None
    if mask_vol is not None:
        if slice_idx < mask_vol.shape[2]:
            contours = extract_contours(mask_vol[:, :, slice_idx])

    for mod_key in sorted(modality_vols.keys()):
        vol = modality_vols[mod_key]
        ct_u8 = render_modality_slice(vol, slice_idx, wl, ww)
        rgb = to_rgb(ct_u8)
        if contours:
            rgb = overlay_contours(rgb, contours)
        rgb = add_white_border(rgb)
        images.append(rgb)
    return make_montage_1x4(images)


def render_case_slice(
    case_id: str,
    case_dir: Path,
    label: int,
    slice_idx: int,
    wl: float,
    ww: float,
    mask_vol: Optional[np.ndarray] = None,
) -> np.ndarray:
    """渲染一个 case 的单层切片，4 模态 1x4 拼接。"""
    modality_vols = _load_modality_volumes(case_dir)
    montage = _render_single_slice_montage(modality_vols, slice_idx, wl, ww, mask_vol)
    montage_with_title = add_title_bar(montage, case_id, label, slice_idx)
    return montage_with_title


def render_case_all_slices(
    case_id: str,
    case_dir: Path,
    label: int,
    wl: float,
    ww: float,
    mask_vol: Optional[np.ndarray] = None,
) -> np.ndarray:
    """渲染一个 case 的所有切片到一张大图上，按 slice 顺序纵向排列。"""
    modality_vols = _load_modality_volumes(case_dir)
    z_dim = list(modality_vols.values())[0].shape[2]

    slice_montages = []
    for idx in range(z_dim):
        m = _render_single_slice_montage(modality_vols, idx, wl, ww, mask_vol)
        slice_montages.append(m)
        # 层与层之间添加灰色分隔线
        if idx < z_dim - 1:
            gap = np.full((SLICE_GAP, m.shape[1], 3), 200, dtype=np.uint8)
            slice_montages.append(gap)

    full_image = np.concatenate(slice_montages, axis=0)
    full_image_with_title = add_title_bar(full_image, case_id, label)
    return full_image_with_title


def main():
    parser = argparse.ArgumentParser(
        description='Visualize classification dataset NIfTI images with labels.'
    )
    parser.add_argument(
        '--image_dir',
        type=Path,
        default=Path('LIFT/data/classification_dataset/images'),
        help='Directory containing case_xxx subdirectories with modality NIfTI files.',
    )
    parser.add_argument(
        '--label_dir',
        type=Path,
        default=Path('LIFT/data/classification_dataset/labels'),
        help='Directory containing label txt files (train_fold*.txt, val_fold*.txt).',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('LIFT/visualization'),
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
    parser.add_argument(
        '--mask_dir',
        type=Path,
        default=Path('data/nnUNet_raw/Dataset001_HCC/labelsTr'),
        help='Directory containing segmentation mask NIfTI files (case_xxx.nii.gz).',
    )
    parser.add_argument(
        '--no_contour',
        action='store_true',
        help='Disable contour overlay on images.',
    )
    args = parser.parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f'Image directory not found: {args.image_dir}')
    if not args.label_dir.exists():
        raise FileNotFoundError(f'Label directory not found: {args.label_dir}')

    label_map = load_all_labels(args.label_dir)
    print(f'Loaded labels for {len(label_map)} cases from {args.label_dir}')

    if args.case_id:
        case_dirs = [args.image_dir / args.case_id]
        if not case_dirs[0].exists():
            raise FileNotFoundError(f'Case directory not found: {case_dirs[0]}')
    else:
        case_dirs = sorted([d for d in args.image_dir.iterdir() if d.is_dir()])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # mask 目录存在性检查
    has_mask_dir = not args.no_contour and args.mask_dir and args.mask_dir.exists()

    for case_dir in case_dirs:
        case_id = case_dir.name
        if case_id not in label_map:
            print(f'Warning: No label found for {case_id}, skipping')
            continue

        label = label_map[case_id]

        mod_files = [
            f for f in case_dir.iterdir()
            if f.is_file() and f.name.endswith(('.nii', '.nii.gz'))
        ]
        if not mod_files:
            print(f'Warning: No NIfTI files found in {case_dir}, skipping')
            continue

        first_vol = load_nii(sorted(mod_files)[0])
        z_dim = first_vol.shape[2]

        # 加载并裁切 mask（按与预处理相同的 ROI+Margin 逻辑）
        mask_vol = None
        if has_mask_dir:
            mask_file = args.mask_dir / f'{case_id}.nii.gz'
            if mask_file.exists():
                try:
                    mask_vol = load_and_crop_mask(mask_file, margin=(0, 15, 15))
                    if mask_vol.shape[:2] != first_vol.shape[:2]:
                        print(f'Warning: Cropped mask shape {mask_vol.shape} mismatch with image {first_vol.shape} for {case_id}, skipping contour')
                        mask_vol = None
                except Exception as e:
                    print(f'Warning: Failed to load/crop mask for {case_id}: {e}')
            else:
                print(f'Warning: Mask file not found for {case_id}: {mask_file}')

        if args.slice_n.lower() == 'all':
            case_output_dir = output_dir / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)
            full_image = render_case_all_slices(case_id, case_dir, label, args.wl, args.ww, mask_vol)
            out_path = case_output_dir / f'{case_id}-all-slices.png'
            Image.fromarray(full_image).save(out_path)
            print(f'Saved all {z_dim} slices in one image for {case_id} -> {out_path}')
        else:
            if args.slice_n.lower() == 'middle':
                slice_idx = z_dim // 2
            else:
                slice_idx = int(args.slice_n)

            montage = render_case_slice(case_id, case_dir, label, slice_idx, args.wl, args.ww, mask_vol)
            case_output_dir = output_dir / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)
            out_path = case_output_dir / f'{case_id}-slice-{slice_idx:03d}.png'
            Image.fromarray(montage).save(out_path)
            print(f'Saved {case_id} slice {slice_idx} -> {out_path}')


if __name__ == '__main__':
    main()
