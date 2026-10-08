"""
nnUNet 数据集可视化脚本：将 CT 图像与标签叠加显示。

功能：
    读取 nnUNet 格式的 imagesTr 和 labelsTr 数据，
    将 CT 图像按指定窗宽窗位渲染后，叠加 ROI 轮廓，
    生成带标注的 2x2 模态拼图 PNG 图片。
    支持单张切片查看和全部切片批量导出。

使用示例：
    # 查看单个病例的某一层切片
    python visualisation.py --case_id case_001 --slice_n 50

    # 导出单个病例的所有切片
    python visualisation.py --case_id case_1730 --slice_n all

    # 仅渲染单个模态（0000）
    python visualisation.py --case_id case_525 --slice_n all --modality 0

    # 自定义窗宽窗位和轮廓粗细
    python visualisation.py --case_id case_001 --slice_n 50 --wl 60 --ww 400 --thickness 2
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


def resolve_requested_modalities(case_modalities: dict, modality_arg: str) -> dict:
    """根据输入参数确定当前需要渲染哪些模态。"""
    if modality_arg == 'all':
        if len(case_modalities) != 4:
            raise FileNotFoundError('Case does not have 4 modalities in imageTr')
        return case_modalities

    modality_key = f'{int(modality_arg):04d}'
    if modality_key not in case_modalities:
        raise FileNotFoundError(f'Modality {modality_arg} not found for case')
    return {modality_key: case_modalities[modality_key]}


def make_montage_2x2(images):
    """将四张模态图拼接成 2x2 图像。"""
    if len(images) != 4:
        raise ValueError(f'Expected 4 modality images, got {len(images)}')

    top = np.concatenate([images[0], images[1]], axis=1)
    bottom = np.concatenate([images[2], images[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def prepare_case_output_dir(output_dir: Path, case_id: str) -> Path:
    """按病例创建输出目录，不存在时自动新建。"""
    case_output_dir = output_dir / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)
    return case_output_dir


def load_case_volumes(case_id: str, modality_paths: dict, label_path: Path):
    """读取单个病例的标签体积和所需模态体积。"""
    label_vol = load_nii(label_path)
    modality_vols = {}
    for modality in sorted(modality_paths.keys()):
        image_path = modality_paths[modality]
        image_vol = load_nii(image_path)
        if image_vol.shape != label_vol.shape:
            raise ValueError(
                f'Shape mismatch in {case_id}: {image_path.name} {image_vol.shape} vs '
                f'{label_path.name} {label_vol.shape}'
            )
        modality_vols[modality] = image_vol
    return modality_vols, label_vol


def find_label_bboxes(label_slice: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    """为切片上的每个非零标签值计算一个外接框。"""
    bboxes = []
    for label_value in sorted(np.unique(label_slice).astype(int)):
        if label_value <= 0:
            continue

        ys, xs = np.where(label_slice == label_value)
        if ys.size == 0 or xs.size == 0:
            continue

        bboxes.append(
            (
                label_value,
                (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            )
        )
    return bboxes


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


def annotate_slice(
    rgb_slice: np.ndarray,
    modality: str,
    label_slice: np.ndarray,
    color=(255, 0, 0),
) -> np.ndarray:
    """在渲染后的 RGB 切片上绘制模态文字和类别编号。"""
    image = Image.fromarray(rgb_slice)
    draw = ImageDraw.Draw(image)
    font = load_annotation_font(ANNOTATION_FONT_SIZE)
    image_w, image_h = image.size
    padding = 4

    # 在图像右下角标出当前图块对应的模态名称。
    modality_text = MODALITY_LABELS.get(modality, modality)
    modality_bbox = draw.textbbox((0, 0), modality_text, font=font)
    modality_w = modality_bbox[2] - modality_bbox[0]
    modality_h = modality_bbox[3] - modality_bbox[1]
    draw.text(
        (image_w - modality_w - padding, image_h - modality_h - padding),
        modality_text,
        fill=color,
        font=font,
    )

    # 对每个非背景标签，计算 ROI 外接框，并在框的右下角写出类别数字。
    for label_value, (x_min, y_min, x_max, y_max) in find_label_bboxes(label_slice):
        label_text = str(label_value)
        label_bbox = draw.textbbox((0, 0), label_text, font=font)
        label_w = label_bbox[2] - label_bbox[0]
        label_h = label_bbox[3] - label_bbox[1]
        # 直接用外接框右下角作为文字左上角，避免数字压在 ROI 边缘上。
        text_x = min(max(x_max, 0), max(image_w - label_w, 0))
        text_y = min(max(y_max, 0), max(image_h - label_h, 0))
        draw.text((text_x, text_y), label_text, fill=color, font=font)

    return np.asarray(image)


def render_case_slice(
    case_id: str,
    modality_vols: dict,
    label_vol: np.ndarray,
    slice_idx: int,
    window_level: float,
    window_width: float,
    contour_thickness: int,
) -> np.ndarray:
    """渲染一个病例的单层切片，并叠加轮廓和文字标注。"""
    z_dim = label_vol.shape[2]

    if slice_idx < 0 or slice_idx >= z_dim:
        raise ValueError(
            f'Slice index out of range for {case_id}: {slice_idx}, valid [0, {z_dim - 1}]'
        )

    label_slice = label_vol[:, :, slice_idx]
    overlays = []

    for modality in sorted(modality_vols.keys()):
        image_vol = modality_vols[modality]
        ct_slice = image_vol[:, :, slice_idx]
        ct_u8 = hu_window(ct_slice, window_level, window_width)
        overlay = overlay_contour(
            ct_u8,
            label_slice,
            contour_color=(255, 0, 0),
            contour_thickness=contour_thickness,
        )
        overlays.append(annotate_slice(overlay, modality=modality, label_slice=label_slice))

    if len(overlays) == 1:
        return overlays[0]

    montage = make_montage_2x2(overlays)
    return montage


def main(default_input_dir: Path, default_output_dir: Path):
    parser = argparse.ArgumentParser(
        description='Visualize 4 modalities in imageTr and overlay labels from labelsTr.'
    )
    parser.add_argument(
        '--input_dir',
        type=Path,
        default=default_input_dir,
        help='Dataset root directory containing imageTr and labelsTr.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=default_output_dir,
        help='Output directory for 2x2 montage PNGs.',
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
        required=True,
        help='Z-axis slice index n, or "all" to save all slices of this case.',
    )
    parser.add_argument('--wl', type=float, default=60.0, help='Window level for CT display.')
    parser.add_argument('--ww', type=float, default=400.0, help='Window width for CT display.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Contour thickness in pixels (>=1).',
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='If set, save image to output_dir. Otherwise show directly without saving.',
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='all',
        choices=['0', 'all'],
        help='Modality to visualize: "0" for case_xxx_0000, "all" for the original 4-modality montage.',
    )
    args = parser.parse_args()

    if args.thickness < 1:
        raise ValueError('--thickness must be >= 1')

    if not args.input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {args.input_dir}')

    image_dir = args.input_dir / 'imagesTs'
    label_dir = args.input_dir / 'labelsTs'
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f'Expected subfolders imagesTs and labelsTs under: {args.input_dir}'
        )

    case_groups = find_case_groups(image_dir)
    if not case_groups:
        raise FileNotFoundError(f'No modality image files found in: {image_dir}')

    if args.case_id not in case_groups:
        raise FileNotFoundError(f'Case not found in imageTr: {args.case_id}')

    case_modalities = case_groups[args.case_id]
    try:
        selected_modalities = resolve_requested_modalities(case_modalities, args.modality)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'{exc}: {args.case_id}') from exc

    label_path = find_label_path(label_dir, args.case_id)
    if label_path is None:
        raise FileNotFoundError(f'Label not found in labelsTr for case: {args.case_id}')

    modality_vols, label_vol = load_case_volumes(
        case_id=args.case_id,
        modality_paths=selected_modalities,
        label_path=label_path,
    )

    if args.slice_n.lower() == 'all':
        case_output_dir = prepare_case_output_dir(args.output_dir, args.case_id)
        z_dim = label_vol.shape[2]
        for idx in range(z_dim):
            montage = render_case_slice(
                case_id=args.case_id,
                modality_vols=modality_vols,
                label_vol=label_vol,
                slice_idx=idx,
                window_level=args.wl,
                window_width=args.ww,
                contour_thickness=args.thickness,
            )
            out_path = case_output_dir / f'{args.case_id}-slice-{idx:03d}.png'
            Image.fromarray(montage).save(out_path)
        print(f'Saved all slices for {args.case_id}: {z_dim} images')
        print(f'Output directory: {case_output_dir.resolve()}')
        return

    try:
        slice_idx = int(args.slice_n)
    except ValueError as exc:
        raise ValueError('--slice_n must be an integer or "all"') from exc

    montage = render_case_slice(
        case_id=args.case_id,
        modality_vols=modality_vols,
        label_vol=label_vol,
        slice_idx=slice_idx,
        window_level=args.wl,
        window_width=args.ww,
        contour_thickness=args.thickness,
    )
    image = Image.fromarray(montage)

    if args.save:
        case_output_dir = prepare_case_output_dir(args.output_dir, args.case_id)
        out_path = case_output_dir / f'{args.case_id}-slice-{slice_idx:03d}.png'
        image.save(out_path)
        print(f'Saved: {out_path.resolve()}')
    else:
        image.show(title=f'{args.case_id}-slice-{slice_idx:03d}')
        print(f'Shown: {args.case_id}-slice-{slice_idx:03d}')


if __name__ == '__main__':
    INPUT_DIR = Path("data/nnUNet_raw/Dataset003_HCC")
    OUTPUT_DIR = Path("./visualization_output")

    main(INPUT_DIR, OUTPUT_DIR)

# python visualisation.py --case_id case_001 --slice_n 50

# 保存该 case 全部层（会自动保存，不需要 --save）
# python visualisation.py --case_id case_1730 --slice_n all
# python visualisation.py --case_id case_525 --slice_n all --modality 0
