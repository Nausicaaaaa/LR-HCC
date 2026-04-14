from __future__ import annotations

from pathlib import Path
import re

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def build_case_file_index(case_dir: Path) -> dict[str, Path]:
    """提取病例文件夹中的所有 nii.gz 文件名并去后缀。"""
    file_index: dict[str, Path] = {}
    for p in case_dir.iterdir():
        if p.is_file() and p.name.lower().endswith(".nii.gz"):
            file_index[p.name[:-7].lower()] = p
    return file_index


def find_label_by_alias(case_files: dict[str, Path], alias: str) -> Path | None:
    """
    按 alias 查找标签文件（大小写不敏感，支持额外后缀）:
    - 精确: a-label
    - 前缀: a-label-*, a-label_*, a-label.*
    """
    base = f"{alias}-label"
    exact = case_files.get(base)
    if exact is not None:
        return exact

    valid_prefixes = (f"{base}-", f"{base}_", f"{base}.")
    for key in sorted(case_files.keys()):
        if key.startswith(valid_prefixes):
            return case_files[key]
    return None


def save_relabelled_mask(mask_img: nib.Nifti1Image, dst_label: Path, roi_value: int) -> None:
    """
    将原始标签重映射为:
    - 背景: 0
    - ROI: roi_value
    """
    mask = mask_img.get_fdata()
    relabelled = np.zeros(mask.shape, dtype=np.uint8)
    relabelled[mask > 0] = np.uint8(roi_value)
    out = nib.Nifti1Image(relabelled, affine=mask_img.affine, header=mask_img.header)
    nib.save(out, str(dst_label))


def resample_image_to_reference(
    src_img: nib.Nifti1Image,
    ref_img: nib.Nifti1Image,
    is_label: bool = False,
) -> nib.Nifti1Image:
    """
    将 src_img 重采样到 ref_img 的空间网格

    保证：
    - shape一致
    - spacing一致
    - orientation一致
    - affine一致

    Parameters
    ----------
    src_img : Nifti1Image
        原始图像
    ref_img : Nifti1Image
        参考图像
    is_label : bool
        是否为标签（决定插值方式）

    Returns
    -------
    Nifti1Image
        resampled image
    """

    # 1 将图像转换为标准RAS方向（右侧、前侧、上侧），确保坐标系一致
    src_img = nib.as_closest_canonical(src_img)
    ref_img = nib.as_closest_canonical(ref_img)

    # 2 如果 shape + affine 已经一致就不resample
    if (
        src_img.shape == ref_img.shape
        and np.allclose(src_img.affine, ref_img.affine)
    ):
        return src_img

    # 3 选择插值方式
    order = 0 if is_label else 1

    # 4 执行 resample
    resampled = resample_from_to(
        src_img,
        (ref_img.shape, ref_img.affine),
        order=order,
    )

    data = resampled.get_fdata()

    # 5 处理 dtype
    if is_label:
        data = np.rint(data)  # 防止0.999这种情况
        data = data.astype(np.uint8)
    else:
        data = data.astype(np.float32)

    # 6 创建新 NIfTI（避免 header 冲突）
    new_img = nib.Nifti1Image(data, ref_img.affine)

    # 7 保留 qform / sform
    new_img.set_qform(ref_img.affine)
    new_img.set_sform(ref_img.affine)

    return new_img


def merge_labels_binary(label_imgs: list[nib.Nifti1Image]) -> nib.Nifti1Image:
    """将同一病例的多个模态标签合并为二值前景(交集)。"""
    ref = label_imgs[0]
    merged = np.ones(ref.shape, dtype=bool)
    for img in label_imgs:
        merged &= img.get_fdata() > 0
    merged = merged.astype(np.uint8)
    return nib.Nifti1Image(merged, affine=ref.affine, header=ref.header.copy())


def load_existing_mapping(mapping_txt: Path) -> tuple[list[str], int]:
    """
    读取已有 mapping 文件，返回:
    - 已有数据行（不含表头）
    - 当前最大 case 编号（若无则为 0)
    """
    if not mapping_txt.exists():
        return [], 0

    lines = mapping_txt.read_text(encoding="utf-8").splitlines()
    if not lines:
        return [], 0

    header = "original_case_name\tconverted_case_name\tgroup"
    data_lines = lines[1:] if lines[0].strip() == header else lines

    max_case_idx = 0
    case_pat = re.compile(r"^case_(\d+)$")
    for line in data_lines:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        m = case_pat.match(parts[1].strip())
        if m:
            max_case_idx = max(max_case_idx, int(m.group(1)))

    return data_lines, max_case_idx


def preprocess(
    path_HCC: str,
    path_noHCC: str,
    path_liangxing: str,
    path_nnUNet_raw: str,
    min_merged_label_voxels: int = 1,
) -> None:
    """
    按顺序读取 path_HCC、path_noHCC、path_liangxing 下的病例。
    每个病例要求匹配 4 个模态及其对应 alias-label（a/d/p/s|ps）。
    在转存前，先根据空间信息将 4 个模态和 4 个标签统一到同一网格。

    标签映射:
    - path_HCC 的 ROI -> 1
    - path_noHCC 的 ROI -> 2
    - path_liangxing 的 ROI -> 3
    - 背景 -> 0
    """
    dst_images_tr = Path(path_nnUNet_raw) / "imagesTr"
    dst_labels_tr = Path(path_nnUNet_raw) / "labelsTr"
    mapping_txt = Path("case_name_mapping.txt")
    skip_data_txt = Path("case_skip_data.txt")
    small_label_txt = Path("small_merged_labels.tsv")
    dst_images_tr.mkdir(parents=True, exist_ok=True)
    dst_labels_tr.mkdir(parents=True, exist_ok=True)

    required_modalities = [
        (["a"], 0),
        (["d"], 1),
        (["p"], 2),
        (["s", "ps"], 3),
    ]

    source_groups = [
        ("HCC", Path(path_HCC), 1),
        ("noHCC", Path(path_noHCC), 2),
        ("liangxing", Path(path_liangxing), 3),
    ]

    header = "original_case_name\tconverted_case_name\tgroup"
    existing_mapping_lines, max_case_idx = load_existing_mapping(mapping_txt) #mapping_txt续写
    case_idx = max_case_idx + 1
    mapping_lines: list[str] = []
    skipped_cases: list[tuple[str, str, str]] = []
    small_label_cases: list[tuple[str, str, int, int, str]] = []
    roi_label_names = {
        1: "HCC",
        2: "noHCC",
        3: "liangxing",
    }

    for group_name, src_root, roi_value in source_groups:
        if not src_root.exists():
            print(f"[SKIP] Source path does not exist: {src_root}")
            continue

        for case_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            case_files = build_case_file_index(case_dir) # 文件名转小写 去后缀

            matched_items: list[tuple[int, str, Path, Path]] = []
            missing_modalities: list[str] = []
            for aliases, mod_idx in required_modalities:
                selected: tuple[int, str, Path, Path] | None = None
                for alias in aliases:
                    img_file = case_files.get(alias)
                    label_file = find_label_by_alias(case_files, alias) #支持额外后缀的标签文件
                    if img_file is not None and label_file is not None:
                        selected = (mod_idx, alias, img_file, label_file)
                        break

                if selected is None:
                    missing_modalities.append("/".join(a.upper() for a in aliases))
                else:
                    matched_items.append(selected)

            if len(matched_items) != 4:
                reason = f"missing_modality_or_label:{','.join(missing_modalities)}"
                skipped_cases.append((group_name, case_dir.name, reason))
                continue

            matched_items.sort(key=lambda x: x[0])

            try:
                modality_imgs = {
                    mod_idx: nib.load(str(img_path))
                    for mod_idx, _, img_path, _ in matched_items
                }
                label_imgs = {
                    mod_idx: nib.load(str(label_path))
                    for mod_idx, _, _, label_path in matched_items
                }
            except Exception as e:
                skipped_cases.append((group_name, case_dir.name, f"nifti_read_error:{e}"))
                continue

            # 以 A 通道(0000)作为统一空间参考
            ref_img = modality_imgs[0] 
            
            # 初始化存储重采样后图像的字典和列表
            resampled_modalities: dict[int, nib.Nifti1Image] = {}
            resampled_labels: list[nib.Nifti1Image] = []
            for mod_idx, alias, _, _ in matched_items:
                src_mod = modality_imgs[mod_idx]
                src_lbl = label_imgs[mod_idx]

                resampled_mod = resample_image_to_reference(src_mod, ref_img, is_label=False)
                resampled_lbl = resample_image_to_reference(src_lbl, ref_img, is_label=True)

                resampled_modalities[mod_idx] = resampled_mod
                resampled_labels.append(resampled_lbl)
                print(f"[RESAMPLE] ({group_name}) {case_dir.name} alias={alias.upper()} -> ref=A")

            merged_label = merge_labels_binary(resampled_labels)
            merged_voxel_count = int(np.count_nonzero(merged_label.get_fdata()))
            converted_case_name = f"case_{case_idx:03d}"
            if merged_voxel_count < min_merged_label_voxels:
                small_label_cases.append(
                    (
                        converted_case_name,
                        case_dir.name,
                        merged_voxel_count,
                        roi_value,
                        roi_label_names.get(roi_value, "unknown"),
                    )
                )
                mapping_lines.append(f"{case_dir.name}\t{converted_case_name}\t{group_name}")
                print(
                    f"[SKIP SAVE] ({group_name}) {case_dir.name} merged label too small: "
                    f"{merged_voxel_count} < {min_merged_label_voxels}"
                )
                case_idx += 1
                continue

            for mod_idx in range(4):
                dst_file = dst_images_tr / f"case_{case_idx:03d}_{mod_idx:04d}.nii.gz"
                nib.save(resampled_modalities[mod_idx], str(dst_file))
                print(f"[COPY] ({group_name}) {case_dir.name} modality_{mod_idx:04d} -> {dst_file}")

            merged_label = merge_labels_binary(resampled_labels) #将4个模态标签合并为二值前景(交集)
            merged_voxel_count = int(np.count_nonzero(merged_label.get_fdata()))
            converted_case_name = f"case_{case_idx:03d}"
            if merged_voxel_count < min_merged_label_voxels:
                small_label_cases.append(
                    (
                        converted_case_name,
                        case_dir.name,
                        merged_voxel_count,
                        roi_value,
                        roi_label_names.get(roi_value, "unknown"),
                    )
                )
                mapping_lines.append(f"{case_dir.name}\t{converted_case_name}\t{group_name}")
                print(
                    f"[SKIP SAVE] ({group_name}) {case_dir.name} merged label too small: "
                    f"{merged_voxel_count} < {min_merged_label_voxels}"
                )
                case_idx += 1
                continue

            dst_label = dst_labels_tr / f"case_{case_idx:03d}.nii.gz"
            save_relabelled_mask(merged_label, dst_label, roi_value=roi_value) #重映射标签值
            print(
                f"[COPY] ({group_name}) {case_dir.name} merged_4labels -> {dst_label} "
                f"(background=0, roi={roi_value}, nonzero_voxels={merged_voxel_count})"
            )

            mapping_lines.append(f"{case_dir.name}\t{converted_case_name}\t{group_name}")
            case_idx += 1

    if not mapping_txt.exists():
        mapping_txt.write_text(header + "\n", encoding="utf-8")
    if mapping_lines:
        with mapping_txt.open("a", encoding="utf-8") as f:
            for line in mapping_lines:
                f.write(line + "\n")
    print(f"[SAVE] Mapping file: {mapping_txt} (appended {len(mapping_lines)} rows)")

    group_counts = {"HCC": 0, "noHCC": 0, "liangxing": 0}
    all_mapping_lines = existing_mapping_lines + mapping_lines
    for line in all_mapping_lines:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2] in group_counts:
            group_counts[parts[2]] += 1

    print("[COUNT] case_name_mapping.txt group counts:")
    print(f"  HCC: {group_counts['HCC']}")
    print(f"  noHCC: {group_counts['noHCC']}")
    print(f"  liangxing: {group_counts['liangxing']}")

    if not skip_data_txt.exists():
        skip_data_txt.write_text("group\tcase_folder\treason\n", encoding="utf-8")
    if skipped_cases:
        with skip_data_txt.open("a", encoding="utf-8") as f:
            for g, c, r in skipped_cases:
                f.write(f"{g}\t{c}\t{r}\n")
    print(f"[SAVE] Skipped case list: {skip_data_txt} (appended {len(skipped_cases)} rows)")

    if not small_label_txt.exists():
        small_label_txt.write_text(
            "case_id\toriginal_case_name\tnonzero_voxel_count\tlabel_value\tlabel_name\n",
            encoding="utf-8",
        )
    if small_label_cases:
        with small_label_txt.open("a", encoding="utf-8") as f:
            for case_id, original_case_name, voxel_count, label_value, label_name in small_label_cases:
                f.write(
                    f"{case_id}\t{original_case_name}\t{voxel_count}\t{label_value}\t{label_name}\n"
                )
    print(
        f"[SAVE] Small merged label list: {small_label_txt} "
        f"(threshold={min_merged_label_voxels}, appended {len(small_label_cases)} rows)"
    )


if __name__ == "__main__":
    # 按实际数据位置修改以下路径
    path_HCC = r"I:\已勾画CT\训练和测试\HCC\西南HCC(有瘤周）"
    path_noHCC = r""
    path_liangxing = r""

    path_nnUNet_raw = r"nnUNet\nnUNet_raw\Dataset003_HCC"

    min_merged_label_voxels = 500
    preprocess(
        path_HCC,
        path_noHCC,
        path_liangxing,
        path_nnUNet_raw,
        min_merged_label_voxels=min_merged_label_voxels,
    )
