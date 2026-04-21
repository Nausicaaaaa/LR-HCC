from __future__ import annotations

from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def build_case_file_index(case_dir: Path) -> dict[str, Path]:
    """提取病例文件夹中的所有 `.nii.gz` 文件名并去掉后缀。"""
    file_index: dict[str, Path] = {}
    for path in case_dir.iterdir():
        if path.is_file() and path.name.lower().endswith(".nii.gz"):
            file_index[path.name[:-7].lower()] = path
    return file_index


def find_label_by_alias(case_files: dict[str, Path], alias: str) -> Path | None:
    """
    按 alias 查找标签文件（大小写不敏感，支持额外后缀）:
    - 精确: `a-label`
    - 前缀: `a-label-*`, `a-label_*`, `a-label.*`
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


def resample_image_to_reference(
    src_img: nib.Nifti1Image,
    ref_img: nib.Nifti1Image,
    is_label: bool = False,
) -> nib.Nifti1Image:
    """将 `src_img` 重采样到 `ref_img` 的空间网格。"""
    src_img = nib.as_closest_canonical(src_img)
    ref_img = nib.as_closest_canonical(ref_img)

    if src_img.shape == ref_img.shape and np.allclose(src_img.affine, ref_img.affine):
        return src_img

    order = 0 if is_label else 1
    resampled = resample_from_to(src_img, (ref_img.shape, ref_img.affine), order=order)
    data = resampled.get_fdata()

    if is_label:
        data = np.rint(data).astype(np.uint8)
    else:
        data = data.astype(np.float32)

    new_img = nib.Nifti1Image(data, ref_img.affine)
    new_img.set_qform(ref_img.affine)
    new_img.set_sform(ref_img.affine)
    return new_img


def merge_labels_union_binary(label_imgs: list[nib.Nifti1Image]) -> nib.Nifti1Image:
    """将多个标签做并集，所有非 0 体素统一置为 1。"""
    ref = label_imgs[0]
    merged = np.zeros(ref.shape, dtype=bool)
    for img in label_imgs:
        merged |= img.get_fdata() > 0
    merged = merged.astype(np.uint8)
    return nib.Nifti1Image(merged, affine=ref.affine, header=ref.header.copy())


def save_binary_mask(mask_img: nib.Nifti1Image, dst_label: Path) -> None:
    """保存二值标签，背景为 0，前景统一为 1。"""
    mask = mask_img.get_fdata()
    binary = np.zeros(mask.shape, dtype=np.uint8)
    binary[mask > 0] = 1
    out = nib.Nifti1Image(binary, affine=mask_img.affine, header=mask_img.header)
    nib.save(out, str(dst_label))


def load_mapping_records(mapping_txt: Path) -> list[tuple[str, str, str]]:
    """
    读取 `case_name_mapping.txt`，返回:
    `(original_case_name, converted_case_name, group)` 列表。
    """
    if not mapping_txt.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_txt}")

    lines = mapping_txt.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []

    header = "original_case_name\tconverted_case_name\tgroup"
    data_lines = lines[1:] if lines[0].strip() == header else lines

    records: list[tuple[str, str, str]] = []
    for line in data_lines:
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 3:
            continue
        records.append((parts[0], parts[1], parts[2]))
    return records


def load_bad_roi_cases(bad_roi_tsv: Path) -> tuple[set[str], set[str]]:
    """
    读取 `case_badROI.tsv`，返回:
    - 坏 ROI 的 converted case id 集合
    - 坏 ROI 的 original case name 集合（如果文件中存在且不是 `xx`）
    """
    bad_case_ids: set[str] = set()
    bad_original_names: set[str] = set()

    if not bad_roi_tsv.exists():
        return bad_case_ids, bad_original_names

    lines = bad_roi_tsv.read_text(encoding="utf-8").splitlines()
    if not lines:
        return bad_case_ids, bad_original_names

    header = "case_id\toriginal_case_name\tnonzero_voxel_count\tlabel_value\tlabel_name"
    data_lines = lines[1:] if lines[0].strip() == header else lines

    for line in data_lines:
        parts = [part.strip() for part in line.split("\t")]
        if not parts:
            continue

        case_id = parts[0]
        if case_id:
            bad_case_ids.add(case_id)

        if len(parts) >= 2:
            original_name = parts[1]
            if original_name and original_name.lower() != "xx":
                bad_original_names.add(original_name)

    return bad_case_ids, bad_original_names


def find_case_directory(src_root: Path, case_name: str) -> Path | None:
    """在源目录及其子目录中查找病例文件夹。"""
    direct_path = src_root / case_name
    if direct_path.exists():
        return direct_path

    for subdir in src_root.rglob("*"):
        if not subdir.is_dir():
            continue
        if subdir.name == case_name or subdir.name.lower() == case_name.lower():
            return subdir
    return None


def load_reference_image(
    dataset_root: Path,
    converted_case_name: str,
    fallback_ref_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """
    优先使用数据集内已存在的 A 通道图像作为参考网格，
    若不存在则退回原始 A 图像。
    """
    image_ref = dataset_root / "imagesTr" / f"{converted_case_name}_0000.nii.gz"
    if image_ref.exists():
        return nib.load(str(image_ref))
    return fallback_ref_img


def preprocess(
    path_HCC: str,
    path_noHCC: str,
    path_liangxing: str,
    path_nnUNet_raw: str,
    mapping_txt: str = "case_name_mapping.txt",
    bad_roi_tsv: str = "case_badROI.tsv",
    label_output_subdir: str = "labelsTr_AP_union",
) -> None:
    """
    基于 `case_name_mapping.txt` 提取 A/P 两个模态的标签，并生成二值并集标签。

    规则:
    - 仅处理 `case_name_mapping.txt` 中已有的病例
    - 排除 `case_badROI.tsv` 中记录的病例
    - 不导出图像，只导出标签
    - 标签重采样时优先对齐到数据集内现有的 A 图像 (`imagesTr/case_xxx_0000.nii.gz`)
    - 若数据集内 A 图像不存在，则退回对齐到原始 A 图像
    - A/P 标签做并集，所有非 0 体素统一赋值为 1
    """
    dataset_root = Path(path_nnUNet_raw)
    dst_labels_tr = dataset_root / label_output_subdir
    dst_labels_tr.mkdir(parents=True, exist_ok=True)

    mapping_records = load_mapping_records(Path(mapping_txt))
    bad_case_ids, bad_original_names = load_bad_roi_cases(Path(bad_roi_tsv))

    print(f"[INFO] Loaded {len(mapping_records)} mapping records from {mapping_txt}")
    print(
        f"[INFO] Loaded {len(bad_case_ids)} bad ROI case ids and "
        f"{len(bad_original_names)} original case names from {bad_roi_tsv}"
    )

    source_groups = {
        "HCC": Path(path_HCC),
        "noHCC": Path(path_noHCC),
        "liangxing": Path(path_liangxing),
    }

    required_modalities = ["a", "p"]
    processed_count = 0
    skipped_cases: list[tuple[str, str, str, str]] = []

    for original_name, converted_name, group in mapping_records:
        if converted_name in bad_case_ids or original_name in bad_original_names:
            skipped_cases.append((group, original_name, converted_name, "bad_roi_case"))
            print(f"[SKIP] ({group}) {original_name} -> {converted_name} is listed in {bad_roi_tsv}")
            continue

        src_root = source_groups.get(group)
        if src_root is None or not src_root.exists():
            skipped_cases.append((group, original_name, converted_name, "source_path_not_found"))
            print(f"[SKIP] ({group}) {original_name} -> source path does not exist: {src_root}")
            continue

        case_dir = find_case_directory(src_root, original_name)
        if case_dir is None:
            skipped_cases.append((group, original_name, converted_name, "directory_not_found"))
            print(f"[SKIP] ({group}) {original_name} -> case directory not found")
            continue

        case_files = build_case_file_index(case_dir)

        matched_items: list[tuple[str, Path, Path]] = []
        missing_modalities: list[str] = []
        for alias in required_modalities:
            image_file = case_files.get(alias)
            label_file = find_label_by_alias(case_files, alias)
            if image_file is None or label_file is None:
                missing_modalities.append(alias.upper())
                continue
            matched_items.append((alias, image_file, label_file))

        if len(matched_items) != len(required_modalities):
            reason = f"missing_modality_or_label:{','.join(missing_modalities)}"
            skipped_cases.append((group, original_name, converted_name, reason))
            print(f"[SKIP] ({group}) {original_name} -> {reason}")
            continue

        try:
            modality_imgs = {
                alias: nib.load(str(image_path))
                for alias, image_path, _ in matched_items
            }
            label_imgs = {
                alias: nib.load(str(label_path))
                for alias, _, label_path in matched_items
            }
        except Exception as exc:
            skipped_cases.append((group, original_name, converted_name, f"nifti_read_error:{exc}"))
            print(f"[SKIP] ({group}) {original_name} -> nifti_read_error: {exc}")
            continue

        ref_img = load_reference_image(
            dataset_root=dataset_root,
            converted_case_name=converted_name,
            fallback_ref_img=modality_imgs["a"],
        )

        resampled_labels: list[nib.Nifti1Image] = []
        for alias, _, _ in matched_items:
            src_label = label_imgs[alias]
            resampled_label = resample_image_to_reference(src_label, ref_img, is_label=True)
            resampled_labels.append(resampled_label)
            print(f"[RESAMPLE] ({group}) {original_name} alias={alias.upper()} -> ref=A")

        merged_label = merge_labels_union_binary(resampled_labels)
        merged_voxel_count = int(np.count_nonzero(merged_label.get_fdata()))

        if merged_voxel_count == 0:
            skipped_cases.append((group, original_name, converted_name, "empty_merged_label"))
            print(f"[SKIP] ({group}) {original_name} -> merged AP label is empty")
            continue

        dst_label = dst_labels_tr / f"{converted_name}.nii.gz"
        save_binary_mask(merged_label, dst_label)
        processed_count += 1

        print(
            f"[SAVE] ({group}) {original_name} -> {dst_label} "
            f"(nonzero_voxels={merged_voxel_count})"
        )

    print("\n[INFO] Processing complete:")
    print(f"  - Mapping records: {len(mapping_records)}")
    print(f"  - Saved labels: {processed_count}")
    print(f"  - Skipped cases: {len(skipped_cases)}")

    if skipped_cases:
        skip_data_txt = Path("case_skip_data_AP.tsv")
        if not skip_data_txt.exists():
            skip_data_txt.write_text(
                "group\toriginal_case_name\tconverted_case_name\treason\n",
                encoding="utf-8",
            )
        with skip_data_txt.open("a", encoding="utf-8") as file:
            for group, original_name, converted_name, reason in skipped_cases:
                file.write(f"{group}\t{original_name}\t{converted_name}\t{reason}\n")
        print(f"[SAVE] Skipped case list: {skip_data_txt}")


if __name__ == "__main__":
    # Fill these paths before running the script.

    path_HCC = r"I:\已勾画CT\训练和测试\HCC"
    path_noHCC = r"I:\已勾画CT\训练和测试\恶性非HCC"
    path_liangxing = r"I:\已勾画CT\训练和测试\良性"
    path_nnUNet_raw = r"data\nnUNet_raw\Dataset002_HCC"

    preprocess(
        path_HCC,
        path_noHCC,
        path_liangxing,
        path_nnUNet_raw,
        mapping_txt="case_name_mapping.txt",
        bad_roi_tsv="case_badROI.tsv",
        label_output_subdir="labelsTr_AP_union",
    )
