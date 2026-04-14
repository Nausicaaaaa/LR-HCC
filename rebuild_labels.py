from __future__ import annotations

import argparse
from pathlib import Path
import re

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def remove_known_nii_suffix(name: str) -> str:
    lower_name = name.lower()
    if lower_name.endswith(".nii.gz"):
        return name[:-7]
    if lower_name.endswith(".nii"):
        return name[:-4]
    return name


def build_case_file_index(case_dir: Path) -> dict[str, Path]:
    file_index: dict[str, Path] = {}
    for p in case_dir.iterdir():
        if not p.is_file():
            continue
        lower_name = p.name.lower()
        if lower_name.endswith(".nii.gz") or lower_name.endswith(".nii"):
            file_index[remove_known_nii_suffix(p.name).lower()] = p
    return file_index


def find_label_by_alias(case_files: dict[str, Path], alias: str) -> Path | None:
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
    mask = mask_img.get_fdata()
    relabelled = np.zeros(mask.shape, dtype=np.uint8)
    relabelled[mask > 0] = np.uint8(roi_value)
    out = nib.Nifti1Image(relabelled, affine=mask_img.affine, header=mask_img.header.copy())
    out.set_qform(mask_img.affine)
    out.set_sform(mask_img.affine)
    nib.save(out, str(dst_label))


def resample_image_to_reference(
    src_img: nib.Nifti1Image,
    ref_img: nib.Nifti1Image,
    is_label: bool = False,
) -> nib.Nifti1Image:
    src_img = nib.as_closest_canonical(src_img)
    ref_img = nib.as_closest_canonical(ref_img)

    if src_img.shape == ref_img.shape and np.allclose(src_img.affine, ref_img.affine):
        return src_img

    order = 0 if is_label else 1
    resampled = resample_from_to(
        src_img,
        (ref_img.shape, ref_img.affine),
        order=order,
    )

    data = resampled.get_fdata()
    if is_label:
        data = np.rint(data).astype(np.uint8)
    else:
        data = data.astype(np.float32)

    new_img = nib.Nifti1Image(data, ref_img.affine)
    new_img.set_qform(ref_img.affine)
    new_img.set_sform(ref_img.affine)
    return new_img


def merge_labels_intersection(label_imgs: list[nib.Nifti1Image]) -> nib.Nifti1Image:
    ref = label_imgs[0]
    mask_stack = np.stack([img.get_fdata() > 0 for img in label_imgs], axis=0)
    merged = np.all(mask_stack, axis=0).astype(np.uint8)
    return nib.Nifti1Image(merged, affine=ref.affine, header=ref.header.copy())


def merge_labels_union_hu(
    label_imgs: list[nib.Nifti1Image],
    hu_img: nib.Nifti1Image,
    hu_min: float,
    hu_max: float,
) -> nib.Nifti1Image:
    ref = label_imgs[0]
    union_mask = np.zeros(ref.shape, dtype=bool)
    for img in label_imgs:
        union_mask |= img.get_fdata() > 0

    hu_data = hu_img.get_fdata(dtype=np.float32)
    hu_mask = (hu_data >= hu_min) & (hu_data <= hu_max)
    merged = (union_mask & hu_mask).astype(np.uint8)
    return nib.Nifti1Image(merged, affine=ref.affine, header=ref.header.copy())


def load_mapping(mapping_txt: Path) -> list[tuple[str, str, str]]:
    if not mapping_txt.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_txt}")

    lines = mapping_txt.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []

    header = "original_case_name\tconverted_case_name\tgroup"
    data_lines = lines[1:] if lines[0].strip() == header else lines

    rows: list[tuple[str, str, str]] = []
    case_pat = re.compile(r"^case_\d+$")
    for line in data_lines:
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 3:
            continue
        original_case_name, converted_case_name, group_name = parts[:3]
        if not case_pat.match(converted_case_name):
            continue
        rows.append((original_case_name, converted_case_name, group_name))
    return rows


def build_case_dir_index(search_root: Path) -> dict[str, list[Path]]:
    case_dir_index: dict[str, list[Path]] = {}
    for path in search_root.rglob("*"):
        if not path.is_dir():
            continue
        case_dir_index.setdefault(path.name.lower(), []).append(path)
    return case_dir_index


def resolve_case_dir(case_dirs: list[Path]) -> Path | None:
    if not case_dirs:
        return None
    if len(case_dirs) == 1:
        return case_dirs[0]
    return None


def get_group_roi_value(group_name: str) -> int:
    group_to_value = {
        "HCC": 1,
        "noHCC": 2,
        "liangxing": 3,
    }
    if group_name not in group_to_value:
        raise ValueError(f"Unknown group name: {group_name}")
    return group_to_value[group_name]


def rebuild_labels(
    path_hcc: Path,
    path_nohcc: Path,
    path_liangxing: Path,
    mapping_txt: Path,
    output_label_dir: Path,
    strategy: str,
    hu_modality: int,
    hu_min: float,
    hu_max: float,
) -> None:
    output_label_dir.mkdir(parents=True, exist_ok=True)
    skipped_log = output_label_dir.parent / f"{output_label_dir.name}_skipped.txt"

    required_modalities = [
        (["a"], 0),
        (["d"], 1),
        (["p"], 2),
        (["s", "ps"], 3),
    ]
    source_groups = {
        "HCC": path_hcc,
        "noHCC": path_nohcc,
        "liangxing": path_liangxing,
    }

    mapping_rows = load_mapping(mapping_txt)
    skipped_cases: list[tuple[str, str, str]] = []
    saved_count = 0
    print(f"[INFO] Loaded {len(mapping_rows)} rows from {mapping_txt}")

    group_case_dir_indices: dict[str, dict[str, list[Path]]] = {}
    for group_name, src_root in source_groups.items():
        if not src_root.exists():
            print(f"[WARN] Source path does not exist: {src_root}")
            group_case_dir_indices[group_name] = {}
            continue
        print(f"[SCAN] Building case index for {group_name}: {src_root}")
        group_case_dir_indices[group_name] = build_case_dir_index(src_root)

    for original_case_name, converted_case_name, group_name in mapping_rows:
        src_root = source_groups.get(group_name)
        if src_root is None or not src_root.exists():
            skipped_cases.append((group_name, original_case_name, "source_path_not_found"))
            continue

        case_dir_candidates = group_case_dir_indices.get(group_name, {}).get(
            original_case_name.lower(),
            [],
        )
        case_dir = resolve_case_dir(case_dir_candidates)
        if case_dir is None:
            reason = "case_dir_not_found" if not case_dir_candidates else "multiple_case_dirs_matched"
            skipped_cases.append((group_name, original_case_name, reason))
            continue

        case_files = build_case_file_index(case_dir)
        matched_items: list[tuple[int, str, Path, Path]] = []
        missing_modalities: list[str] = []
        for aliases, mod_idx in required_modalities:
            selected: tuple[int, str, Path, Path] | None = None
            for alias in aliases:
                img_file = case_files.get(alias)
                label_file = find_label_by_alias(case_files, alias)
                if img_file is not None and label_file is not None:
                    selected = (mod_idx, alias, img_file, label_file)
                    break

            if selected is None:
                missing_modalities.append("/".join(a.upper() for a in aliases))
            else:
                matched_items.append(selected)

        if len(matched_items) != 4:
            reason = f"missing_modality_or_label:{','.join(missing_modalities)}"
            skipped_cases.append((group_name, original_case_name, reason))
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
        except Exception as exc:
            skipped_cases.append((group_name, original_case_name, f"nifti_read_error:{exc}"))
            continue

        ref_img = modality_imgs[0]
        resampled_labels: list[nib.Nifti1Image] = []
        for mod_idx, _, _, _ in matched_items:
            src_lbl = label_imgs[mod_idx]
            resampled_lbl = resample_image_to_reference(src_lbl, ref_img, is_label=True)
            resampled_labels.append(resampled_lbl)

        if strategy == "intersection":
            merged_label = merge_labels_intersection(resampled_labels)
        else:
            hu_ref_img = resample_image_to_reference(
                modality_imgs[hu_modality],
                ref_img,
                is_label=False,
            )
            merged_label = merge_labels_union_hu(
                resampled_labels,
                hu_ref_img,
                hu_min=hu_min,
                hu_max=hu_max,
            )

        dst_label = output_label_dir / f"{converted_case_name}.nii.gz"
        roi_value = get_group_roi_value(group_name)
        save_relabelled_mask(merged_label, dst_label, roi_value=roi_value)
        saved_count += 1
        print(
            f"[SAVE] ({group_name}) {original_case_name} -> {dst_label.name} "
            f"strategy={strategy}, foreground={int(np.count_nonzero(merged_label.get_fdata()))}"
        )

    skipped_lines = ["group\tcase_folder\treason"]
    skipped_lines.extend(f"{g}\t{c}\t{r}" for g, c, r in skipped_cases)
    skipped_log.write_text("\n".join(skipped_lines) + "\n", encoding="utf-8")

    print(f"[DONE] Saved {saved_count} labels to {output_label_dir}")
    print(f"[DONE] Skipped {len(skipped_cases)} cases, log saved to {skipped_log}")
    for group_name, case_name, reason in skipped_cases[:10]:
        print(f"[SKIP] ({group_name}) {case_name}: {reason}")
    if len(skipped_cases) > 10:
        print(f"[SKIP] ... and {len(skipped_cases) - 10} more cases")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild labels only, using existing case_name_mapping.txt and the same "
            "case matching / resampling logic as preprocess.py."
        )
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("case_name_mapping.txt"),
        help="Path to case_name_mapping.txt generated by preprocess.py.",
    )
    parser.add_argument("--hcc-root", type=Path, required=True, help="Root directory for HCC cases.")
    parser.add_argument(
        "--nohcc-root",
        type=Path,
        required=True,
        help="Root directory for noHCC cases.",
    )
    parser.add_argument(
        "--liangxing-root",
        type=Path,
        required=True,
        help="Root directory for liangxing cases.",
    )
    parser.add_argument(
        "--output-label-dir",
        type=Path,
        default=None,
        help="Output labels directory. Default depends on strategy.",
    )
    parser.add_argument(
        "--strategy",
        choices=["intersection", "union_hu"],
        required=True,
        help="Label merge strategy.",
    )
    parser.add_argument(
        "--hu-modality",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Which modality image to use for HU filtering in union_hu mode. Default: 0 (A).",
    )
    parser.add_argument(
        "--hu-min",
        type=float,
        default=-20.0,
        help="Lower HU bound for union_hu mode.",
    )
    parser.add_argument(
        "--hu-max",
        type=float,
        default=200.0,
        help="Upper HU bound for union_hu mode.",
    )
    args = parser.parse_args()

    if args.strategy == "union_hu" and args.hu_min > args.hu_max:
        raise ValueError("--hu-min must be <= --hu-max")

    output_label_dir = args.output_label_dir
    if output_label_dir is None:
        output_label_dir = Path("nnUNet") / "nnUNet_raw" / "Dataset001_HCC" / f"labelsTr_{args.strategy}"

    rebuild_labels(
        path_hcc=args.hcc_root,
        path_nohcc=args.nohcc_root,
        path_liangxing=args.liangxing_root,
        mapping_txt=args.mapping,
        output_label_dir=output_label_dir,
        strategy=args.strategy,
        hu_modality=args.hu_modality,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
    )


if __name__ == "__main__":
    main()
