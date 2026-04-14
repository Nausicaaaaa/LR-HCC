from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_MODALITIES = [
    (["a"], 0, "A"),
    (["d"], 1, "D"),
    (["p"], 2, "P"),
    (["s", "ps"], 3, "S/PS"),
]


def build_case_file_index(case_dir: Path) -> dict[str, Path]:
    file_index: dict[str, Path] = {}
    for p in case_dir.iterdir():
        if p.is_file() and p.name.lower().endswith(".nii.gz"):
            file_index[p.name[:-7].lower()] = p
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


def load_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "shape_check.py requires nibabel. Please install it in the current Python environment first."
        ) from exc
    return nib


def collect_case_shapes(src_root: Path, group_name: str) -> tuple[list[str], list[str]]:
    report_lines: list[str] = []
    skipped_lines: list[str] = []

    if not src_root.exists():
        skipped_lines.append(f"[SKIP_GROUP] group={group_name} reason=source_path_not_found path={src_root}")
        return report_lines, skipped_lines

    nib = load_nibabel()

    for case_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        case_files = build_case_file_index(case_dir)

        matched_items: list[tuple[int, str, str, Path, Path]] = []
        missing_modalities: list[str] = []
        for aliases, mod_idx, display_name in REQUIRED_MODALITIES:
            selected: tuple[int, str, str, Path, Path] | None = None
            for alias in aliases:
                img_file = case_files.get(alias)
                label_file = find_label_by_alias(case_files, alias)
                if img_file is not None and label_file is not None:
                    selected = (mod_idx, display_name, alias, img_file, label_file)
                    break

            if selected is None:
                missing_modalities.append(display_name)
            else:
                matched_items.append(selected)

        if len(matched_items) != 4:
            skipped_lines.append(
                f"[SKIP_CASE] group={group_name} case={case_dir.name} "
                f"reason=missing_modality_or_label:{','.join(missing_modalities)}"
            )
            continue

        matched_items.sort(key=lambda x: x[0])

        case_lines = [f"[CASE] group={group_name} case={case_dir.name}"]
        read_failed = False
        for _, display_name, alias, img_path, label_path in matched_items:
            try:
                img = nib.load(str(img_path))
                lbl = nib.load(str(label_path))
            except Exception as exc:
                skipped_lines.append(
                    f"[SKIP_CASE] group={group_name} case={case_dir.name} "
                    f"reason=nifti_read_error:{exc}"
                )
                read_failed = True
                break

            case_lines.append(
                f"  {display_name:<4} alias={alias.upper():<2} "
                f"image_shape={tuple(img.shape)} label_shape={tuple(lbl.shape)} "
                f"image_file={img_path.name} label_file={label_path.name}"
            )

        if read_failed:
            continue

        report_lines.extend(case_lines)
        report_lines.append("")

    return report_lines, skipped_lines


def generate_shape_report(
    path_hcc: str,
    path_nohcc: str,
    path_liangxing: str,
    output_txt: str,
) -> Path:
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_groups = [
        ("HCC", Path(path_hcc)),
        ("noHCC", Path(path_nohcc)),
        ("liangxing", Path(path_liangxing)),
    ]

    all_report_lines: list[str] = []
    all_skipped_lines: list[str] = []
    total_complete_cases = 0

    for group_name, src_root in source_groups:
        group_report_lines, group_skipped_lines = collect_case_shapes(src_root, group_name)
        complete_cases = sum(1 for line in group_report_lines if line.startswith("[CASE]"))
        total_complete_cases += complete_cases

        all_report_lines.append(f"===== {group_name} =====")
        all_report_lines.append(f"source_path={src_root}")
        all_report_lines.append(f"complete_cases={complete_cases}")
        all_report_lines.append("")
        all_report_lines.extend(group_report_lines)

        if group_skipped_lines:
            all_report_lines.append(f"----- {group_name} skipped -----")
            all_report_lines.extend(group_skipped_lines)
            all_report_lines.append("")

        all_skipped_lines.extend(group_skipped_lines)

    summary_lines = [
        "===== SUMMARY =====",
        f"total_complete_cases={total_complete_cases}",
        f"total_skipped_entries={len(all_skipped_lines)}",
        "",
    ]

    output_path.write_text("\n".join(summary_lines + all_report_lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    default_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Scan all cases with the same modality-matching logic as preprocess.py and save modality shapes to a txt report."
    )
    parser.add_argument(
        "--path_hcc",
        type=str,
        default=r"I:\已勾画CT\训练和测试\HCC\附二HCC",
        help="HCC source directory.",
    )
    parser.add_argument(
        "--path_nohcc",
        type=str,
        default=r"I:\已勾画CT\训练和测试\恶性非HCC\附二恶性非HCC",
        help="noHCC source directory.",
    )
    parser.add_argument(
        "--path_liangxing",
        type=str,
        default=r"I:\已勾画CT\训练和测试\良性\附二良性\有病理",
        help="liangxing source directory.",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=str(default_root / "modality_shape_report.txt"),
        help="Output txt path.",
    )
    args = parser.parse_args()

    output_path = generate_shape_report(
        path_hcc=args.path_hcc,
        path_nohcc=args.path_nohcc,
        path_liangxing=args.path_liangxing,
        output_txt=args.output_txt,
    )
    print(f"[SAVE] Shape report: {output_path}")


if __name__ == "__main__":
    main()
