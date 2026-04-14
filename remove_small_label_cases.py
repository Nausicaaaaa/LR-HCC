from __future__ import annotations

from pathlib import Path
import csv
import json
import argparse

import nibabel as nib
import numpy as np


DATASET_DIR = Path("nnUNet/nnUNet_raw/Dataset001_HCC")
IMAGE_DIR = DATASET_DIR / "imagesTr"
DEFAULT_DELETE_LABEL_DIRS = [
    DATASET_DIR / "labelsTr",
    DATASET_DIR / "labelsTr_intersection",
]
THRESHOLD = 500


def case_id_from_label_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def collect_small_cases(label_dir: Path, threshold: int) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for label_path in sorted(label_dir.glob("*.nii.gz")):
        case_id = case_id_from_label_path(label_path)
        data = np.asanyarray(nib.load(str(label_path)).dataobj)
        nonzero_voxels = int(np.count_nonzero(data))
        if nonzero_voxels >= threshold:
            continue

        label_values = [int(v) for v in np.unique(data) if v != 0]
        cases.append(
            {
                "case_id": case_id,
                "label_voxel_count": nonzero_voxels,
                "label_tags": label_values,
                "label_path": label_path,
            }
        )
    return cases


def write_report(cases: list[dict[str, object]], report_path: Path) -> None:
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["case_id", "label_voxel_count", "label_tags"])
        for case in cases:
            writer.writerow(
                [
                    case["case_id"],
                    case["label_voxel_count"],
                    json.dumps(case["label_tags"], ensure_ascii=False),
                ]
            )


def remove_case_files(
    cases: list[dict[str, object]],
    remove_images: bool,
    delete_label_dirs: list[Path],
) -> None:
    for case in cases:
        case_id = str(case["case_id"])
        for label_dir in delete_label_dirs:
            label_path = label_dir / f"{case_id}.nii.gz"
            if label_path.exists():
                label_path.unlink()

        if remove_images:
            for image_path in sorted(IMAGE_DIR.glob(f"{case_id}_*.nii.gz")):
                image_path.unlink()


def update_dataset_json() -> None:
    dataset_json_path = DATASET_DIR / "dataset.json"
    if not dataset_json_path.exists():
        return

    with dataset_json_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset["numTraining"] = len(list((DATASET_DIR / "labelsTr").glob("*.nii.gz")))

    with dataset_json_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-dir",
        default=str(DATASET_DIR / "labelsTr"),
        help="Directory containing label NIfTI files.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DATASET_DIR / "removed_small_label_cases.tsv"),
        help="Output TSV path.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=THRESHOLD,
        help="Remove cases with nonzero voxel count below this threshold.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matching cases without deleting files.",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Do not delete imagesTr files or update dataset.json.",
    )
    parser.add_argument(
        "--delete-label-dirs",
        nargs="*",
        default=[str(p) for p in DEFAULT_DELETE_LABEL_DIRS],
        help="Label directories to delete matching case ids from.",
    )
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    report_path = Path(args.report_path)
    delete_label_dirs = [Path(p) for p in args.delete_label_dirs]

    cases = collect_small_cases(label_dir, args.threshold)
    write_report(cases, report_path)

    if not args.dry_run:
        remove_case_files(
            cases,
            remove_images=not args.keep_images,
            delete_label_dirs=delete_label_dirs,
        )
        if not args.keep_images:
            update_dataset_json()

    print(f"label_dir={label_dir}")
    print(f"threshold={args.threshold}")
    print(f"removed_cases={len(cases)}")
    print(f"report={report_path}")
    print(f"dry_run={args.dry_run}")
    for case in cases:
        print(
            f"{case['case_id']}\t{case['label_voxel_count']}\t{case['label_tags']}"
        )


if __name__ == "__main__":
    main()
