import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

OUTPUT_DIR = Path(r"./visualization_output")
DEFAULT_INPUT_ROOT = Path(r"I:\\")

MODALITY_SPECS = [
    ("A", ["a"]),
    ("D", ["d"]),
    ("P", ["p"]),
    ("S", ["s", "ps"]),
]


def strip_nii_suffix(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".nii.gz"):
        return filename[:-7]
    if name.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def load_nii(path: Path) -> np.ndarray:
    nii = nib.load(str(path))
    return np.asarray(nii.get_fdata(), dtype=np.float32)


def normalize_to_uint8(image_slice: np.ndarray) -> np.ndarray:
    image_slice = np.nan_to_num(image_slice, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(np.min(image_slice))
    max_val = float(np.max(image_slice))
    if max_val <= min_val:
        return np.zeros(image_slice.shape, dtype=np.uint8)
    scaled = (image_slice - min_val) / (max_val - min_val)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def inner_pixels(mask: np.ndarray) -> np.ndarray:
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return inner


def dilate4(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(0, iterations)):
        expanded = out.copy()
        expanded[1:, :] |= out[:-1, :]
        expanded[:-1, :] |= out[1:, :]
        expanded[:, 1:] |= out[:, :-1]
        expanded[:, :-1] |= out[:, 1:]
        out = expanded
    return out


def resize_mask_if_needed(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool)
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (target_shape[1], target_shape[0]),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(resized) > 0


def overlay_contour(
    image_u8: np.ndarray,
    label_slice: np.ndarray,
    contour_color: tuple[int, int, int] = (255, 0, 0),
    contour_thickness: int = 1,
) -> np.ndarray:
    rgb = np.stack([image_u8, image_u8, image_u8], axis=-1)
    mask = label_slice > 0
    if not np.any(mask):
        return rgb

    mask = resize_mask_if_needed(mask, image_u8.shape)
    boundary = mask & (~inner_pixels(mask))
    if contour_thickness > 1:
        boundary = dilate4(boundary, contour_thickness - 1)

    rgb[boundary] = np.array(contour_color, dtype=np.uint8)
    return rgb


def build_case_file_index(case_dir: Path) -> dict[str, Path]:
    file_index: dict[str, Path] = {}
    for path in sorted(case_dir.iterdir()):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if lower_name.endswith(".nii.gz") or lower_name.endswith(".nii"):
            file_index[strip_nii_suffix(lower_name)] = path
    return file_index


def find_label_path(case_files: dict[str, Path], alias: str) -> Path | None:
    candidates = [
        f"{alias}-label",
        f"{alias}_label",
        f"{alias}label",
    ]
    for candidate in candidates:
        exact = case_files.get(candidate)
        if exact is not None:
            return exact

    prefixes = tuple(
        prefix
        for candidate in candidates
        for prefix in (f"{candidate}-", f"{candidate}_", f"{candidate}.")
    )
    for key in sorted(case_files.keys()):
        if key.startswith(prefixes):
            return case_files[key]
    return None


def find_modality_files(case_dir: Path) -> dict[str, tuple[Path, Path]]:
    case_files = build_case_file_index(case_dir)
    modality_files: dict[str, tuple[Path, Path]] = {}

    for modality_name, aliases in MODALITY_SPECS:
        selected_pair: tuple[Path, Path] | None = None
        for alias in aliases:
            image_path = case_files.get(alias)
            label_path = find_label_path(case_files, alias)
            if image_path is not None and label_path is not None:
                selected_pair = (image_path, label_path)
                break
        if selected_pair is None:
            raise FileNotFoundError(
                f"Missing image or label for modality {modality_name} in {case_dir}"
            )
        modality_files[modality_name] = selected_pair

    return modality_files


def score_case_dir(case_dir: Path) -> int:
    try:
        return len(find_modality_files(case_dir))
    except FileNotFoundError:
        return 0


def resolve_case_dir(input_root: Path, case_id: str) -> Path:
    direct = input_root / case_id
    candidates: list[Path] = []
    if direct.is_dir():
        candidates.append(direct)

    for path in input_root.rglob(case_id):
        if path.is_dir() and path not in candidates:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"Cannot find folder named {case_id} under {input_root}")

    best = max(candidates, key=score_case_dir)
    if score_case_dir(best) == 0:
        raise FileNotFoundError(
            f"Found folder(s) for {case_id}, but none contains complete A/D/P/S(or PS) image-label pairs."
        )
    return best


def save_modality_slices(
    case_id: str,
    modality_name: str,
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    contour_thickness: int,
) -> int:
    image_vol = load_nii(image_path)
    label_vol = load_nii(label_path)

    if image_vol.ndim != 3 or label_vol.ndim != 3:
        raise ValueError(
            f"Only 3D NIfTI is supported, but got {image_path.name}:{image_vol.shape}, "
            f"{label_path.name}:{label_vol.shape}"
        )

    slice_count = min(image_vol.shape[2], label_vol.shape[2])
    if slice_count == 0:
        raise ValueError(f"No valid slices found in {image_path} or {label_path}")

    for slice_idx in range(slice_count):
        image_slice = image_vol[:, :, slice_idx]
        label_slice = label_vol[:, :, slice_idx]
        image_u8 = normalize_to_uint8(image_slice)
        overlay = overlay_contour(
            image_u8=image_u8,
            label_slice=label_slice,
            contour_thickness=contour_thickness,
        )
        out_path = output_dir / f"{modality_name}-slice-{slice_idx:03d}.png"
        Image.fromarray(overlay).save(out_path)

    return slice_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a case folder under I:\\ by --ID, overlay labels on A/D/P/S(or PS) NIfTI slices, and save PNGs."
    )
    parser.add_argument("--ID", required=True, help="Case folder name to search for.")
    parser.add_argument(
        "--input_root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=r"Root directory used to search case folders. Default: I:\ ",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output root directory.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=1,
        help="Contour thickness in pixels, must be >= 1.",
    )
    args = parser.parse_args()

    if args.thickness < 1:
        raise ValueError("--thickness must be >= 1")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    case_dir = resolve_case_dir(args.input_root, args.ID)
    modality_files = find_modality_files(case_dir)

    case_output_dir = args.output_dir / args.ID
    case_output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[str] = []
    for modality_name, (image_path, label_path) in modality_files.items():
        saved_count = save_modality_slices(
            case_id=args.ID,
            modality_name=modality_name,
            image_path=image_path,
            label_path=label_path,
            output_dir=case_output_dir,
            contour_thickness=args.thickness,
        )
        summary.append(
            f"{modality_name}: {saved_count} slices | image={image_path.name} | label={label_path.name}"
        )

    print(f"[CASE] {args.ID}")
    print(f"[SOURCE] {case_dir}")
    print(f"[OUTPUT] {case_output_dir.resolve()}")
    for line in summary:
        print(f"[SAVE] {line}")


if __name__ == "__main__":
    main()

# python visualization_original.py --ID CT171573 --input_root I:\ --output_dir ./visualization_output --thickness 1