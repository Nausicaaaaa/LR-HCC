from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TRAINER_DIRS = [
    PROJECT_ROOT / "nnunetv2" / "training" / "nnUNetTrainer",
    PROJECT_ROOT / "nnunetv2" / "training" / "nnUnetTrainer",
]
LOCAL_LOSS_DIR = PROJECT_ROOT / "nnunetv2" / "training" / "loss"


def _filtered_sys_path() -> list[str]:
    repo_root = str(PROJECT_ROOT.resolve()).casefold()
    filtered: list[str] = []

    for entry in sys.path:
        candidate = Path(entry) if entry else Path.cwd()
        try:
            resolved = str(candidate.resolve()).casefold()
        except OSError:
            filtered.append(entry)
            continue

        if resolved != repo_root:
            filtered.append(entry)

    return filtered


def _import_official_module(module_name: str):
    original_sys_path = sys.path[:]
    sys.path = _filtered_sys_path()
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path = original_sys_path


def _append_package_path(package, extra_path: Path) -> None:
    if not extra_path.is_dir():
        return

    package_path = package.__path__
    extra_path_str = str(extra_path)

    if extra_path_str in list(package_path):
        return

    if hasattr(package_path, "append"):
        package_path.append(extra_path_str)
    else:
        package.__path__ = [*list(package_path), extra_path_str]


def _load_module_from_file(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def _find_local_trainer_class(trainer_name: str):
    for trainer_dir in LOCAL_TRAINER_DIRS:
        if not trainer_dir.is_dir():
            continue

        for file_path in trainer_dir.rglob("*.py"):
            if file_path.name == "__init__.py":
                continue

            relative_module = file_path.relative_to(trainer_dir).with_suffix("").parts
            module_name = ".".join(
                ("nnunetv2", "training", "nnUNetTrainer", *relative_module)
            )
            module = _load_module_from_file(module_name, file_path)
            trainer_class = getattr(module, trainer_name, None)
            if trainer_class is not None:
                return trainer_class

    return None


def _patch_trainer_lookup(run_training_module) -> None:
    original_recursive_find = run_training_module.recursive_find_python_class

    def patched_recursive_find_python_class(folder, class_name, current_module):
        if current_module == "nnunetv2.training.nnUNetTrainer":
            trainer_class = _find_local_trainer_class(class_name)
            if trainer_class is not None:
                return trainer_class

        return original_recursive_find(folder, class_name, current_module)

    run_training_module.recursive_find_python_class = patched_recursive_find_python_class

    try:
        find_class_module = _import_official_module("nnunetv2.utilities.find_class_by_name")
    except ModuleNotFoundError:
        return

    find_class_module.recursive_find_python_class = patched_recursive_find_python_class


def main() -> None:
    try:
        run_training_module = _import_official_module("nnunetv2.run.run_training")
        trainer_package = _import_official_module("nnunetv2.training.nnUNetTrainer")
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "nnunetv2"
        raise SystemExit(
            f"Cannot import {missing_module}. Please activate the nnU-Net environment before training."
        ) from exc

    for trainer_dir in LOCAL_TRAINER_DIRS:
        _append_package_path(trainer_package, trainer_dir)

    try:
        loss_package = _import_official_module("nnunetv2.training.loss")
    except ModuleNotFoundError:
        loss_package = None

    if loss_package is not None:
        _append_package_path(loss_package, LOCAL_LOSS_DIR)

    _patch_trainer_lookup(run_training_module)

    entrypoint = getattr(run_training_module, "run_training_entry", None)
    if entrypoint is None:
        entrypoint = getattr(run_training_module, "main", None)

    if entrypoint is None:
        raise SystemExit("Could not find nnU-Net training entrypoint.")

    entrypoint()


if __name__ == "__main__":
    main()
