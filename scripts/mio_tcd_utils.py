"""Shared, notebook-only utilities for the MIO-TCD Localization workflow.

This module deliberately has no dependency on the production video pipeline.
Raw MIO-TCD files are read only; all generated artefacts live below PROJECT_ROOT.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
CLASS_TO_ID = {name: index for index, name in enumerate(TARGET_NAMES)}
PROJECT_TO_COCO_ID = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 7}
CLASS_MAPPING = {
    "car": "car", "bus": "bus", "bicycle": "bicycle", "motorcycle": "motorcycle",
    "pedestrian": "person", "pickup_truck": "truck", "single_unit_truck": "truck",
    "articulated_truck": "truck",
}
IGNORED_CLASSES = {"work_van", "motorized_vehicle", "non-motorized_vehicle"}
CSV_COLUMNS = ["image_id", "original_class", "x1", "y1", "x2", "y2"]


def resolve_dataset_root(override=None) -> Path:
    """Resolve MIO_TCD_ROOT first, then an optional local project config."""
    configured = os.environ.get("MIO_TCD_ROOT") or override
    config_path = PROJECT_ROOT / "data" / "mio_tcd" / "dataset_config.yaml"
    if not configured and config_path.is_file():
        configured = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("dataset_root")
    if not configured:
        raise RuntimeError("Set MIO_TCD_ROOT, add data/mio_tcd/dataset_config.yaml, or set DATASET_ROOT_OVERRIDE.")
    root = Path(configured).expanduser()
    required = [root, root / "train", root / "gt_train.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("MIO-TCD dataset is incomplete: " + ", ".join(missing))
    return root.resolve()


def read_annotations(dataset_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(dataset_root / "gt_train.csv", header=None, names=CSV_COLUMNS,
                        dtype={"image_id": str, "original_class": str})
    if frame.shape[1] != len(CSV_COLUMNS) or frame.empty:
        raise ValueError("gt_train.csv must be non-empty and use image_id,class,x1,y1,x2,y2 with no header.")
    frame["mapped_class"] = frame["original_class"].map(CLASS_MAPPING)
    for col in ["x1", "y1", "x2", "y2"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def image_path(train_dir: Path, image_id: str) -> Path | None:
    """Find one image without recursively scanning the raw dataset."""
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"):
        candidate = train_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    matches = list(train_dir.glob(f"{image_id}.*"))
    return matches[0].resolve() if len(matches) == 1 and matches[0].is_file() else None


def build_inventory(annotations: pd.DataFrame, dataset_root: Path, progress=None) -> pd.DataFrame:
    rows = []
    groups = annotations.groupby("image_id", sort=True)
    iterator = progress(groups, total=groups.ngroups, desc="Inspecting images") if progress else groups
    for image_id, group in iterator:
        path = image_path(dataset_root / "train", image_id)
        width = height = 0
        exists = path is not None
        if exists:
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                exists = False
        finite = group[["x1", "y1", "x2", "y2"]].notna().all(axis=1)
        bbox_ok = finite & (group.x2 > group.x1) & (group.y2 > group.y1)
        if exists:
            bbox_ok &= (group.x1 >= 0) & (group.y1 >= 0) & (group.x2 <= width) & (group.y2 <= height)
        targets = group.mapped_class.isin(TARGET_NAMES)
        classes = set(group.loc[targets & bbox_ok, "mapped_class"])
        row = {"image_id": image_id, "image_path": str(path) if path else "", "width": width, "height": height,
               "num_objects": len(group), "num_target_objects": int((targets & bbox_ok).sum()),
               "ignored_object_count": int(group.original_class.isin(IGNORED_CLASSES).sum()),
               "invalid_bbox_count": int((~bbox_ok).sum()), "source_classes": ";".join(sorted(group.original_class.unique())),
               "valid": bool(exists and bbox_ok.all() and not group.duplicated(CSV_COLUMNS[1:]).any())}
        row.update({f"has_{name}": name in classes for name in TARGET_NAMES})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


def assert_inventory(inventory: pd.DataFrame) -> None:
    if inventory.empty or not inventory.image_id.is_unique:
        raise ValueError("Inventory must contain unique image IDs.")
    if "valid" not in inventory.columns:
        raise ValueError("Inventory must contain a valid column.")


def filter_valid_inventory(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude invalid images as whole image-level units before splitting."""
    if inventory.empty or not inventory.image_id.is_unique or "valid" not in inventory.columns:
        raise ValueError("Inventory must contain unique image IDs and a valid column.")
    valid_mask = inventory["valid"]
    if valid_mask.dtype != bool:
        normalized = valid_mask.astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError("Inventory valid column must contain only true/false values.")
        valid_mask = normalized.eq("true")
    eligible = inventory.loc[valid_mask].copy().reset_index(drop=True)
    excluded = inventory.loc[~valid_mask].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No valid images remain after inventory filtering.")
    return eligible, excluded


def _greedy_multilabel_split(inventory: pd.DataFrame, seed=42, ratios=(.70, .15, .15)) -> pd.Series:
    """Deterministic quota-aware fallback when iterstrat is unavailable.

    It assigns rare-label images first and minimizes class/image quota deviation.
    """
    rng = random.Random(seed)
    labels = inventory[[f"has_{name}" for name in TARGET_NAMES]].astype(int).to_numpy()
    n = len(inventory); split_names = ["train", "val", "test"]
    desired_n = np.array(ratios) * n
    desired_labels = np.outer(np.array(ratios), labels.sum(axis=0))
    order = list(range(n)); rng.shuffle(order)
    order.sort(key=lambda i: (labels[i].sum(), (labels[i] / np.maximum(labels.sum(axis=0), 1)).sum()), reverse=True)
    assigned = np.full(n, -1, dtype=int); counts = np.zeros(3); class_counts = np.zeros((3, len(TARGET_NAMES)))
    for i in order:
        costs = []
        for s in range(3):
            size_cost = ((counts[s] + 1 - desired_n[s]) / max(desired_n[s], 1)) ** 2
            label_cost = np.square((class_counts[s] + labels[i] - desired_labels[s]) / np.maximum(desired_labels[s], 1)).sum()
            overflow = max(0, counts[s] + 1 - np.ceil(desired_n[s])) * 100
            costs.append(size_cost + label_cost + overflow)
        choice = int(np.argmin(costs)); assigned[i] = choice; counts[choice] += 1; class_counts[choice] += labels[i]
    return pd.Series([split_names[index] for index in assigned], index=inventory.index, name="split")


def create_split(inventory: pd.DataFrame, seed=42, ratios=(.70, .15, .15)) -> pd.DataFrame:
    assert_inventory(inventory)
    inventory, _ = filter_valid_inventory(inventory)
    manifest = inventory.copy()
    manifest["split"] = _greedy_multilabel_split(manifest, seed, ratios)
    if manifest.image_id.duplicated().any() or manifest.split.isna().any():
        raise ValueError("Split assignment failed uniqueness validation.")
    return manifest


def split_report(manifest: pd.DataFrame, annotations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_report = manifest.groupby("split").size().rename("images").reindex(["train", "val", "test"], fill_value=0).to_frame()
    image_report["ratio"] = image_report.images / max(len(manifest), 1)
    lookup = manifest.set_index("image_id").split
    mapped = annotations.loc[annotations.mapped_class.isin(TARGET_NAMES)].copy()
    mapped["split"] = mapped.image_id.map(lookup)
    bbox = mapped.pivot_table(index="split", columns="mapped_class", values="image_id", aggfunc="size", fill_value=0).reindex(index=["train", "val", "test"], columns=TARGET_NAMES, fill_value=0)
    return image_report, bbox


def save_split(manifest: pd.DataFrame, dataset_root: Path, seed=42, ratios=(.70, .15, .15), force=False,
               excluded: pd.DataFrame | None = None) -> Path:
    output = PROJECT_ROOT / "data" / "mio_tcd" / "splits"; output.mkdir(parents=True, exist_ok=True)
    expected = [output / name for name in ("train.txt", "val.txt", "test.txt", "split_manifest.csv", "split_config.yaml")]
    if any(path.exists() for path in expected) and not force:
        raise FileExistsError("Split outputs already exist. Set FORCE_REBUILD=True only to intentionally replace them.")
    for split in ("train", "val", "test"):
        paths = manifest.loc[manifest.split.eq(split), "image_path"]
        (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    columns = ["image_id", "image_path", "split", "width", "height", "num_objects", "num_target_objects"] + [f"has_{name}" for name in TARGET_NAMES]
    manifest[columns].to_csv(output / "split_manifest.csv", index=False)
    if excluded is not None:
        excluded.to_csv(output / "excluded_invalid_images.csv", index=False)
    config = {"dataset": "MIO-TCD Localization", "split_version": 1, "seed": seed,
              "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
              "class_mapping": CLASS_MAPPING, "ignored_classes": sorted(IGNORED_CLASSES),
              "selection_method": "deterministic quota-aware multilabel greedy fallback; image-level, no sequence metadata available",
              "excluded_invalid_images": 0 if excluded is None else len(excluded),
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_root": str(dataset_root)}
    (output / "split_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output


def _remove_generated_path(path: Path) -> None:
    """Remove only a generated link/view, never its raw-data target."""
    is_junction = bool(getattr(path, "is_junction", lambda: False)())
    if is_junction:
        os.rmdir(path)
    elif path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _create_directory_view(source: Path, target: Path, force=False) -> bool:
    """Create one zero-copy directory view; return False for per-file fallback."""
    if target.exists() or target.is_symlink() or bool(getattr(target, "is_junction", lambda: False)()):
        try:
            if os.path.samefile(source, target):
                return True
        except OSError:
            pass
        if not force:
            raise FileExistsError(f"Image view already exists: {target}")
        _remove_generated_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target, target_is_directory=True)
        return True
    except OSError:
        pass
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(target), str(source)],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0 and target.exists():
            return True
    target.mkdir(parents=True, exist_ok=True)
    return False


def _create_hardlink(source: Path, target: Path) -> None:
    if target.exists():
        if os.path.samefile(source, target):
            return
        raise FileExistsError(f"Image reference already exists: {target}")
    try:
        os.link(source, target)
    except OSError as exc:
        raise RuntimeError(f"Cannot create zero-copy reference for {source}.") from exc


def _normalize_manifest_image_ids(manifest: pd.DataFrame) -> pd.DataFrame:
    """Recover exact string IDs (including leading zeroes) from image filenames."""
    if "image_path" not in manifest.columns:
        raise ValueError("Split manifest must contain image_path.")
    normalized = manifest.copy()
    normalized["image_id"] = normalized["image_path"].map(lambda value: PureWindowsPath(str(value)).stem)
    if normalized.image_id.eq("").any() or not normalized.image_id.is_unique:
        raise ValueError("Cannot recover unique image IDs from split manifest image paths.")
    return normalized


def prepare_yolo(annotations: pd.DataFrame, manifest: pd.DataFrame, force=False) -> Path:
    root = PROJECT_ROOT / "data" / "mio_tcd" / "yolo"; label_root = root / "labels"
    if root.exists() and any(label_root.glob("*/*.txt")) and not force:
        raise FileExistsError("YOLO labels already exist. Set FORCE_REBUILD=True to intentionally replace them.")
    if force and label_root.exists():
        _remove_generated_path(label_root)
    manifest = _normalize_manifest_image_ids(manifest)
    lookup = manifest.set_index("image_id")
    if not lookup.index.is_unique or set(manifest.split) - {"train", "val", "test"}:
        raise ValueError("Invalid split manifest.")
    required = {"image_id", "image_path", "split", "width", "height"}
    if required - set(manifest.columns):
        raise ValueError(f"Split manifest lacks columns: {sorted(required - set(manifest.columns))}. Re-run notebook 02.")
    mapped = annotations[annotations.mapped_class.isin(TARGET_NAMES)].merge(
        manifest[["image_id", "width", "height"]], on="image_id", how="inner", validate="many_to_one")
    mapped["class_id"] = mapped.mapped_class.map(CLASS_TO_ID)
    mapped["xc"] = (mapped.x1 + mapped.x2) / 2 / mapped.width
    mapped["yc"] = (mapped.y1 + mapped.y2) / 2 / mapped.height
    mapped["w"] = (mapped.x2 - mapped.x1) / mapped.width
    mapped["h"] = (mapped.y2 - mapped.y1) / mapped.height
    coords = mapped[["xc", "yc", "w", "h"]]
    if not (coords.notna().all().all() and coords.ge(0).all().all() and coords.le(1).all().all()
            and mapped.w.gt(0).all() and mapped.h.gt(0).all()):
        raise ValueError("Mapped annotations contain invalid normalized bounding boxes.")
    mapped["yolo_line"] = [f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}"
                           for class_id, x, y, w, h in mapped[["class_id", "xc", "yc", "w", "h"]].itertuples(index=False, name=None)]
    labels_by_image = mapped.groupby("image_id", sort=False).yolo_line.agg("\n".join).to_dict()
    from tqdm.auto import tqdm
    for split in ("train", "val", "test"):
        folder = label_root / split; folder.mkdir(parents=True, exist_ok=True)
        image_folder = root / "images" / split
        directory_view = _create_directory_view(Path(manifest.iloc[0].image_path).parent, image_folder, force)
        references = []
        records = manifest.loc[manifest.split.eq(split)]
        for record in tqdm(records.itertuples(index=False), total=len(records), desc=f"Writing {split} labels"):
            source = Path(record.image_path)
            reference = image_folder / source.name
            if not directory_view:
                _create_hardlink(source, reference)
            references.append(str(reference.absolute()))
            text = labels_by_image.get(record.image_id, "")
            (folder / f"{record.image_id}.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
        (root / f"{split}.txt").write_text("\n".join(references) + "\n", encoding="utf-8")
    config = {"path": str(PROJECT_ROOT), "train": str((root / "train.txt").resolve()),
              "val": str((root / "val.txt").resolve()), "test": str((root / "test.txt").resolve()),
              "names": {i: name for i, name in enumerate(TARGET_NAMES)}}
    (root / "mio_tcd.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    validate_yolo(manifest, label_root)
    return root


def validate_yolo(manifest: pd.DataFrame, label_root: Path) -> None:
    manifest = _normalize_manifest_image_ids(manifest)
    if manifest.image_path.duplicated().any() or manifest.image_id.duplicated().any():
        raise ValueError("Duplicate image in manifest.")
    for record in manifest.itertuples():
        if not Path(record.image_path).is_file(): raise FileNotFoundError(record.image_path)
        label = label_root / record.split / f"{record.image_id}.txt"
        if not label.is_file(): raise FileNotFoundError(label)
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5 or int(fields[0]) not in range(len(TARGET_NAMES)):
                raise ValueError(f"Invalid YOLO label: {label}")
            coords = [float(value) for value in fields[1:]]
            if not (all(0 <= value <= 1 for value in coords) and coords[2] > 0 and coords[3] > 0):
                raise ValueError(f"Out-of-range YOLO bbox: {label}")


def prepare_coco_baseline_eval(manifest: pd.DataFrame, yolo_root: Path, model_names: dict,
                               force=False) -> Path:
    """Remap six-class labels to native COCO IDs for valid pretrained evaluation."""
    manifest = _normalize_manifest_image_ids(manifest)
    test_manifest = manifest.loc[manifest.split.eq("test")].copy()
    if test_manifest.empty:
        raise ValueError("Frozen test split is empty.")
    root = PROJECT_ROOT / "data" / "mio_tcd" / "yolo_coco_baseline"
    yaml_path = root / "mio_tcd_coco_eval.yaml"
    label_root = root / "labels" / "test"
    if yaml_path.is_file() and label_root.is_dir() and not force:
        return yaml_path
    if force and root.exists():
        _remove_generated_path(root)
    label_root.mkdir(parents=True, exist_ok=True)
    image_view = root / "images" / "test"
    source_dir = Path(test_manifest.iloc[0].image_path).parent
    directory_view = _create_directory_view(source_dir, image_view, force=True)
    references = []
    from tqdm.auto import tqdm
    for record in tqdm(test_manifest.itertuples(index=False), total=len(test_manifest),
                       desc="Preparing COCO-ID baseline labels"):
        source_image = Path(record.image_path)
        reference = image_view / source_image.name
        if not directory_view:
            _create_hardlink(source_image, reference)
        references.append(str(reference.absolute()))
        source_label = yolo_root / "labels" / "test" / f"{record.image_id}.txt"
        if not source_label.is_file():
            raise FileNotFoundError(source_label)
        converted = []
        for line in source_label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5 or int(fields[0]) not in PROJECT_TO_COCO_ID:
                raise ValueError(f"Invalid project-class label: {source_label}")
            converted.append(" ".join([str(PROJECT_TO_COCO_ID[int(fields[0])]), *fields[1:]]))
        (label_root / f"{record.image_id}.txt").write_text(
            "\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
    test_list = root / "test.txt"
    test_list.write_text("\n".join(references) + "\n", encoding="utf-8")
    config = {"path": str(PROJECT_ROOT), "test": str(test_list.absolute()),
              "names": {int(index): name for index, name in model_names.items()}}
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path


def metric_tables(results, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    box = results.box
    overall = pd.DataFrame([{"model": model_name, "precision": float(box.mp), "recall": float(box.mr),
                             "mAP50": float(box.map50), "mAP50-95": float(box.map),
                             "latency_ms": float(sum(getattr(results, "speed", {}).values()))}])
    class_ids = [int(value) for value in box.ap_class_index]
    names = {int(index): name for index, name in results.names.items()}
    per = pd.DataFrame({"class": [names[index] for index in class_ids],
                        "precision": np.asarray(box.p), "recall": np.asarray(box.r),
                        "mAP50": np.asarray(box.ap50), "mAP50-95": np.asarray(box.ap)})
    per = per.loc[per["class"].isin(TARGET_NAMES)].set_index("class").reindex(TARGET_NAMES).reset_index()
    return overall, per
