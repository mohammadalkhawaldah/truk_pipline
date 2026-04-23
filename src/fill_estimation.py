from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src import config


def ensure_size_model_compat() -> None:
    # Some locally-trained truck-size weights were serialized with custom class names.
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.modules.head as head

    if not hasattr(head, "Segment26"):
        head.Segment26 = head.Segment
    if not hasattr(block, "Proto26"):
        block.Proto26 = block.Proto


def load_size_seg_model(model_path) -> YOLO:
    ensure_size_model_compat()
    return YOLO(str(model_path))


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int, int] | tuple[int, int]) -> np.ndarray:
    target_h = int(target_shape[0])
    target_w = int(target_shape[1])
    return cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def calculate_fill_percentage(box_mask: np.ndarray, content_mask: np.ndarray) -> float:
    content_mask = content_mask & box_mask
    kernel = np.ones((5, 5), np.uint8)
    content_mask = cv2.morphologyEx(
        content_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    ).astype(bool)

    box_rows = np.any(box_mask, axis=1)
    content_rows = np.any(content_mask, axis=1)

    if not np.any(box_rows):
        return 0.0

    box_top = int(np.argmax(box_rows))
    box_bottom = int(len(box_rows) - 1 - np.argmax(box_rows[::-1]))

    if not np.any(content_rows):
        return 0.0

    content_top = int(np.argmax(content_rows))
    content_top = max(content_top, box_top)

    box_height = box_bottom - box_top
    content_height = box_bottom - content_top
    if box_height <= 0:
        return 0.0

    fill = (content_height / box_height) * 100.0
    return float(max(0.0, min(fill, 100.0)))


def _normalize_label(label: str) -> str:
    return " ".join(str(label).replace("_", " ").replace("-", " ").lower().split())


def _run_size_segmentation(
    truck_crop_bgr: Any,
    size_model,
    size_names: dict[int, str],
    seg_conf_threshold: float = config.SIZE_SEG_CONF_THRESHOLD,
    box_class_names: list[str] | None = None,
    content_class_names: list[str] | None = None,
) -> dict[str, Any]:
    if truck_crop_bgr is None or getattr(truck_crop_bgr, "size", 0) == 0:
        return {
            "status": "empty_truck_crop",
            "fill_percentage": None,
            "box_detected": False,
            "content_detected": False,
            "labels": [],
            "box_mask_resized": None,
            "content_mask_resized": None,
        }

    box_targets = {
        _normalize_label(x) for x in (box_class_names if box_class_names is not None else config.SIZE_BOX_CLASS_NAMES)
    }
    content_targets = {
        _normalize_label(x)
        for x in (content_class_names if content_class_names is not None else config.SIZE_CONTENT_CLASS_NAMES)
    }

    seg_result = size_model.predict(
        source=truck_crop_bgr,
        device="cpu",
        conf=float(seg_conf_threshold),
        verbose=False,
    )[0]

    boxes = seg_result.boxes
    masks = seg_result.masks
    if boxes is None or len(boxes) == 0 or masks is None or masks.data is None:
        return {
            "status": "no_segmentation_detected",
            "fill_percentage": 0.0,
            "box_detected": False,
            "content_detected": False,
            "labels": [],
            "box_mask_resized": None,
            "content_mask_resized": None,
        }

    mask_data = masks.data.cpu().numpy()
    labels: list[str] = []
    truck_box_mask = None
    content_mask = None
    box_count = 0
    content_count = 0

    classes = boxes.cls.cpu().numpy()
    for index, cls in enumerate(classes):
        class_name = size_names.get(int(cls), f"class_{int(cls)}")
        labels.append(class_name)
        normalized = _normalize_label(class_name)
        if normalized in box_targets:
            truck_box_mask = mask_data[index]
            box_count += 1
        elif normalized in content_targets:
            content_mask = mask_data[index]
            content_count += 1

    if truck_box_mask is None:
        return {
            "status": "no_box_detected",
            "fill_percentage": 0.0,
            "box_detected": False,
            "content_detected": content_mask is not None,
            "labels": labels,
            "box_detection_count": box_count,
            "content_detection_count": content_count,
            "box_mask_resized": None,
            "content_mask_resized": None,
        }

    box_mask_resized = resize_mask(truck_box_mask, truck_crop_bgr.shape)
    content_detected = content_mask is not None
    if content_detected:
        content_mask_resized = resize_mask(content_mask, truck_crop_bgr.shape)
        fill_percentage = calculate_fill_percentage(box_mask_resized, content_mask_resized)
        content_area_px = int(content_mask_resized.sum())
    else:
        fill_percentage = 0.0
        content_area_px = 0
        content_mask_resized = None

    return {
        "status": "ok",
        "fill_percentage": float(fill_percentage),
        "box_detected": True,
        "content_detected": bool(content_detected),
        "labels": labels,
        "box_detection_count": box_count,
        "content_detection_count": content_count,
        "box_area_px": int(box_mask_resized.sum()),
        "content_area_px": int(content_area_px),
        "box_mask_resized": box_mask_resized,
        "content_mask_resized": content_mask_resized,
    }


def estimate_fill_for_truck_crop(
    truck_crop_bgr: Any,
    size_model,
    size_names: dict[int, str],
    seg_conf_threshold: float = config.SIZE_SEG_CONF_THRESHOLD,
    box_class_names: list[str] | None = None,
    content_class_names: list[str] | None = None,
) -> dict[str, Any]:
    result = _run_size_segmentation(
        truck_crop_bgr=truck_crop_bgr,
        size_model=size_model,
        size_names=size_names,
        seg_conf_threshold=seg_conf_threshold,
        box_class_names=box_class_names,
        content_class_names=content_class_names,
    )
    result.pop("box_mask_resized", None)
    result.pop("content_mask_resized", None)
    return result


def render_fill_overlay_for_truck_crop(
    truck_crop_bgr: Any,
    size_model,
    size_names: dict[int, str],
    seg_conf_threshold: float = config.SIZE_SEG_CONF_THRESHOLD,
    box_class_names: list[str] | None = None,
    content_class_names: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    result = _run_size_segmentation(
        truck_crop_bgr=truck_crop_bgr,
        size_model=size_model,
        size_names=size_names,
        seg_conf_threshold=seg_conf_threshold,
        box_class_names=box_class_names,
        content_class_names=content_class_names,
    )
    overlay_crop = None
    if truck_crop_bgr is not None and getattr(truck_crop_bgr, "size", 0) > 0:
        overlay_crop = truck_crop_bgr.copy()
        box_mask_resized = result.get("box_mask_resized")
        content_mask_resized = result.get("content_mask_resized")
        if box_mask_resized is not None:
            overlay = overlay_crop.copy()
            overlay[box_mask_resized] = (255, 0, 0)
            if content_mask_resized is not None:
                overlay[content_mask_resized] = (0, 255, 0)
            overlay_crop = cv2.addWeighted(overlay, 0.4, overlay_crop, 0.6, 0)

    result.pop("box_mask_resized", None)
    result.pop("content_mask_resized", None)
    return overlay_crop, result
