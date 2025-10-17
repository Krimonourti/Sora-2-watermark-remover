from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class WatermarkRemovalError(RuntimeError):
    """Raised when the watermark cannot be removed."""


@dataclass
class WatermarkDetection:
    top_left: Tuple[int, int]
    size: Tuple[int, int]
    confidence: float

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        x, y = self.top_left
        w, h = self.size
        return x, y, w, h


DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "sora2_template.pgm"


def _load_template(template_path: Path) -> np.ndarray:
    if not template_path.exists():
        raise WatermarkRemovalError(
            "Watermark template not found. Please provide 'sora2_template.pgm' in the assets directory."
        )

    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise WatermarkRemovalError("Unable to read watermark template image")
    return template


def _detect_watermark(gray_frame: np.ndarray, template: np.ndarray, threshold: float = 0.35) -> Optional[WatermarkDetection]:
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return None

    h, w = template.shape[:2]
    return WatermarkDetection(top_left=(max_loc[0], max_loc[1]), size=(w, h), confidence=float(max_val))


def _build_inpaint_mask(frame_shape: Tuple[int, int], detection: WatermarkDetection) -> np.ndarray:
    mask = np.zeros(frame_shape, dtype=np.uint8)
    x, y, w, h = detection.bbox
    mask[y : y + h, x : x + w] = 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def process_video(
    input_path: Path,
    output_dir: Path,
    template_path: Optional[Path] = None,
    threshold: float = 0.35,
) -> Path:
    """Process a video to remove the Sora 2 watermark."""

    if not input_path.exists():
        raise WatermarkRemovalError("Input video does not exist")

    template = _load_template(template_path or DEFAULT_TEMPLATE)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise WatermarkRemovalError("Could not open input video")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_clean.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        capture.release()
        raise WatermarkRemovalError("Could not initialise video writer")

    detection: Optional[WatermarkDetection] = None

    while True:
        success, frame = capture.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if detection is None:
            detection = _detect_watermark(gray, template, threshold)
            if detection is None:
                writer.release()
                capture.release()
                output_path.unlink(missing_ok=True)
                raise WatermarkRemovalError(
                    "Unable to detect the Sora 2 watermark. Try adjusting the template or threshold."
                )
        else:
            updated = _detect_watermark(gray, template, threshold)
            if updated is not None and updated.confidence >= detection.confidence:
                detection = updated

        mask = _build_inpaint_mask(gray.shape, detection)
        cleaned = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        writer.write(cleaned)

    capture.release()
    writer.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise WatermarkRemovalError("Failed to generate cleaned video")

    return output_path
