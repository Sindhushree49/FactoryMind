"""Image loading and normalisation for visual inspection.

MVTec AD images are RGB PNGs of varying size (7 px to 1024 px per side). This
module resizes them consistently and records exactly what was done, so an
inspection report can state the preprocessing that produced its numbers.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_SIDE = 32
MAX_BYTES = 25 * 1024 * 1024
ALLOWED_FORMATS = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}


class InvalidImage(ValueError):
    """Raised when an upload is not a usable, licence-clean image."""


@dataclass
class PreparedImage:
    rgb: np.ndarray                      # HxWx3 uint8
    original_size: tuple[int, int]
    resized_to: tuple[int, int]
    was_resized: bool
    original_format: str
    filename: str
    mean: float
    std: float
    warnings: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "original_size": list(self.original_size),
            "resized_to": list(self.resized_to),
            "was_resized": self.was_resized,
            "format": self.original_format,
            "pixel_mean": round(self.mean, 4),
            "pixel_std": round(self.std, 4),
            "warnings": self.warnings,
        }


def validate_upload(filename: str, data: bytes) -> None:
    if not data:
        raise InvalidImage("Uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise InvalidImage(
            f"File is {len(data) / 1e6:.1f} MB; the limit is {MAX_BYTES / 1e6:.0f} MB."
        )
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_FORMATS:
        raise InvalidImage(
            f"Unsupported file type '.{extension}'. Allowed: {', '.join(sorted(ALLOWED_FORMATS))}"
        )


def load_and_prepare(data: bytes, filename: str, image_size: int = 256) -> PreparedImage:
    """Decode, convert to RGB, and resize to ``image_size`` on the short side."""
    import cv2

    validate_upload(filename, data)
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise InvalidImage("The uploaded file could not be decoded as an image.")

    warnings: list[str] = []

    # Normalise exotic channel layouts to 3-channel BGR.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        warnings.append("Alpha channel discarded.")
    elif image.shape[2] != 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2BGR)

    original_h, original_w = image.shape[:2]
    if min(original_h, original_w) < MIN_SIDE:
        raise InvalidImage(
            f"Image is {original_w}x{original_h} px; the minimum usable side is {MIN_SIDE} px."
        )

    # Resize the short side to image_size while preserving aspect ratio, then
    # centre-crop to a square. MVTec images are already square, but user uploads
    # are not, so this keeps patch geometry consistent across sources.
    scale = image_size / min(original_h, original_w)
    new_w, new_h = max(image_size, int(round(original_w * scale))), max(
        image_size, int(round(original_h * scale))
    )
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    side = min(new_w, new_h)
    top = (new_h - side) // 2
    left = (new_w - side) // 2
    cropped = resized[top : top + side, left : left + side]
    final = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if (original_w, original_h) != (image_size, image_size):
        warnings.append(
            f"Resized from {original_w}x{original_h} to {image_size}x{image_size} "
            f"(short-side scale + centre crop)."
        )

    rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    grey = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    return PreparedImage(
        rgb=rgb,
        original_size=(original_w, original_h),
        resized_to=(image_size, image_size),
        was_resized=True,
        original_format=(filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"),
        filename=filename,
        mean=float(grey.mean()),
        std=float(grey.std()),
        warnings=warnings,
    )


def encode_png(rgb: np.ndarray) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise InvalidImage("Failed to encode the result image as PNG.")
    return buffer.tobytes()
