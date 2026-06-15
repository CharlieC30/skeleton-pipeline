"""Step 2: binarize with Otsu thresholding.

Uses one global threshold for the whole 3D volume.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tifffile
from natsort import natsorted
from skimage.filters import threshold_otsu

STEP_NAME = "02_threshold"
STEP_DESCRIPTION = "Threshold"


def calculate_stack_otsu_threshold(image: np.ndarray, logger: logging.Logger) -> float:
    """Compute single Otsu threshold from entire 3D stack.

    Args:
        image: 3D numpy array with shape (Z, Y, X).
        logger: Logger instance.

    Returns:
        Otsu threshold value.
    """
    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image, got {image.ndim}D. Run the normalize step first."
        )

    # Determine histogram range based on dtype
    if image.dtype == np.uint8:
        hist_range = (0, 256)
        bins = 256
    else:
        logger.warning(
            f"Input dtype is {image.dtype}, expected uint8. "
            "Consider running the normalize step first."
        )
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            hist_range = (info.min, info.max + 1)
            bins = min(256, info.max - info.min + 1)
        else:
            hist_range = (float(image.min()), float(image.max()))
            bins = 256

    # Compute histogram to avoid high memory usage
    counts, _ = np.histogram(image, bins=bins, range=hist_range)
    if np.count_nonzero(counts) <= 1:
        raise ValueError(
            "Cannot compute Otsu threshold: the image is empty or constant-valued "
            "(only one intensity present). Check the input or the normalize step."
        )
    threshold = threshold_otsu(hist=counts)
    return float(threshold)


def apply_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    """Apply threshold to create binary mask (0/255).

    Args:
        image: Input numpy array.
        threshold: Threshold value.

    Returns:
        Binary uint8 array with values 0 or 255.
    """
    return (image >= threshold).astype(np.uint8) * 255


def run(input_path: str, output_dir: str, config: dict, logger: logging.Logger) -> str:
    """Run step 2: threshold each volume into a binary mask.

    Args:
        input_path: Path to input (previous step output directory).
        output_dir: Base output directory.
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        Output directory path for this step.
    """
    # Input is the previous step's output directory
    input_dir = Path(input_path)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path = Path(output_dir) / STEP_NAME
    output_path.mkdir(parents=True, exist_ok=True)

    method = config.get("threshold", {}).get("method", "otsu")
    if method != "otsu":
        raise ValueError(f"Unsupported threshold method: {method}")

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_path}")

    tif_files = natsorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() in (".tif", ".tiff")]
    )

    if not tif_files:
        raise ValueError(f"No TIF files found in: {input_dir}")

    logger.info(f"Found {len(tif_files)} TIF files")

    for idx, tif_file in enumerate(tif_files, start=1):
        logger.info(f"[{idx}/{len(tif_files)}] Processing: {tif_file.name}")

        image = tifffile.imread(str(tif_file))
        logger.debug(f"  Shape: {image.shape}, dtype: {image.dtype}")

        threshold = calculate_stack_otsu_threshold(image, logger)
        logger.debug(f"  Otsu threshold: {threshold:.2f}")

        binary = apply_threshold(image, threshold)

        base = tif_file.stem.removesuffix("_normalized")
        out_file = output_path / f"{base}_threshold.tif"
        tifffile.imwrite(str(out_file), binary, imagej=True, metadata={"axes": "ZYX"})
        logger.debug(f"  Saved to: {out_file}")

    return str(output_path)
