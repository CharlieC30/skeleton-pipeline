"""Step 3: clean the binary mask with 3D morphological filters.

Workflow: remove small objects, opening, closing, fill holes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import tifffile
from natsort import natsorted
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
from skimage.morphology import remove_small_objects

STEP_NAME = "03_clean"
STEP_DESCRIPTION = "Clean"


def get_structure_3d(radius: int) -> np.ndarray:
    """Generate 3D cubic structure element.

    Args:
        radius: Radius of structure element.

    Returns:
        3D array of ones with shape (2*radius+1, 2*radius+1, 2*radius+1).
    """
    size = 2 * radius + 1
    return np.ones((size, size, size))


def clean_mask(
    image: np.ndarray,
    opening_radius: int = 1,
    closing_radius: int = 2,
    min_size: int = 64,
    skip_remove_small: bool = False,
    skip_fill_holes: bool = False,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Clean binary mask with morphological operations.

    Workflow: remove small objects, opening, closing, fill holes.

    Args:
        image: Input 3D binary mask with shape (Z, Y, X).
        opening_radius: Radius for opening operation (0 to skip).
        closing_radius: Radius for closing operation (0 to skip).
        min_size: Min object size in voxels.
        skip_remove_small: Skip remove small objects step.
        skip_fill_holes: Skip fill holes step.
        logger: Logger instance.

    Returns:
        Cleaned binary mask (uint8, values 0 or 255).
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got {image.ndim}D.")

    binary = image > 0

    if not skip_remove_small:
        if logger:
            logger.debug(f"Removing small objects (min_size={min_size})")
        t0 = time.time()
        binary = remove_small_objects(binary, min_size=min_size)
        if logger:
            logger.debug(f"  Completed in {time.time() - t0:.1f}s")

    if opening_radius > 0:
        struct = get_structure_3d(opening_radius)
        if logger:
            logger.debug(f"Applying binary opening (radius={opening_radius})")
        t0 = time.time()
        binary = binary_opening(binary, structure=struct)
        if logger:
            logger.debug(f"  Completed in {time.time() - t0:.1f}s")

    if closing_radius > 0:
        struct = get_structure_3d(closing_radius)
        if logger:
            logger.debug(f"Applying binary closing (radius={closing_radius})")
        t0 = time.time()
        binary = binary_closing(binary, structure=struct)
        if logger:
            logger.debug(f"  Completed in {time.time() - t0:.1f}s")

    if not skip_fill_holes:
        if logger:
            logger.debug("Filling holes")
        t0 = time.time()
        binary = binary_fill_holes(binary, structure=np.ones((3, 3, 3)))
        if logger:
            logger.debug(f"  Completed in {time.time() - t0:.1f}s")

    return binary.astype(np.uint8) * 255


def run(input_path: str, output_dir: str, config: dict, logger: logging.Logger) -> str:
    """Run step 3: clean each binary mask.

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

    # Get config parameters
    clean_config = config.get("clean", {})
    opening_radius = clean_config.get("opening_radius", 1)
    closing_radius = clean_config.get("closing_radius", 2)
    min_size = clean_config.get("min_size", 64)

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(
        f"Parameters: opening={opening_radius}, closing={closing_radius}, min_size={min_size}"
    )

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

        cleaned = clean_mask(
            image,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
            min_size=min_size,
            logger=logger,
        )

        base = tif_file.stem.removesuffix("_threshold")
        out_file = output_path / f"{base}_clean.tif"
        tifffile.imwrite(str(out_file), cleaned, imagej=True, metadata={"axes": "ZYX"})
        logger.debug(f"  Saved to: {out_file}")

    return str(output_path)
