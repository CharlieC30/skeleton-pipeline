"""Step 1: normalize intensities and convert to uint8.

Handles ImageJ virtual stacks and accepts a single 3D TIF or a folder of 2D slices.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import tifffile
from natsort import natsorted

STEP_NAME = "01_normalize"
STEP_DESCRIPTION = "Normalize"


def load_and_check_tif(path: str, logger: logging.Logger) -> np.ndarray:
    """Load a TIF file, handling the ImageJ virtual stack format.

    Args:
        path: Path to the TIF file.
        logger: Logger instance.

    Returns:
        3D array with shape (Z, Y, X).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as a 3D stack.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with tifffile.TiffFile(path) as tif:
            num_pages = len(tif.pages)
            is_imagej = tif.is_imagej

            if is_imagej:
                metadata = tif.imagej_metadata or {}
                expected_slices = metadata.get("slices", 1)

                # ImageJ virtual stack: single page but metadata indicates multiple slices
                if num_pages == 1 and expected_slices > 1:
                    logger.debug(
                        f"Detected ImageJ virtual stack ({expected_slices} slices)"
                    )

                    page = tif.pages[0]
                    height, width = page.shape
                    base_dtype = page.dtype
                    data_offset = page.dataoffsets[0]

                    # Map to big-endian dtype for ImageJ compatibility
                    dtype_map = {
                        np.dtype("float32"): np.dtype(">f4"),
                        np.dtype("float64"): np.dtype(">f8"),
                        np.dtype("uint16"): np.dtype(">u2"),
                        np.dtype("int16"): np.dtype(">i2"),
                    }
                    read_dtype = dtype_map.get(base_dtype, base_dtype)

                    with open(path, "rb") as f:
                        f.seek(data_offset)
                        all_data = np.fromfile(f, dtype=read_dtype)

                    pixels_per_slice = height * width
                    total_complete_slices = len(all_data) // pixels_per_slice

                    if total_complete_slices == 0:
                        raise ValueError(
                            "Insufficient data for even one complete slice"
                        )

                    complete_data = all_data[: total_complete_slices * pixels_per_slice]
                    image = complete_data.reshape(
                        (total_complete_slices, height, width)
                    )

                    if image.dtype.byteorder == ">":
                        image = image.astype(image.dtype.newbyteorder("="))

                    logger.debug(f"Loaded shape {image.shape}, dtype {image.dtype}")
                    return image

        image = tifffile.imread(path)

        if image.ndim == 2:
            raise ValueError(
                f"Input is 2D ({image.shape}). "
                "Skeletonization requires 3D data. "
                "For 2D slice sequence, provide folder path instead."
            )
        elif image.ndim != 3:
            raise ValueError(f"Expected 3D array, got {image.ndim}D")

        logger.debug(f"Loaded shape {image.shape}, dtype {image.dtype}")
        return image

    except Exception as e:
        raise ValueError(f"Failed to read TIF file {path}: {e}") from e


def normalize_array(
    array: np.ndarray,
    method: str = "minmax",
    percentile_low: float = 0.0,
    percentile_high: float = 100.0,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Normalize array values to the [0, 1] range.

    Args:
        array: Input array.
        method: 'minmax' or 'percentile'.
        percentile_low: Lower percentile for clipping (percentile method).
        percentile_high: Upper percentile for clipping (percentile method).
        logger: Logger instance.

    Returns:
        Float array with values in [0, 1].

    Raises:
        ValueError: If method is not 'minmax' or 'percentile'.
    """
    if method == "minmax":
        arr_min, arr_max = array.min(), array.max()
    elif method == "percentile":
        arr_min = np.percentile(array, percentile_low)
        arr_max = np.percentile(array, percentile_high)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if arr_min == arr_max:
        if logger:
            logger.debug("Array has constant value, returning zeros")
        return np.zeros_like(array, dtype=np.float64)

    clipped = np.clip(array, arr_min, arr_max)
    normalized = (clipped.astype(np.float64) - arr_min) / (arr_max - arr_min)
    if logger:
        logger.debug(
            f"Normalized using {method} (range [{arr_min:.2f}, {arr_max:.2f}])"
        )
    return normalized


def convert_to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert normalized [0, 1] array to uint8 [0, 255].

    Args:
        array: Input array with values in range [0, 1].

    Returns:
        Array with values in range [0, 255] as uint8 type.
    """
    if array.dtype == np.uint8:
        return array
    return (array * 255).astype(np.uint8)


def load_2d_sequence(folder_path: str, logger: logging.Logger) -> np.ndarray | None:
    """Stack a folder of 2D TIF slices into a 3D array.

    Args:
        folder_path: Path to the folder of TIF files.
        logger: Logger instance.

    Returns:
        3D array stacked from the slices, or None if the folder holds 3D TIFs
        (i.e. not a 2D sequence).

    Raises:
        ValueError: If no TIFs are found, only a single 2D file exists, or the
            slices have mismatched shapes.
    """
    tif_files = [
        f for f in os.listdir(folder_path) if f.lower().endswith((".tif", ".tiff"))
    ]
    tif_files = natsorted(tif_files)

    if not tif_files:
        raise ValueError(f"No TIF files found in {folder_path}")

    # Check first file
    first_path = os.path.join(folder_path, tif_files[0])
    try:
        first = load_and_check_tif(first_path, logger)
        return None  # It's 3D, not a 2D sequence
    except ValueError as e:
        if "2D" in str(e):
            first = tifffile.imread(first_path)
        else:
            raise

    if len(tif_files) == 1:
        raise ValueError(
            f"Single 2D file found ({first.shape}). "
            "Skeletonization requires 3D data with multiple slices."
        )

    logger.debug(f"Detected 2D sequence ({len(tif_files)} files)")

    # Stack all slices
    slices = [first]
    for f in tif_files[1:]:
        img = tifffile.imread(os.path.join(folder_path, f))
        if img.ndim != 2:
            raise ValueError(f"Mixed dimensions in sequence: {f}")
        if img.shape != first.shape:
            raise ValueError(f"Shape mismatch: {f}")
        slices.append(img)

    stack = np.stack(slices, axis=0)
    logger.debug(f"Stacked {len(slices)} slices, shape {stack.shape}")
    return stack


def run(input_path: str, output_dir: str, config: dict, logger: logging.Logger) -> str:
    """Run step 1: normalize the input to uint8.

    Args:
        input_path: Path to input TIF file or directory.
        output_dir: Output directory for this step.
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        Output directory path.
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir)
    step_output = output_dir / STEP_NAME
    step_output.mkdir(parents=True, exist_ok=True)

    # Get config parameters
    normalize_config = config.get("normalize", {})
    normalize_method = normalize_config.get("normalize_method", "minmax")
    percentile_low = normalize_config.get("percentile_low", 0.0)
    percentile_high = normalize_config.get("percentile_high", 100.0)

    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {step_output}")
    logger.info(f"Normalize: method={normalize_method}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    norm_kwargs = {
        "method": normalize_method,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "logger": logger,
    }

    if input_path.is_file():
        # Single file
        if input_path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Input file is not a TIF: {input_path}")

        logger.info(f"Processing: {input_path.name}")
        image = load_and_check_tif(str(input_path), logger)
        logger.debug(f"  Shape: {image.shape}, dtype: {image.dtype}")

        normalized = normalize_array(image, **norm_kwargs)
        image_uint8 = convert_to_uint8(normalized)

        output_path = step_output / f"{input_path.stem}_normalized.tif"
        tifffile.imwrite(
            str(output_path), image_uint8, imagej=True, metadata={"axes": "ZYX"}
        )
        logger.debug(f"  Saved to: {output_path}")

    elif input_path.is_dir():
        # Try 2D sequence first
        stack = load_2d_sequence(str(input_path), logger)

        if stack is not None:
            # 2D sequence
            logger.info(f"Processing 2D sequence from: {input_path}")
            normalized = normalize_array(stack, **norm_kwargs)
            image_uint8 = convert_to_uint8(normalized)

            output_path = step_output / f"{input_path.name}_normalized.tif"
            tifffile.imwrite(
                str(output_path), image_uint8, imagej=True, metadata={"axes": "ZYX"}
            )
            logger.debug(f"  Saved to: {output_path}")
        else:
            # Multiple 3D TIFs
            tif_files = natsorted(
                [
                    f
                    for f in input_path.iterdir()
                    if f.suffix.lower() in (".tif", ".tiff")
                ]
            )

            if not tif_files:
                raise ValueError(f"No TIF files found in: {input_path}")

            logger.info(f"Found {len(tif_files)} TIF files")

            for idx, tif_file in enumerate(tif_files, start=1):
                logger.info(f"[{idx}/{len(tif_files)}] Processing: {tif_file.name}")
                image = load_and_check_tif(str(tif_file), logger)
                logger.debug(f"  Shape: {image.shape}, dtype: {image.dtype}")

                normalized = normalize_array(image, **norm_kwargs)
                image_uint8 = convert_to_uint8(normalized)

                output_path = step_output / f"{tif_file.stem}_normalized.tif"
                tifffile.imwrite(
                    str(output_path), image_uint8, imagej=True, metadata={"axes": "ZYX"}
                )
                logger.debug(f"  Saved to: {output_path}")

    else:
        raise ValueError(f"Input path is neither file nor directory: {input_path}")

    return str(step_output)
