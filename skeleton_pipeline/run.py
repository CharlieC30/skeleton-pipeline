#!/usr/bin/env python3
"""Skeleton Pipeline - Main Entry Point.

A complete pipeline for 3D image skeletonization and analysis.

Usage:
    python -m skeleton_pipeline --input data/examples/input/sample_input.tif
    python -m skeleton_pipeline --input data/examples/input/sample_input.tif --output /custom/path
    python -m skeleton_pipeline --help
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from skeleton_pipeline.steps import (
    analyze,
    clean,
    normalize,
    skeletonize,
    threshold,
)
from skeleton_pipeline.utils import (
    format_duration,
    get_output_dir,
    load_config,
    resolve_input_path,
    setup_logging,
)

# Step registry, in execution order
STEPS = [normalize, threshold, clean, skeletonize, analyze]


def run_pipeline(
    input_path: str,
    output_dir: str,
    config: dict,
    logger: logging.Logger,
) -> None:
    """Run the full skeleton pipeline.

    Args:
        input_path: Path to input TIF file or directory.
        output_dir: Output directory.
        config: Configuration dictionary.
        logger: Logger instance.
    """
    logger.info("=" * 60)
    logger.info("Skeleton Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("-" * 60)

    total_start = time.time()
    step_times = []
    current_input = input_path

    for step_num, step_module in enumerate(STEPS, start=1):
        step_desc = step_module.STEP_DESCRIPTION
        logger.info("")
        logger.info(f"[Step {step_num}/{len(STEPS)}] {step_desc}")
        logger.info("-" * 40)

        step_start = time.time()
        try:
            current_input = step_module.run(
                input_path=current_input,
                output_dir=output_dir,
                config=config,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"Step {step_num} ({step_desc}) failed: {e}")
            raise

        step_elapsed = time.time() - step_start
        step_times.append((step_num, step_desc, step_elapsed))
        logger.info(f"Step {step_num} completed in {format_duration(step_elapsed)}")

    # Summary
    total_elapsed = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    for step_num, step_desc, elapsed in step_times:
        logger.info(f"  Step {step_num} ({step_desc}): {format_duration(elapsed)}")
    logger.info("-" * 60)
    logger.info(f"Total time: {format_duration(total_elapsed)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point for the skeleton pipeline.

    Parses command line arguments and runs the full pipeline.
    Run with --help to see all options.
    """
    parser = argparse.ArgumentParser(
        description="Skeleton Pipeline - 3D image skeletonization and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the pipeline
  python -m skeleton_pipeline --input data/examples/input/sample_input.tif

  # Specify output directory
  python -m skeleton_pipeline --input data/examples/input/sample_input.tif --output /path/to/output

  # Use a custom config
  python -m skeleton_pipeline --input data/examples/input/sample_input.tif --config skeleton_pipeline/config/filopodia.yaml

Steps:
  1. Normalize     - Normalize and convert TIF to uint8
  2. Threshold     - Binarize using Otsu's method
  3. Clean         - Morphological operations
  4. Skeletonize   - Extract skeleton using Kimimaro
  5. Analyze       - Analyze trunk and branches
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input TIF file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory (default: data/output/TIMESTAMP/, set by config output.base_dir)",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Configuration file (default: config/examples.yaml)",
    )
    parser.add_argument(
        "--log-file",
        help="Log file path",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logger = setup_logging(level=args.log_level, log_file=args.log_file)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        sys.exit(1)

    try:
        input_path = str(resolve_input_path(args.input))
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    output_dir = str(get_output_dir(config, args.output))

    # Create output directory and save config backup for reproducibility
    os.makedirs(output_dir, exist_ok=True)
    config_backup_path = os.path.join(output_dir, "config_used.yaml")
    if args.config:
        shutil.copy(args.config, config_backup_path)
    else:
        default_config = Path(__file__).parent / "config" / "examples.yaml"
        shutil.copy(default_config, config_backup_path)
    logger.info(f"Config saved to: {config_backup_path}")

    try:
        run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            config=config,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
