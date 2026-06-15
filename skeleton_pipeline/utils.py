"""Skeleton pipeline utilities."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import yaml

# Base directory (skeleton_pipeline/)
BASE_DIR = Path(__file__).parent.resolve()
# Project root (repository root)
PROJECT_ROOT = BASE_DIR.parent


def load_config(config_path: Path | None = None) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file. If None, uses default config/examples.yaml.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: Config file not found.
        ValueError: Empty config file.
    """
    if config_path is None:
        config_path = BASE_DIR / "config" / "examples.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty config file: {config_path}")

    return config


def get_output_dir(config: dict, custom_output: str | None = None) -> Path:
    """Get output directory based on config or custom path.

    Args:
        config: Configuration dictionary.
        custom_output: Optional custom output path (overrides config).

    Returns:
        Output directory path.
    """
    if custom_output:
        output_dir = Path(custom_output)
    else:
        output_config = config.get("output", {})
        base_dir = output_config.get("base_dir", "../data/output")

        # Resolve relative to BASE_DIR
        if not os.path.isabs(base_dir):
            output_dir = BASE_DIR / base_dir
        else:
            output_dir = Path(base_dir)

        # Add timestamp subdirectory if configured
        if output_config.get("use_timestamp", True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir / timestamp

    return output_dir.resolve()


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Setup logging to console and optionally to file.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = level_map.get(level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "1m 23s" or "45.2s".
    """
    if seconds >= 60:
        total = round(seconds)
        minutes, secs = divmod(total, 60)
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


def resolve_input_path(input_path: str) -> Path:
    """Resolve input path to absolute path.

    Args:
        input_path: Input file or directory path.

    Returns:
        Resolved absolute path.

    Raises:
        FileNotFoundError: If path doesn't exist.
    """
    path = Path(input_path)

    # If relative, try resolving from current directory first
    if not path.is_absolute():
        # Try from current directory
        if path.exists():
            return path.resolve()
        # Try from project root
        project_path = PROJECT_ROOT / path
        if project_path.exists():
            return project_path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return path.resolve()
