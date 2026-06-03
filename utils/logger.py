import logging
import platform
import subprocess
import sys
from pathlib import Path


def setup_logger(log_file=None):
    """Create a logger that writes to stdout and optionally a file."""
    logger = logging.getLogger("cyclegan")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def get_git_commit():
    """Return current Git commit hash if available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def log_environment(logger, config=None):
    """Log runtime environment without adding heavy dependencies."""
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "none"
    except Exception:
        torch_version = "unavailable"
        cuda_available = False
        gpu_name = "none"
    logger.info("Git commit: %s", get_git_commit())
    logger.info("Python: %s", platform.python_version())
    logger.info("PyTorch: %s", torch_version)
    logger.info("CUDA available: %s", cuda_available)
    logger.info("GPU: %s", gpu_name)
    if config is not None:
        summary_keys = [
            "name",
            "dataset_name",
            "model_type",
            "n_blocks",
            "batch_size",
            "lr",
            "epochs",
            "decay_start_epoch",
            "lambda_cycle",
            "lambda_identity",
            "use_amp",
            "num_workers",
        ]
        logger.info("Config summary: %s", {key: config.get(key) for key in summary_keys})
