from copy import deepcopy
from pathlib import Path

import yaml


DEFAULTS = {
    "device": "cuda",
    "seed": 42,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "save_epoch_freq": 10,
    "sample_freq": 500,
    "log_freq": 100,
    "use_amp": False,
    "cudnn_benchmark": True,
    "cudnn_deterministic": False,
    "batch_size": 1,
    "image_size": 256,
    "load_size": 286,
    "pool_size": 50,
    "lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "epochs": 200,
    "decay_start_epoch": 100,
    "lambda_cycle": 10,
    "lambda_identity": 5,
    "use_cycle": True,
    "use_identity": True,
    "direction": "AtoB",
}


REQUIRED_FIELDS = ["dataset_name", "dataroot", "model_type"]


def load_config(config_path):
    """Load a YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    cfg["_config_path"] = str(config_path)
    return cfg


def apply_overrides(cfg, overrides):
    """Apply CLI overrides to config."""
    cfg = deepcopy(cfg)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def resolve_config(cfg, experiment_name=None):
    """Validate, normalize, and fill default config values."""
    resolved = deepcopy(DEFAULTS)
    resolved.update(cfg)
    if experiment_name is not None:
        resolved["name"] = experiment_name
    resolved.setdefault("name", resolved.get("dataset_name", "experiment"))

    missing = [field for field in REQUIRED_FIELDS if field not in resolved or resolved[field] in (None, "")]
    if missing:
        raise KeyError(f"Missing required config fields: {missing}")

    model_type = resolved["model_type"]
    if model_type not in {"baseline", "lightweight", "dwconv9", "custom"}:
        raise ValueError("model_type must be one of: baseline, lightweight, dwconv9, custom")
    if model_type == "baseline":
        resolved["n_blocks"] = 9
    elif model_type == "lightweight":
        resolved["n_blocks"] = 6
    elif model_type == "dwconv9":
        resolved["n_blocks"] = 9
    else:
        resolved.setdefault("n_blocks", 9)

    resolved["dataroot"] = str(Path(resolved["dataroot"]))
    resolved["results_dir"] = str(Path("results") / resolved["name"])
    resolved["checkpoint_dir"] = str(Path("checkpoints") / resolved["name"])

    if int(resolved["num_workers"]) == 0:
        resolved["persistent_workers"] = False
        resolved["prefetch_factor"] = None
    if resolved["device"] == "cpu":
        resolved["pin_memory"] = False
    if not resolved.get("use_identity", True):
        resolved["lambda_identity"] = 0
    if not resolved.get("use_cycle", True):
        resolved["lambda_cycle"] = 0
    return resolved


def save_resolved_config(cfg):
    """Save final config to results/{name}/config_resolved.yaml."""
    output = Path(cfg["results_dir"]) / "config_resolved.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return output


def load_and_resolve_config(config_path, experiment_name=None, overrides=None, save=True):
    cfg = load_config(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    cfg = resolve_config(cfg, experiment_name)
    if save:
        save_resolved_config(cfg)
    return cfg


def dataloader_kwargs(cfg, shuffle=True, drop_last=False):
    """Build DataLoader kwargs from resolved config."""
    kwargs = {
        "batch_size": cfg["batch_size"],
        "shuffle": shuffle,
        "num_workers": cfg["num_workers"],
        "pin_memory": cfg["pin_memory"],
        "drop_last": drop_last,
    }
    if cfg["num_workers"] > 0:
        kwargs["persistent_workers"] = cfg["persistent_workers"]
        if cfg["prefetch_factor"] is not None:
            kwargs["prefetch_factor"] = cfg["prefetch_factor"]
    return kwargs

