import random

import numpy as np
import torch


def set_seed(seed=42, deterministic=False, benchmark=True):
    """Set Python, NumPy, PyTorch, and cuDNN reproducibility settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

