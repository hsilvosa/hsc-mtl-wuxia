import numpy as np
import torch


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_mixed_precision(device: str):
    return torch.cuda.amp.autocast() if device == "cuda" else __import__("contextlib").nullcontext()
