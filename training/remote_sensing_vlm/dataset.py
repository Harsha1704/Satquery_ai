"""Dataset-root, hardware, and reproducibility helpers with no model imports."""
from __future__ import annotations
import os, platform, random, shutil
from pathlib import Path
from typing import Any

def data_root() -> Path:
    return Path(os.getenv("SATQUERY_DATA_ROOT", "data")).expanduser().resolve()

def hardware() -> dict[str, Any]:
    import torch
    return {"os": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
            "free_disk_bytes": shutil.disk_usage(data_root().anchor or ".").free}

def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except ImportError: pass
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
