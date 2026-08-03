from pathlib import Path
from .config import BaseConfig

config = BaseConfig(
    dataset_dir=Path("/path/to/processed_data/wuxia_zh_en_clean"),
    output_dir=Path("/path/to/models/small100_wuxia"),
    model_ckpt="alirezamsh/small100",
)
