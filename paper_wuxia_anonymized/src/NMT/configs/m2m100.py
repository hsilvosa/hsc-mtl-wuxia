from pathlib import Path
from .config import M2M100Config

config = M2M100Config(
    dataset_dir=Path("/path/to/processed_data/wuxia_zh_en_clean"),
    output_dir=Path("/path/to/models/m2m100_wuxia"),
    model_ckpt="facebook/m2m100_418M",
)
