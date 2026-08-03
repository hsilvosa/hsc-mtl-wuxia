from pathlib import Path
from .config import MarianMTConfig

config = MarianMTConfig(
    dataset_dir=Path("/path/to/processed_data/wuxia_zh_en_clean"),
    output_dir=Path("/path/to/models/marianmt_wuxia"),
    model_ckpt="Helsinki-NLP/opus-mt-zh-en",
)
