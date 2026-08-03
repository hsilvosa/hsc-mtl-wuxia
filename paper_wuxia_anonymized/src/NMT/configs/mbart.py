from pathlib import Path
from NMT.configs.config import MBartConfig

model_specific_dir = Path("/path/to/models/mbart50_wuxia")
config = MBartConfig(
    dataset_dir=Path("/path/to/processed_data/wuxia_zh_en_clean"),
    output_dir=model_specific_dir,
    model_ckpt="facebook/mbart-large-50-many-to-many-mmt",
)
