from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass


@dataclass
class BaseConfig:
    dataset_dir: Path
    output_dir: Path
    model_ckpt: str
    src_col: str = "zh"
    tgt_col: str = "en"
    seed: int = 42
    max_source_length: int = 128
    max_target_length: int = 128
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    fraction: float = 1.0


@dataclass
class MarianMTConfig(BaseConfig):
    model_ckpt: str = "Helsinki-NLP/opus-mt-zh-en"


@dataclass
class M2M100Config(BaseConfig):
    src_lang: str = "zh"
    tgt_lang: str = "en"
    model_ckpt: str = "facebook/m2m100_418M"


@dataclass
class MBartConfig(BaseConfig):
    src_lang: str = "zh_CN"
    tgt_lang: str = "en_XX"
    model_ckpt: str = "facebook/mbart-large-50-many-to-many-mmt"


@dataclass
class MT5Config(BaseConfig):
    model_ckpt: str = "google/mt5-small"
    use_instruction_prefix: bool = True
    translation_prefix: str = "translate Chinese to English: "
