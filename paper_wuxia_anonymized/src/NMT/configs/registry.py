from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = BASE_DIR / "models_output"
CONFIGS_DIR = BASE_DIR / "configs"

MODEL_CONFIGS = {
    "marianmt": "Helsinki-NLP/opus-mt-zh-en",
    "m2m100": "facebook/m2m100_418M",
    "mbart": "facebook/mbart-large-50-many-to-many-mmt",
    "mt5": "google/mt5-small",
    "small100": "alirezamsh/small100",
}

def get_config(model_name: str):
    ckpt = MODEL_CONFIGS[model_name]
    return {
        "dataset_dir": BASE_DIR / "processed_data" / "wuxia_zh_en_clean",
        "output_dir": BASE_DIR / "models" / f"{model_name}_wuxia",
        "model_ckpt": ckpt,
    }
