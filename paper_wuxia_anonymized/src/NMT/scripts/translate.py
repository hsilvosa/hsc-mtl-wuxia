import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[2]))

from NMT.utils.reproducibility import set_seed
from NMT.utils.generation import generate_text
from NMT.utils.data import select_fraction
from NMT.configs.config import (
    MarianMTConfig,
    M2M100Config,
    MBartConfig,
    MT5Config,
    BaseConfig,
)

MODEL_CONFIGS = {
    "marianmt": MarianMTConfig,
    "m2m100": M2M100Config,
    "mbart": MBartConfig,
    "mt5": MT5Config,
    "small100": BaseConfig,
}

MODEL_DIR_NAMES = {
    "marianmt": "marianmt_wuxia",
    "m2m100": "m2m100_wuxia",
    "mbart": "mbart50_wuxia",
    "mt5": "mt5_small",
    "small100": "small100_wuxia",
}


def find_latest_checkpoint(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return candidates[-1]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


TRANSLATE_DIR = get_project_root() / "src" / "NMT" / "evaluation" / "translate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_CONFIGS.keys(), help="Model name")
    parser.add_argument("--dataset-dir", default=str(get_project_root() / "processed_data" / "wuxia_zh_en_clean"))
    parser.add_argument("--out-file", default="output.txt", help="Filename (always saved in src/NMT/evaluation/translate/)")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--num-sentences", type=int, default=None, help="Limit translation to first N sentences")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg_cls = MODEL_CONFIGS[args.model]
    model_dir_name = MODEL_DIR_NAMES[args.model]
    model_dir = get_project_root() / "models" / model_dir_name
    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(model_dir)

    cfg = cfg_cls(
        dataset_dir=Path(args.dataset_dir),
        output_dir=model_dir,
        model_ckpt=str(checkpoint_dir),
    )

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from datasets import load_from_disk
    dataset = load_from_disk(str(cfg.dataset_dir))
    test_ds = dataset["test"]
    test_ds = select_fraction(test_ds, args.fraction, cfg.seed)

    src_texts = [str(x[cfg.src_col]) for x in test_ds]
    if args.num_sentences is not None:
        src_texts = src_texts[: args.num_sentences]

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model_ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(cfg.model_ckpt))
    model.to(device)
    model.eval()

    preds = generate_text(
        model,
        tokenizer,
        src_texts,
        max_length=args.max_length,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
    )

    out_path = TRANSLATE_DIR / args.out_file
    TRANSLATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for src, hyp in zip(src_texts, preds):
            f.write(f"SRC: {src}\nPRED: {hyp}\n\n")
    print(f"Translations guardadas in {out_path}")


if __name__ == "__main__":
    main()
