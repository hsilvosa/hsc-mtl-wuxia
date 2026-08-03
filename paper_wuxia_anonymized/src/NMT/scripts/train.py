import argparse
import sys
from pathlib import Path
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

sys.path.append(str(Path(__file__).resolve().parents[2]))

from NMT.utils.reproducibility import set_seed
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


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_dataset(cfg):
    from datasets import load_from_disk
    dataset = load_from_disk(str(cfg.dataset_dir))
    return dataset["train"], dataset["validation"], dataset["test"]


def preprocess_function(examples, tokenizer, cfg):
    inputs = [str(x) for x in examples[cfg.src_col]]
    targets = [str(x) for x in examples[cfg.tgt_col]]

    model_inputs = tokenizer(
        inputs,
        max_length=cfg.max_source_length,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=cfg.max_target_length,
            truncation=True,
            padding="max_length",
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train(args):
    cfg_cls = MODEL_CONFIGS[args.model]
    model_dir_name = MODEL_DIR_NAMES[args.model]
    output_dir = Path(args.output_dir) if args.output_dir else get_project_root() / "models" / model_dir_name

    cfg = cfg_cls(
        dataset_dir=Path(args.dataset_dir),
        output_dir=output_dir,
        model_ckpt=args.model_ckpt or cfg_cls.__dataclass_fields__["model_ckpt"].default,
        fraction=args.fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    train_ds, val_ds, test_ds = load_dataset(cfg)
    train_ds = select_fraction(train_ds, cfg.fraction, cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model_ckpt))

    if hasattr(cfg, "translation_prefix") and cfg.use_instruction_prefix:
        def add_prefix(example):
            example[cfg.src_col] = cfg.translation_prefix + str(example[cfg.src_col])
            return example
        train_ds = train_ds.map(add_prefix)
        val_ds = val_ds.map(add_prefix)

    train_tok = train_ds.map(lambda ex: preprocess_function(ex, tokenizer, cfg), batched=True)
    val_tok = val_ds.map(lambda ex: preprocess_function(ex, tokenizer, cfg), batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(str(cfg.model_ckpt))
    model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        fp16=(device == "cuda"),
        push_to_hub=False,
        logging_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    print("Starting training...")
    trainer.train()

    best_dir = cfg.output_dir / "best"
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Model saved in {best_dir}")
    return best_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_CONFIGS.keys(), help="Model name")
    parser.add_argument("--dataset-dir", default=str(get_project_root() / "processed_data" / "wuxia_zh_en_clean"))
    parser.add_argument("--output-dir", default=None, help="Output directory (default: models/{model_dir})")
    parser.add_argument("--model-ckpt", default=None)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        return

    if "train" in sys.argv or args.model in MODEL_CONFIGS:
        train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
