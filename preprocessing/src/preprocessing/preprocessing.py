
import argparse
import random
import unicodedata
from pathlib import Path
import pandas as pd
import string
import ssl
import sys


def import_dataset_classes():
    if sys.platform != 'win32':
        from datasets import Dataset, DatasetDict

        return Dataset, DatasetDict

    import certifi

    original_create_default_context = ssl.create_default_context

    def create_certifi_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        *,
        cafile=None,
        capath=None,
        cadata=None,
    ):
        if cafile is None and capath is None and cadata is None:
            cafile = certifi.where()
        return original_create_default_context(
            purpose,
            cafile=cafile,
            capath=capath,
            cadata=cadata,
        )

    ssl.create_default_context = create_certifi_context
    try:
        from datasets import Dataset, DatasetDict
    finally:
        ssl.create_default_context = original_create_default_context

    return Dataset, DatasetDict

# Lista de puntuación a conservar (puedes ajustarla si quieres menos)
ALLOWED_PUNCT = set(string.punctuation) | {"，", "。", "！", "？", "、", "；", "：", "“", "”", "‘", "’", "—", "…"}

def is_letter_or_mark(ch: str) -> bool:
    # Letras y marcas (pinyin con tildes, caracteres chinos)
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "M")

def is_space(ch: str) -> bool:
    return unicodedata.category(ch) == "Zs" or ch in [" ", "\t"]

def is_allowed_punct(ch: str) -> bool:
    return ch in ALLOWED_PUNCT

def clean_text(s: str) -> str:
    if not s:
        return ""
    kept = []
    for ch in s:
        if is_letter_or_mark(ch) or is_space(ch) or is_allowed_punct(ch):
            kept.append(ch)
    out = "".join(kept)

    # Quitar puntuación al inicio y final
    while out and is_allowed_punct(out[0]):
        out = out[1:]
    while out and is_allowed_punct(out[-1]):
        out = out[:-1]

    # Normalizar espacios
    out = " ".join(out.split())

    # Si solo queda puntuación, eliminar
    if all(is_allowed_punct(ch) or is_space(ch) for ch in out):
        return ""

    return out.strip()

def load_and_clean(input_path: Path, sep=";"):
    srcs, tgts = [], []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if sep not in line:
                continue
            left, right = line.split(sep, 1)
            zh_raw = left.strip()
            en_raw = right.strip()
            zh = clean_text(zh_raw)
            en = clean_text(en_raw)
            if not zh or not en:
                continue
            srcs.append(zh)
            tgts.append(en)
    return pd.DataFrame({"zh": srcs, "en": tgts})

def main():
    Dataset, DatasetDict = import_dataset_classes()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_txt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input_txt)
    df = load_and_clean(input_path)

    if len(df) < 10:
        raise ValueError("El dataset tiene muy pocas líneas tras la limpieza.")

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n = len(df)
    n_test = int(n * args.test_size)
    n_val = int(n * args.val_size)
    n_train = n - n_test - n_val

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df  = df.iloc[n_train + n_val:].reset_index(drop=True)

    ds = DatasetDict(
        train=Dataset.from_pandas(train_df),
        validation=Dataset.from_pandas(val_df),
        test=Dataset.from_pandas(test_df),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"Guardado DatasetDict en: {out_dir}")
    print({k: len(v) for k, v in ds.items()})

if __name__ == "__main__":
    main()

# al mismo nivel 
# python preprocess.py --input_txt /data/dataset.txt --output_dir processed_data/wuxia_zh_en_clean
