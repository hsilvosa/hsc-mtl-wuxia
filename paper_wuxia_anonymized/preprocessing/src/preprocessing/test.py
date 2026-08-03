from datasets import load_from_disk
import pandas as pd

dataset_path = r"preprocess_data/wuxia_zh_en_clean"
dataset = load_from_disk(dataset_path)

# If is DatasetDict (several splits)
if hasattr(dataset, "keys"):
    first_split = list(dataset.keys())[0]
    df = dataset[first_split].to_pandas()
else:
    df = dataset.to_pandas()

# Mostrar primeras filas
print(df.head())
