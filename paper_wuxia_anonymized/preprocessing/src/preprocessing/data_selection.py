import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import re
import random


# Paths relative to the location of the script (.../data/src/preprocessing/)
PROJECT_DIR = Path(__file__).resolve().parents[2]              # .../data
REPO_DIR = PROJECT_DIR.parent                                  # .../CORPUS
DATA_DIR = PROJECT_DIR / "inputs" / "scores"                   # CSVs of scores chrF
ORIGINAL_DATASET_PATH = REPO_DIR / "processed_data" / "wuxia_zh_en_clean"

# PARAMETERS
K_SAMPLES_TARGET = 100_000
MIN_SCORE_THRESHOLD = 20.0
LOW_QUANTILE = 1 / 3
HIGH_QUANTILE = 2 / 3
RANDOM_SEED = 42


def load_local_dataset(path):
    """Loads datasets locales evitando certificados corruptos of Windows."""
    if sys.platform != 'win32':
        from datasets import load_from_disk

        return load_from_disk(str(path))

    import certifi
    import ssl

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
        from datasets import load_from_disk
    finally:
        ssl.create_default_context = original_create_default_context

    return load_from_disk(str(path))



def clean_column_data(series):
    s = series.astype(str)
    
    return pd.to_numeric(s, errors='coerce')

def load_and_merge_scores(data_dir_path):
    """
    Loads CSVs with:
    If a column appears good but has some values > 100 (outliers),
    filters those rows
    """
    search_pattern = str(data_dir_path / "scores_*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"Without files in: {search_pattern}")
        sys.exit(1)
        
    print(f"Files found ({len(csv_files)}):")
    
    df_final = None

    for f in csv_files:
        filename = Path(f).name
        model_name = Path(f).stem.replace("scores_", "")
        
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f" Error leyendo {filename}: {e}")
            continue

        # 1. of column to indice
        cols_to_drop = [c for c in df.columns if 'Unnamed' in c or c.lower() == 'index']
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        valid_metric_col = None
        candidates = [c for c in df.columns if c != 'id']
        
        # 2. iterate for find the score
        for col in candidates:

            df[col] = clean_column_data(df[col])
            
            if df[col].isnull().all():
                continue
            
            max_val = df[col].max()
            mean_val = df[col].mean()
            
            if mean_val > 500: 
                continue 
            if max_val > 100.1:
                bad_rows = df[col] > 100.1
                pct_bad = bad_rows.mean()
                
                if pct_bad > 0.5: 
                    continue
                else:
                    df.loc[bad_rows, col] = np.nan 
            
            if any(x in col.lower() for x in ['chrf']):
                valid_metric_col = col
                break
            
            if valid_metric_col is None:
                valid_metric_col = col

        if valid_metric_col is None:
            continue

        # 3. cleaning final
        # Remove rows where the score is NaN (by conversion failed or filter >100)
        n_before = len(df)
        df = df.dropna(subset=[valid_metric_col])
        

        print(f" {model_name}: Columna '{valid_metric_col}' OK (Max: {df[valid_metric_col].max():.2f})")

        # 4. mergeo
        df = df.rename(columns={valid_metric_col: f'score_{model_name}'})
        df_clean = df[['id', f'score_{model_name}']]
        
        if df_final is None:
            df_final = df_clean
        else:
            df_final = df_final.merge(df_clean, on='id')

    return df_final

def calculate_stats(df):
    score_cols = [c for c in df.columns if c.startswith('score_')]
    print(f"Calculating statistics over {len(score_cols)} models.")
    
    df['mean_score'] = df[score_cols].mean(axis=1)
    
    if len(score_cols) > 1:
        df['variance_score'] = df[score_cols].var(axis=1, ddof=1)
        df['std_score'] = df[score_cols].std(axis=1)
    else:
        df['variance_score'] = 0.0
        df['std_score'] = 0.0
        
    return df

def perform_balanced_sampling(df, k_target):
    """
    Selects the three strata described in the paper:

    * hard: mean in the tertile lower and variance in the tertile lower;
    * intermediate: variance in the tertile upper;
    * easy: mean in the tertile upper and variance in the tertile lower.

    The examples that not meet not of these conditions not is seleccionan.
    """
    print("\nStarting stratified sampling by mean and variance...")

    mean_low, mean_high = df['mean_score'].quantile(
        [LOW_QUANTILE, HIGH_QUANTILE]
    )
    variance_low, variance_high = df['variance_score'].quantile(
        [LOW_QUANTILE, HIGH_QUANTILE]
    )

    print(
        "   -> Umbrales (terciles): "
        f"media baja <= {mean_low:.4f}, media alta >= {mean_high:.4f}, "
        f"variance baja <= {variance_low:.4f}, "
        f"variance alta >= {variance_high:.4f}"
    )

    hard_mask = (
        (df['mean_score'] <= mean_low)
        & (df['variance_score'] <= variance_low)
    )
    intermediate_mask = df['variance_score'] >= variance_high
    strict_easy_mask = (
        (df['mean_score'] >= mean_high)
        & (df['variance_score'] <= variance_low)
    )

    base_target, remainder = divmod(k_target, 3)
    targets = {
        difficulty_bin: base_target + (1 if difficulty_bin < remainder else 0)
        for difficulty_bin in (0, 1, 2)
    }

    # Completes the stratum facil without duplicados relajando only the variance.
    strict_easy_count = int(strict_easy_mask.sum())
    easy_target = targets[2]
    if strict_easy_count < easy_target:
        easy_pool = df.loc[
            (df['mean_score'] >= mean_high) & ~intermediate_mask
        ].nsmallest(easy_target, 'variance_score')
        if len(easy_pool) < easy_target:
            raise ValueError(
                f"Only there are {len(easy_pool)} samples faciles unicas "
                f"available for a target of {easy_target}."
            )
        easy_mask = df.index.isin(easy_pool.index)
        effective_variance_limit = easy_pool['variance_score'].max()
        print(
            f"   -> Estrato facil ampliado: {strict_easy_count} candidates "
            f"estrictas; {easy_target} candidates final with "
            f"variance <= {effective_variance_limit:.4f}."
        )
    else:
        easy_mask = strict_easy_mask

    candidates = []
    for difficulty_bin, mask in (
        (0, hard_mask),
        (1, intermediate_mask),
        (2, easy_mask),
    ):
        stratum = df.loc[mask].copy()
        stratum['difficulty_bin'] = difficulty_bin
        candidates.append(stratum)

    df_candidates = pd.concat(candidates, ignore_index=False)
    counts = (
        df_candidates['difficulty_bin']
        .value_counts()
        .reindex([0, 1, 2], fill_value=0)
    )
    print(f"   -> Candidates by stratum:\n{counts}")
    print(f"   -> Samples outside of the three strata: {len(df) - len(df_candidates)}")

    if (counts == 0).any():
        empty_bins = counts[counts == 0].index.tolist()
        raise ValueError(
            f"Not there are samples candidates for the strata {empty_bins}. "
            "Checks that exist scores of several models and enough "
            "variation in the data."
        )

    target_counts = pd.Series(targets)
    insufficient_bins = counts[counts < target_counts]
    if not insufficient_bins.empty:
        raise ValueError(
            "No there are suficientes samples for complete the strata: "
            f"{insufficient_bins.to_dict()}. Objetivos: {targets}."
        )

    sampled_strata = []
    for difficulty_bin in (0, 1, 2):
        target = targets[difficulty_bin]
        stratum = df_candidates[
            df_candidates['difficulty_bin'] == difficulty_bin
        ]
        sampled_strata.append(
            stratum.sample(
                n=target,
                random_state=RANDOM_SEED,
            )
        )

    df_balanced = pd.concat(sampled_strata, ignore_index=False)
    if df_balanced['id'].duplicated().any():
        raise ValueError("The seleccion contains identificadores duplicados.")
    print(
        f"   -> Target maximum: {k_target} samples "
        f"(semilla {RANDOM_SEED})."
    )
    return df_balanced

def inspect_data(df_selected, raw_ds):
    print("\n" + "="*50)
    print("INSPECTION VISUAL")
    print("="*50)
    
    df_sorted = df_selected.sort_values(by='mean_score')
    
    def print_examples(subset, title):
        print(f"\n--- {title} ---")
        for _, row in subset.iterrows():
            idx = int(row['id'])
            try:
                item = raw_ds['train'][idx]
                print(
                    f"[ID: {idx}] Media: {row['mean_score']:.2f}; "
                    f"variance: {row['variance_score']:.2f} "
                    f"(Bin {row['difficulty_bin']})"
                )
                print(f"   ZH: {item.get('zh', '???')}")
                print(f"   EN: {item.get('en', '???')}")
                print("-" * 20)
            except:
                pass

    print_examples(df_sorted.head(3), f"TOP 3 MORE HARD (>{MIN_SCORE_THRESHOLD})")
    print_examples(df_sorted.tail(3), "TOP 3 MORE EASY")
    


def split_selected_indices(df_selected):
    """Splits the seleccion 80/10/10 keeping balanced the strata."""
    split_sizes = {
        'train': int(len(df_selected) * 0.8),
        'validation': int(len(df_selected) * 0.1),
    }
    split_sizes['test'] = len(df_selected) - sum(split_sizes.values())

    bins = (0, 1, 2)
    indices_by_bin = {}
    for difficulty_bin in bins:
        stratum = df_selected[df_selected['difficulty_bin'] == difficulty_bin]
        indices_by_bin[difficulty_bin] = (
            stratum.sample(frac=1.0, random_state=RANDOM_SEED + difficulty_bin)
            ['id']
            .astype(int)
            .tolist()
        )

    result = {split: [] for split in split_sizes}
    offsets = {difficulty_bin: 0 for difficulty_bin in bins}
    for split_number, (split, split_size) in enumerate(split_sizes.items()):
        base_size, remainder = divmod(split_size, len(bins))
        for position, difficulty_bin in enumerate(bins):
            start = offsets[difficulty_bin]
            if split_number == len(split_sizes) - 1:
                size = len(indices_by_bin[difficulty_bin]) - start
            else:
                size = base_size + (1 if position < remainder else 0)
            end = start + size
            result[split].extend(indices_by_bin[difficulty_bin][start:end])
            offsets[difficulty_bin] = end
        random.Random(RANDOM_SEED + 100 + split_number).shuffle(result[split])

    all_indices = [index for indices in result.values() for index in indices]
    sizes_are_correct = all(
        len(result[split]) == expected_size
        for split, expected_size in split_sizes.items()
    )
    if (
        not sizes_are_correct
        or len(all_indices) != len(df_selected)
        or len(set(all_indices)) != len(all_indices)
    ):
        raise ValueError("The splits no contain exactamente the IDs selected.")

    return result


def print_table_statistics(df_balanced):
    print("\n" + "="*50)
    print("STATISTICS FOR THE TABLE (RANGOS CHRF)")
    print("="*50)
    
    # Agrupar by bin and sacar count, minimum, maximum and mean
    stats = df_balanced.groupby('difficulty_bin', observed=False).agg(
            Muestras=('mean_score', 'count'),
            CHRF_Min=('mean_score', 'min'),
            CHRF_Max=('mean_score', 'max'),
            CHRF_Medio=('mean_score', 'mean'),
            Varianza_Media=('variance_score', 'mean'),
            Desviacion_Estandar_Media=('std_score', 'mean')
        )
    
    # Rename the indices for greater clarity (asumiendo qcut ascendente)
    nombres_estratos = {
        0: 'Hard (Tercil lower)', 
        1: 'Intermedias (Tercil medio)', 
        2: 'Easy (Tercil upper)'
    }
    stats.index = stats.index.map(nombres_estratos)
    
    print(stats.to_string(float_format="%.2f"))
    print("="*50)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='execute', choices=['execute', 'study'])
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory of output for the mode execute.',
    )
    args = parser.parse_args()

    print(f"=== MODE: {args.mode.upper()} ===")
    
    # 1. Load
    df_final = load_and_merge_scores(DATA_DIR)
    if df_final is None or len(df_final) == 0:
        print("Error: Could not load no data item valid.")
        return

    # 2. Calcular
    df_final = calculate_stats(df_final)
    
    # 3. Filter of Calidad
    print(f"\n Aplicando filtro (Min Score >= {MIN_SCORE_THRESHOLD})...")
    df_filtered = df_final[df_final['mean_score'] >= MIN_SCORE_THRESHOLD].copy()
    print(f"    Samples valid: {len(df_filtered)}")

    if len(df_filtered) == 0:
        print("Error: All the samples fueron filtered.")
        return

    # 4. Sampling
    df_selected = perform_balanced_sampling(df_filtered, K_SAMPLES_TARGET)
    selected_indices = df_selected['id'].values
    print(f"Final selection: {len(selected_indices)} samples.")

    # 5. The dataset original only is required for save the seleccion.
    # In mode estudio is uses exclusively for show examples cualitativos.
    raw_ds = None
    try:
        print("\nLoading the original dataset...")
        raw_ds = load_local_dataset(ORIGINAL_DATASET_PATH)
    except Exception as exc:
        if args.mode == 'execute':
            raise
        print(
            "\nWarning: could not load the original dataset; "
            "is skips the inspeccion of examples."
        )
        print(f"Motivo: {exc}")

    if args.mode == 'study':
            print_table_statistics(df_selected)
            if raw_ds is not None:
                inspect_data(df_selected, raw_ds)
            
            # Configure figure with 2 panels (Left: Original, Right: Final)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # This shows the natural bias of Internet data, which likely contains many easy samples.
            axes[0].hist(df_filtered['mean_score'], bins=50, color='royalblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[0].set_title(f"BEFORE: Distribution Original (Filtered)\nTotal: {len(df_filtered)} samples", fontsize=12, fontweight='bold')
            axes[0].set_xlabel("Dificultad (Score Medio)", fontsize=10)
            axes[0].set_ylabel("Number of Sentences", fontsize=10)
            axes[0].grid(axis='y', alpha=0.3)
            
            # This shows the effect of balancing the three strata.
            axes[1].hist(df_selected['mean_score'], bins=50, color='limegreen', alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[1].set_title(f"AFTER: Distribution Final (Balanced)\nTotal: {len(df_selected)} samples", fontsize=12, fontweight='bold')
            axes[1].set_xlabel("Dificultad (Score Medio)", fontsize=10)
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    elif args.mode == 'execute':
        split_indices = split_selected_indices(df_selected)
        original_train = raw_ds['train']
        raw_ds['train'] = original_train.select(split_indices['train'])
        raw_ds['validation'] = original_train.select(split_indices['validation'])
        raw_ds['test'] = original_train.select(split_indices['test'])
        output_path = (
            Path(args.output_dir)
            if args.output_dir
            else REPO_DIR / "processed_data" / "wuxia_selected_100k"
        )
        if output_path.exists():
            raise FileExistsError(
                f"The output directory already exists: {output_path}. "
                "Uses --output_dir with a path new."
            )
        raw_ds.save_to_disk(str(output_path))
        print(f"Saved in: {output_path}")
        print({split: len(dataset) for split, dataset in raw_ds.items()})

    
if __name__ == "__main__":
    main()
