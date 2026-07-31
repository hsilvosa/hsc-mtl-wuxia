import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from datasets import load_from_disk
from pathlib import Path
import argparse
import sys
import re


# Rutas relativas a la ubicación del script (.../data/src/preprocessing/)
PROJECT_DIR = Path(__file__).resolve().parents[2]              # .../data
REPO_DIR = PROJECT_DIR.parent                                  # .../CORPUS
DATA_DIR = PROJECT_DIR / "inputs" / "scores"                   # CSVs de scores chrF
ORIGINAL_DATASET_PATH = REPO_DIR / "processed_data" / "wuxia_zh_en_clean"

# PARÁMETROS
K_SAMPLES_TARGET = 100_000
MIN_SCORE_THRESHOLD = 20.0
LOW_QUANTILE = 1 / 3
HIGH_QUANTILE = 2 / 3
RANDOM_SEED = 42



def clean_column_data(series):
    s = series.astype(str)
    
    return pd.to_numeric(s, errors='coerce')

def load_and_merge_scores(data_dir_path):
    """
    Carga CSVs con:
    Si una columna parece buena pero tiene algunos valores > 100 (outliers),
    filtra esas filas
    """
    search_pattern = str(data_dir_path / "scores_*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"Sin archivos en: {search_pattern}")
        sys.exit(1)
        
    print(f"Archivos encontrados ({len(csv_files)}):")
    
    df_final = None

    for f in csv_files:
        filename = Path(f).name
        model_name = Path(f).stem.replace("scores_", "")
        
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f" Error leyendo {filename}: {e}")
            continue

        # 1. de columna a indice
        cols_to_drop = [c for c in df.columns if 'Unnamed' in c or c.lower() == 'index']
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        valid_metric_col = None
        candidates = [c for c in df.columns if c != 'id']
        
        # 2. iterar para encontrar el score
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

        # 3. limpieza final
        # Borramos filas donde la nota sea NaN (por conversión fallida o filtro >100)
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
    print(f"Calculando estadísticas sobre {len(score_cols)} modelos.")
    
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
    Selecciona los tres estratos descritos en el artículo:

    * difíciles: media en el tercil inferior y varianza en el tercil inferior;
    * intermedias: varianza en el tercil superior;
    * fáciles: media en el tercil superior y varianza en el tercil inferior.

    Los ejemplos que no cumplen ninguna de estas condiciones no se seleccionan.
    """
    print("\nIniciando sampling estratificado por media y varianza...")

    mean_low, mean_high = df['mean_score'].quantile(
        [LOW_QUANTILE, HIGH_QUANTILE]
    )
    variance_low, variance_high = df['variance_score'].quantile(
        [LOW_QUANTILE, HIGH_QUANTILE]
    )

    print(
        "   -> Umbrales (terciles): "
        f"media baja <= {mean_low:.4f}, media alta >= {mean_high:.4f}, "
        f"varianza baja <= {variance_low:.4f}, "
        f"varianza alta >= {variance_high:.4f}"
    )

    hard_mask = (
        (df['mean_score'] <= mean_low)
        & (df['variance_score'] <= variance_low)
    )
    intermediate_mask = df['variance_score'] >= variance_high
    easy_mask = (
        (df['mean_score'] >= mean_high)
        & (df['variance_score'] <= variance_low)
    )

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
    print(f"   -> Candidatas por estrato:\n{counts}")
    print(f"   -> Muestras fuera de los tres estratos: {len(df) - len(df_candidates)}")

    if (counts == 0).any():
        empty_bins = counts[counts == 0].index.tolist()
        raise ValueError(
            f"No hay muestras candidatas para los estratos {empty_bins}. "
            "Comprueba que existan puntuaciones de varios modelos y suficiente "
            "variación en los datos."
        )

    base_target, remainder = divmod(k_target, 3)
    sampled_strata = []
    for difficulty_bin in (0, 1, 2):
        target = base_target + (1 if difficulty_bin < remainder else 0)
        stratum = df_candidates[
            df_candidates['difficulty_bin'] == difficulty_bin
        ]
        sampled_strata.append(
            stratum.sample(
                n=min(len(stratum), target),
                random_state=RANDOM_SEED,
            )
        )

    df_balanced = pd.concat(sampled_strata, ignore_index=False)
    print(
        f"   -> Objetivo máximo: {k_target} muestras "
        f"(semilla {RANDOM_SEED})."
    )
    return df_balanced

def inspect_data(df_selected, raw_ds):
    print("\n" + "="*50)
    print("INSPECCIÓN VISUAL")
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
                    f"varianza: {row['variance_score']:.2f} "
                    f"(Bin {row['difficulty_bin']})"
                )
                print(f"   ZH: {item.get('zh', '???')}")
                print(f"   EN: {item.get('en', '???')}")
                print("-" * 20)
            except:
                pass

    print_examples(df_sorted.head(3), f"TOP 3 MÁS DIFÍCILES (>{MIN_SCORE_THRESHOLD})")
    print_examples(df_sorted.tail(3), "TOP 3 MÁS FÁCILES")
    


def print_table_statistics(df_balanced):
    print("\n" + "="*50)
    print("ESTADÍSTICAS PARA LA TABLA (RANGOS CHRF)")
    print("="*50)
    
    # Agrupar por bin y sacar recuento, mínimo, máximo y media
    stats = df_balanced.groupby('difficulty_bin', observed=False).agg(
            Muestras=('mean_score', 'count'),
            CHRF_Min=('mean_score', 'min'),
            CHRF_Max=('mean_score', 'max'),
            CHRF_Medio=('mean_score', 'mean'),
            Varianza_Media=('variance_score', 'mean'),
            Desviacion_Estandar_Media=('std_score', 'mean')
        )
    
    # Renombrar los índices para mayor claridad (asumiendo qcut ascendente)
    nombres_estratos = {
        0: 'Difíciles (Tercil inferior)', 
        1: 'Intermedias (Tercil medio)', 
        2: 'Fáciles (Tercil superior)'
    }
    stats.index = stats.index.map(nombres_estratos)
    
    print(stats.to_string(float_format="%.2f"))
    print("="*50)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='execute', choices=['execute', 'study'])
    args = parser.parse_args()

    print(f"=== MODO: {args.mode.upper()} ===")
    
    # 1. Cargar
    df_final = load_and_merge_scores(DATA_DIR)
    if df_final is None or len(df_final) == 0:
        print("Error: No se pudo cargar ningún dato válido.")
        return

    # 2. Calcular
    df_final = calculate_stats(df_final)
    
    # 3. Filtro de Calidad
    print(f"\n Aplicando filtro (Min Score >= {MIN_SCORE_THRESHOLD})...")
    df_filtered = df_final[df_final['mean_score'] >= MIN_SCORE_THRESHOLD].copy()
    print(f"    Muestras válidas: {len(df_filtered)}")

    if len(df_filtered) == 0:
        print("Error: Todas las muestras fueron filtradas.")
        return

    # 4. Sampling
    df_selected = perform_balanced_sampling(df_filtered, K_SAMPLES_TARGET)
    selected_indices = df_selected['id'].values
    print(f"Selección final: {len(selected_indices)} muestras.")

    # 5. Dataset Original
    print(f"\nCargando dataset original...")
    raw_ds = load_from_disk(str(ORIGINAL_DATASET_PATH))

    if args.mode == 'study':
            print_table_statistics(df_selected)
            inspect_data(df_selected, raw_ds)
            
            # Configurar figura con 2 paneles (Izquierda: Original, Derecha: Final)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Esto muestra el sesgo natural de los datos de internet (seguramente muchas fáciles)
            axes[0].hist(df_filtered['mean_score'], bins=50, color='royalblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[0].set_title(f"ANTES: Distribución Original (Filtrada)\nTotal: {len(df_filtered)} muestras", fontsize=12, fontweight='bold')
            axes[0].set_xlabel("Dificultad (Score Medio)", fontsize=10)
            axes[0].set_ylabel("Número de Frases", fontsize=10)
            axes[0].grid(axis='y', alpha=0.3)
            
            # Esto muestra cómo has forzado la igualdad de oportunidades
            axes[1].hist(df_selected['mean_score'], bins=50, color='limegreen', alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[1].set_title(f"DESPUÉS: Distribución Final (Balanced)\nTotal: {len(df_selected)} muestras", fontsize=12, fontweight='bold')
            axes[1].set_xlabel("Dificultad (Score Medio)", fontsize=10)
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    elif args.mode == 'execute':
        new_train = raw_ds['train'].select(selected_indices)
        raw_ds['train'] = new_train
        output_path = REPO_DIR / "processed_data" / f"wuxia_selected_{len(new_train)}"
        raw_ds.save_to_disk(str(output_path))
        print(f"Guardado en: {output_path}")

    
if __name__ == "__main__":
    main()
