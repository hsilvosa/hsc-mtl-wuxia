
import os
import re
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
# nltk.download('punkt')  

try:
    from .nm_aligner import align_segments_nm, empty_alignment_statistics, print_alignment_metrics
except ImportError:
    from nm_aligner import align_segments_nm, empty_alignment_statistics, print_alignment_metrics

MAX_SOURCE_GROUP = 7
MAX_TARGET_GROUP = 7






def get_device():
    """
    Retorna el dispositivo disponible: GPU (cuda) si existe, o CPU en caso contrario.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# Configura el modelo LaBSE para ejecutarse en GPU si está disponible
device = get_device()
print(f"Usando dispositivo: {device}")
model = SentenceTransformer('sentence-transformers/LaBSE', device=str(device))


def list_file_pairs(input_dir="."):
    """
    Busca y empareja archivos .ch.txt y .en.txt con el mismo prefijo numérico.
    
    Args:
        input_dir (str): Directorio donde buscar los archivos.
    
    Returns:
        list of tuple: Lista de tuplas (path_chino, path_ingles)
    """
    files = os.listdir(input_dir)
    en_files = {}
    ch_files = {}

    for f in files:
        match = re.match(r"(\d+)(en|ch)\.txt", f)
        if match:
            num, lang = match.groups()
            idx = int(num)
            if lang == 'en':
                en_files[idx] = os.path.join(input_dir, f)
            else:
                ch_files[idx] = os.path.join(input_dir, f)

    # solo empareja si ambos existen
    keys = sorted(set(en_files) & set(ch_files))
    return [(ch_files[k], en_files[k]) for k in keys]

def fix_broken_words(text):
    """
    Para palabras que en el textro original vienen rotas, "F ang", "c ultivation", se corrigen.

    Args:
        text (str): texto roto

    Returns:
        texto: texto limpio
    """
    # detecta letra + palabra
    pattern = re.compile(r'\b([a-zA-Z])\s+([a-z]{1,})\b')

    # palabras que si son válidas solas no deben unirse 
    no_merge_heads = {'a', 'i', 'he', 'she', 'we', 'you', 'it', 'an'}

    def merge_if_valid(match):
        head = match.group(1).lower()
        tail = match.group(2).lower()
        merged = head + tail
        if head in no_merge_heads:
            return match.group(0)  
        if merged in word_set:
            return merged
        return match.group(0)

    prev = ""
    while prev != text:
        prev = text
        text = pattern.sub(merge_if_valid, text)
    return text

def load_and_clean_text(path_ch, path_en):
    """
    Carga y limpia textos chino y ingles desde sus respectivos archivos.
    
    Elimina líneas vacías, paréntesis, notas y referencias numéricas.
    
    Args:
        path_ch (str): Ruta del archivo chino.
        path_en (str): Ruta del archivo ingles.
    
    Returns:
        tuple: Texto chino limpio, texto ingles limpio.
    """
    #  texto chino
    lines_ch = []
    with open(path_ch, 'r', encoding='utf-8', errors='ignore') as fch:
        for line in fch:
            s = line.lstrip()
            if not s or s.startswith('(') or s.startswith('（'):
                continue  # omite líneas vacías o con notas
            s = s.replace("&amp;amp; {}", "").strip()  # quita secuencia HTML raras
            lines_ch.append(s)
    text_ch = " ".join(lines_ch)
    text_ch = re.sub(r'（[^）]*）', '', text_ch)  # elimina notas entre paréntesis chinos

    #  texto ingles
    lines_en = []
    with open(path_en, 'r', encoding='utf-8', errors='ignore') as fen:
        for line in fen:
            s = line.lstrip()
            if not s or s.startswith('('):
                continue  # omite líneas vacías o notas
            cleaned = re.sub(r"\(\d+\)", "", line).strip()  # quita referencias tipo (1)
            cleaned = re.sub(r"\s{2,}", " ", cleaned)  # normaliza espacios múltiples
            lines_en.append(cleaned)
    text_en = " ".join(lines_en)
    text_en = fix_broken_words(text_en)

    return text_ch, text_en



def split_quotes(text: str, quote_chars: str):
    """
    Separa el texto en torno a cada comilla de `quote_chars`,
    dejando siempre la comilla como token independiente.
    """
    # captura cada comilla (ASCII o Unicode)
    pattern = f"([{re.escape(quote_chars)}])"
    parts = re.split(pattern, text)
    # filtra strings vacíos
    return [p for p in parts if p]

def segment_text(text_ch: str, text_en: str):
    """
    Segmenta el chino y el inglés de forma agresiva:
    - Separa por puntuación.
    - Elimina las comillas (“ ” ").
    """
    quote_chars = '"“”'

    # segmentacion chino
    ch_tokens = split_quotes(text_ch, quote_chars)
    segments_ch = []
    for tok in ch_tokens:
        if tok in quote_chars:
            continue  
        subs = re.split(r'(?<=[。！？；：:])', tok)
        segments_ch.extend([s.strip() for s in subs if s.strip()])

    # segmentacion ingles
    en_tokens = split_quotes(text_en, quote_chars)
    segments_en = []
    for tok in en_tokens:
        if tok in quote_chars:
            continue 
        for sent in sent_tokenize(tok):
            subs = re.split(r'(?<=[\.!\?:;])\s+', sent)
            segments_en.extend([s.strip() for s in subs if s.strip()])

    return segments_ch, segments_en


def get_embeddings(texts):
    """
    Devuelve embeddings para una lista de textos, usando caché para evitar
    recomputar repeticiones entre archivos.

    Args:
        texts (list of str): Lista de textos.

    Returns:
        np.ndarray: Embeddings de los textos en el mismo orden.
    """
    to_compute = []
    computed_indices = []

    # Identificar textos no cacheados
    for i, text in enumerate(texts):
        if text not in embedding_cache:
            to_compute.append(text)
            computed_indices.append(i)

    # Codificar solo los que no están en caché
    if to_compute:
        new_embeds = model.encode(to_compute, normalize_embeddings=True)
        for idx, text in zip(computed_indices, to_compute):
            embedding_cache[text] = new_embeds[idx - computed_indices[0]]

    # Recuperar embeddings en orden original
    return np.array([embedding_cache[text] for text in texts])

def align_segments(segments_ch, segments_en):
    """
    Alinea segmentos y devuelve texto + puntajes de similitud.
    """
    
    aligned, scores, statistics = align_segments_nm(
        segments_ch, segments_en, model,
        max_source_group=MAX_SOURCE_GROUP,
        max_target_group=MAX_TARGET_GROUP,
        skip_penalty=globals().get("skip_penalty", -0.5),
    )
    return aligned, scores, statistics

    # Legacy implementation retained below for reference.
    MAX_ALIGNMENT_MULTIPLIER = 7
    global skip_penalty 

    vecs_ch = model.encode(segments_ch, normalize_embeddings=True)
    vecs_en = model.encode(segments_en, normalize_embeddings=True)

    M, N = len(segments_ch), len(segments_en)

    vecs_ch_multi = {} 
    vecs_en_multi = {} 
    sim_matrices = {}

    #  Generación de Matrices  
    sim_matrices['1-1'] = np.dot(vecs_ch, vecs_en.T)

    for K in range(2, MAX_ALIGNMENT_MULTIPLIER + 1):
        if M >= K:
            combined_ch = [" ".join(segments_ch[i:i+K]) for i in range(M - K + 1)]
            vecs_ch_multi[K] = model.encode(combined_ch, normalize_embeddings=True)
            sim_matrices[f'{K}-1'] = np.dot(vecs_ch_multi[K], vecs_en.T)

        if N >= K:
            combined_en = [" ".join(segments_en[j:j+K]) for j in range(N - K + 1)]
            vecs_en_multi[K] = model.encode(combined_en, normalize_embeddings=True)
            sim_matrices[f'1-{K}'] = np.dot(vecs_ch, vecs_en_multi[K].T)

    # DP 
    DP = np.full((M+1, N+1), -1e9)
    DP[0, 0] = 0.0
    backpointer = [[None] * (N+1) for _ in range(M+1)]

    for i in range(M+1):
        for j in range(N+1):
            for K in range(1, MAX_ALIGNMENT_MULTIPLIER + 1):
                for L in range(1, MAX_ALIGNMENT_MULTIPLIER + 1):
                    action = f'{K}-{L}'
                    if K > 1 and L > 1: continue
                    
                    if K == 1 and L == 1:
                        if i > 0 and j > 0 and action in sim_matrices:
                            score = DP[i-1][j-1] + sim_matrices[action][i-1, j-1]
                            if score > DP[i, j]:
                                DP[i, j] = score
                                backpointer[i][j] = (action, i-1, j-1)
                        continue 
                    
                    if K >= 1 and L >= 1 and action in sim_matrices:
                        if i >= K and j >= L:
                            score = DP[i-K][j-L] + sim_matrices[action][i-K, j-L]
                            if score > DP[i, j]:
                                DP[i, j] = score
                                backpointer[i][j] = (action, i-K, j-L)

            if i > 0:
                score = DP[i-1][j] + skip_penalty
                if score > DP[i, j]:
                    DP[i, j] = score
                    backpointer[i][j] = ("skip_ch", i-1, j)
            if j > 0:
                score = DP[i][j-1] + skip_penalty
                if score > DP[i, j]:
                    DP[i, j] = score
                    backpointer[i][j] = ("skip_en", i, j-1)

    # --- Reconstrucción (MODIFICADO) ---
    aligned = []
    aligned_scores = []  # NUEVA LISTA PARA PUNTAJES
    i, j = M, N
    
    alignment_stats = {
        f'{K}-{L}': 0 
        for K in range(1, MAX_ALIGNMENT_MULTIPLIER + 1) 
        for L in range(1, MAX_ALIGNMENT_MULTIPLIER + 1)
        if K == 1 or L == 1
    }
    alignment_stats.update({
        "skip_ch": 0, "skip_en": 0,
        "total_segments_ch": M, "total_segments_en": N
    }) 

    while i > 0 or j > 0:
        if backpointer[i][j] is None: break
        
        action, pi, pj = backpointer[i][j]
        alignment_stats[action] += 1
        
        if '-' in action:
            K, L = map(int, action.split('-'))
            
            # Recuperar el puntaje exacto de la matriz correspondiente
            # pi y pj son los índices iniciales del bloque, que coinciden con la matriz
            similarity_score = float(sim_matrices[action][pi, pj])
            aligned_scores.append(similarity_score)

            combined_ch = ' '.join(segments_ch[pi:pi+K]) 
            combined_en = ' '.join(segments_en[pj:pj+L])
            
            aligned.append((combined_ch, combined_en))
            i, j = pi, pj 
        
        elif action == "skip_ch":
            i -= 1
        elif action == "skip_en":
            j -= 1
            
    aligned.reverse()
    aligned_scores.reverse() # Invertimos también los puntajes
    
    # Devolvemos aligned, aligned_scores y stats
    return aligned, aligned_scores, alignment_stats


def calculate_and_print_metrics(global_stats, total_elapsed):
    print_alignment_metrics(global_stats, total_elapsed)
    return
    """
    Calcula métricas derivadas y las imprime, incluyendo combinaciones dinámicas K:1 y 1:L.
    """
    
    MAX_ALIGNMENT_MULTIPLIER = 7
    
    print("\n## Estadísticas Globales de Alineamiento y Rendimiento")
    print("-" * 60)
    
    M = global_stats['total_segments_ch']
    N = global_stats['total_segments_en']
    
    print(f"**Tiempo Total de Procesamiento:** {total_elapsed:.2f} segundos")
    print(f"**Segmentos Chinos Totales (M):** {M}")
    print(f"**Segmentos Ingleses Totales (N):** {N}")

    
    total_aligned_pairs = 0
    ch_used_in_pairs = 0
    en_used_in_pairs = 0
    
    print("\n# Frecuencia de Acciones")
    
    # Calcular y mostrar las combinaciones dinámicas
    alignment_counts = {}
    for K in range(1, MAX_ALIGNMENT_MULTIPLIER + 1):
        for L in range(1, MAX_ALIGNMENT_MULTIPLIER + 1):
            if K == 1 or L == 1:
                key = f'{K}-{L}'
                count = global_stats.get(key, 0)
                if count > 0:
                    alignment_counts[key] = count
                    total_aligned_pairs += count
                    ch_used_in_pairs += count * K
                    en_used_in_pairs += count * L
                    print(f"Alineamientos {key} (Ch: {K}, En: {L}): {count}")

    print(f"Pares Alineados Totales: {total_aligned_pairs}")
    
    # Métricas de salto
    skip_ch = global_stats.get('skip_ch', 0)
    skip_en = global_stats.get('skip_en', 0)
    
    print(f"Segmentos Chinos Saltados (`skip_ch`): {skip_ch}")
    print(f"Segmentos Ingleses Saltados (`skip_en`): {skip_en}")
    
    ch_unaccounted = M - (ch_used_in_pairs + skip_ch)
    en_unaccounted = N - (en_used_in_pairs + skip_en)
    
    ch_skipped_percent = (skip_ch / M) * 100 if M else 0
    en_skipped_percent = (skip_en / N) * 100 if N else 0
    
    print("\n# Impacto de la Penalización de Salto (`skip_penalty`)")
    print(f"% Chinos Saltados: {ch_skipped_percent:.2f}% del total de segmentos chinos.")
    print(f"% Ingleses Saltados: {en_skipped_percent:.2f}% del total de segmentos ingleses.")

    if ch_unaccounted != 0 or en_unaccounted != 0:
        print(f"\nSegmentos no contabilizados (Ch: {ch_unaccounted}, En: {en_unaccounted}).")
    
    print("-" * 60)
    
    
    

def labse_similarity(text1, text2):
    """
    Calcula la similitud de coseno entre dos textos usando LaBSE.
    """
    vec1, vec2 = get_embeddings([text1, text2])
    return float(np.dot(vec1, vec2.T))


def process_all_files(input_dir="."):
    """
    Procesa archivos, alinea y guarda texto y similitudes en dos archivos separados.
    """
    pairs = list_file_pairs(input_dir)
    all_aligned = []

    global_stats = empty_alignment_statistics(MAX_SOURCE_GROUP, MAX_TARGET_GROUP)

    output_scores_file = "final_awe_similitudes_3.txt"

    # Abrimos ambos archivos simultáneamente
    with open(output_file, 'w', encoding='utf-8') as fout, \
         open(output_scores_file, 'w', encoding='utf-8') as fscore:
        
        for path_ch, path_en in pairs:
            print(f"Procesando par: {path_ch} + {path_en}")
            start = time.perf_counter()

            text_ch, text_en = load_and_clean_text(path_ch, path_en)
            seg_ch, seg_en = segment_text(text_ch, text_en)
            
            # Alineación inicial (ahora desempaquetamos 3 valores)
            aligned, scores, file_stats = align_segments(seg_ch, seg_en) 
            
            for key, value in file_stats.items():
                global_stats[key] = global_stats.get(key, 0) + value
            
            # Iteramos sobre pares y puntajes al mismo tiempo
            for (ch_sub, en_sub), score in zip(aligned, scores):
                fout.write(f"{ch_sub} ; {en_sub}\n")
                # Escribimos el score con 5 decimales (puedes ajustar el .5f)
                fscore.write(f"{score:.5f}\n")

            elapsed = time.perf_counter() - start
            print(f"  Tiempo: {elapsed:.2f}s")
            all_aligned.extend(aligned)

    calculate_and_print_metrics(global_stats, time.perf_counter() - total_start)

    return all_aligned





if __name__ == '__main__':
    
    
    # Configuración 
    ch_suffix = "ch.txt"
    en_suffix = "en.txt"
        
    output_dir = os.path.join("data", "awe", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_awe.txt")
    
    skip_penalty = -0.2  # penalización por saltarse segmentos
    embedding_cache = {}  # Diccionario global: texto -> embedding
    
    # Palabras para corrección del texto
    word_set = set(words.words())

    total_start = time.perf_counter()
    resultados = process_all_files("data/awe/segmented/chapter")
    total_elapsed = time.perf_counter() - total_start
    print(f"Proceso completado. Tiempo total: {total_elapsed:.2f}s con {len(resultados)} segmentos alineados.")

