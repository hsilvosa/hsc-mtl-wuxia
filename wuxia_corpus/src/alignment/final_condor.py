
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
import contractions

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
    
    text = contractions.fix(text)
    
    text = re.sub(r"\b(\w+)[’']s\b(?=\s|$)", r"\1", text)            
            
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
    quote_chars = '"“”‘’\''
    
    
    
    
    
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
    Alinea segmentos de texto en chino y ingles usando embeddings de LaBSE
    y programación dinámica.

    El algoritmo busca la mejor correspondencia entre pares de segmentos basándose
    en la similitud de coseno entre sus embeddings. Considera los siguientes emparejamientos:
    
    - 1-1: un segmento chino con uno ingles
    - 1-2: un segmento chino con dos segmentos ingleses
    - 2-1: dos segmentos chinos con uno ingles
    - 1-3: un segmento chino con tres segmentos ingleses
    - 3-1: tres segmentos chinos con un segmento ingles
    - 1-4: un segmento chino con cuatro segmentos ingleses
    - 4-1: cuatro segmentos chinos con un segmento ingles
    - Saltos (omisiones) penalizados

    Args:
        segments_ch (list): Lista de segmentos en chino.
        segments_en (list): Lista de segmentos en ingles.

    Returns:
        list of tuple: Lista alineada de pares (segmento_ch, segmento_en).
    """

    aligned, _, statistics = align_segments_nm(
        segments_ch, segments_en, model,
        max_source_group=MAX_SOURCE_GROUP,
        max_target_group=MAX_TARGET_GROUP,
        skip_penalty=globals().get("skip_penalty", -0.5),
    )
    return aligned, statistics

    # Legacy implementation retained below for reference.
    # Generar embeddings normalizados
    vecs_ch = model.encode(segments_ch, normalize_embeddings=True)
    vecs_en = model.encode(segments_en, normalize_embeddings=True)

    M, N = len(segments_ch), len(segments_en)

    # Vectores combinados para grupos de segmentos
    vecs_ch2 = vecs_ch3 = vecs_ch4 = None
    vecs_en2 = vecs_en3 = vecs_en4 = None

    # Combinaciones de chino
    if M > 1:
        combined_ch_segments_2 = [segments_ch[i] + " " + segments_ch[i+1] for i in range(M-1)]
        vecs_ch2 = model.encode(combined_ch_segments_2, normalize_embeddings=True)
    if M > 2:
        combined_ch_segments_3 = [segments_ch[i] + " " + segments_ch[i+1] + " " + segments_ch[i+2] for i in range(M-2)]
        vecs_ch3 = model.encode(combined_ch_segments_3, normalize_embeddings=True)
    if M > 3:
        combined_ch_segments_4 = [
            segments_ch[i] + " " + segments_ch[i+1] + " " + segments_ch[i+2] + " " + segments_ch[i+3]
            for i in range(M-3)
        ]
        vecs_ch4 = model.encode(combined_ch_segments_4, normalize_embeddings=True)

    # Combinaciones de ingles
    if N > 1:
        combined_en_segments_2 = [segments_en[j] + " " + segments_en[j+1] for j in range(N-1)]
        vecs_en2 = model.encode(combined_en_segments_2, normalize_embeddings=True)
    if N > 2:
        combined_en_segments_3 = [segments_en[j] + " " + segments_en[j+1] + " " + segments_en[j+2] for j in range(N-2)]
        vecs_en3 = model.encode(combined_en_segments_3, normalize_embeddings=True)
    if N > 3:
        combined_en_segments_4 = [
            segments_en[j] + " " + segments_en[j+1] + " " + segments_en[j+2] + " " + segments_en[j+3]
            for j in range(N-3)
        ]
        vecs_en4 = model.encode(combined_en_segments_4, normalize_embeddings=True)

    # Matrices de similitud
    sim_1to1 = np.dot(vecs_ch, vecs_en.T)
    sim_1to2 = np.dot(vecs_ch, vecs_en2.T) if vecs_en2 is not None else None
    sim_2to1 = np.dot(vecs_ch2, vecs_en.T) if vecs_ch2 is not None else None
    sim_1to3 = np.dot(vecs_ch, vecs_en3.T) if vecs_en3 is not None else None
    sim_3to1 = np.dot(vecs_ch3, vecs_en.T) if vecs_ch3 is not None else None
    sim_1to4 = np.dot(vecs_ch, vecs_en4.T) if vecs_en4 is not None else None
    sim_4to1 = np.dot(vecs_ch4, vecs_en.T) if vecs_ch4 is not None else None

    # Inicializar DP
    DP = np.full((M+1, N+1), -1e9)
    DP[0, 0] = 0.0
    backpointer = [[None] * (N+1) for _ in range(M+1)]

    # Relleno de DP
    for i in range(M+1):
        for j in range(N+1):
            if i > 0 and j > 0:
                score = DP[i-1][j-1] + sim_1to1[i-1, j-1]
                if score > DP[i, j]:
                    DP[i, j] = score
                    backpointer[i][j] = ("1-1", i-1, j-1)
            if i > 0 and j > 1 and sim_1to2 is not None:
                score = DP[i-1][j-2] + sim_1to2[i-1, j-2]
                if score > DP[i, j]:
                    DP[i, j] = score
                    backpointer[i][j] = ("1-2", i-1, j-2)
            if i > 1 and j > 0 and sim_2to1 is not None:
                score = DP[i-2][j-1] + sim_2to1[i-2, j-1]
                if score > DP[i, j]:
                    DP[i, j] = score
                    backpointer[i][j] = ("2-1", i-2, j-1)
            if i > 0 and j > 2 and sim_1to3 is not None:
                score = DP[i-1][j-3] + sim_1to3[i-1, j-3]
                if score > DP[i, j]:
                    DP[i, j] = score
                    backpointer[i][j] = ("1-3", i-1, j-3)
            if i > 2 and j > 0 and sim_3to1 is not None:
                score = DP[i-3][j-1] + sim_3to1[i-3, j-1]
                if score > DP[i, j]:
                    DP[i, j] = score
                    backpointer[i][j] = ("3-1", i-3, j-1)
            if i > 0 and j > 3 and sim_1to4 is not None:
                score = DP[i-1][j-4] + sim_1to4[i-1, j-4]
                if score > DP[i, j]:
                    DP[i, j] = score
                    backpointer[i][j] = ("1-4", i-1, j-4)
            if i > 3 and j > 0 and sim_4to1 is not None:
                score = DP[i-4][j-1] + sim_4to1[i-4, j-1]
                if score > DP[i, j]:
                    DP[i, j] = score
                    backpointer[i][j] = ("4-1", i-4, j-1)
            # Saltos con penalización
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

    # Reconstrucción del camino óptimo
    aligned = []
    i, j = M, N
    
    alignment_stats = {
        "1-1": 0, "1-2": 0, "2-1": 0, "1-3": 0, "3-1": 0, 
        "1-4":0, "4-1":0,
        "skip_ch": 0, "skip_en": 0,
        "total_segments_ch": M, "total_segments_en": N
    } 

    while i > 0 or j > 0:
        action, pi, pj = backpointer[i][j]  
        
  
        alignment_stats[action] += 1
        if action == "1-1":
            aligned.append((segments_ch[pi], segments_en[pj]))
            i, j = pi, pj
        elif action == "1-2":
            combined = segments_en[pj] + ' ' + segments_en[pj+1]
            aligned.append((segments_ch[pi], combined))
            i, j = pi, pj
        elif action == "2-1":
            combined = segments_ch[pi] + ' ' + segments_ch[pi+1]
            aligned.append((combined, segments_en[pj]))
            i, j = pi, pj
        elif action == "1-3":
            combined = ' '.join(segments_en[pj:pj+3])
            aligned.append((segments_ch[pi], combined))
            i -= 1; j -= 3
        elif action == "3-1":
            combined = ' '.join(segments_ch[pi:pi+3])
            aligned.append((combined, segments_en[pj]))
            i -= 3; j -= 1
        elif action == "1-4":
            combined = ' '.join(segments_en[pj:pj+4])
            aligned.append((segments_ch[pi], combined))
            i -= 1; j -= 4
        elif action == "4-1":
            combined = ' '.join(segments_ch[pi:pi+4])
            aligned.append((combined, segments_en[pj]))
            i -= 4; j -= 1
        elif action == "skip_ch":
            i -= 1
        else:  # skip_en
            j -= 1
    aligned.reverse()
    return aligned, alignment_stats


def labse_similarity(text1, text2):
    """
    Calcula la similitud de coseno entre dos textos usando LaBSE.
    """
    vec1, vec2 = get_embeddings([text1, text2])
    return float(np.dot(vec1, vec2.T))

def calculate_and_print_metrics(global_stats, total_elapsed):
    print_alignment_metrics(global_stats, total_elapsed)
    return
    """
    Calcula métricas derivadas y las imprime, ofreciendo un resumen del proceso de alineamiento.

    Args:
        global_stats (dict): Diccionario acumulado de contadores de alineamiento.
        total_elapsed (float): Tiempo total empleado en el proceso.
    """
    print("\n## Estadísticas Globales de Alineamiento y Rendimiento")
    print("-" * 60)
    
    M = global_stats['total_segments_ch']
    N = global_stats['total_segments_en']
    
    print(f"**Tiempo Total de Procesamiento:** {total_elapsed:.2f} segundos")
    print(f"**Segmentos Chinos Totales (M):** {M}")
    print(f"**Segmentos Ingleses Totales (N):** {N}")

    
    total_aligned_pairs = (global_stats["1-1"] + global_stats["1-2"] + 
                        global_stats["2-1"] + global_stats["1-3"] + 
                        global_stats["3-1"] + global_stats["4-1"]+ global_stats["1-4"])

    ch_used_in_pairs = (global_stats["1-1"] * 1 + global_stats["1-2"] * 1 + 
                        global_stats["2-1"] * 2 + global_stats["1-3"] * 1 + 
                        global_stats["3-1"] * 3 + global_stats["1-4"] * 1 + 
                        global_stats["4-1"] * 4)                             
                        
    en_used_in_pairs = (global_stats["1-1"] * 1 + global_stats["1-2"] * 2 + 
                        global_stats["2-1"] * 1 + global_stats["1-3"] * 3 + 
                        global_stats["3-1"] * 1 + global_stats["1-4"] * 4 + 
                        global_stats["4-1"] * 1)
    
    ch_unaccounted = M - (ch_used_in_pairs + global_stats['skip_ch'])
    en_unaccounted = N - (en_used_in_pairs + global_stats['skip_en'])

    print("\n# Frecuencia de Acciones")
    print(f"Pares Alineados Totales: {total_aligned_pairs}")
    print(f"Alineamientos 1-1: {global_stats['1-1']}")
    print(f"Alineamientos 1-2 (Ch: 1, En: 2): {global_stats['1-2']}")
    print(f"Alineamientos 2-1 (Ch: 2, En: 1): {global_stats['2-1']}")
    print(f"Alineamientos 1-3 (Ch: 1, En: 3): {global_stats['1-3']}")
    print(f"Alineamientos 3-1 (Ch: 3, En: 1): {global_stats['3-1']}")
    print(f"Alineamientos 1-4 (Ch: 1, En: 4): {global_stats['1-4']}")
    print(f"Alineamientos 4-1 (Ch: 4, En: 1): {global_stats['4-1']}")
    print(f"Segmentos Chinos Saltados (`skip_ch`): {global_stats['skip_ch']}")
    print(f"Segmentos Ingleses Saltados (`skip_en`): {global_stats['skip_en']}")
    
    ch_skipped_percent = (global_stats['skip_ch'] / M) * 100 if M else 0
    en_skipped_percent = (global_stats['skip_en'] / N) * 100 if N else 0
    
    print("\n# Impacto de la Penalización de Salto (`skip_penalty`)")
    print(f"% Chinos Saltados: {ch_skipped_percent:.2f}% del total de segmentos chinos.")
    print(f"% Ingleses Saltados: {en_skipped_percent:.2f}% del total de segmentos ingleses.")

    if ch_unaccounted != 0 or en_unaccounted != 0:
        print(f"\nSegmentos no contabilizados (Ch: {ch_unaccounted}, En: {en_unaccounted}).")
    
    print("-" * 60)
    
def process_all_files(input_dir=".", output_file="final_default.txt"):
    """
    Procesa todos los archivos en el directorio: carga textos, segmenta en cláusulas optimizado,
    alinea una sola vez con embeddings multilingües y guarda el resultado.
    """
    pairs = list_file_pairs(input_dir)
    all_aligned = []


    global_stats = empty_alignment_statistics(MAX_SOURCE_GROUP, MAX_TARGET_GROUP)

    with open(output_file, 'w', encoding='utf-8') as fout:
        for path_ch, path_en in pairs:
            print(f"Procesando par: {path_ch} + {path_en}")
            start = time.perf_counter()

            text_ch, text_en = load_and_clean_text(path_ch, path_en)
            seg_ch, seg_en = segment_text(text_ch, text_en)
            # Alineación inicial
            aligned, file_stats = align_segments(seg_ch, seg_en) 
            
            for key, value in file_stats.items():
                global_stats[key] = global_stats.get(key, 0) + value
                
            for ch_sub, en_sub in aligned:
                fout.write(f"{ch_sub} ; {en_sub}\n")

            elapsed = time.perf_counter() - start
            print(f"  Tiempo: {elapsed:.2f}s")
            all_aligned.extend(aligned)
    calculate_and_print_metrics(global_stats, time.perf_counter() - total_start) # total_start debe ser accesible

    return all_aligned, global_stats





# Bloque principal de ejecución
if __name__ == '__main__':
    
    
    ch_suffix = "ch.txt"
    en_suffix = "en.txt"
    
    skip_penalty = -0.5  # penalización por saltarse segmentos
    embedding_cache = {}  # Diccionario global: texto -> embedding

    # Palabras para corrección del texto
    word_set = set(words.words())
    
    
    global_total_stats = empty_alignment_statistics(MAX_SOURCE_GROUP, MAX_TARGET_GROUP)
    
            
    output_dir = os.path.join("data", "condor", "processed")
    os.makedirs(output_dir, exist_ok=True)
    base_input_dir = os.path.join("data", "condor", "segmented")
        
        
    total_process_start = time.perf_counter()
    
    for i in [1, 2, 3]:
        output_file_name = os.path.join(output_dir, f"final_condor_{i}_stats.txt")
        chapter_dir = os.path.join(base_input_dir, f"Condor{i}")
        
        total_start = time.perf_counter()

        resultados_capitulo, stats_capitulo = process_all_files(chapter_dir, output_file=output_file_name)        
        total_elapsed = time.perf_counter() - total_start
        
        for key in global_total_stats:
            global_total_stats[key] += stats_capitulo.get(key, 0)
            
        print(
            f"\nProceso completado para Condor{i}. "
            f"Tiempo del capítulo: {total_elapsed:.2f}s con {len(resultados_capitulo)} segmentos alineados."
        )
