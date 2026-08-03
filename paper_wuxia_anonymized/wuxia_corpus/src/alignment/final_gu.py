
import os
import re
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
nltk.download('punkt')  

try:
    from .nm_aligner import align_segments_nm, empty_alignment_statistics, print_alignment_metrics
except ImportError:  # Allow direct execution from this directory.
    from nm_aligner import align_segments_nm, empty_alignment_statistics, print_alignment_metrics

MAX_SOURCE_GROUP = 7
MAX_TARGET_GROUP = 7






def get_device():
    """
    Returns the device available: GPU (cuda) if exists, o CPU otherwise.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# Configures the model LaBSE for run in GPU when available
device = get_device()
print(f"Usando dispositivo: {device}")
model = SentenceTransformer('sentence-transformers/LaBSE', device=str(device))


def list_file_pairs(input_dir="."):
    """
    Finds and pairs files .ch.txt and .en.txt with the same prefix numeric.
    
    Args:
        input_dir (str): Directory where find the files.
    
    Returns:
        list of tuple: List of tuples (path_chino, path_ingles)
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

    # only pairs if both exist
    keys = sorted(set(en_files) & set(ch_files))
    return [(ch_files[k], en_files[k]) for k in keys]

def fix_broken_words(text):
    """
    For words that in the textro original are rotas, "F ang", "c ultivation", is corrigen.

    Args:
        text (str): text roto

    Returns:
        text: text limpio
    """
    # detects letter + word
    pattern = re.compile(r'\b([a-zA-Z])\s+([a-z]{1,})\b')

    # words that if are valid alone not must be joined 
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
    Loads and cleans texts Chinese and English from their respective files.
    
    Removes lines empty, parentheses, notes and references numeric.
    
    Args:
        path_ch (str): Path of the file Chinese.
        path_en (str): Path of the file English.
    
    Returns:
        tuple: Text Chinese limpio, text English limpio.
    """
    #  text Chinese
    lines_ch = []
    with open(path_ch, 'r', encoding='utf-8', errors='ignore') as fch:
        for line in fch:
            s = line.lstrip()
            if not s or s.startswith('(') or s.startswith('（'):
                continue  # skips lines empty or with notes
            s = s.replace("&amp;amp; {}", "").strip()  # removes secuencia HTML raras
            lines_ch.append(s)
    text_ch = " ".join(lines_ch)
    text_ch = re.sub(r'（[^）]*）', '', text_ch)  # removes notes between parentheses Chinese

    #  text English
    lines_en = []
    with open(path_en, 'r', encoding='utf-8', errors='ignore') as fen:
        for line in fen:
            s = line.lstrip()
            if not s or s.startswith('('):
                continue  # skips lines empty or notes
            cleaned = re.sub(r"\(\d+\)", "", line).strip()  # removes references type (1)
            cleaned = re.sub(r"\s{2,}", " ", cleaned)  # normaliza espacios multiple
            lines_en.append(cleaned)
    text_en = " ".join(lines_en)
    text_en = fix_broken_words(text_en)

    return text_ch, text_en



def split_quotes(text: str, quote_chars: str):
    """
    Splits the text around each comilla of `quote_chars`,
    leaving siempre the comilla as token independiente.
    """
    # captures each comilla (ASCII or Unicode)
    pattern = f"([{re.escape(quote_chars)}])"
    parts = re.split(pattern, text)
    # filters strings empty
    return [p for p in parts if p]

def segment_text(text_ch: str, text_en: str):
    """
    Segments the Chinese and the English of way aggressively:
    - Splits by score.
    - Removes the quotation marks (“ ” ").
    """
    quote_chars = '"“”'

    # segmentacion Chinese
    ch_tokens = split_quotes(text_ch, quote_chars)
    segments_ch = []
    for tok in ch_tokens:
        if tok in quote_chars:
            continue  
        subs = re.split(r'(?<=[。！？；：:])', tok)
        segments_ch.extend([s.strip() for s in subs if s.strip()])

    # segmentacion English
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
    Returns embeddings for a list of texts, using cache for avoid
    recompute repetitions between files.

    Args:
        texts (list of str): List of texts.

    Returns:
        np.ndarray: Embeddings of the texts in the same order.
    """
    to_compute = []
    computed_indices = []

    # Identify texts not cached
    for i, text in enumerate(texts):
        if text not in embedding_cache:
            to_compute.append(text)
            computed_indices.append(i)

    # Codificar only the that not are in cache
    if to_compute:
        new_embeds = model.encode(to_compute, normalize_embeddings=True)
        for idx, text in zip(computed_indices, to_compute):
            embedding_cache[text] = new_embeds[idx - computed_indices[0]]

    # Retrieve embeddings in order original
    return np.array([embedding_cache[text] for text in texts])


def align_segments(segments_ch, segments_en):
    """
    Aligns segments of text in Chinese and English using embeddings of LaBSE
    and programming dynamic.

    The algorithm finds the best match between pairs of segments based
    in the similarity of cosine between their embeddings. Considera the following pairings:
    
    - 1-1: to segment Chinese with one English
    - 1-2: to segment Chinese with two segments English
    - 2-1: two segments Chinese with one English
    - 1-3: to segment Chinese with three segments English
    - 3-1: three segments Chinese with to segment English
    - 1-4: to segment Chinese with four segments English
    - 4-1: four segments Chinese with to segment English
    - Skips (omisiones) penalizados

    Args:
        segments_ch (list): List of segments in Chinese.
        segments_en (list): List of segments in English.

    Returns:
        list of tuple: List alineada of pairs (segmento_ch, segmento_en).
    """

    aligned, _, statistics = align_segments_nm(
        segments_ch, segments_en, model,
        max_source_group=MAX_SOURCE_GROUP,
        max_target_group=MAX_TARGET_GROUP,
        skip_penalty=globals().get("skip_penalty", -0.5),
    )
    return aligned, statistics




def labse_similarity(text1, text2):
    """
    Calculates the similarity of cosine between two texts using LaBSE.
    """
    vec1, vec2 = get_embeddings([text1, text2])
    return float(np.dot(vec1, vec2.T))

def calculate_and_print_metrics(global_stats, total_elapsed):
    """
    Calculates metrics derived and the prints, providing a summary of the process of alignment.

    Args:
        global_stats (dict): Dictionary accumulated of counters of alignment.
        total_elapsed (float): Total time used in the process.
    """
    print_alignment_metrics(global_stats, total_elapsed)
    return


    
def process_all_files(input_dir="."):
    """
    Processes all files in the directory: loads texts, segments in clauses optimized,
    aligns once with embeddings multilingual and saves the result.
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
            # Alignment initial
            aligned, file_stats = align_segments(seg_ch, seg_en) 
            
            for key, value in file_stats.items():
                global_stats[key] = global_stats.get(key, 0) + value
                
            for ch_sub, en_sub in aligned:
                fout.write(f"{ch_sub} ; {en_sub}\n")

            elapsed = time.perf_counter() - start
            print(f"  Time: {elapsed:.2f}s")
            all_aligned.extend(aligned)
    calculate_and_print_metrics(global_stats, time.perf_counter() - total_start) # total_start must be accessible

    return all_aligned





# Block main of execution
if __name__ == '__main__':
    
    
    # Configuration 
    ch_suffix = "ch.txt"
    en_suffix = "en.txt"
    
    output_dir = os.path.join("data", "gu", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_gu.txt")
    
    
    skip_penalty = -0.5  # penalty by skip segments
    embedding_cache = {}  # Dictionary global: text -> embedding
    
    # Words for correction of the text
    word_set = set(words.words())

    total_start = time.perf_counter()
    resultados = process_all_files("data/gu/segmented/chapter")
    total_elapsed = time.perf_counter() - total_start
    print(f"Process completed. Total time: {total_elapsed:.2f}s with {len(resultados)} segments aligned.")

