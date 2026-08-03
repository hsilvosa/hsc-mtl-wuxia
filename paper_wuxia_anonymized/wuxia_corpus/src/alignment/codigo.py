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
    from .nm_aligner import align_segments_nm
except ImportError:  # Allow direct execution from this directory.
    from nm_aligner import align_segments_nm

MAX_SOURCE_GROUP = 7
MAX_TARGET_GROUP = 7


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model LaBSE
device = get_device()
print(f"Usando dispositivo: {device}")
model = SentenceTransformer('sentence-transformers/LaBSE', device=str(device))


def list_file_pairs(input_dir="."):
    files = os.listdir(input_dir)
    en_files, ch_files = {}, {}
    for f in files:
        m = re.match(r"(\d+)(en|ch)\.txt", f)
        if m:
            num, lang = int(m.group(1)), m.group(2)
            if lang == 'en':   en_files[num] = os.path.join(input_dir, f)
            else:              ch_files[num] = os.path.join(input_dir, f)
    keys = sorted(set(en_files) & set(ch_files))
    return [(ch_files[k], en_files[k]) for k in keys]


def fix_broken_words(text):
    pattern = re.compile(r'\b([a-zA-Z])\s+([a-z]{1,})\b')
    no_merge = {'a','i','he','she','we','you','it','an'}
    def merge_if_valid(m):
        h, t = m.group(1).lower(), m.group(2).lower()
        if h in no_merge: return m.group(0)
        if h + t in word_set: return h + t
        return m.group(0)
    prev = None
    while prev != text:
        prev, text = text, pattern.sub(merge_if_valid, text)
    return text


def load_and_clean_text(path_ch, path_en):
    # Chinese
    lines = []
    with open(path_ch, 'r', encoding='utf-8', errors='ignore') as f:
        for L in f:
            s = L.strip()
            if not s or s.startswith(('(', '（')): continue
            lines.append(s.replace("&amp;amp; {}", ""))
    text_ch = re.sub(r'（[^）]*）', '', " ".join(lines))

    # English
    lines = []
    with open(path_en, 'r', encoding='utf-8', errors='ignore') as f:
        for L in f:
            s = L.strip()
            if not s or s.startswith('('): continue
            s = re.sub(r"\(\d+\)", "", s)
            lines.append(s)
    text_en = fix_broken_words(" ".join(lines))

    return text_ch, text_en


def split_quotes(text, quote_chars='"“”'):
    parts = re.split(f"([{re.escape(quote_chars)}])", text)
    return [p for p in parts if p]


def segment_text(text_ch, text_en):
    # Chinese: split in each 。！？；：:
    ch_tokens = split_quotes(text_ch)
    seg_ch = []
    for tok in ch_tokens:
        if tok in '"“”': continue
        for s in re.split(r'(?<=[。！？；：:])', tok):
            if s.strip(): seg_ch.append(s.strip())

    # English: split in each . ! ? : ;
    en_tokens = split_quotes(text_en)
    seg_en = []
    for tok in en_tokens:
        if tok in '"“”': continue
        for sent in sent_tokenize(tok):
            for s in re.split(r'(?<=[\.!\?:;])\s+', sent):
                if s.strip(): seg_en.append(s.strip())

    return seg_ch, seg_en


def merge_segments_ch(segments):
    """
    Joins in Chinese dialogue + narrator when the segundo segment starts with 说道/问道/…
    """
    merged, i = [], 0
    tag_re = re.compile(r'(?:说道|问道|答道|喊道|叫道)')
    while i < len(segments):
        if i+1 < len(segments) \
           and segments[i].endswith('。') \
           and tag_re.search(segments[i+1]):
            merged.append(segments[i] + " " + segments[i+1])
            i += 2
        else:
            merged.append(segments[i])
            i += 1
    return merged


def merge_segments_en(segments):
    """
    Joins in English when to segment ends in comma/;/: and the next
    starts in lowercase (continuation narrative inside of dialogue).
    """
    merged, i = [], 0
    while i < len(segments):
        if i+1 < len(segments) \
           and re.search(r'[,;:]\s*$', segments[i]) \
           and segments[i+1] and segments[i+1][0].islower():
            merged.append(segments[i].rstrip() + " " + segments[i+1])
            i += 2
        else:
            merged.append(segments[i])
            i += 1
    return merged


def get_embeddings(texts):
    to_compute, idxs = [], []
    for k, t in enumerate(texts):
        if t not in embedding_cache:
            to_compute.append(t)
            idxs.append(k)
    if to_compute:
        embs = model.encode(to_compute, normalize_embeddings=True)
        for i, t in zip(idxs, to_compute):
            embedding_cache[t] = embs[idxs.index(i)]
    return np.array([embedding_cache[t] for t in texts])


def align_segments(segments_ch, segments_en):
    """
    Aligns with DP permitiendo combinations 1-1,1-2,2-1,1-3,3-1,1-4,4-1
    without no filter extra.
    """
    aligned, _, _ = align_segments_nm(
        segments_ch, segments_en, model,
        max_source_group=MAX_SOURCE_GROUP,
        max_target_group=MAX_TARGET_GROUP,
        skip_penalty=globals().get("skip_penalty", -0.5),
    )
    return aligned

    # Legacy implementation retained below for reference.
    M, N = len(segments_ch), len(segments_en)
    vecs_ch = model.encode(segments_ch, normalize_embeddings=True)
    vecs_en = model.encode(segments_en, normalize_embeddings=True)

    # combinations Chinese
    vecs_ch2 = model.encode([segments_ch[i]+" "+segments_ch[i+1] for i in range(M-1)],
                             normalize_embeddings=True) if M>1 else None
    vecs_ch3 = model.encode([segments_ch[i]+" "+segments_ch[i+1]+" "+segments_ch[i+2]
                             for i in range(M-2)], normalize_embeddings=True) if M>2 else None
    vecs_ch4 = model.encode([segments_ch[i]+" "+segments_ch[i+1]+" "+segments_ch[i+2]+" "+segments_ch[i+3]
                             for i in range(M-3)], normalize_embeddings=True) if M>3 else None

    # combinations English
    vecs_en2 = model.encode([segments_en[j]+" "+segments_en[j+1] for j in range(N-1)],
                             normalize_embeddings=True) if N>1 else None
    vecs_en3 = model.encode([segments_en[j]+" "+segments_en[j+1]+" "+segments_en[j+2]
                             for j in range(N-2)], normalize_embeddings=True) if N>2 else None
    vecs_en4 = model.encode([segments_en[j]+" "+segments_en[j+1]+" "+segments_en[j+2]+" "+segments_en[j+3]
                             for j in range(N-3)], normalize_embeddings=True) if N>3 else None

    # similitudes
    sim_1to1 = np.dot(vecs_ch, vecs_en.T)
    sim_1to2 = np.dot(vecs_ch, vecs_en2.T) if vecs_en2 is not None else None
    sim_2to1 = np.dot(vecs_ch2, vecs_en.T) if vecs_ch2 is not None else None
    sim_1to3 = np.dot(vecs_ch, vecs_en3.T) if vecs_en3 is not None else None
    sim_3to1 = np.dot(vecs_ch3, vecs_en.T) if vecs_ch3 is not None else None
    sim_1to4 = np.dot(vecs_ch, vecs_en4.T) if vecs_en4 is not None else None
    sim_4to1 = np.dot(vecs_ch4, vecs_en.T) if vecs_ch4 is not None else None

    # DP init
    DP = np.full((M+1, N+1), -1e9)
    DP[0,0] = 0.0
    back = [[None]*(N+1) for _ in range(M+1)]

    # DP fill
    for i in range(M+1):
        for j in range(N+1):
            if i>0 and j>0:
                s = DP[i-1][j-1] + sim_1to1[i-1,j-1]
                if s>DP[i][j]:
                    DP[i][j], back[i][j] = s, ("1-1", i-1, j-1)
            if i>0 and j>1 and sim_1to2 is not None:
                s = DP[i-1][j-2] + sim_1to2[i-1,j-2]
                if s>DP[i][j]:
                    DP[i][j], back[i][j] = s, ("1-2", i-1, j-2)
            if i>1 and j>0 and sim_2to1 is not None:
                s = DP[i-2][j-1] + sim_2to1[i-2,j-1]
                if s>DP[i][j]:
                    DP[i][j], back[i][j] = s, ("2-1", i-2, j-1)
            if i>0 and j>2 and sim_1to3 is not None:
                s = DP[i-1][j-3] + sim_1to3[i-1,j-3]
                if s>DP[i][j]:
                    DP[i][j], back[i][j] = s, ("1-3", i-1, j-3)
            if i>2 and j>0 and sim_3to1 is not None:
                s = DP[i-3][j-1] + sim_3to1[i-3,j-1]
                if s>DP[i][j]:
                    DP[i][j], back[i][j] = s, ("3-1", i-3, j-1)
            if i>0 and j>3 and sim_1to4 is not None:
                s = DP[i-1][j-4] + sim_1to4[i-1,j-4]
                if s>DP[i][j]:
                    DP[i][j], back[i][j] = s, ("1-4", i-1, j-4)
            if i>3 and j>0 and sim_4to1 is not None:
                s = DP[i-4][j-1] + sim_4to1[i-4,j-1]
                if s>DP[i][j]:
                    DP[i][j], back[i][j] = s, ("4-1", i-4, j-1)
            if i>0:
                s = DP[i-1][j] + skip_penalty
                if s>DP[i][j]:
                    DP[i][j], back[i][j] = s, ("skip_ch", i-1, j)
            if j>0:
                s = DP[i][j-1] + skip_penalty
                if s>DP[i][j]:
                    DP[i][j], back[i][j] = s, ("skip_en", i, j-1)

    # reconstruction
    aligned = []
    i, j = M, N
    while i>0 or j>0:
        act, pi, pj = back[i][j]
        if act=="1-1":
            aligned.append((segments_ch[pi], segments_en[pj] ))
            i, j = pi, pj
        elif act=="1-2":
            txt = segments_en[pj] + " " + segments_en[pj+1]
            aligned.append((segments_ch[pi], txt))
            i, j = pi, pj
        elif act=="2-1":
            txt = segments_ch[pi] + " " + segments_ch[pi+1]
            aligned.append((txt, segments_en[pj]))
            i, j = pi, pj
        elif act=="1-3":
            txt = " ".join(segments_en[pj:pj+3])
            aligned.append((segments_ch[pi], txt))
            i, j = pi, pj
        elif act=="3-1":
            txt = " ".join(segments_ch[pi:pi+3])
            aligned.append((txt, segments_en[pj]))
            i, j = pi, pj
        elif act=="1-4":
            txt = " ".join(segments_en[pj:pj+4])
            aligned.append((segments_ch[pi], txt))
            i, j = pi, pj
        elif act=="4-1":
            txt = " ".join(segments_ch[pi:pi+4])
            aligned.append((txt, segments_en[pj]))
            i, j = pi, pj
        elif act=="skip_ch":
            i -= 1
        else:
            j -= 1

    aligned.reverse()
    return aligned


def process_all_files(input_dir="."):
    pairs = list_file_pairs(input_dir)
    all_aligned = []
    with open(output_file, 'w', encoding='utf-8') as fout:
        for ch_f, en_f in pairs:
            print(f"Procesando: {ch_f} + {en_f}")
            text_ch, text_en = load_and_clean_text(ch_f, en_f)
            seg_ch, seg_en = segment_text(text_ch, text_en)
            seg_ch = merge_segments_ch(seg_ch)
            seg_en = merge_segments_en(seg_en)
            aligned = align_segments(seg_ch, seg_en)
            for c, e in aligned:
                fout.write(f"{c} ; {e}\n")
            all_aligned.extend(aligned)
    return all_aligned


if __name__ == '__main__':
    output_file = "final_ISSTH.txt"
    skip_penalty = -0.5
    embedding_cache = {}
    word_set = set(words.words())

    start = time.perf_counter()
    resultados = process_all_files(".")
    print(f"Finished in {time.perf_counter()-start:.2f}s with {len(resultados)} segments.")
