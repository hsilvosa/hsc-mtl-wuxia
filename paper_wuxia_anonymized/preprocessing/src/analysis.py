import os
import re
import math
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent   
CORPUS_DIR  = PROJECT_DIR / "inputs" / "corpus"         # corpus paralelo ZH-IN
MAIN_STATS  = PROJECT_DIR / "outputs" / "main" / "statistics"
MAIN_GRAPHS = PROJECT_DIR / "outputs" / "main" / "graphs"
HF_STATS    = PROJECT_DIR / "outputs" / "hf" / "statistics"
HF_GRAPHS   = PROJECT_DIR / "outputs" / "hf" / "graphs"

try:
    from pyvis.network import Network
except ImportError:
    print("Error: Instala pyvis (pip install pyvis)")
    exit()

try:
    import jieba
    import jieba.posseg as pseg
except ImportError:
    print("Error: Instala jieba (pip install jieba)")
    exit()

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.util import ngrams
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True) 
except ImportError:
    print("Error: Instala nltk (pip install nltk)")
    exit()

try:
    from lexicalrichness import LexicalRichness
except ImportError:
    print("Error: Instala lexicalrichness (pip install lexicalrichness)")
    exit()

# CONFIGURATION AND PATTERNS WUXIA

STOP_WORDS_EN = set(stopwords.words('english'))
REGEX_FACTIONS = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s(?:Sect|Faction|Clan|School|Valley|Pavilion|Palace|Court|Mountain))\b'
REGEX_TECHNIQUES = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s(?:Art|Skill|Technique|Mantra|Method|Sword|Fist|Palm|Finger|Step|Scripture|Manual|Formation))\b'
REGEX_CHARS = r'\b([A-Z][a-z]+\s[A-Z][a-z]+)\b'
STOP_WORDS_CHARS = {"The", "This", "That", "Some", "Many", "All", "His", "Her", "My", "Your", "Elder", "Patriarch", "Senior", "Junior", "Brother", "Sister", "Outer", "Inner", "Core"}

sns.set_theme(style="whitegrid", palette="muted")

def is_valid_character(name):
    words = name.split()
    if words[0] in STOP_WORDS_CHARS: return False
    if any(suffix in name for suffix in ['Sect', 'Clan', 'Art', 'Skill', 'Valley', 'Technique']): return False
    return True

# FUNCIONES 

def calc_sttr(tokens, chunk_size=1000):
    if len(tokens) < chunk_size: return (len(set(tokens)) / len(tokens)) * 100 if tokens else 0
    ttrs = [len(set(tokens[i : i + chunk_size])) / chunk_size for i in range(0, len(tokens) - chunk_size + 1, chunk_size)]
    return (sum(ttrs) / len(ttrs)) * 100

def calc_lexical_density_en(tokens):
    tagged = nltk.pos_tag(tokens)
    content_words = [w for w, tag in tagged if tag.startswith(('NN', 'VB', 'JJ', 'RB'))]
    return (len(content_words) / len(tokens)) * 100 if tokens else 0

def calc_lexical_density_zh(zh_text):
    words = pseg.cut(zh_text)
    content_tags = ('n', 'v', 'a', 'd') 
    content_count = total_count = 0
    for w, flag in words:
        if w.strip() and not re.match(r'[^\w\u4e00-\u9fff]', w): 
            total_count += 1
            if flag.startswith(content_tags): content_count += 1
    return (content_count / total_count) * 100 if total_count else 0

def compute_metrics_and_print(en_words, zh_words, zh_full_text, title="ANALYSIS STATISTICAL ADVANCED OF THE CORPUS WUXIA"):
    en_tokens_cnt, en_types_cnt = len(en_words), len(set(en_words))
    zh_tokens_cnt, zh_types_cnt = len(zh_words), len(set(zh_words))

    guiraud_en = en_types_cnt / math.sqrt(en_tokens_cnt) if en_tokens_cnt else 0
    guiraud_zh = zh_types_cnt / math.sqrt(zh_tokens_cnt) if zh_tokens_cnt else 0

    sttr_en = calc_sttr(en_words, 1000)
    sttr_zh = calc_sttr(zh_words, 1000)

    lex_den_en = calc_lexical_density_en(en_words[:int(en_tokens_cnt * 0.1)] if en_tokens_cnt > 1000000 else en_words)
    lex_den_zh = calc_lexical_density_zh(zh_full_text[:500000])

    # Hapax Legomena (Words that aparecen exactamente 1 time)
    freq_en = Counter(en_words)
    hapax_en = sum(1 for v in freq_en.values() if v == 1)
    hapax_pct_en = (hapax_en / en_types_cnt) * 100 if en_types_cnt else 0
    
    freq_zh = Counter(zh_words)
    hapax_zh = sum(1 for v in freq_zh.values() if v == 1)
    hapax_pct_zh = (hapax_zh / zh_types_cnt) * 100 if zh_types_cnt else 0

    try:
        lex_en, lex_zh = LexicalRichness(" ".join(en_words)), LexicalRichness(" ".join(zh_words))
        mtld_en, mtld_zh = lex_en.mtld(threshold=0.72), lex_zh.mtld(threshold=0.72)
        hdd_en, hdd_zh = lex_en.hdd(draws=42), lex_zh.hdd(draws=42)
    except:
        mtld_en = mtld_zh = hdd_en = hdd_zh = 0.0

    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)
    print("\n---  ENGLISH (Text Meta) ---")
    print(f"Total Tokens:           {en_tokens_cnt:,}")
    print(f"Total Types:            {en_types_cnt:,}")
    print(f"TTR Standard:            {(en_types_cnt/en_tokens_cnt)*100:.2f}%")
    print(f"STTR (Blocks of 1k):   {sttr_en:.2f}%")
    print(f"Index Guiraud (R):     {guiraud_en:.2f}")
    print(f"MTLD:                   {mtld_en:.2f}")
    print(f"VOCD-D (HD-D):          {hdd_en:.2f}")
    print(f"Density Lexical:        {lex_den_en:.2f}%")
    print(f"Hapax Legomena:         {hapax_en:,} ({hapax_pct_en:.2f}% of the vocabulario)")
    
    print("\n---  CHINESE (Text Origen) ---")
    print(f"Total Tokens:           {zh_tokens_cnt:,}")
    print(f"Total Types:            {zh_types_cnt:,}")
    print(f"TTR Standard:            {(zh_types_cnt/zh_tokens_cnt)*100:.2f}%")
    print(f"STTR (Blocks of 1k):   {sttr_zh:.2f}%")
    print(f"Index Guiraud (R):     {guiraud_zh:.2f}")
    print(f"MTLD:                   {mtld_zh:.2f}")
    print(f"VOCD-D (HD-D):          {hdd_zh:.2f}")
    print(f"Density Lexical:        {lex_den_zh:.2f}%")
    print(f"Hapax Legomena:         {hapax_zh:,} ({hapax_pct_zh:.2f}% of the vocabulario)")
    print("\n" + "="*60 + "\n")

    return {'sttr_en': sttr_en, 'sttr_zh': sttr_zh, 'den_en': lex_den_en, 'den_zh': lex_den_zh}

# FUNCIONES OF CHARTS (PYVIS AND STATISTICS)
def generate_statistical_charts(metrics_data, entities_data, lens_data, freq_data, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    # 1. Riqueza y Densidad
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['STTR', 'Density Lexical']
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, [metrics_data['sttr_en'], metrics_data['den_en']], width, label='English', color='#4c72b0')
    ax.bar(x + width/2, [metrics_data['sttr_zh'], metrics_data['den_zh']], width, label='Chinese', color='#c44e52')
    ax.set_ylabel('Porcentaje (%)')
    ax.set_title('Comparison Lexical')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(os.path.join(out_folder, '1_riqueza_densidad.png'), dpi=300); plt.close()

    # 2. Histograma KDE (Longitud)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(lens_data['en_lens'], fill=True, label='English (Words/sentence)', color='#4c72b0')
    sns.kdeplot(lens_data['zh_lens'], fill=True, label='Chinese (Tokens/sentence)', color='#c44e52')
    plt.xlim(0, max(np.percentile(lens_data['en_lens'], 99), np.percentile(lens_data['zh_lens'], 99)))
    plt.title('Distribution of Length of Sentences')
    plt.legend()
    plt.savefig(os.path.join(out_folder, '2_distribucion_longitud.png'), dpi=300); plt.close()

    # 3. Ley of Zipf
    en_freqs = sorted(list(Counter(freq_data['en_words']).values()), reverse=True)[:10000]
    zh_freqs = sorted(list(Counter(freq_data['zh_words']).values()), reverse=True)[:10000]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(en_freqs)+1), en_freqs, label='English', color='#4c72b0')
    plt.plot(range(1, len(zh_freqs)+1), zh_freqs, label='Chinese', color='#c44e52')
    plt.xscale('log'); plt.yscale('log')
    plt.title('Ley of Zipf')
    plt.legend()
    plt.savefig(os.path.join(out_folder, '3_ley_de_zipf.png'), dpi=300); plt.close()

    # 4. Top 15 Factions
    top15 = Counter([x for sub in entities_data['factions'] for x in sub]).most_common(15)
    if top15:
        names, counts = [item[0] for item in top15][::-1], [item[1] for item in top15][::-1]
        plt.figure(figsize=(10, 7))
        plt.barh(names, counts, color='#55a868')
        plt.title('Top 15 Factions More Mentioned')
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, '4_top15_facciones.png'), dpi=300); plt.close()

    # 5. Top 15 Trigramas (English)
    # Extract n-gramas of size 3 and the contamos
    trigramas_en = list(ngrams(freq_data['en_words'], 3))
    top15_trigramas = Counter(trigramas_en).most_common(15)
    if top15_trigramas:
        names = [" ".join(tg[0]) for tg in top15_trigramas][::-1]
        counts = [tg[1] for tg in top15_trigramas][::-1]
        plt.figure(figsize=(10, 7))
        plt.barh(names, counts, color='#8c564b')
        plt.title('Top 15 Trigramas More Frecuentes (English)')
        plt.xlabel('Frequency of Occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, '5_top15_trigramas.png'), dpi=300); plt.close()

def create_interactive_graph(entities_per_line, filepath, node_color, custom_targets=None):
    if custom_targets is None:
        all_ents = [ent for sub in entities_per_line for ent in sub]
        targets = {ent: count for ent, count in Counter(all_ents).most_common(50)}
    else:
        targets = custom_targets
        
    if len(targets) < 2: return

    edges = []
    for ents in entities_per_line:
        filt = [e for e in ents if e in targets]
        if len(filt) > 1:
            for i in range(len(filt)):
                for j in range(i + 1, len(filt)):
                    edges.append(tuple(sorted([filt[i], filt[j]])))

    net = Network(height='800px', width='100%', bgcolor='#1a1a1a', font_color='white', cdn_resources='remote')
    max_c = max(targets.values()) if targets.values() else 1
    for ent, c in targets.items():
        net.add_node(ent, label=ent, title=f"Frequency: {c}", size=10 + (c / max_c)*40, color=node_color)
    for (n1, n2), w in Counter(edges).items():
        if n1 in targets and n2 in targets:
            net.add_edge(n1, n2, value=w, title=f"Co-ocurrencias: {w}", color='#888888')

    net.repulsion(node_distance=150)
    net.save_graph(filepath)


# LOGIC MAIN OF THE CORPUS
def analyze_local_corpus():
    
    print(" STARTING STAGE 1: CORPUS LOCAL MAIN")
    
    
    # 1. Analysis Global Statistical
    en_words, zh_words, zh_full_text = [], [], ""
    lens_data = {'en_lens': [], 'zh_lens': []}
    
    with open(CORPUS_DIR / 'dataset.txt', 'r', encoding='utf-8-sig') as f:
        for line in f:
            if ';' not in line: continue
            parts = line.strip().split(';', 1)
            zh_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', parts[0].strip())
            zh_full_text += zh_clean + " "
            zh_tokens = list(jieba.cut(zh_clean))
            en_tokens = re.findall(r'\b[a-zA-Z]+\b', parts[1].strip().lower())
            
            zh_words.extend(zh_tokens); en_words.extend(en_tokens)
            lens_data['zh_lens'].append(len(zh_tokens)); lens_data['en_lens'].append(len(en_tokens))

    metrics = compute_metrics_and_print(en_words, zh_words, zh_full_text, "CORPUS PRINCIPAL (dataset.txt)")

    # 2. Extraction of Entidades by Libro
    libros = {"AWE": ['final_awe.txt'], "CONDOR": ['final_condor_1.txt', 'final_condor_2.txt', 'final_condor_3.txt'], "GU": ['final_gu.txt']}
    book_data = {}
    glob_ents = {'factions': [], 'techniques': [], 'chars': []}

    for b_name, files in libros.items():
        fac, tech, char = [], [], []
        for fp in files:
            fp = CORPUS_DIR / fp
            if not os.path.exists(fp): continue
            with open(fp, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    if ';' not in line: continue
                    en_t = line.split(';', 1)[1].strip()
                    f_m = re.findall(REGEX_FACTIONS, en_t); t_m = re.findall(REGEX_TECHNIQUES, en_t)
                    c_m = [c for c in re.findall(REGEX_CHARS, en_t) if is_valid_character(c)]
                    if f_m: fac.append(f_m)
                    if t_m: tech.append(t_m)
                    if c_m: char.append(c_m)
        
        book_data[b_name] = {'factions': fac, 'techniques': tech, 'chars': char}
        glob_ents['factions'].extend(fac); glob_ents['techniques'].extend(tech); glob_ents['chars'].extend(char)
        
        o_dir = os.path.join(MAIN_GRAPHS, b_name)
        os.makedirs(o_dir, exist_ok=True)
        create_interactive_graph(fac, os.path.join(o_dir, f"grafo_facciones_{b_name.lower()}.html"), "#4287f5")
        create_interactive_graph(tech, os.path.join(o_dir, f"grafo_tecnicas_{b_name.lower()}.html"), "#f54242")
        create_interactive_graph(char, os.path.join(o_dir, f"grafo_personajes_{b_name.lower()}.html"), "#42f58d")

    # 3. Grafos and Global statistics
    generate_statistical_charts(metrics, glob_ents, lens_data, {'en_words': en_words, 'zh_words': zh_words}, MAIN_STATS)
    
    t_fac, t_tech, t_char = set(), set(), set()
    for d in book_data.values():
        t_fac.update([e for e, c in Counter([x for s in d['factions'] for x in s]).most_common(30)])
        t_tech.update([e for e, c in Counter([x for s in d['techniques'] for x in s]).most_common(30)])
        t_char.update([e for e, c in Counter([x for s in d['chars'] for x in s]).most_common(30)])
    
    o_dir = os.path.join(MAIN_GRAPHS, "GLOBAL")
    os.makedirs(o_dir, exist_ok=True)
    create_interactive_graph(glob_ents['factions'], os.path.join(o_dir, "grafo_facciones_global.html"), "#4287f5", dict(Counter([x for s in glob_ents['factions'] for x in s if x in t_fac])))
    create_interactive_graph(glob_ents['techniques'], os.path.join(o_dir, "grafo_tecnicas_global.html"), "#f54242", dict(Counter([x for s in glob_ents['techniques'] for x in s if x in t_tech])))
    create_interactive_graph(glob_ents['chars'], os.path.join(o_dir, "grafo_personajes_global.html"), "#42f58d", dict(Counter([x for s in glob_ents['chars'] for x in s if x in t_char])))


# LOGIC OF THE DATASET HUGGING FACE 
#
def analyze_hf_dataset(hf_path):
    
    print(" STARTING STAGE 2: DATASET HUGGING FACE (GLOBAL)")
    
    
    try:
        from datasets import load_from_disk, DatasetDict
        ds = load_from_disk(hf_path)
    except Exception as e:
        print(f"Error loading the Hugging Face dataset from {hf_path}.\nMake sure the datasets library is installed and check the path.")
        return

    en_words, zh_words, zh_full_text = [], [], ""
    lens_data = {'en_lens': [], 'zh_lens': []}
    glob_ents = {'factions': [], 'techniques': [], 'chars': []}

    if isinstance(ds, dict) or hasattr(ds, 'keys'):
        splits = list(ds.keys())
    else:
        ds = {'default': ds}
        splits = ['default']

    print(f"Particiones detectadas: {splits}. Processing and unificando data...")

    for split in splits:
        print(f" -> Analizando split: '{split}'...")
        for row in ds[split]:
            en_t, zh_t = "", ""
            if 'translation' in row and isinstance(row['translation'], dict):
                en_t = row['translation'].get('en', '')
                zh_t = row['translation'].get('zh', '')
            elif 'en' in row and 'zh' in row:
                en_t = row['en']
                zh_t = row['zh']
            else:
                try:
                    en_t = str(list(row.values())[1])
                    zh_t = str(list(row.values())[0])
                except:
                    continue

            if not en_t or not zh_t: continue

            zh_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', zh_t)
            zh_full_text += zh_clean + " "
            zh_tokens = list(jieba.cut(zh_clean))
            en_tokens = re.findall(r'\b[a-zA-Z]+\b', en_t.lower())
            
            zh_words.extend(zh_tokens); en_words.extend(en_tokens)
            lens_data['zh_lens'].append(len(zh_tokens)); lens_data['en_lens'].append(len(en_tokens))

            f_m = re.findall(REGEX_FACTIONS, en_t); t_m = re.findall(REGEX_TECHNIQUES, en_t)
            c_m = [c for c in re.findall(REGEX_CHARS, en_t) if is_valid_character(c)]
            if f_m: glob_ents['factions'].append(f_m)
            if t_m: glob_ents['techniques'].append(t_m)
            if c_m: glob_ents['chars'].append(c_m)

    metrics = compute_metrics_and_print(en_words, zh_words, zh_full_text, f"DATASET HUGGING FACE (Unificado: {', '.join(splits)})")

    print("Generating Charts and Statistics of HF...")
    generate_statistical_charts(metrics, glob_ents, lens_data, {'en_words': en_words, 'zh_words': zh_words}, HF_STATS)

    graf_dir = HF_GRAPHS
    os.makedirs(graf_dir, exist_ok=True)

    create_interactive_graph(glob_ents['factions'], os.path.join(graf_dir, "grafo_facciones_hf.html"), "#4287f5")
    create_interactive_graph(glob_ents['techniques'], os.path.join(graf_dir, "grafo_tecnicas_hf.html"), "#f54242")
    create_interactive_graph(glob_ents['chars'], os.path.join(graf_dir, "grafo_personajes_hf.html"), "#42f58d")


if __name__ == "__main__":
    # FASE 1 
    analyze_local_corpus()
    
    # FASE 2
    HF_PATH = PROJECT_DIR.parent / "processed_data" / "wuxia_selected_100000"
    analyze_hf_dataset(HF_PATH)
    
    print("\n All processes completed successfully!")
