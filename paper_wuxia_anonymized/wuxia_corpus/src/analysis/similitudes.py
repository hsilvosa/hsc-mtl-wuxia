import os
from sentence_transformers import SentenceTransformer, util

INPUT_FILE_1 = "gu10.txt"    # The format: Chinese ; English
INPUT_FILE_2 = "gu_ch10.txt"  # The format of 3 lines
OUTPUT_FILE = "similarity_results100.txt"  
FILTER_THRESHOLD = 0.15                  

print("Loading model LaBSE...")
model = SentenceTransformer('sentence-transformers/LaBSE')

def clean_text_for_csv(text):
    """Cleans the text for preserve the format CSV with ;"""
    if not text:
        return ""
    text = text.replace('\n', ' ').replace('\r', '')
    text = text.replace(';', ',') 
    return text.strip()

def calculate_similarity(text1, text2):
    """Calculates the similarity cosine."""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings1, embeddings2).item()

def print_stats(total_score, total_count, filtered_score, filtered_count, filename):
    """Function auxiliar for print the statistics in console"""
    print(f"\n--- Statistics for: {filename} ---")
    
    # 1. Mean Global (All the content)
    if total_count > 0:
        avg_total = total_score / total_count
        print(f"   [GLOBAL]")
        print(f"   > Total sentences processed: {total_count}")
        print(f"   > Mean of similarity:      {avg_total:.4f}")
    else:
        print("   > No there are sentences for process.")

    # 2. Mean Filtered (Only it useful)
    if filtered_count > 0:
        avg_filtered = filtered_score / filtered_count
        print(f"   [FILTERED (Similarity >= {FILTER_THRESHOLD})]")
        print(f"   > Sentences that pasan the filter: {filtered_count} (de {total_count})")
        print(f"   > Mean filtered:             {avg_filtered:.4f}")
    else:
        print(f"   [FILTERED] Not sentence exceeded the threshold of {FILTER_THRESHOLD}")

def process_files():
    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig') as out_f:
        out_f.write("file;Chinese;English;similarity\n")

        # FILE 1
        if os.path.exists(INPUT_FILE_1):
            print(f"Procesando {INPUT_FILE_1}...")
            with open(INPUT_FILE_1, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Variables for mean GLOBAL
            total_score = 0.0
            total_count = 0
            
            # Variables for mean FILTERED
            filtered_score = 0.0
            filtered_count = 0

            for line in lines:
                line = line.strip()
                if not line: continue

                parts = line.split(';')
                if len(parts) >= 2:
                    raw_zh = parts[0]
                    raw_en = ";".join(parts[1:])
                    
                    score = calculate_similarity(raw_zh, raw_en)
                    
                    # Update Global
                    total_score += score
                    total_count += 1
                    
                    # Update Filtrados (If exceeds the threshold)
                    if score >= FILTER_THRESHOLD:
                        filtered_score += score
                        filtered_count += 1
                    
                    out_zh = clean_text_for_csv(raw_zh)
                    out_en = clean_text_for_csv(raw_en)
                    out_f.write(f"{INPUT_FILE_1};{out_zh};{out_en};{score:.5f}\n")
            
            # Print report
            print_stats(total_score, total_count, filtered_score, filtered_count, INPUT_FILE_1)

        else:
            print(f"Warning: Was not found {INPUT_FILE_1}")

        # PROCESSING FILE 2

        if os.path.exists(INPUT_FILE_2):
            print(f"Procesando {INPUT_FILE_2}...")
            with open(INPUT_FILE_2, 'r', encoding='utf-8') as f:
                raw_lines = f.read().splitlines()

            total_score = 0.0
            total_count = 0
            filtered_score = 0.0
            filtered_count = 0

            for i in range(0, len(raw_lines), 3):
                if i + 1 >= len(raw_lines):
                    break
                
                raw_zh = raw_lines[i]
                raw_en = raw_lines[i+1]

                score = calculate_similarity(raw_zh, raw_en)
                
                # Update Global
                total_score += score
                total_count += 1
                
                # Update Filtrados
                if score >= FILTER_THRESHOLD:
                    filtered_score += score
                    filtered_count += 1
                
                out_zh = clean_text_for_csv(raw_zh)
                out_en = clean_text_for_csv(raw_en)
                out_f.write(f"{INPUT_FILE_2};{out_zh};{out_en};{score:.5f}\n")

            print_stats(total_score, total_count, filtered_score, filtered_count, INPUT_FILE_2)

        else:
            print(f"Warning: Was not found {INPUT_FILE_2}")

    print(f"\n--- Finished. Details saved in: {OUTPUT_FILE} ---")

if __name__ == "__main__":


    process_files()
