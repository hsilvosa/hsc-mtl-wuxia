import os
import argparse

import time
import random
import csv
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from huggingface_hub import login

# 
# Global Settings
# 
PERFORM_HF_LOGIN = True
N_SAMPLES = 1000
SEED = 42
BATCH_SIZE = 64
# PROMPT_IDS = [0, 1, 2, 3, 4, 5]

SHOTS_CONFIG = {
    "0shot": 0,
    "1shot": 1,
    "2shot": 2, 
    "3shot": 3,
    "5shot": 5,
    "10shot":10
}

def parse_args():
    """Parses command-line arguments for prompt selection."""
    parser = argparse.ArgumentParser(description="Run LLM evaluation with specified prompt IDs.")
    parser.add_argument(
        '--prompts',
        nargs='+',  # Accepts one or more arguments
        type=int,
        required=True,
        help='List of prompt IDs (integers) to evaluate.'
    )
    return parser.parse_args()
# 
# Configuration Class
# 
@dataclass
class Config:
    """Configuration for paths and model settings."""
    
    lustre_base: Path = Path(os.environ.get("LUSTRE", "/tmp/lustre_fallback"))
    store_base: Path = Path(os.environ.get("STORE", "/tmp/store_fallback"))
    
    dataset_dir: Path = lustre_base / "dataset" / "wuxia_zh_en_clean"    
    
    translate_dir: Path = store_base / "results" 
    
    translate_file: str = "gemma3.txt"
    results_file: str = "llm_results.txt"
    
    model_ckpt: str = "google/gemma-3-12b-it"
    #"google/gemma-3-27b-it"
    
    
PROMPT_TEMPLATES = [
    {
        "system_msg": (
            "You are a Chinese-to-English translator. "
            "Output only the translated text inside [[[ and ]]]."
        ),
        "user_instruction": (
            "Translate the following Chinese text into English:"
        ),
    },
    {
        "system_msg": (
            "Your role is an assistant transforming Chinese wuxia text into gripping English fantasy prose. "
            "Adapt idioms and cultural references to be easily understood by Western readers while keeping the specific wuxia flavor. "
            "Prioritize narrative pacing and emotional impact over word-for-word accuracy. "
            "Do not output anything other than the translation enclosed in triple brackets."
            "Format: [[[Translation]]]"
        ),
        "user_instruction": (
            "Translate the following Chinese text into immersive, modern English fantasy prose:"
        ),
    },
    {
        "system_msg": (
            "You are an expert literary translator of Chinese wuxia fiction. "
            "Retain the poetic flow, cultural symbolism, and epic atmosphere characteristic of wuxia narratives. "
            "The goal is to produce English reflecting the depth of Chinese idioms. "
            "Strictly enclose the output in [[[ and ]]]."
        ),
        "user_instruction": (
            "Translate the following Chinese passage into refined, literary English that conveys the spirit of classic wuxia storytelling:"
        ),
    },
    {
        "system_msg": (
            "You are functioning as a seq2seq neural translation model trained for Chinese-to-English tasks. "
            "You must produce a one-to-one translation of the source text without commentary, explanation, or rephrasing. "
            "Do not include introductions or stylistic variations, just output the translation. "
            "The translated text must be enclosed between [[[ and ]]] for clear identification."
        ),
        "user_instruction": (
            "Provide a direct English translation of the following Chinese text, maintaining word order and fidelity as much as possible, similar to a seq2seq model:"
        ),
    },
    {
        "system_msg": (
            "You are a bilingual scholar translating Chinese martial arts literature into English. "
            "Your translation must be accurate, faithful, and formal. "
            "Do not include introductions or stylistic variations, just output the translation. "
            "Enclose the entire translation between [[[ and ]]] so it can be clearly extracted."
        ),
        "user_instruction": (
            "Translate the following text into precise English, maintaining original meanings and cultural nuances:"
        ),
    },
    {
        "system_msg": (
            "You are a literal translation engine. "
            "Translate the Chinese text into English maintaining the original sentence structure and word order as closely as possible, even if it results in stiff English. "
            "Treat idioms literally (e.g., translate 'nine deaths one life' literally, do not adapt to 'narrow escape'). "
            "Enclose the result strictly in [[[ and ]]]."
        ),
        "user_instruction": (
            "Provide a literal, structure-preserving translation of the following text:"
        ),
    }
]


# 
# Uitls
# 

def set_deterministic_seed(seed_value):
    """Sets seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def load_model_resources(config: Config):
    """Loads tokenizer and model for Gemma 3."""
    print(f"Loading tokenizer: {config.model_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt, trust_remote_code=True)
    tokenizer.padding_side = "left"
    
    print(f"Loading model: {config.model_ckpt}...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        config.model_ckpt, 
        device_map="auto",       
        dtype=torch.bfloat16, 
        trust_remote_code=True
    )

    print(f"Model loaded on device: {model.device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    return tokenizer, model
def translate_batch_gemma(
    chinese_texts, 
    tokenizer, 
    model, 
    system_msg, 
    user_instruction, 
    shots=None, 
    max_new_tokens=256
):
    """
    Version vectorizada for Gemma 3.
    Processes to list of texts simultaneously.
    """
    prompts_as_strings = []
    
    # 1. Prepare each prompt individually as text
    for text in chinese_texts:
        full_content = f"{system_msg}\n\n{user_instruction}\n"
        
        if shots:
            examples = "\n".join(
                f"Example {i}\nChinese: {ex['zh']}\nEnglish: {ex['en']}\n"
                for i, ex in enumerate(shots, 1)
            )
            full_content += f"\nBelow are example translations to guide your style:\n{examples}\n"
            full_content += "Use these examples for reference, but DO NOT repeat them."

        full_content += f"\nConvert all Chinese personal names... into pinyin...\nChinese: {text}\nEnglish:"

        messages = [{"role": "user", "content": full_content}]
        
        # Apply the template but WITHOUT tokenize yet (tokenize=False)
        # This nos gives the string formatted with the tokens special of Gemma (<start_of_turn>, etc.)
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts_as_strings.append(prompt_str)

    # 2. Tokenize the batch completo (with Padding)
    inputs = tokenizer(
        prompts_as_strings,
        return_tensors="pt",
        padding=True,       # Pads to the longest sequence in the batch
        truncation=True,
        max_length=4096     
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    # 3. Generation in Batch
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,      
        temperature=0.3,     
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 4. Decoding
    generated_tokens = [out[input_length:] for out in output_ids]
    output_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return [t.strip() for t in output_texts]

def build_shots_dataset(dataset, num_shots, split="train"):
    """Extracts 'num_shots' examples from the dataset."""
    if num_shots == 0:
        return []
    total_rows = len(dataset[split])
    return [dataset[split][i % total_rows] for i in range(num_shots)]



# 
# Main Execution
# 
def main():
    
    args = parse_args()

    PROMPT_IDS = args.prompts
    
    if PERFORM_HF_LOGIN:
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

            if hf_token:
                print("Token of Hugging Face found in the environment. Starting session...")
                login(token=hf_token.strip("'\""))
            else:
                login()
        
    cfg = Config()
    
    os.environ["HF_HOME"] = str(Path(os.environ.get("STORE", "/tmp/store_fallback")) / ".huggingface")

    set_deterministic_seed(SEED)

    print("Loading dataset from disk...")
    try:
        raw_ds = load_from_disk(str(cfg.dataset_dir))
        print(raw_ds)
    except Exception as e:
        print(f"Error loading dataset at {cfg.dataset_dir}: {e}")
        return

    # Load Model & Tokenizer
    tokenizer, model = load_model_resources(cfg)

    # Prepare output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}.txt"
    time_log_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}_timing.txt"
    
    print(f"Results will be saved to: {output_file_path}")
    print(f"Timing logs will be saved to: {time_log_path}")

    all_time_records = []
    test_ds_len = len(raw_ds["test"])

    with open(output_file_path, "w", encoding="utf-8") as f_out:
        
        for prompt_idx in PROMPT_IDS:
            current_prompt = PROMPT_TEMPLATES[prompt_idx]
            sys_msg = current_prompt["system_msg"]
            usr_instr = current_prompt["user_instruction"]
            
            # Header log for file
            f_out.write(f"\n\n##############################################\n")
            f_out.write(f"### PROMPT {prompt_idx}\n")
            f_out.write(f"### {sys_msg[:120]}...\n")
            f_out.write(f"##############################################\n")
            
            prompt_total_time = 0.0

            for mode_name, n_shots in SHOTS_CONFIG.items():
                f_out.write(f"\n=== {mode_name.upper()} ({n_shots}-shot) ===\n\n")
                
                mode_times = []
                shot_examples = build_shots_dataset(raw_ds, n_shots, split="train")
                
                # --- BATCHING LOOP ---
                batch_starts = list(range(0, N_SAMPLES, BATCH_SIZE))
                progress_desc = f"Prompt {prompt_idx} -> {mode_name}"
                
                for i in tqdm(batch_starts, desc=progress_desc):
                    # 1. Prepare batch
                    current_batch_indices = range(i, min(i + BATCH_SIZE, N_SAMPLES))
                    batch_samples = [raw_ds["test"][j % test_ds_len] for j in current_batch_indices]
                    batch_zh_texts = [s["zh"] for s in batch_samples]
                    
                    start_ts = time.perf_counter()
                    try:
                        # 2. Inference in Batch
                        translations = translate_batch_gemma(
                            chinese_texts=batch_zh_texts,
                            tokenizer=tokenizer,
                            model=model,
                            system_msg=sys_msg,
                            user_instruction=usr_instr,
                            shots=shot_examples
                        )
                    except Exception as exc:
                        translations = [f"[Error: {exc}]"] * len(batch_zh_texts)
                        print(f"Batch Error: {exc}")
                    
                    batch_elapsed = time.perf_counter() - start_ts
                    avg_per_sample = batch_elapsed / len(batch_zh_texts)
                    
                    # Save times individual
                    mode_times.extend([avg_per_sample] * len(batch_zh_texts))
                    
                    # 3. Writing of results of the batch
                    for idx_in_batch, trans_text in enumerate(translations):
                        global_idx = current_batch_indices[idx_in_batch]
                        original = batch_samples[idx_in_batch]
                        
                        f_out.write(f"--- Case {global_idx+1} ---\n")
                        f_out.write(f"[ZH]: {original['zh']}\n")
                        f_out.write(f"[REF]: {original['en']}\n")
                        f_out.write(f"[GEN]: {trans_text}\n")
                        f_out.write(f"[TIME]: {avg_per_sample:.2f} s (Batch Avg)\n")
                

                # Calculate stats for this mode (Shot Config)
                if mode_times:
                    avg_t = sum(mode_times) / len(mode_times)
                    total_t = sum(mode_times)
                    prompt_total_time += total_t
                    
                    all_time_records.append((prompt_idx, mode_name, n_shots, avg_t, total_t))
                    
                    f_out.write(f"\n>>> Mean Time {mode_name}: {avg_t:.2f} s | Total: {total_t:.1f} s\n")
                    f_out.flush() 

            f_out.write(f"\n##### TOTAL PROMPT {prompt_idx}: {prompt_total_time/60:.2f} min #####\n")

    
    print("\nTIME SUMMARY (Per Prompt)")
    summary_map = {}
    for pid, _, _, _, t_total in all_time_records:
        summary_map.setdefault(pid, 0.0)
        summary_map[pid] += t_total

    with open(time_log_path, "w", encoding="utf-8") as f_time:
        f_time.write("=== TIME SUMMARY (Per Prompt) ===\n")
        for pid, total_sec in summary_map.items():
            line = f"Prompt {pid}: {total_sec/60:.2f} min"
            print(line)
            f_time.write(line + "\n")

    print(f"\nDone. Results: {output_file_path}")
if __name__ == "__main__":
    main()