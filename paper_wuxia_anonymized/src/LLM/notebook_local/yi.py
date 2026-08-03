
import os
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# 
# Global Settings
# 
PERFORM_HF_LOGIN = False
N_SAMPLES = 1
SEED = 42
PROMPT_IDS_TO_EVALUATE = [1]#, 1, 2, 3, 4, 5]

SHOTS_CONFIG = {
    "0shot": 0,
    "1shot": 1,
    "2shot": 2,
    "3shot": 3,
}

@dataclass
class Config:
    """Configuration for paths and model settings."""

    base_dir: Path = Path.cwd().parent
    
    dataset_dir: Path = base_dir / "processed_data" / "wuxia_zh_en_clean"
    output_dir: Path = base_dir / "models" / "yi"
    
    evaluation_dir: Path = base_dir / "evaluation"
    translate_dir: Path = base_dir / "evaluation" / "translate" / "llm"
    
    translate_file: str = "yi.txt"
    results_file: str = "llm_results.txt"
    
    # Model
    model_ckpt: str = "zai-org/GLM-4-9B-0414" #"01-ai/Yi-1.5-6B"#


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
# Utils
# 
def set_deterministic_seed(seed_value):
    """Sets seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def load_model_resources(config: Config):
    """Loads tokenizer and model for Yi-1.5."""
    print(f"Loading tokenizer: {config.model_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt, trust_remote_code=True)



    model = AutoModelForCausalLM.from_pretrained(
        config.model_ckpt, 
        device_map="auto",
        dtype=torch.bfloat16, 
        trust_remote_code=True
    )

    print(f"Model loaded on device: {model.device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    return tokenizer, model

def translate_with_prompt_yi(chinese_text, tokenizer, model, system_msg, user_instruction, shots=None, max_new_tokens=256):
    """
    Logic of translation optimized for Yi-1.5-Chat.
    Uses format instructional direct for avoid repetitions of the input.
    """
    
 # 1. Prepare Examples (Few-Shot)
    examples_str = ""
    if shots:
        examples_str = "\n\n=== EXAMPLES ===\n"
        for i, ex in enumerate(shots, 1):
            examples_str += f"Input: {ex['zh']}\nOutput: {ex['en']}\n"
        examples_str += "=== END EXAMPLES ===\n"

    # 2. Build un ONLY message of user
    full_prompt = (
        f"{system_msg}\n"
        f"INSTRUCTION: {user_instruction}\n"
        f"RULE: Do not repeat the Chinese text. Output ONLY the English translation.\n"
        f"{examples_str}\n"
        f"=== TRANSLATE THIS ===\n"
        f"Input: {chinese_text}\n"
        f"English Output:" 
    )

    # 3. Structure Chat (All in User)
    messages = [
        {"role": "user", "content": full_prompt}
    ]

    # 4. Tokenization
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True, 
        return_dict=True
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    # 5. Generation with Parameters Anti-Repetition
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,       
            top_p=0.9,
            repetition_penalty=1.3, 
            pad_token_id=tokenizer.eos_token_id
        )

    # 6. Decoding
    generated_tokens = output_ids[0][input_length:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    cleaned_text = output_text.replace("English Output:", "").strip()
    
    return cleaned_text

def build_shots_dataset(dataset, num_shots, split="train"):
    """Extracts 'num_shots' examples from the dataset."""
    if num_shots == 0:
        return []
    total_rows = len(dataset[split])
    return [dataset[split][i % total_rows] for i in range(num_shots)]


# 
# Main
# 

def main():
    if PERFORM_HF_LOGIN:
        login()
        
    cfg = Config()
    

    
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
    time_log_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}_timing.csv"
    
    print(f"Results will be saved to: {output_file_path}")
    print(f"Timing logs will be saved to: {time_log_path}")

    all_time_records = []
    
    with open(output_file_path, "w", encoding="utf-8") as f_out:
        
        for prompt_idx in PROMPT_IDS_TO_EVALUATE:
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
                
                progress_desc = f"Prompt {prompt_idx} -> {mode_name}"
                
                for i in tqdm(range(N_SAMPLES), desc=progress_desc):
                    sample_data = raw_ds["test"][i % len(raw_ds["test"])]
                    
                    start_ts = time.perf_counter()
                    try:
                        translation_result = translate_with_prompt_yi(
                            chinese_text=sample_data["zh"],
                            tokenizer=tokenizer,
                            model=model,
                            system_msg=sys_msg,
                            user_instruction=usr_instr,
                            shots=shot_examples
                        )
                    except Exception as exc:
                        translation_result = f"[Error in iteration {i}: {exc}]"
                    
                    elapsed = time.perf_counter() - start_ts
                    mode_times.append(elapsed)
                    
                    # Write immediately to file (safe logging)
                    f_out.write(f"--- Case {i+1} ---\n")
                    f_out.write(f"[ZH]: {sample_data['zh']}\n")
                    f_out.write(f"[REF]: {sample_data['en']}\n")
                    f_out.write(f"[GEN]: {translation_result.strip()}\n")
                    f_out.write(f"[TIME]: {elapsed:.2f} s\n")
                
                # Stats calculation
                avg_time = sum(mode_times) / len(mode_times) if mode_times else 0
                total_time = sum(mode_times)
                prompt_total_time += total_time
                
                all_time_records.append((prompt_idx, mode_name, n_shots, avg_time, total_time))
                
                f_out.write(f"\n>>> Mean Time {mode_name}: {avg_time:.2f} s | Total: {total_time:.1f} s\n")
                f_out.flush() 

                f_out.write(f"\n##### TOTAL PROMPT {prompt_idx}: {prompt_total_time/60:.2f} min #####\n")
            
               # Calculate stats for this mode
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