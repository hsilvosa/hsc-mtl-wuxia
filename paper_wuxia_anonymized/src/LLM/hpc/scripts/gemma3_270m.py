import os
import time
import random
import re
import argparse
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# 
# Global Settings
# 
PERFORM_HF_LOGIN = True
N_SAMPLES = 1000
BATCH_SIZE = 32
SEED = 42

SHOTS_CONFIG = {
    "0shot": 0,
    "1shot": 1,
    "2shot": 2, 
    "3shot": 3,
    "5shot": 5,
    "10shot": 10
}

@dataclass
class Config:
    
    lustre_base: Path = Path(os.environ.get("LUSTRE", "/tmp/lustre_fallback"))
    store_base: Path = Path(os.environ.get("STORE", "/tmp/store_fallback"))
    
    dataset_dir: Path = lustre_base / "dataset" / "wuxia_zh_en_clean"    
    
    translate_dir: Path = store_base / "results" 
    # base_dir: Path = Path.cwd().parent
    # dataset_dir: Path = base_dir / "processed_data" / "wuxia_zh_en_clean"
    # # Output in directory gemma
    # output_dir: Path = base_dir / "models" / "gemma"
    # translate_dir: Path = base_dir / "evaluation" / "translate" / "llm"
    
    translate_file: str = "gemma3_270m"
    
    # Checkpoint of Gemma
    model_ckpt: str = "google/gemma-3-270m-it" 

PROMPT_TEMPLATES = [
    {
        "system_msg": (
            "You are a professional Chinese-to-English translator. "
            "Output ONLY the translation inside triple brackets like this: [[[Translation]]]."
            "Do not output conversational text."
        ),
        "user_instruction": "Translate the following Chinese text into English:",
    },
    {
        "system_msg": (
            "Your role is an assistant transforming Chinese wuxia text into gripping English fantasy prose. "
            "Adapt idioms and cultural references. "
            "Format: [[[Translation]]]"
        ),
        "user_instruction": "Translate the following Chinese text into immersive, modern English fantasy prose:",
    },
    {
        "system_msg": (
            "You are an expert literary translator of Chinese wuxia fiction. "
            "Retain the poetic flow and epic atmosphere. "
            "Strictly enclose the output in [[[ and ]]]."
        ),
        "user_instruction": "Translate the following Chinese passage into refined, literary English that conveys the spirit of classic wuxia storytelling:",
    },
    {
        "system_msg": (
            "You are a seq2seq neural translation model. "
            "Produce a one-to-one translation without explanation. "
            "Enclose text between [[[ and ]]]."
        ),
        "user_instruction": "Provide a direct English translation maintaining word order and fidelity:",
    },
    {
        "system_msg": (
            "You are a bilingual scholar. "
            "Translation must be accurate, faithful, and formal. "
            "Enclose the translation between [[[ and ]]]."
        ),
        "user_instruction": "Translate the following text into precise English, maintaining original meanings:",
    },
    {
        "system_msg": (
            "You are a literal translation engine. "
            "Translate maintaining sentence structure even if stiff. "
            "Treat idioms literally. "
            "Enclose result in [[[ and ]]]."
        ),
        "user_instruction": "Provide a literal, structure-preserving translation:",
    }
]

# 
# Utils
# 
def parse_args():
    """Parses command-line arguments for prompt selection."""
    parser = argparse.ArgumentParser(description="Run LLM evaluation with specified prompt IDs.")
    parser.add_argument(
        '--prompts',
        nargs='+',  
        type=int,
        required=True,
        help='List of prompt IDs (integers) to evaluate.'
    )
    return parser.parse_args()

def set_deterministic_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def load_model_resources(config: Config):
    """Loads specific for Gemma 3."""
    print(f"Loading tokenizer: {config.model_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt, trust_remote_code=True)
    
    tokenizer.padding_side = "left" 
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {config.model_ckpt}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_ckpt, 
        device_map="auto",
        dtype=torch.bfloat16, 
        trust_remote_code=True 
    )
    return tokenizer, model

def extract_translation(text):
    """Cleans the text generated using Regex."""
    match = re.search(r"\[\[\[(.*?)\]\]\]", text, re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r"\[\[(.*?)\]\]", text, re.DOTALL)
    if match: return match.group(1).strip()
    # Cleaning of fallback if falla the regex
    text = text.replace("The output is:", "").replace("Translation:", "").strip()
    return text

def translate_batch_gemma(chinese_texts, tokenizer, model, system_msg, user_instruction, shots=None, max_new_tokens=512):
    """
    Function of inference by batches specific for Gemma.
    """
    # 1. Prepare Few-Shot String (common for all the batch)
    examples_str = ""
    if shots:
        examples_str = "\n\n=== EXAMPLES ===\n"
        for ex in shots:
            examples_str += f"Input: {ex['zh']}\nOutput: [[[{ex['en']}]]]\n"
        examples_str += "=== END EXAMPLES ===\n"

    # 2. Build Prompts
    batch_prompts_content = []
    for txt in chinese_texts:
        content = (
            f"{system_msg}\n\n"
            f"INSTRUCTION: {user_instruction}\n"
            f"{examples_str}\n"
            f"=== TRANSLATE THIS ===\n"
            f"Input: {txt}\n"
            f"Output:"
        )
        batch_prompts_content.append(content)

    # 3. Apply Chat Template (without tokenize yet)
    formatted_prompts = []
    for content in batch_prompts_content:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted)

    # 4. Tokenize Batch
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    # 5. Generation
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,       # Baja temperatura for consistencia
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    # 6. Decode
    generated_texts = tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)
    
    # 7. Clean
    cleaned_results = [extract_translation(txt) for txt in generated_texts]
    
    return cleaned_results

def build_shots_dataset(dataset, num_shots, split="train"):
    if num_shots == 0: return []
    total_rows = len(dataset[split])
    return [dataset[split][i % total_rows] for i in range(num_shots)]

# 
# Main
# 
def main():
    args = parse_args()
    PROMPT_IDS = args.prompts
    
    if PERFORM_HF_LOGIN:
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token: 
            print("Logging in with token from env...")
            login(token=hf_token.strip("'\""))
        else: 
            print("Logging in interactively...")
            login()
        
    cfg = Config()
    # Configure HF HOME
    os.environ["HF_HOME"] = str(Path(os.environ.get("STORE", "/tmp/store_fallback")) / ".huggingface")
    cfg.translate_dir.mkdir(parents=True, exist_ok=True)
    
    set_deterministic_seed(SEED)

    print("Loading dataset from disk...")
    try:
        raw_ds = load_from_disk(str(cfg.dataset_dir))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy dataset for testing...")
        from datasets import Dataset, DatasetDict
        dummy_data = [{"zh": f"你好 {i}", "en": f"Hello {i}"} for i in range(100)]
        raw_ds = DatasetDict({"train": Dataset.from_list(dummy_data), "test": Dataset.from_list(dummy_data)})

    tokenizer, model = load_model_resources(cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}.txt"
    time_log_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}_time.txt" 
    
    print(f"Results: {output_file_path}")
    print(f"Batch Size: {BATCH_SIZE}")

    all_time_records = []
    test_ds_len = len(raw_ds["test"])
    
    with open(output_file_path, "w", encoding="utf-8") as f_out:
        
        for prompt_idx in PROMPT_IDS:
            if prompt_idx < 0 or prompt_idx >= len(PROMPT_TEMPLATES):
                print(f"Skipping invalid prompt ID: {prompt_idx}")
                continue

            current_prompt = PROMPT_TEMPLATES[prompt_idx]
            sys_msg = current_prompt["system_msg"]
            usr_instr = current_prompt["user_instruction"]
            
            f_out.write(f"\n\n##############################################\n")
            f_out.write(f"### PROMPT {prompt_idx}\n")
            f_out.write(f"### {sys_msg[:100]}...\n")
            f_out.write(f"##############################################\n")
            
            prompt_total_time = 0.0

            for mode_name, n_shots in SHOTS_CONFIG.items():
                f_out.write(f"\n=== {mode_name.upper()} ({n_shots}-shot) ===\n\n")
                
                mode_times = []
                shot_examples = build_shots_dataset(raw_ds, n_shots, split="train")
                
                batch_starts = list(range(0, N_SAMPLES, BATCH_SIZE))
                progress_desc = f"Prompt {prompt_idx} -> {mode_name}"
                
                for i in tqdm(batch_starts, desc=progress_desc):
                    # 1. Prepare batch (with cycle if N_SAMPLES > dataset)
                    current_batch_indices = range(i, min(i + BATCH_SIZE, N_SAMPLES))
                    batch_samples = [raw_ds["test"][j % test_ds_len] for j in current_batch_indices]
                    batch_zh_texts = [s["zh"] for s in batch_samples]
                    
                    start_ts = time.perf_counter()
                    try:
                        # 2. Inference in Batch (Using the logic of Gemma)
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
                        print(f"Batch error: {exc}")
                    
                    elapsed = time.perf_counter() - start_ts
                    avg_per_sample = elapsed / len(batch_zh_texts)
                    
                    # Save times individual (average of the batch)
                    mode_times.extend([avg_per_sample] * len(batch_zh_texts))
                    
                    # 3. Escribir results
                    for idx_in_batch, trans in enumerate(translations):
                        global_idx = current_batch_indices[idx_in_batch]
                        original = batch_samples[idx_in_batch]
                        
                        f_out.write(f"--- Case {global_idx+1} ---\n")
                        f_out.write(f"[ZH]: {original['zh']}\n")
                        f_out.write(f"[REF]: {original['en']}\n")
                        f_out.write(f"[GEN]: {trans}\n")
                        f_out.write(f"[TIME]: {avg_per_sample:.2f} s (Batch Avg)\n")
                
                # Summary of the mode
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