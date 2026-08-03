#!/usr/bin/env python

import argparse
import os
import time
import random
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig, 
    StoppingCriteria, 
    StoppingCriteriaList,
    BitsAndBytesConfig
)
from huggingface_hub import login
import re

PERFORM_HF_LOGIN = False 

N_SAMPLES = 10
SEED = 42
# PROMPT_IDS_TO_EVALUATE = [0] #, 1, 2, 3, 4]

SHOTS_CONFIG = {
    "0shot": 0 }
#     "1shot": 1,
#     "3shot": 3, 
#     "5shot": 5,
# }

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


@dataclass
class Config:
    """Configuration for paths and model settings."""

    base_dir: Path = Path.cwd().parent
    
    dataset_dir: Path = base_dir / "processed_data" / "wuxia_zh_en_clean"
    output_dir: Path = base_dir / "models" / "qwen3"
    evaluation_dir: Path = base_dir / "evaluation"
    translate_dir: Path = base_dir / "evaluation" / "translate" / "llm"
    
    translate_file: str = "deepseek.txt"
    results_file: str = "llm_results.txt"
    
    model_ckpt: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    def create_structure(self):
        """Ensures necessary directory structure exists."""
        self.base_dir.mkdir(exist_ok=True)
        for subdir in [self.output_dir, self.translate_dir, self.dataset_dir.parent]:
            subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"Base Directory: {self.base_dir.resolve()}")
        print(f"Translation Output Directory: {self.translate_dir.resolve()}")


PROMPT_TEMPLATES = [
    {
        "system_msg": (
            "Your role will be that of an assistant that translates Chinese wuxia text into fluent English prose."
            "Your translations should read as if they were written by a native English novelist while preserving the atmosphere and meaning of ancient martial worlds."
            "Avoid literal phrasing and focus on emotional tone and narrative flow. "
            "Always enclose your final English translation between triple square brackets [[[ and ]]] so it can be easily identified."
        ),
        "user_instruction": (
            "Translate the following Chinese text into natural English, making it smooth and stylistically consistent with modern fantasy novels:"
        ),
    },
    {
        "system_msg": (
            "You are an expert literary translator of Chinese wuxia fiction. "
            "Retain the poetic flow, cultural symbolism, and epic atmosphere characteristic of wuxia narratives. "
            "The goal is to produce English reflecting the depth of Chinese idioms. "
            "Surround the complete translated text with [[[ and ]]] to clearly mark the translation."
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
            "You are simulating a traditional sequence-to-sequence translation model working on chinese to english translations. "
            "Translate the input text directly into English without adding explanations, literary style, or rewording. "
            "Produce a literal, token-aligned translation suitable for machine translation evaluation. "
            "The output must be wrapped between [[[ and ]]] so it can be parsed automatically."
        ),
        "user_instruction": (
            "Translate the following Chinese text literally into English, preserving structure and meaning with minimal adaptation:"
        ),
    }
]


class StopOnSubsequence(StoppingCriteria):
    """Custom stopping criteria for Qwen generation."""
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = torch.tensor(stop_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        length = self.stop_ids.numel()
        
        stop_ids_on_device = self.stop_ids.to(seq.device)
        if seq.numel() >= length and torch.equal(seq[-length:], stop_ids_on_device):
            return True
        return False

def set_deterministic_seed(seed_value):
    """Sets seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def load_model_resources(config: Config):
    """Loads tokenizer and model with specific Qwen configurations."""
    print(f"Loading model: {config.model_ckpt}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
            config.model_ckpt, 
            device_map="auto",
            dtype=torch.bfloat16,  
            trust_remote_code=True
        )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        

    model.generation_config.use_cache = True
    
    print(f"Model loaded on: {model.device}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return tokenizer, model

@torch.no_grad()

def translate_with_prompt_qwen(chinese_text, tokenizer, model, system_msg, user_instruction, shots=None, max_new_tokens=1024):
    """
    Implementation estricta of the recommendations of Hugging Face for DeepSeek-R1.
    Ref: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    """
    
    # 1. Build Examples (Few-Shot)
    examples_text = ""
    if shots:
        examples_text = "\n\nExamples:\n"
        for i, ex in enumerate(shots, 1):
            examples_text += f"Input: {ex['zh']}\nOutput: [[[{ex['en']}]]]\n"

    # 2. CONSTRUCTION OF THE PROMPT (ALL IN USER)
    # HF Recommends: "Avoid adding to system prompt; all instructions should be contained within the user prompt."
    # HF Recommends: "include to directive... put your final answer within \boxed{}" (Use [[[ ]]] here)
    
    full_prompt = (
        f"{system_msg}\n\n"
        f"INSTRUCTION: {user_instruction}\n"
        f"You strictly translate Chinese to English.\n"
        f"Please reason step by step, and put your final translation within triple brackets like this: [[[Translation]]].\n"
        f"{examples_text}\n"
        f"Input: {chinese_text}\n"
    )

    # 3. Structure of Messages (ONLY USER)
    messages = [
        {"role": "user", "content": full_prompt}
    ]

    # 4. Tokenization
    # Forzamos the start of generation. 
    # Score: R1 usually add <think> only, but the prompt "reason step by step" help.
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    # 5. Generation (Parameters HF)
    # HF Recommends: "Set the temperature within the range of 0.5-0.7 (0.6 is recommended)"
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens, # R1 needs espacio for pensar
            do_sample=True,
            temperature=0.6,        # <--- RECOMENDADO BY HF
            top_p=0.95,
            repetition_penalty=1.1, # Ligero, no agresivo
            pad_token_id=tokenizer.eos_token_id
        )

    # 6. Decoding
    generated_tokens = output_ids[0][input_len:]
    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # 7. EXTRACTION ROBUST
    # DeepSeek R1 generates: <think> ... </think> ... [[[Answer]]]
    
    # TO. Intentamos extraer it that there are in the brackets [[[ ]]]
    match = re.search(r"\[\[\[(.*?)\]\]\]", full_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # B. Fallback: If fail the brackets, stop the reasoning and return the remainder
    if "</think>" in full_output:
        return full_output.split("</think>")[-1].strip()
        
    return full_output.strip()

def build_shots_dataset(dataset, num_shots, split="train"):
    """Extracts 'num_shots' examples from the dataset."""
    if num_shots == 0:
        return []
    total_rows = len(dataset[split])
    return [dataset[split][i % total_rows] for i in range(num_shots)]


def main():
    
    
    args = parse_args()

    PROMPT_IDS = args.prompts
    
    if PERFORM_HF_LOGIN:
        login()
        
    set_deterministic_seed(SEED)
    
    cfg = Config()
    cfg.create_structure()
    print(f"Configuration:\n{cfg}")

    print("Loading dataset from disk...")
    try:
        raw_ds = load_from_disk(str(cfg.dataset_dir))
        print(raw_ds)
    except Exception as e:
        print(f"Error loading dataset at {cfg.dataset_dir}: {e}")
        return

    tokenizer, model = load_model_resources(cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}.txt"
    time_log_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}_time.txt"
    
    print(f"Results will be saved to: {output_file_path}")

    all_time_records = []
    
    with open(output_file_path, "w", encoding="utf-8") as f_out:
        
        for prompt_idx in PROMPT_IDS:
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
                
                mode_latencies = []
                
                shot_examples = build_shots_dataset(raw_ds, n_shots, split="train")
                

                progress_desc = f"Prompt {prompt_idx} [{mode_name}]"
                for i in tqdm(range(N_SAMPLES), desc=progress_desc):
                    sample_data = raw_ds["test"][i % len(raw_ds["test"])]
                    
                    start_ts = time.perf_counter()
                    try:
                        translation_result = translate_with_prompt_qwen(
                            chinese_text=sample_data["zh"],
                            tokenizer=tokenizer,
                            model=model,
                            shots=shot_examples,
                            system_msg=sys_msg,
                            user_instruction=usr_instr
                        )
                    except Exception as exc:
                        translation_result = f"[Error in iteration {i}: {exc}]"
                    
                    end_ts = time.perf_counter()
                    elapsed = end_ts - start_ts
                    mode_latencies.append(elapsed)
                    
                    f_out.write(f"--- Case {i+1} ---\n")
                    f_out.write(f"[ZH]: {sample_data['zh']}\n")
                    f_out.write(f"[REF]: {sample_data['en']}\n")
                    f_out.write(f"[GEN]: {translation_result}\n")
                    f_out.write(f"[TIME]: {elapsed:.2f} s\n")
                
                # Calculate stats for this mode
                if mode_latencies:
                    avg_t = sum(mode_latencies) / len(mode_latencies)
                    total_t = sum(mode_latencies)
                    prompt_total_time += total_t
                    
                    all_time_records.append((prompt_idx, mode_name, n_shots, avg_t, total_t))
                    
                    f_out.write(f"\n>>> Mean Time {mode_name}: {avg_t:.2f} s | Total: {total_t:.1f} s\n")
                    f_out.flush() # Ensure write to disk

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