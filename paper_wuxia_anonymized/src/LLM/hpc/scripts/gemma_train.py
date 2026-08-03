import os
import argparse
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
    Gemma3ForConditionalGeneration,
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from huggingface_hub import login
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)

# Integration CodeCarbon
try:
    from codecarbon import OfflineEmissionsTracker 
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


# 
# Global Settings
# 
PERFORM_HF_LOGIN = True
N_SAMPLES = 64
SEED = 42
BATCH_SIZE = 16

SHOTS_CONFIG = {
    "0shot": 0,
    "1shot": 1,
    "2shot": 2, 
    "3shot": 3,
    "5shot": 5,
    "10shot":10
}

# --- Configuration of Training ---
TRAIN_SAMPLE_SIZE = None 

def parse_args():
    """Parses command-line arguments for prompt selection."""
    parser = argparse.ArgumentParser(description="Run LLM evaluation with specified prompt IDs.")
    parser.add_argument(
        '--prompts',
        nargs='*', 
        type=int,
        default=[],
        help='List of prompt IDs (integers) to evaluate.'
    )
    parser.add_argument(
        '--do_finetuning',
        action='store_true',
        help='Enable fine-tuning (QLoRA) before evaluation. If not set, runs inference only.'
    )
    parser.add_argument(
        '--track_emissions',
        action='store_true',
        help='Enables the mode benchmark for measure emissions of CO2 (Micro-training).'
    )
    parser.add_argument(
        '--benchmark_steps',
        type=int,
        default=50,
        help='Number of steps of training for the measurement of emissions (Default: 50).'
    )
    parser.add_argument(
        '--country_code',
        type=str,
        default="ESP",
        help='Code ISO of the country for calculate the intensity of carbon (e.g.. ESP, USES, FRA).'
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
    
    dataset_dir: Path = lustre_base / "dataset" / "wuxia_selected_100000"    
    
    translate_dir: Path = store_base / "results" 
    finetune_output_dir: Path = store_base / "models" / "gemma"

    
    translate_file: str = "gemma3.txt"
    results_file: str = "llm_results.txt"
    
    model_ckpt: str = "google/gemma-3-27b-it" 
    
    # --- Configuration of Training ---
    do_finetuning: bool = False
    
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    
    num_train_epochs: int = 1
    
    per_device_train_batch_size: int = 4 
    gradient_accumulation_steps: int = 16

    # New field for control steps maximum (Tracking)
    max_steps: int = -1 
    
    
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
    """Loads tokenizer and model for Gemma 3 (BF16 Native - No 4bit)."""
    print(f"Loading tokenizer: {config.model_ckpt}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Pad token asignado: {tokenizer.pad_token}")
    
    # Padding side logic: right for training, left for inference (restored later)
    tokenizer.padding_side = "right" 
    
    print(f"Loading model: {config.model_ckpt} (Native BFloat16)...")
    
    # Use Cache setting logic
    use_cache_setting = False if config.do_finetuning else True

    # LOADS NATIVA IN BFLOAT16 (Estable in A100)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        config.model_ckpt, 
        dtype=torch.bfloat16,  
        device_map="auto",       
        trust_remote_code=True
    )

    model.config.use_cache = use_cache_setting

    print(f"Model loaded on device: {model.device} | Use Cache: {use_cache_setting}")
    
    return tokenizer, model

# 
# Training Functions
# 

def format_data_for_training(example, tokenizer):
    system_msg = "You are an expert translator specializing in Chinese Wuxia fiction to English."
    user_msg = f"Translate the following Chinese text to English:\n{example['zh']}"
    assistant_msg = example['en'] 

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512, 
        padding="max_length"
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def run_finetuning(cfg: Config, model, tokenizer, dataset):
    print("\n" + "="*40)
    print("STARTING STAGE OF FINE-TUNING (LoRA Standard) - GEMMA 3")
    if cfg.max_steps > 0:
        print(f"MODE TRACKING / BENCHMARK: Limitado to {cfg.max_steps} pasos.")
    print("="*40)
    
    # Target modules for Gemma
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads() 
    model.config.use_cache = False 
    model.print_trainable_parameters()
    
    print("Preprocesando dataset for training...")
    train_data = dataset["train"]
    
    if TRAIN_SAMPLE_SIZE is not None and TRAIN_SAMPLE_SIZE < len(train_data):
        print(f"Limiting training to {TRAIN_SAMPLE_SIZE} examples.")
        train_data = train_data.select(range(TRAIN_SAMPLE_SIZE))
    else:
        print(f"Using ALL the dataset of training: {len(train_data)} examples.")
        
    mapped_dataset = train_data.map(
        lambda x: format_data_for_training(x, tokenizer),
        batched=False,
        remove_columns=train_data.column_names
    )
    
    args_dict = {
        "output_dir": str(cfg.finetune_output_dir),
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "logging_steps": 5 if cfg.max_steps > 0 else 10,
        "num_train_epochs": cfg.num_train_epochs,
        "save_strategy": "steps", 
        "save_steps": 500,
        "save_total_limit": 2,
        "fp16": False,
        "bf16": True, # BFloat16 for A100
        "optim": "adamw_torch",
        "report_to": "none",
        "gradient_checkpointing": True 
    }
    
    if cfg.max_steps > 0:
        args_dict["max_steps"] = cfg.max_steps
        args_dict["save_strategy"] = "no"

    training_args = TrainingArguments(**args_dict)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mapped_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )
    
    print("Training...")
    trainer.train()
    print("Fine-tuning completed.")
    
    if cfg.max_steps == -1:
        print(f"Saving adapter in: {cfg.finetune_output_dir}")
        model.save_pretrained(str(cfg.finetune_output_dir))
    else:
        print("Mode Benchmark: Not is saved weights in disk.")
        
    model.config.use_cache = True
    model.eval()
    
    return model

# 
# Inference Functions
# 
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
    Version vectorizada for Gemma 3 with correction of STOP TOKENS.
    """
    prompts_as_strings = []
    
    # 1. Prepare each prompt
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
        
        # apply_chat_template adds <start_of_turn>model to the final automatically
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts_as_strings.append(prompt_str)

    # 2. Tokenize
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts_as_strings,
        return_tensors="pt",
        padding=True,       
        truncation=True,
        max_length=4096     
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    # --- CORRECTION CRITICAL: STOP TOKENS ---
    # Gemma needs stop when finds <end_of_turn> or EOS.
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    # 3. Generation
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,         # Greedy for stability
        temperature=None,     
        top_p=None,
        repetition_penalty=1.0,  # Without penalty for avoid NaNs in A100
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators # <--- HERE FORZAMOS THE PARADA
    )

    # 4. Decoding and Cleaning
    generated_tokens = [out[input_length:] for out in output_ids]
    output_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # Restauramos padding original
    tokenizer.padding_side = old_padding

    # Cleaning of safety: If the model hallucinates "model" or repeats, stop.
    cleaned_results = []
    for t in output_texts:
        t = t.split("model\n")[0] 
        t = t.split("<start_of_turn>")[0]
        t = t.strip()
        cleaned_results.append(t)

    return cleaned_results

def build_shots_dataset(dataset, num_shots, split="train"):
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
    
    tracker = None
    if args.track_emissions:
        if not CODECARBON_AVAILABLE:
            print("ERROR CRITICAL: 'codecarbon' not installed. Runs 'pip install codecarbon'.")
            return
        
        print("\n" + "!"*60)
        print(f" MODE TRACKING ACTIVADO: Midiendo emissions ({args.country_code})")
        print(f" Pasos de benchmark: {args.benchmark_steps}")
        print("!"*60 + "\n")
        
        # Iniciamos the tracker
        tracker = OfflineEmissionsTracker(
            project_name="gemma_benchmark",
            country_iso_code=args.country_code,
            output_dir=str(Config().store_base)
        )
        tracker.start()
    
    if PERFORM_HF_LOGIN:
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                print("Token of Hugging Face found in the environment. Starting session...")
                login(token=hf_token.strip("'\""))
            else:
                login()
        
    cfg = Config()
    cfg.do_finetuning = args.do_finetuning
    
    if args.track_emissions:
        cfg.do_finetuning = True
        cfg.max_steps = args.benchmark_steps
    
    os.environ["HF_HOME"] = str(Path(os.environ.get("STORE", "/tmp/store_fallback")) / ".huggingface")
    cfg.translate_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic_seed(SEED)
    print(f"Configuration Mode (Finetuning): {cfg.do_finetuning}")

    print("Loading dataset from disk...")
    try:
        raw_ds = load_from_disk(str(cfg.dataset_dir))
    except Exception as e:
        print(f"Error loading dataset at {cfg.dataset_dir}: {e}")
        return

    # Load Model & Tokenizer
    tokenizer, model = load_model_resources(cfg)

    # 2. Management of Fine-tuning or Loads of Adapter
    if cfg.do_finetuning:
        model = run_finetuning(cfg, model, tokenizer, raw_ds)
        
        if args.track_emissions:
            emissions = tracker.stop()
            print("\n" + "*"*50)
            print(f"RESULTS OF THE MICRO-TRAINING ({args.benchmark_steps} steps)")
            print(f"Emissions detectadas: {emissions:.6f} kg CO2")
            print(f"File saved in: {cfg.store_base}/gemmaemissions.csv")
            print("*"*50)
            print("\n--- CALCULATION OF EXTRAPOLATION ---")
            print(f"Actual_Emissions = {emissions:.6f} * (Actual_Total_Time / Micro_Training_Time)")
            print("Exiting of the script (Tracking finished).")
            return 
    else:
        print(f"Mode Inference: Searching adapter trained in {cfg.finetune_output_dir}...")
        adapter_config_path = cfg.finetune_output_dir / "adapter_config.json"
        
        if adapter_config_path.exists():
            print("Adapter found. Loading weights of LoRA...")
            try:
                # Loads of LoRA standard over model BF16
                model = PeftModel.from_pretrained(model, str(cfg.finetune_output_dir))
                model.eval()
                print("Model Fine-Tuned loaded successfully!")
            except Exception as e:
                print(f"Error loading the adapter: {e}")
                print("WARNING: The base model will be used without fine-tuning.")
        else:
            print("WARNING: Was not found to adapter trained.")
            print("Is will proceed utilizando the model BASE.")

    if not PROMPT_IDS:
        print("No prompts provided in arguments. Exiting.")
        return

    # Prepare output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    has_adapter = (cfg.finetune_output_dir / "adapter_config.json").exists()
    ft_suffix = "_FT" if (cfg.do_finetuning or has_adapter) else "_BASE"

    output_file_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}{ft_suffix}.txt"
    time_log_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}{ft_suffix}_timing.txt"
    
    print(f"Results will be saved to: {output_file_path}")
    print(f"Timing logs will be saved to: {time_log_path}")

    all_time_records = []
    test_ds_len = len(raw_ds["test"])

    # Cleaning of cache
    torch.cuda.empty_cache()

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
                        if "CUDA" in str(exc): break
                    
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
            f_time.write(f"Prompt {pid}: {total_sec/60:.2f} min\n")

    print(f"\nDone. Results: {output_file_path}")
if __name__ == "__main__":
    main()
