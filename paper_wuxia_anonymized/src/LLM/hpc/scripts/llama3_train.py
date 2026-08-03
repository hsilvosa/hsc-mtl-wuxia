import argparse
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
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from huggingface_hub import login
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
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
N_SAMPLES = 50
SEED = 42

SHOTS_CONFIG = {
    "0shot": 0
    # "1shot": 1,
    # "2shot": 2, 
    # "3shot": 3,
    # "5shot": 5,
    # "10shot":10
}

DO_FINETUNING = False  
TRAIN_SAMPLE_SIZE = None 

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM evaluation or Carbon Tracking Benchmark.")
    parser.add_argument(
        '--prompts',
        nargs='*', # Optional if only hacemos tracking
        type=int,
        default=[],
        help='List of prompt IDs (integers) to evaluate.'
    )
    # --- Nuevos Arguments for CodeCarbon ---
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
    dataset_dir: Path = lustre_base / "dataset" / "wuxia_chunk_data"    
    translate_dir: Path = store_base / "results" 
    finetune_output_dir: Path = store_base / "models" / "llama3"
    
    translate_file: str = "llama3_finetuned.txt"
    results_file: str = "llm_results.txt"
    
    # Model
    model_ckpt: str = "meta-llama/Llama-3.3-70B-Instruct"
    
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    
    num_train_epochs: int = 1
    
    per_device_train_batch_size: int = 4 
    gradient_accumulation_steps: int = 16
    
    eval_batch_size: int = 2
    
    # New field for control steps maximum (Tracking)
    max_steps: int = -1 
    
    
PROMPT_TEMPLATES = [
    {
        "system_msg": (
            "You are a Chinese-to-English translator. "
            "CRITICAL INSTRUCTION: You must translate the ENTIRE input text. "
            "Do not summarize. Do not skip any sentences. Translate line by line until the very end."
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
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def load_model_resources(config: Config):
    print(f"Loading tokenizer: {config.model_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Pad token asignado: {tokenizer.pad_token}")
    
    # Inicialmente right padding for training
    tokenizer.padding_side = "right" 

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading BASE model (4-bit quantized): {config.model_ckpt}...")
    # Determinamos if use cache depending on if vamos to train or not
    use_cache_setting = True
    # If DO_FINETUNING is True (by variable global o by tracking), desactivamos cache
    # Score: DO_FINETUNING is evaluates in main, but here need saber if are in mode train.
    # Assume that if is calls to this function, the main already configured the environment.
    
    model = AutoModelForCausalLM.from_pretrained(
            config.model_ckpt,
            quantization_config=bnb_config,
            device_map="auto",
            use_cache=False # It ponemos to False by default, is reactiva in inference
        )
    
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model loaded on device: {model.device}")
    
    return tokenizer, model

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
    print("STARTING STAGE OF FINE-TUNING (QLoRA)")
    if cfg.max_steps > 0:
        print(f"MODE TRACKING / BENCHMARK: Limitado to {cfg.max_steps} pasos.")
    print("="*40)
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
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
    
    # Configure arguments of training, respetando max_steps if exists
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
        "bf16": True, 
        "optim": "paged_adamw_32bit", 
        "report_to": "none"
    }

    # If is benchmark, ajustamos parameters
    if cfg.max_steps > 0:
        args_dict["max_steps"] = cfg.max_steps
        args_dict["save_strategy"] = "no" # No save checkpoints in benchmark
        
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
    
    # Only save the model if NOT is to benchmark
    if cfg.max_steps == -1:
        print(f"Saving adapter in: {cfg.finetune_output_dir}")
        model.save_pretrained(str(cfg.finetune_output_dir))
    else:
        print("Mode Benchmark: Not is saved weights in disk.")
    
    # Reactivar cache for inference futura
    model.config.use_cache = True
    model.eval()
    
    return model

# 
# Batch Inference Logic
# 
def translate_batch_llama(zh_texts, tokenizer, model, system_msg, user_instruction, shots=None, max_new_tokens=2048):
    """
    Performs inference in batch for a list of texts in Chinese.
    """
    
    # 1. Prepare the contexto of shots (examples) once
    shots_context = ""
    if shots:
        examples = "\n".join(
            f"Example {i}\nChinese: {ex['zh']}\nEnglish: {ex['en']}\n"
            for i, ex in enumerate(shots, 1)
        )
        shots_context = (
            f"\n\nBelow are example translations to guide your style:\n"
            f"{examples}\n"
            "Use these examples for reference, but DO NOT repeat them or mention them in your output."
        )
    
    final_system_msg = system_msg + shots_context

    # 2. Build the list of prompts completos (format chat)
    batch_prompts_text = []
    
    for zh_text in zh_texts:
        user_msg = (
            f"{user_instruction}\n"
            f"Convert all Chinese personal names, place names, and martial arts terms into pinyin romanization (do not translate them into English). Respond ONLY with the English translation:\n"
            f"Chinese: {zh_text}\nEnglish:"
        )
        
        messages = [
            {"role": "system", "content": final_system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # apply_chat_template with tokenize=False returns the string formatted
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        batch_prompts_text.append(formatted_prompt)

    # 3. Tokenize the batch
    # IMPORTANT: For generation in batch, the padding must be LEFT
    # Save the configuration original for restaurarla after if is necessary
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    inputs = tokenizer(
        batch_prompts_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048 
    ).to(model.device)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 4. Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False, # Greedy decoding
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # 5. Decode only the nuevos tokens generados
    # Stop the portion that corresponds to the prompt of input
    generated_sequences = generated_ids[:, input_ids.shape[1]:]
    
    decoded_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
    
    # Restaurar padding side (good practice)
    tokenizer.padding_side = original_padding_side
    
    return [text.strip() for text in decoded_texts]

def build_shots_dataset(dataset, num_shots, split="train"):
    if num_shots == 0:
        return []
    total_rows = len(dataset[split])
    return [dataset[split][i % total_rows] for i in range(num_shots)]


# 
# Main Execution
# 
def main():
    global DO_FINETUNING
    
    args = parse_args()
    PROMPT_IDS = args.prompts
    
    # ---------------------------------------------------------
    # CONFIGURATION TRACKING
    # ---------------------------------------------------------
    tracker = None
    if args.track_emissions:
        if not CODECARBON_AVAILABLE:
            print("ERROR CRITICAL: 'codecarbon' not installed. Runs 'pip install codecarbon'.")
            return
        
        print("\n" + "!"*60)
        print(f" MODE TRACKING ACTIVADO: Midiendo emissions ({args.country_code})")
        print(f" Pasos de benchmark: {args.benchmark_steps}")
        print("!"*60 + "\n")
        
        # Forzamos that is ejecute the logic of training
        DO_FINETUNING = True
        
        # Iniciamos the tracker
        tracker = OfflineEmissionsTracker(
            project_name="llama3_benchmark",
            country_iso_code=args.country_code,
            output_dir=str(Config().store_base)
        )
        tracker.start()

    if PERFORM_HF_LOGIN:
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                login(token=hf_token.strip("'\""))
            else:
                login()
        
    cfg = Config()
    
    # If are trackeando, inyectamos the steps maximum in the config
    if args.track_emissions:
        cfg.max_steps = args.benchmark_steps
    
    set_deterministic_seed(SEED)

    print("Loading dataset from disk...")
    try:
        raw_ds = load_from_disk(str(cfg.dataset_dir))
        print(raw_ds)
    except Exception as e:
        print(f"Error loading dataset at {cfg.dataset_dir}: {e}")
        return

    # 1. Load Tokenizer and Model Base
    tokenizer, model = load_model_resources(cfg)

    # 2. Management of Fine-tuning or Loads of Adapter
    if DO_FINETUNING:
        # Here is runs the training (completo o benchmark)
        model = run_finetuning(cfg, model, tokenizer, raw_ds)
        
        # IF IS MODE TRACKING, STOP HERE
        if args.track_emissions:
            emissions = tracker.stop()
            print("\n" + "*"*50)
            print(f"RESULTS OF THE MICRO-TRAINING ({args.benchmark_steps} steps)")
            print(f"Emissions detectadas: {emissions:.6f} kg CO2")
            print(f"File saved in: {cfg.store_base}/emissions.csv")
            print("*"*50)
            print("\n--- CALCULATION OF EXTRAPOLATION ---")
            print(f"Actual_Emissions = {emissions:.6f} * (Actual_Total_Time / Micro_Training_Time)")
            print("Exiting of the script (Tracking finished).")
            return # <--- IMPORTANT: Exit for not run inference
    else:
        print(f"Mode Inference: Searching adapter trained in {cfg.finetune_output_dir}...")
        adapter_config_path = cfg.finetune_output_dir / "adapter_config.json"
        
        if adapter_config_path.exists():
            print("Adapter found. Loading weights of LoRA...")
            try:
                model = PeftModel.from_pretrained(model, str(cfg.finetune_output_dir))
                model.eval()
                print("Model Fine-Tuned loaded successfully!")
            except Exception as e:
                print(f"Error loading the adapter: {e}")
                print("WARNING: The base model will be used without fine-tuning.")
        else:
            print("WARNING: Was not found to adapter trained.")
            print("Is will proceed utilizando the model BASE.")
        
    # ---------------------------------------------------------
    # INFERENCE / EVALUATION
    # (Only is runs if NOT are in mode tracking)
    # ---------------------------------------------------------
    
    if not args.prompts:
        print("No prompts provided in arguments. Exiting.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ft_suffix = "_FT" if (DO_FINETUNING or (cfg.finetune_output_dir / "adapter_config.json").exists()) else "_BASE"
    output_file_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}{ft_suffix}.txt"
    time_log_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}{ft_suffix}_time.txt"
    
    print(f"Results will be saved to: {output_file_path}")
    print(f"Using Inference Batch Size: {cfg.eval_batch_size}")
    
    all_time_records = []
    
    # Preparamos the subset of test
    test_dataset = raw_ds["test"]
    if N_SAMPLES < len(test_dataset):
        test_dataset = test_dataset.select(range(N_SAMPLES))
    
    total_examples = len(test_dataset)

    with open(output_file_path, "w", encoding="utf-8") as f_out:
        
        for prompt_idx in PROMPT_IDS:
            current_prompt = PROMPT_TEMPLATES[prompt_idx]
            sys_msg = current_prompt["system_msg"]
            usr_instr = current_prompt["user_instruction"]
            
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
                
                # --- LOGICA OF BATCHING ---
                for i in tqdm(range(0, total_examples, cfg.eval_batch_size), desc=progress_desc):
                    
                    batch_slice = test_dataset[i : i + cfg.eval_batch_size]
                    zh_batch = batch_slice["zh"]
                    en_batch = batch_slice["en"]
                    
                    start_ts = time.perf_counter()
                    try:
                        translations_batch = translate_batch_llama(
                            zh_texts=zh_batch,
                            tokenizer=tokenizer,
                            model=model,
                            system_msg=sys_msg,
                            user_instruction=usr_instr,
                            shots=shot_examples
                        )
                    except Exception as exc:
                        translations_batch = [f"[Error in batch: {exc}]"] * len(zh_batch)
                    
                    batch_elapsed = time.perf_counter() - start_ts
                    avg_time_per_sample = batch_elapsed / len(zh_batch)
                    
                    for _ in range(len(zh_batch)):
                        mode_times.append(avg_time_per_sample)

                    for j, (src, ref, gen) in enumerate(zip(zh_batch, en_batch, translations_batch)):
                        global_idx = i + j + 1
                        f_out.write(f"--- Case {global_idx} ---\n")
                        f_out.write(f"[ZH]: {src}\n")
                        f_out.write(f"[REF]: {ref}\n")
                        f_out.write(f"[GEN]: {gen}\n")
                        f_out.write(f"[TIME]: {avg_time_per_sample:.2f} s (batch avg)\n")

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
