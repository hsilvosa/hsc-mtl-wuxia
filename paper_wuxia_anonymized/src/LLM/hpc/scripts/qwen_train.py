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

# 
# Global Settings
# 
PERFORM_HF_LOGIN = False 

N_SAMPLES = 1000
SEED = 42
BATCH_SIZE = 32

SHOTS_CONFIG = {
    "0shot": 0,
    "1shot": 1,
    "2shot": 2, 
    "3shot": 3,
    "5shot": 5, 
    "10shot": 10
}

# Configuration of training (Sample size)
TRAIN_SAMPLE_SIZE = None # None for use all the dataset

def parse_args():
    """Parses command-line arguments for prompt selection and finetuning mode."""
    parser = argparse.ArgumentParser(description="Run LLM evaluation with specified prompt IDs.")
    
    parser.add_argument(
        '--prompts',
        nargs='+',  # Accepts one or more arguments
        type=int,
        required=True,
        help='List of prompt IDs (integers) to evaluate.'
    )
    
    parser.add_argument(
        '--do_finetuning',
        action='store_true',
        help='Enable fine-tuning (QLoRA) before evaluation. If not set, runs inference only.'
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
    finetune_output_dir: Path = store_base / "models" / "qwen3"
    
    translate_file: str = "qwen3.txt"
    results_file: str = "llm_results.txt"
    
    model_ckpt: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # --- Configuration of Training ---
    do_finetuning: bool = False  # Is will be updated from args
    
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4 
    gradient_accumulation_steps: int = 16


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
    print(f"Loading tokenizer: {config.model_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "right" # For training; in inference it cambias to left dynamically
    
    use_cache_setting = False if config.do_finetuning else True

    if config.do_finetuning:
        # MODE TRAINING: Use 4-bit for save VRAM in QLoRA
        print("Loading model in 4-bit for Fine-Tuning...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_ckpt, 
            quantization_config=bnb_config,
            device_map="auto",       
            trust_remote_code=True,
            use_cache=use_cache_setting
        )
    else:
        print("Loading model in bfloat16 ...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_ckpt, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="sdpa",
            device_map="auto",       
            trust_remote_code=True,
            use_cache=use_cache_setting
        )
    
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    
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
    print("STARTING STAGE OF FINE-TUNING (QLoRA) - QWEN")
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
    
    training_args = TrainingArguments(
        output_dir=str(cfg.finetune_output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=10,
        num_train_epochs=cfg.num_train_epochs,
        save_strategy="steps", 
        save_steps= 100,
        save_total_limit=15,
        fp16=False,
        bf16=True, 
        optim="paged_adamw_32bit", 
        report_to="none" 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mapped_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )
    
    print("Training...")
    from transformers.trainer_utils import get_last_checkpoint

    print("Training...")
    
    # Check if exists a checkpoint previous in the directory of output
    last_checkpoint = get_last_checkpoint(str(cfg.finetune_output_dir))
    
    if last_checkpoint is not None:
        print(f"Checkpoint detected! Resuming training from: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No valid checkpoints were found. starting from scratch.")
        trainer.train()
        
    print("Fine-tuning completed.")
    
    print(f"Saving adapter in: {cfg.finetune_output_dir}")
    model.save_pretrained(str(cfg.finetune_output_dir))
    
    model.config.use_cache = True
    model.eval()
    
    return model


# 
# Batch Inference Logic
# 

@torch.no_grad()
def translate_batch_qwen(
    chinese_texts,
    tokenizer,
    model,
    shots=None,
    system_msg=None,
    user_instruction=None,
    max_new_tokens=256,
):
    device = model.device
    prompts = []

    # 1. Preparation of Prompts
    if system_msg is None: system_msg = "You are a translator."
    if user_instruction is None: user_instruction = "Translate this:"

    # Pre-calculate the string of examples
    shots_str = ""
    if shots:
        examples_text = "\n".join(
            f"Example {i}\nChinese: {ex['zh']}\nEnglish: {ex['en']}\n"
            for i, ex in enumerate(shots, 1)
        )
        shots_str = (
            f"Below are example translations to guide your style:\n"
            f"{examples_text}\n"
            "Use these examples for reference, but DO NOT repeat them."
        )

    system_msg_final = f"{system_msg}\n\n{shots_str}" if shots else system_msg

    # Build list of prompts
    for text in chinese_texts:
        prompt_str = (
            f"<|im_start|>system\n{system_msg_final}\n<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_instruction}\n"
            f"Convert all Chinese personal names, place names, and martial arts terms into pinyin romanization. Respond ONLY with the English translation:\n"
            f"Chinese: {text}\nEnglish:"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append(prompt_str)

    # 2. Tokenization in Batch
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)

    # 3. Generation
    gen_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True 
    )

    # 4. Decoding
    gen_ids = [output[len(inputs.input_ids[i]):] for i, output in enumerate(gen_ids)]
    output_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    
    tokenizer.padding_side = original_padding

    return [t.strip() for t in output_texts]

def build_shots_dataset(dataset, num_shots, split="train"):
    if num_shots == 0:
        return []
    total_rows = len(dataset[split])
    return [dataset[split][i % total_rows] for i in range(num_shots)]

def main():
    # --- CAMBIO HERE: Lectura of args ---
    args = parse_args()
    PROMPT_IDS = args.prompts
    
    if PERFORM_HF_LOGIN:
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            login(token=hf_token.strip("'\""))
        else:
            login()

    cfg = Config()
    # Update the configuration with the argument of the parser
    cfg.do_finetuning = args.do_finetuning 
    
    os.environ["HF_HOME"] = str(Path(os.environ.get("STORE", "/tmp/store_fallback")) / ".huggingface")
    cfg.translate_dir.mkdir(parents=True, exist_ok=True)
    
    set_deterministic_seed(SEED)
    print(f"Configuration Mode (Finetuning): {cfg.do_finetuning}")
    print(f"Output Dir: {cfg.finetune_output_dir}")

    # Loads of Dataset
    print("Loading dataset from disk...")
    try:
        raw_ds = load_from_disk(str(cfg.dataset_dir))
    except Exception as e:
        print(f"Error loading dataset at {cfg.dataset_dir}: {e}")
        return

    # 1. Load Tokenizer and Model
    tokenizer, model = load_model_resources(cfg)

    # 2. Management of Fine-tuning or Loads of Adapter
    if cfg.do_finetuning:
        model = run_finetuning(cfg, model, tokenizer, raw_ds)
        has_adapter = True
    else:
        print(f"Mode Inference: Searching adapter trained in {cfg.finetune_output_dir}...")
        
        adapter_dir = cfg.finetune_output_dir
        adapter_config_path = adapter_dir / "adapter_config.json"
        
        # If not is in the root, buscamos in the subdirectories of checkpoints
        if not adapter_config_path.exists():
            print("Was not found in the root. Searching checkpoints intermedios...")
            # Filtramos only directories that empiecen by "checkpoint-"
            checkpoints = [d for d in adapter_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            
            if checkpoints:
                # Sort the carpetas by the number of step for select the last
                checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
                latest_checkpoint = checkpoints[-1]
                
                if (latest_checkpoint / "adapter_config.json").exists():
                    print(f"Checkpoint found! Using the more recent: {latest_checkpoint.name}")
                    adapter_dir = latest_checkpoint
                    adapter_config_path = adapter_dir / "adapter_config.json"
        
        has_adapter = adapter_config_path.exists()
        
        if has_adapter:
            print(f"Loading LoRA weights from: {adapter_dir}...")
            try:
                model = PeftModel.from_pretrained(model, str(adapter_dir))
                model.eval()
                print("Model Fine-Tuned loaded successfully!")
            except Exception as e:
                print(f"Error loading the adapter: {e}")
                print("WARNING: The base model will be used without fine-tuning.")
                has_adapter = False
        else:
            print("WARNING: No trained adapter or valid checkpoints were found.")
            print("Is will proceed utilizando the model BASE.")

    # Preparation of Files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Identify suffix successfully using the variable has_adapter updated
    ft_suffix = "_FT" if has_adapter else "_BASE"
    
    output_file_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}{ft_suffix}.txt"
    time_log_path = cfg.translate_dir / f"{cfg.translate_file}_{timestamp}{ft_suffix}_time.txt"
    
    print(f"Results will be saved to: {output_file_path}")
    print(f"Batch Size: {BATCH_SIZE}")

    all_time_records = []
    test_ds_len = len(raw_ds["test"])
    
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
                
                mode_times = []
                shot_examples = build_shots_dataset(raw_ds, n_shots, split="train")
                
                batch_starts = list(range(0, N_SAMPLES, BATCH_SIZE))
                
                progress_desc = f"Prompt {prompt_idx} [{mode_name}]"
                
                for i in tqdm(batch_starts, desc=progress_desc):
                    current_batch_indices = range(i, min(i + BATCH_SIZE, N_SAMPLES))
                    
                    batch_samples = [raw_ds["test"][j % test_ds_len] for j in current_batch_indices]
                    batch_zh_texts = [s["zh"] for s in batch_samples]
                    
                    start_ts = time.perf_counter()
                    try:
                        translations = translate_batch_qwen(
                            chinese_texts=batch_zh_texts,
                            tokenizer=tokenizer,
                            model=model,
                            shots=shot_examples,
                            system_msg=sys_msg,
                            user_instruction=usr_instr
                        )
                    except Exception as exc:
                        translations = [f"[Error: {exc}]"] * len(batch_zh_texts)
                        print(f"Batch Error: {exc}")
                    
                    end_ts = time.perf_counter()
                    
                    batch_elapsed = end_ts - start_ts
                    avg_per_sample = batch_elapsed / len(batch_zh_texts)
                    
                    mode_times.extend([avg_per_sample] * len(batch_zh_texts))
                    
                    for idx_in_batch, trans_text in enumerate(translations):
                        global_idx = current_batch_indices[idx_in_batch]
                        original_data = batch_samples[idx_in_batch]
                        
                        f_out.write(f"--- Case {global_idx+1} ---\n")
                        f_out.write(f"[ZH]: {original_data['zh']}\n")
                        f_out.write(f"[REF]: {original_data['en']}\n")
                        f_out.write(f"[GEN]: {trans_text}\n")
                        f_out.write(f"[TIME]: {avg_per_sample:.2f} s (Batch Avg)\n")
                
                if mode_times:
                    avg_t = sum(mode_times) / len(mode_times)
                    total_t = sum(mode_times)
                    prompt_total_time += total_t
                    
                    all_time_records.append((prompt_idx, mode_name, n_shots, avg_t, total_t))
                    
                    f_out.write(f"\n>>> Mean Time {mode_name}: {avg_t:.2f} s | Total Process Time: {total_t:.1f} s\n")
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