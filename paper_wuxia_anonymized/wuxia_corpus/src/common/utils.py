"""
Shared utilities for corpus processing pipeline.
Common functions used across preprocessing and alignment scripts.
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
from config_loader import get_config


def setup_logging(script_name, dataset_name=None):
    """Setup logging for any script"""
    config = get_config()
    logs_dir = config.get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{script_name}"
    if dataset_name:
        log_name = f"{dataset_name}_{log_name}"
    log_file = logs_dir / f"{log_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(script_name)


def load_text_file(file_path, encoding='utf-8', errors='ignore'):
    """Load text from file with error handling"""
    try:
        with open(file_path, 'r', encoding=encoding, errors=errors) as f:
            return f.read()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading {file_path}: {e}")
        return None


def save_text_file(content, file_path, encoding='utf-8'):
    """Save text to file with error handling"""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error saving to {file_path}: {e}")
        return False


def get_file_pairs(directory, pattern_ch=r"(\d+)ch\.txt", pattern_en=r"(\d+)en\.txt"):
    """Find and pair Chinese and English files by numeric prefix"""
    files = os.listdir(directory)
    ch_files = {}
    en_files = {}
    
    for f in files:
        ch_match = re.match(pattern_ch, f)
        en_match = re.match(pattern_en, f)
        
        if ch_match:
            ch_files[int(ch_match.group(1))] = os.path.join(directory, f)
        if en_match:
            en_files[int(en_match.group(1))] = os.path.join(directory, f)
    
    pairs = []
    for idx in sorted(set(ch_files.keys()) & set(en_files.keys())):
        pairs.append((ch_files[idx], en_files[idx]))
    
    return pairs


def get_variant_file_pairs(config, dataset_name, variant_name=None):
    """
    Get file pairs for a dataset, supporting multi-variant datasets like CONDOR.
    
    Args:
        config: CorpusConfig instance
        dataset_name: 'awe', 'condor', 'gu', 'issth'
        variant_name: For CONDOR only - 'condor_1', 'condor_2', 'condor_3'
    
    Returns:
        List of (ch_path, en_path) tuples
    """
    dataset_cfg = config.get_dataset_config(dataset_name)
    
    # Handle multi-variant datasets (CONDOR)
    if dataset_name == 'condor' and variant_name:
        variant = next((v for v in dataset_cfg['variants'] if v['name'] == variant_name), None)
        if variant:
            segmented_path = config.base_dir / variant['segmented_dir']
        else:
            raise ValueError(f"Unknown variant: {variant_name}")
    else:
        # Single-variant datasets (AWE, GU, ISSTH)
        segmented_path = config.base_dir / dataset_cfg['segmented_dir']
    
    return get_file_pairs(str(segmented_path))


def clean_text(text, remove_extra_spaces=True, remove_special=False):
    """Clean text with various options"""
    if not text:
        return text
    
    # Remove extra whitespace
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters if requested
    if remove_special:
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    
    return text.strip()


def chunk_text(text, chunk_size=512, overlap=0):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks
