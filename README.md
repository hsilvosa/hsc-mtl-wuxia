# TFG-UDC Hugo Silvosa Cuervo

**Enfoques de traducción automática con modelos de lenguaje en obras *wuxia*** 
Machine Translation Approaches with Language Models in Wuxia Literature

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Academic-TFG](https://img.shields.io/badge/Degree-Data_Science_and_Engineering-darkred.svg)]()

This repository contains the implementation, datasets, pipeline configurations, and evaluation frameworks for the Bachelor's Thesis (**Trabajo de Fin de Grado - TFG**) titled *"Enfoques de traducción automática con modelos de lenguaje en obras wuxia"* (Facultade de Informática, Universdidade da Coruña). 

The objective of this research is to conduct a rigorous comparative study across three paradigms of Machine Translation—**Statistical Machine Translation (SMT)**, **Neural Machine Translation (NMT)**, and fine-tuned **Large Language Models (LLMs)**—applied to the highly nuanced, culturally rich, and domain-specific literary genre of *Wuxia* (Chinese martial arts fantasy) translated into English.




##  Table of Contents
- [TFG Hugo Silvosa Cuervo](#tfg-hugo-silvosa-cuervo)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
    - [\[EN\]](#en)
    - [\[ES\]](#es)
    - [\[ZH\]](#zh)
  - [Project Overview \& Methodology](#project-overview--methodology)
  - [Dataset \& Corpus Specifics](#dataset--corpus-specifics)
    - [Primary Experimental Dataset (TFG Core)](#primary-experimental-dataset-tfg-core)
  - [Repository Structure](#repository-structure)
  - [Pipeline \& Implementation Modules](#pipeline--implementation-modules)
    - [1. Corpus Alignment Pipeline (wuxia\_corpus/)](#1-corpus-alignment-pipeline-wuxia_corpus)
    - [2. Lexical \& Structural Preprocessing (preprocessing/)](#2-lexical--structural-preprocessing-preprocessing)
    - [3. Translation Engine Implementations](#3-translation-engine-implementations)
      - [SMT Baseline (src/SMT/)](#smt-baseline-srcsmt)
      - [NMT Systems (src/NMT/)](#nmt-systems-srcnmt)
      - [LLM Architectures (src/LLM/)](#llm-architectures-srcllm)
        - [Parameter Size Study](#parameter-size-study)
  - [Evaluation Framework](#evaluation-framework)
    - [Quantitative Evaluation](#quantitative-evaluation)
    - [Automated Qualitative Evaluation (LLM-as-a-Judge)](#automated-qualitative-evaluation-llm-as-a-judge)
  - [Hugging Face Hub Assets](#hugging-face-hub-assets)
  - [Installation \& Requirements](#installation--requirements)
  - [Execution Guide](#execution-guide)
  - [Results](#results)


---

## Abstract

### [EN]
Translating Chinese web literature of the \textit{wuxia} and \textit{xianxia} genres into English can be a complex task due to its heavy idiomatic and cultural load. In this work, we contrast the effectiveness of neural machine translation (NMT) systems against large language models (LLMs) using an ad hoc, domain-specific parallel corpus. To adapt the models to this domain, we trained the NMT models through conventional fine-tuning. Meanwhile, for the LLMs, we experimented with various techniques, such as prompt engineering and in-context learning, culminating in their fine-tuning using QLoRA. Upon evaluation, we observed that while NMT leads in lexical overlap metrics, LLMs achieve comparable semantic equivalence. This analysis is complemented by a comparison with commercial systems and a qualitative evaluation that briefly reviews the main issues encountered in the translations. Thus, the results demonstrate that, compared to the rigidity and contextual limitations of NMT, domain-adapted LLMs manage to prioritize narrative pacing and cultural context, performing in a manner much closer to that of a professional human translator when dealing with highly complex texts.

### [ES]
Traducir literatura web china de los géneros \textit{wuxia} y \textit{xianxia} al inglés puede resultar una tarea compleja debido a su fuerte carga idiomática y cultural. En este trabajo contrastamos la eficacia de los sistemas de traducción automática neuronal (NMT, por sus siglas en inglés) frente a los grandes modelos de lenguaje (LLM, por sus siglas en inglés) mediante el uso de un corpus paralelo ad hoc específico al dominio. Para adaptar los modelos a este dominio, entrenamos los modelos NMT mediante ajuste fino convencional. Por su parte, en los LLMs se experimentó con diferentes técnicas, como la ingeniería de instrucciones y el aprendizaje en contexto, hasta culminar en su ajuste con QLoRA. Tras la evaluación, observamos que, si bien la NMT lidera en métricas de solapamiento léxico, los LLM logran una equivalencia semántica comparable. Este análisis se complementa con una comparativa con sistemas comerciales y una evaluación cualitativa en la que, brevemente, se revisan las principales problématicas encontradas en las traducciones. De esta manera, los resultados demuestran que, frente a la rigidez y las limitaciones contextuales de la NMT, los LLM ajustados al dominio logran priorizar el ritmo narrativo y el contexto cultural, actuando de una manera mucho más cercana a la de un traductor humano profesional en textos de dominios muy específicos.

### [ZH]
由于包含丰富的习语和文化内涵，将“武侠”（\textit{wuxia}）和“仙侠”（\textit{xianxia}）题材的中国网络文学翻译为英文是一项复杂的任务。在本文中，我们使用针对该领域特制的平行语料库，对比了神经机器翻译（NMT）系统与大型语言模型（LLM）的有效性。为了使模型适应此领域，我们使用常规微调（conventional fine-tuning）对NMT模型进行了训练。而在LLM方面，我们实验了多种技术，包括提示工程（prompt engineering）和上下文学习（in-context learning），最终使用QLoRA对其进行了微调。评估结果显示，尽管NMT在词汇重叠度指标上保持领先，但LLM能够达到与之相当的语义对等。此外，我们还将分析结果与商业系统进行了对比，并进行了定性评估，简要回顾了翻译中遇到的主要问题。结果表明，与NMT的僵硬和上下文局限相比，经过领域适应的LLM能够更好地兼顾叙事节奏和文化语境，在处理高度专业领域的文本时，其表现更接近于专业的人类译者。

---

## Project Overview & Methodology

Wuxia literature presents steep challenges for standard machine translation due to culturally bound terms, martial arts techniques, historical naming protocols, fictional factions, and intricate metaphysical world-building concepts. 

This project implements an end-to-end pipeline from raw, unaligned web/book sources to fine-tuned advanced sequence-to-sequence networks and decoder-only LLMs using QLoRA optimizations executed on High-Performance Computing (HPC) nodes.

---

## Dataset & Corpus Specifics

The experimental framework draws text from three Wuxia/Xianxia web novels:
* **AWE**: *A Will Eternal*
* **CONDOR**: *The Condor Trilogy* (Jin Yong - 3 volumes)
* **GU**: *Reverend Insanity* (Guzhenren collection)

### Primary Experimental Dataset (TFG Core)
While the full pipeline is capable of generating massive parallel text blocks, the core empirical research, training cycles, hyperparameter tuning, and cross-model evaluations presented in the thesis text explicitly utilize the dataset located at:
```text
processed_data/wuxia_selected_100000/

    Type: Stratified sample consisting of exactly 100,000 high-confidence parallel sentence pairs.

    Distribution: Balances structural diversity, sentence length variants, and vocabulary richness evenly across the four novel sources.

    Format: Ready-to-use HuggingFace Dataset format splits (train, validation, test) optimized for tokenization and pipeline memory mapping.
```

---

## Repository Structure

```
TFG/
├── wuxia_corpus/              # Corpus construction & paragraph/sentence alignment pipeline
│   ├── src/
│   │   ├── preprocessing/     # Chapter segmentation regex engines (Chinese/English)
│   │   ├── alignment/         # LaBSE embeddings + Dynamic Programming alignment matrix
│   │   ├── analysis/          # Cosine similarity thresholding
│   │   └── common/            # Shared utilities & constants
│   └── data/                  # Raw, segmented, and mid-process unaligned source texts
│
├── preprocessing/             # Statistical analysis, data filtration & dataset splitting
│   ├── inputs/
│   │   ├── corpus/            # Aligned raw parallel datasets (final_*.txt)
│   │   └── scores/            # Token-level and chrF scoring metrics for data selection
│   ├── src/                   # Analysis engine (lexical richness, Type-Token Ratio)
│   └── outputs/               # Generated PNG charts, pyvis HTML interactive entity graphs
│
├── src/                       # Translation Model Engine implementations
│   ├── SMT/                   # Baseline: Statistical Machine Translation (IBM Models 1-3)
│   ├── NMT/                   # Neural Machine Translation: Seq2Seq fine-tuning code
│   │   ├── configs/           # Architectural hyperparameter classes per model
│   │   ├── utils/             # Data loaders, pipeline builders, tokenization wraps
│   │   └── scripts/           # train.py, evaluate.py, translate.py execution entries
│   └── LLM/                   # Large Language Models fine-tuning & prompt-engineering frameworks
│       ├── cesga/             # SLURM script wrappers for HPC batch compute cluster architectures
│       ├── evaluation/        # Raw generation files, automated evaluation metrics plots
│       └── notebook_local/    # Local verification, exploratory Jupyter notebooks
│
├── models/                    # Fine-tuned model outputs, configurations, and adapter checkpoints
│   ├── mbart50_wuxia/         # mBART-50 Many-to-Many fine-tuned checkpoint
│   ├── m2m100_wuxia/          # Facebook M2M100 fine-tuned adapter/weights
│   ├── marianmt_wuxia/        # Helsinki-NLP MarianMT localized model
│   └── mt5_small/             # Google mT5 instruction-adapted checkpoint
│
├── processed_data/            # Highly refined data layers
│   ├── wuxia_zh_en_clean/     # Full master set of cleaned parallel sentence pairs
│   └── wuxia_selected_100000/ # Core stratified 100k sample dataset used in the TFG
│
├── evaluation/                # Downstream metric evaluation assets
│   ├── CO2/                   # Carbon footprint trackers and green computing metrics
│   ├── times/                 # Inference latency, token throughput, generation timing logs
│   └── cualitativa/           # Automated qualitative evaluation datasets (LLM-as-a-Judge outputs plus Blind Human Trials)
├── figures/                   # Architectural drawings, BPE breakdowns, and thesis graphics used in the thesis
└── docs/                      # Thesis text assets, project proposals
```

---

##  Pipeline & Implementation Modules

### 1. Corpus Alignment Pipeline (wuxia_corpus/)

Extracts and structures parallel corpora from raw text:

```bash
# Segmentation: chinese.py and english.py tokenize inputs into structural semantic paragraphs using regex sequences specialized for historical Chinese chapter hooks (第...章, 第...回)
python wuxia_corpus/src/preprocessing/chinese.py --novela awe --input awe_ch.txt --outname segmented/chapter
python wuxia_corpus/src/preprocessing/english.py --novela awe --input awe_en.txt --outname segmented/chapter

# Vector Space Alignment: Processes segments through LaBSE to extract cross-lingual dense vectors
python wuxia_corpus/src/alignment/final_awe.py
```

### 2. Lexical & Structural Preprocessing (preprocessing/)

Builds and measures the linguistic properties of the corpus:

```bash
python preprocessing/src/analysis.py  # Executes complex lexical analysis (STTR, MTLD, HD-D, Guiraud Index)
python preprocessing/src/count.py     # Statistical counting utilities

# Generate corpus
python src/preprocessing/preprocessing.py

# Stratified selection
python src/preprocessing/data_selection.py --mode study     
python src/preprocessing/data_selection.py --mode execute   
```

### 3. Translation Engine Implementations

#### SMT Baseline (src/SMT/)
Implements traditional statistical translation via NLTK's IBM Models (1, 2, 3), tokenized via jieba:

```bash
python src/SMT/smt_test.py train --model ibm2 --train_size 5000
python src/SMT/smt_test.py infer --model ibm2 --infer_size 200
python src/SMT/smt_prob.py  # Query translation probability tables
```

#### NMT Systems (src/NMT/)
Integrates configurations for deep seq2seq models (marianmt, m2m100, mbart50, mt5):

**Models Used in TFG:**
| Model | Checkpoint | Description |
|-------|------------|-------------|
| M2M100 | `facebook/m2m100_418M` | Facebook multilingual model (418M params) |
| MarianMT | `Helsinki-NLP/opus-mt-zh-en` | Helsinki-NLP Chinese-to-English model |
| mBART50 | `facebook/mbart-large-50-many-to-many-mmt` | Facebook multilingual denoising model |
| mT5 Small | `google/mt5-small` | Google multilingual T5 (small variant) |

```bash
# Fine-tune an NMT backbone model using a fraction of the dataset
python src/NMT/scripts/train.py m2m100 --fraction 0.1 --epochs 3

# Compute automated scores (BLEU/chrF) over the test set
python src/NMT/scripts/evaluate.py m2m100

# Generate translation outputs for qualitative analysis
python src/NMT/scripts/translate.py m2m100 --out-file m2m100.txt
```

#### LLM Architectures (src/LLM/)
Scripts tailored for large-scale decoder models (Gemma, Llama 3, Qwen, GLM, Yi, DeepSeek) with QLoRA fine-tuning:

**Models Used in TFG:**
| Model | Checkpoint | Description |
|-------|------------|-------------|
| Qwen 3 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE (30B params) |
| Llama 3.3 | `meta-llama/Llama-3.3-70B-Instruct` | Dense 70B params |
| Gemma 3 | `google/gemma-3-27b-it` | Dense 27B params |
| GLM-4 | `zai-org/GLM-4-32B-0414` | Dense 32B params |

##### Parameter Size Study
Models evaluated for parameter scaling analysis:
| Model | Checkpoint | Parameters |
|-------|------------|------------|
| Gemma 3 270M | `google/gemma-3-270m-it` | 270M |
| Gemma 3 1B | `google/gemma-3-1b-it` | 1B |
| Gemma 3 4B | `google/gemma-3-4b-it` | 4B |
| Gemma 3 12B | `google/gemma-3-12b-it` | 12B |


```bash
# Local (notebooks)
cd src/LLM/notebook_local && jupyter notebook

# HPC (CESGA)
sbatch src/LLM/cesga/BATCH\ CESGA/gemma.sh
sbatch src/LLM/cesga/BATCH\ CESGA/gemma_fine.sh
```

---

## Evaluation Framework

The study relies on a validation paradigm combining automated quantitative data points with advanced qualitative structures.

### Quantitative Evaluation

The quantitative evaluation measures translations across linguistic overlap metrics (SacreBLEU, chrF2, ROUGE-L, METEOR) and semantic proximity (COMET).

#### Large Language Model (LLM) Results (Prompt 0, 0-Shot)

The table below summarizes the comparative performance of the base LLM architectures versus their QLoRA fine-tuned adapters on the Wuxia test set:

| Model | Variant | SacreBLEU | chrF2 | ROUGE-L | METEOR | COMET |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemma 3 (27B)** | Base | 10.64 | 36.46 | 40.29 | 39.41 | 71.50 |
| | **QLoRA Fine-tuned** | **26.28** | **48.47** | **54.54** | **54.64** | **78.59** |
| **Llama 3.3 (70B)**| Base | 9.50 | 30.20 | 30.78 | 30.02 | 62.52 |
| | **QLoRA Fine-tuned** | **26.12** | **48.39** | **54.16** | **54.45** | **77.94** |
| **Qwen 3 (30B-A3B)**| Base | 9.30 | 30.10 | 31.38 | 31.12 | 61.89 |
| | **QLoRA Fine-tuned** | **25.08** | **47.18** | **54.47** | **53.77** | **78.42** |
| **GLM-4 (32B)** | Base | 9.57 | 36.27 | 39.47 | 38.97 | **72.49** |
| | **QLoRA Fine-tuned** | **13.52** | **39.94** | **41.15** | **45.27** | 72.38 |

* **Linguistic Proximity**: BLEU, ROUGE-L, METEOR and chrF scores
* **Semantic Proximity**: COMET score
* **Operational Footprint**: Inference latency, token throughput, parameter counts

### Automated Qualitative Evaluation (LLM-as-a-Judge)

Following the LLM-as-a-Judge paradigm, validation protocols utilize state-of-the-art closed engines (Claude 3.5 Sonnet, Gemini 1.5 Flash, OpenAI GPT-4o-mini) to analyze deep translation mechanics compared to the open source models.


---

## Hugging Face Hub Assets

All datasets and fine-tuned models developed for this project are publicly available on the Hugging Face Hub:

### Datasets
* **[hsc-wuxia-full](https://huggingface.co/datasets/HSilvosa/hsc-wuxia-full)**: The complete aligned Chinese-English parallel corpus of Wuxia literature.
* **[hsc-wuxia-100k](https://huggingface.co/datasets/HSilvosa/hsc-wuxia-100k)**: The core stratified parallel dataset containing exactly 100,000 sentence pairs used for model training and evaluation.

### Fine-tuned Models

#### Large Language Model (LLM) Adapters
* **[hsc-wuxia-llama-3.3-70b](https://huggingface.co/HSilvosa/hsc-wuxia-llama-3.3-70b)**: QLoRA Adapter fine-tuned on top of `meta-llama/Llama-3.3-70B-Instruct`.
* **[hsc-wuxia-gemma-3-27b](https://huggingface.co/HSilvosa/hsc-wuxia-gemma-3-27b)**: QLoRA Adapter fine-tuned on top of `google/gemma-3-27b-it`.
* **[hsc-wuxia-glm-4-32b](https://huggingface.co/HSilvosa/hsc-wuxia-glm-4-32b)**: QLoRA Adapter fine-tuned on top of `zai-org/GLM-4-32B-0414`.
* **[hsc-wuxia-qwen-3-30b-a3b](https://huggingface.co/HSilvosa/hsc-wuxia-qwen-3-30b-a3b)**: QLoRA Adapter fine-tuned on top of `Qwen/Qwen3-30B-A3B-Instruct-2507`.

#### Neural Machine Translation (NMT) Models
* **[hsc-wuxia-mbart-large-50](https://huggingface.co/HSilvosa/hsc-wuxia-mbart-large-50)**: Fine-tuned `facebook/mbart-large-50-many-to-many-mmt` translation model.
* **[hsc-wuxia-m2m100-418m](https://huggingface.co/HSilvosa/hsc-wuxia-m2m100-418m)**: Fine-tuned `facebook/m2m100_418M` translation model.
* **[hsc-wuxia-marianmt-zh-en](https://huggingface.co/HSilvosa/hsc-wuxia-marianmt-zh-en)**: Fine-tuned `Helsinki-NLP/opus-mt-zh-en` translation model.
* **[hsc-wuxia-mt5-small](https://huggingface.co/HSilvosa/hsc-wuxia-mt5-small)**: Fine-tuned `google/mt5-small` translation model.

---

## Installation & Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `sentence-transformers`, `datasets`, `peft`, `bitsandbytes`, `nltk`, `jieba`, `lexicalrichness`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `pyvis`, `codecarbon`

---

## Execution Guide

| Step | Command | Description |
|------|---------|-------------|
| 1 | `python wuxia_corpus/src/preprocessing/chinese.py` | Segment Chinese chapters |
| 2 | `python wuxia_corpus/src/preprocessing/english.py` | Segment English chapters |
| 3 | `python wuxia_corpus/src/alignment/final_awe.py` | Align with LaBSE + DP |
| 4 | `python src/preprocessing/preprocessing.py` | Generate corpus |
| 5 | `python src/preprocessing/data_selection.py --mode execute` | Stratified selection of segments |
| 6 | `python src/SMT/smt_test.py train` |  SMT baseline |
| 7 | `python src/NMT/scripts/train.py m2m100` | Fine-tune NMT model |
| 8 | `python src/LLM/cesga/scripts/qwen_train.py --prompts 0 1 2` | LLM inference/fine-tuning (HPC) |
| 9 | `python results/qualitative/human.py` | Run blind human evaluation |
| 10 | `python results/qualitative/llm_judge.py` | Run blind human evaluation |


---

## Results

Complete metrics and charts are available in:
* `preprocessing/outputs/` - Statistical charts and entity graphs
* `evaluation/` - CO2 metrics, inference times, llm results, nmt results, qualitative evaluations
* `src\LLM\evaluation` - detailed LLM results
* `src\NMT\evaluation` - detailed NMT results
