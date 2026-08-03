# Anonymous research code

This directory contains the anonymized code accompanying the paper. It is a
standalone release assembled from the corpus construction, preprocessing, and
model experimentation components of the research repository.

## Contents

- `wuxia_corpus/`: corpus segmentation, alignment, and similarity tools.
- `preprocessing/`: filtering, selection, conversion, and corpus analysis.
- `src/NMT/`: neural machine translation training and evaluation.
- `src/SMT/`: statistical machine translation experiments.
- `src/LLM/`: large language model inference, fine-tuning, and evaluation.

The Jupyter notebooks have been cleared of outputs, execution counters, and
machine-specific metadata. Personal paths, cluster account names, email
addresses, and credentials have been removed. The `requirements.txt` file is
included to document the environment used for the experiments.

## Data and generated artifacts

Copyrighted source texts, PDFs, processed corpora, model checkpoints, logs,
translations, figures, and evaluation outputs are intentionally excluded.
Place locally obtained data in the paths described by
`wuxia_corpus/config.yaml`, or update the configuration for your environment.
Preprocessed Hugging Face datasets are expected under `processed_data/`.

The expected top-level working directories are:

```text
models/
processed_data/
results/
evaluation/
logs/
```

These directories are created by the relevant scripts when possible. Scripts
that consume existing datasets or checkpoints require those inputs to be
provided before execution.

## Running the code

Create and activate a Python environment, then install the recorded
dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run commands from the root of this directory so that all relative paths resolve
consistently. Examples for corpus construction are available in
`wuxia_corpus/README.md`.

The SLURM examples in `src/LLM/hpc/slurm/` are generic templates. Set
`WUXIA_CONDA_ENV` to the desired Conda environment and, for gated Hugging Face
models, export `HUGGING_FACE_HUB_TOKEN` in the submission environment. Create
the `logs/` directory before submitting a job.

## Alignment

The corpus package uses its own monotonic dynamic-programming aligner. It
supports N-M transitions in both directions, with configurable maximum group
sizes, and does not bundle external alignment projects.

