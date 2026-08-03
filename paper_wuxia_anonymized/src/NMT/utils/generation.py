import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def generate_text(
    model,
    tokenizer,
    texts,
    max_length: int = 128,
    num_beams: int = 4,
    batch_size: int = 8,
):
    device = next(model.parameters()).device
    model.eval()
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **encoded,
                max_length=max_length,
                num_beams=num_beams,
            )
        preds.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    return preds


def decode_ids_to_text(dataset, id_col: str):
    return [row[id_col] for row in dataset]


def compute_rougeL_f1(hyp_list, ref_list):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(ref_list, hyp_list)]
    return sum(scores) / len(scores) if scores else 0.0


def compute_meteor(hyp_list, ref_list):
    from nltk.translate.meteor_score import meteor_score
    import nltk
    try:
        nltk.data.find("wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    return sum(meteor_score([r], h) for r, h in zip(ref_list, hyp_list)) / len(hyp_list)


def compute_bleu(hyp_list, ref_list):
    import sacrebleu
    return sacrebleu.corpus_bleu(hyp_list, [ref_list]).score


def compute_chrf(hyp_list, ref_list):
    import sacrebleu
    return sacrebleu.corpus_chrf(hyp_list, [ref_list]).score


def compute_ter(hyp_list, ref_list):
    import sacrebleu
    return sacrebleu.corpus_ter(hyp_list, [ref_list]).score


@torch.inference_mode()
def comet_score_batch(
    batch_src: list[str],
    batch_hyp: list[str],
    batch_ref: list[str],
    model,
    tokenizer,
    device,
    max_length: int = 128,
):
    inputs = []
    for src, hyp, ref in zip(batch_src, batch_hyp, batch_ref):
        inputs.append(f"SRC: {src}\nREF: {ref}\nHYP: {hyp}")
    encoded = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = model(**encoded)
    logits = outputs.logits
    # Assume binary classification: last dim size 2
    probs = torch.softmax(logits, dim=-1)[..., 1].float()
    return probs.cpu().tolist()


_comet_model = None


def get_comet_model():
    global _comet_model
    if _comet_model is None:
        from comet import load_from_checkpoint
        _comet_model = load_from_checkpoint("Unbabel/wmt22-comet-da")
    return _comet_model


def comet_score_batch(
    batch_src: list[str],
    batch_hyp: list[str],
    batch_ref: list[str],
    model,
    tokenizer,
    device,
    max_length: int = 128,
):
    inputs = []
    for src, hyp, ref in zip(batch_src, batch_hyp, batch_ref):
        inputs.append(f"SRC: {src}\nREF: {ref}\nHYP: {hyp}")
    encoded = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = model(**encoded)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[..., 1].float()
    return probs.cpu().tolist()


def compute_comet_from_model(
    src_list: list[str],
    hyp_list: list[str],
    ref_list: list[str],
    model,
    tokenizer,
    device,
    max_length: int = 128,
    batch_size: int = 8,
):
    scores = []
    for i in tqdm(range(0, len(src_list), batch_size), desc="COMET"):
        scores.extend(
            comet_score_batch(
                src_list[i : i + batch_size],
                hyp_list[i : i + batch_size],
                ref_list[i : i + batch_size],
                model,
                tokenizer,
                device,
                max_length=max_length,
            )
        )
    return sum(scores) / len(scores) if scores else 0.0


def compute_comet(src_list: list[str], hyp_list: list[str], ref_list: list[str]):
    model = get_comet_model()
    data = [{"src": s, "ref": r, "mt": h} for s, r, h in zip(src_list, ref_list, hyp_list)]
    seg_scores, _ = model.predict(data)
    return sum(seg_scores) / len(seg_scores) if seg_scores else 0.0


def compute_all_metrics(src_list, hyp_list, ref_list):
    return {
        "bleu": compute_bleu(hyp_list, ref_list),
        "chrf": compute_chrf(hyp_list, ref_list),
        "ter": compute_ter(hyp_list, ref_list),
        "rougeL_f1": compute_rougeL_f1(hyp_list, ref_list),
        "meteor": compute_meteor(hyp_list, ref_list),
        "comet": compute_comet(src_list, hyp_list, ref_list),
    }
