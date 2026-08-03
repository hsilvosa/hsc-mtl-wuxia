# Anonymized Artifact Documentation

## 1. Document purpose and scope

This document describes the datasets, alignment software, trained-model
interfaces, and evaluation artifacts associated with the anonymous research
submission. It is intended to support artifact review and reproducibility. It
does not identify the researchers, their institution, user accounts, local
machines, or computing facilities.

Names and identifiers of public base models are retained because they are
necessary to reproduce the experiments and do not disclose the identity of the
authors. The descriptions below are based on the repository code and its
recorded corpus analysis. They do not claim that every referenced checkpoint,
adapter, generated translation, or source text is included in the release.

## 2. Artifact inventory

| Anonymous artifact | Description | Languages | Release status |
| --- | --- | --- | --- |
| `ANON-CORPUS-WUXIA-ZH-EN` | Parallel literary corpus constructed from four wuxia/xianxia collections | Chinese source, English target | Construction and preprocessing code documented; source texts may require separate lawful access |
| `ANON-ALIGN-NM` | LaBSE-based monotonic dynamic-programming aligner supporting N-M segment transitions | Chinese–English | Source code included |
| `ANON-NMT-*-WUXIA` | Fine-tuning pipelines for five encoder-decoder translation models | Chinese to English | Training and evaluation code included; generated checkpoints are separate artifacts |
| `ANON-SMT-IBM*-WUXIA` | IBM Models 1, 2, and 3 and their lexical translation tables | Chinese to English | Training and inference code included; serialized tables are separate artifacts |
| `ANON-LLM-*-WUXIA` | Prompt-based evaluation and optional parameter-efficient adaptation of public instruction models | Chinese to English | Experimental code included; availability of each adapter or output must be reported separately |
| `ANON-EVAL-WUXIA` | Automatic evaluation scripts, prompt templates, timing records, and plotting utilities | Primarily English outputs with Chinese sources | Code included; result files are separate artifacts |

The term *model* in this document can therefore mean either a public base
checkpoint referenced by the code or a derived checkpoint/adapter produced by
the project. These must not be treated as equivalent. Public model identifiers
name upstream dependencies; anonymous identifiers name the derived research
artifacts.

## 3. Intended task and use

The primary task is automatic translation of Chinese wuxia and xianxia prose
into English. The artifacts are intended for research on domain adaptation,
translation quality, prompt sensitivity, few-shot learning, model scaling, and
the comparison of statistical, neural, and large-language-model approaches.

Appropriate uses include controlled academic evaluation, error analysis,
literary machine-translation research, and assisted drafting where a qualified
human translator reviews the output. The artifacts are not designed as a
general Chinese–English translation service, a source of historical facts, or
an authoritative translator of legal, medical, safety-critical, or personal
communications.

## 4. Data coverage

### 4.1 Domain and sources

The corpus is narrow-domain literary data. It contains four collections marked
in the repository as AWE, CONDOR, GU, and ISSTH. They cover modern web-novel
fantasy and classic or classic-inspired martial-arts fiction. The material is
not a balanced sample of Chinese literature, contemporary Chinese usage, or
general translation domains.

The recorded corpus analysis reports the following aggregate volumes for the
main aligned corpus:

| Side | Tokens | Unique types | Standardized TTR, 1,000-token windows | Lexical density |
| --- | ---: | ---: | ---: | ---: |
| Chinese source | 6,938,915 | 133,374 | 53.53% | 57.58% |
| English target | 9,351,283 | 28,500 | 39.97% | 63.81% |

These figures are values recorded by the repository analysis, not a new audit
performed for this document. Tokenization differs by language, so direct
cross-language comparisons must be interpreted cautiously. A separate
reference dataset is also analyzed in the repository, but its provenance and
licensing require independent documentation before distribution.

### 4.2 Languages and varieties

- Source language: written Chinese represented with Han characters. The code
  does not provide a systematic audit of simplified versus traditional script,
  regional variety, or diachronic variation.
- Target language: written English in literary or web-fiction translation
  style. The corpus does not document a controlled balance among national or
  regional English varieties.
- Translation direction: Chinese to English. Reverse translation is outside
  the documented task, even where an upstream multilingual model technically
  supports it.
- Other languages: multilingual base models may have broader pretraining, but
  the project data and reported evaluation do not establish performance in
  other languages.

### 4.3 Unit of data and alignment

Texts are segmented using Chinese and English punctuation and sentence
boundaries. The built-in aligner embeds contiguous groups with LaBSE and finds
a monotonic path with dynamic programming. It can align any source group of
size N to any target group of size M within independently configured limits;
the default limit is seven segments on each side. Source-only and target-only
transitions represent skipped material.

Aligned pairs are cleaned into `zh` and `en` fields. The cleaning pipeline
retains letters, combining marks, spaces, and selected punctuation, removes
empty pairs, and shuffles examples with a recorded seed. The default basic
split is 80% training, 10% validation, and 10% test.

A selected 100,000-example variant uses mean chrF and cross-model score
variance as proxies for difficulty. Examples below a mean-score threshold of
20 are excluded. The remaining candidates are sampled across three strata:
lower-score/low-variance, high-variance, and higher-score/low-variance. The
selected examples are then divided 80/10/10 while approximately preserving the
three strata. This procedure balances model-estimated difficulty, not topic,
author, demographic group, or linguistic phenomenon.

## 5. Linguistic phenomena represented

The following phenomena are expected from the documented domain and are
explicitly addressed by preprocessing, prompts, or corpus analysis:

| Phenomenon | Coverage and evidence | Annotation status |
| --- | --- | --- |
| Literary narration and dialogue | Long-form fiction is segmented into narrative and quoted material | Present but not labeled by discourse type |
| Wuxia/xianxia terminology | Martial arts, cultivation concepts, sects, techniques, ranks, and culturally specific items are central to the task prompts | Present but no complete terminology inventory is supplied |
| Personal names, place names, and titles | Some prompts request pinyin romanization rather than semantic translation | No named-entity or transliteration labels |
| Idioms and cultural references | Prompt variants explicitly test literal, adaptive, formal, and literary treatment; corpus analysis discusses four-character idioms | No idiom-level gold annotations |
| Register and style | Outputs range from close, word-order-preserving translation to modern fantasy adaptation and literary prose | Controlled by prompt, not independently annotated |
| Lexical diversity and normalization | TTR, standardized TTR, Guiraud, MTLD, HD-D, lexical density, and hapax statistics are computed | Aggregate measurements only |
| Length expansion | The recorded corpus analysis reports an English-to-Chinese token ratio of approximately 1.35 | Aggregate observation; tokenization-dependent |
| Many-to-many sentence correspondence | N-M alignment represents sentence splitting, merging, omissions, and additions | Automatically inferred, not manually validated at full scale |

Coverage is not established for conversational speech, social media, technical
writing, legal or medical language, learner language, speech disfluencies,
systematic code-switching, dialect transcription, or accessibility-oriented
language. Performance on these phenomena must not be inferred from the
reported experiments.

## 6. Demographic and cultural representation

No demographic labels or demographic audit are present in the repository. The
data describes fictional characters rather than recruited human participants.
Consequently, the artifact does not establish coverage rates for gender, age,
disability, socioeconomic status, ethnicity, nationality, religion, sexual
orientation, or other protected or socially salient groups.

The domain is culturally concentrated in Chinese martial-arts and cultivation
fiction. It may contain fictionalized kinship structures, sect and clan
hierarchies, social rank, violence, romance, spiritual or religious imagery,
and historical or pseudo-historical institutions. Their presence must not be
interpreted as representative evidence about real contemporary populations.
English translations may also reflect the choices and biases of translators,
fan-translation communities, editors, or source platforms, none of which are
systematically documented here.

No claim of demographic fairness is made. Any future fairness study should
define relevant groups, create ethically reviewed annotations, evaluate names
and honorifics separately, inspect gendered pronoun resolution, and use human
reviewers familiar with both source culture and target-language translation.

## 7. Model coverage

### 7.1 Neural machine translation models

The unified NMT pipeline references the following public base checkpoints:

| Anonymous derived artifact | Public base checkpoint | Model type |
| --- | --- | --- |
| `ANON-NMT-MARIAN-WUXIA` | `Helsinki-NLP/opus-mt-zh-en` | Bilingual encoder-decoder MT |
| `ANON-NMT-M2M100-WUXIA` | `facebook/m2m100_418M` | Multilingual encoder-decoder MT |
| `ANON-NMT-MBART-WUXIA` | `facebook/mbart-large-50-many-to-many-mmt` | Multilingual denoising encoder-decoder |
| `ANON-NMT-MT5-WUXIA` | `google/mt5-small` | Multilingual text-to-text encoder-decoder |

Default training configuration uses seed 42, maximum source and target lengths
of 128 tokens, batch size 16, ten epochs, learning rate `2e-5`, weight decay
`0.01`, and early stopping after three validation epochs without improvement.
These are code defaults and should be replaced by the actual run manifest when
reporting a specific checkpoint.

### 7.2 Statistical machine translation models

The SMT baseline implements IBM Models 1, 2, and 3 using English whitespace
tokenization and Jieba segmentation for Chinese. The saved artifact is a
lexical translation-probability table rather than a complete phrase-based MT
system. It does not include a language model, reordering decoder, or explicit
terminology constraints, so its output is primarily a lexical baseline.

### 7.3 Large language models

Scripts and exploratory notebooks reference the following public model
families and variants:

- Qwen: 
  Qwen3-30B-A3B-Instruct-2507.
- Llama: Llama-3.3-70B-Instruct.
- Gemma:  Gemma 3 27B.
- GLM: GLM-4-32B-0414.

This list records coverage in code, not proof that every run completed. Several
training scripts support four-bit QLoRA-style adaptation with rank 16, alpha
32, dropout 0.05, learning rate `2e-4`, and one epoch. Derived adapters should
be identified with their exact base revision, dependency versions, effective
training sample count, random seed, and checksum before release.

## 8. Prompt and evaluation coverage

Five principal prompt conditions are documented: neutral translation,
adaptive modern-fantasy prose, literary translation, strict seq2seq-like
fidelity, and formal culturally sensitive translation. Most LLM scripts can
evaluate 0, 1, 2, 3, 5, and 10 in-context examples. Some exploratory scripts
use smaller subsets or only zero-shot evaluation, so result files must record
the precise prompt IDs, shot counts, sample count, seed, and model revision.

Automatic metrics implemented across the project include SacreBLEU, chrF or
chrF++, TER, ROUGE-L, METEOR, BERTScore, and COMET. Runtime is also recorded for
some LLM experiments. These metrics primarily measure overlap or learned
semantic similarity and do not directly establish literary quality, cultural
appropriateness, factual fidelity, or reader preference. Human bilingual
evaluation is needed for those properties.

One LLM evaluation path appears to construct the COMET source field from the
English reference rather than the original Chinese source. COMET scores from
that path should be treated as unverified until the source field is corrected
and the evaluation is rerun. Metric implementations and model versions should
also be frozen in a release manifest because library defaults can change.

## 9. Known limitations and risks

- The corpus is highly domain-specific and cannot establish general-domain
  translation quality.
- Segment-level random splits may place passages from the same novel or chapter
  in different splits. Results therefore may not measure document- or
  work-level generalization, and near-duplicate leakage has not been ruled out.
- Public base models may already have encountered the source works or their
  translations during pretraining. No contamination audit is documented.
- Automatic N-M alignment can merge incorrect passages, omit content, or assign
  a high embedding similarity to semantically different text. Full manual
  validation is not reported.
- Cleaning can remove digits, symbols, or boundary punctuation and may alter
  names, quantities, formatting, or stylistic signals.
- The NMT default length of 128 tokens truncates longer examples unless the
  configuration is changed.
- Difficulty-balanced selection is based on model scores. It can amplify the
  biases shared by the scoring models and does not guarantee balanced topics or
  linguistic constructions.
- Few-shot examples are selected deterministically from the training split in
  several scripts; prompt results may depend strongly on those particular
  examples.
- Literary translations can contain violence, coercion, stereotypes, archaic
  social roles, or other sensitive content inherited from the source material.
- Copyright and distribution permissions for source works and translations
  must be reviewed separately. Code availability does not imply permission to
  redistribute all underlying text or upstream weights.
- Upstream model licenses and acceptable-use policies remain applicable to
  each derived model or adapter.

## 10. Recommended reporting for each released checkpoint

Each model release should add a machine-readable manifest containing:

1. Anonymous artifact identifier and SHA-256 checksum.
2. Exact public base checkpoint and revision.
3. Training-data version and split hashes, without local absolute paths.
4. Number of training, validation, and test examples actually used.
5. Hyperparameters, seed, precision, quantization, and stopping criterion.
6. Software and accelerator configuration at a non-identifying level.
7. Evaluation script version, prompt IDs, shot counts, decoding parameters,
   metric versions, and complete results.
8. Known failed runs or deviations from the default configuration.
9. License and access conditions for code, weights, and data as separate items.

## 11. Anonymization statement

This document intentionally omits author names, affiliations, email addresses,
usernames, local absolute paths, cluster or facility names, repository remotes,
private dataset locations, credentials, and unpublished model-hosting
accounts. Public model and corpus labels are retained only where needed to
describe scientific coverage and reproducibility. No hidden author identifier
is intended in the anonymous artifact names.

## 12. Repository evidence used for this documentation

The description was derived from the corpus README and configuration,
preprocessing and difficulty-selection scripts, NMT configuration and training
scripts, SMT baseline, LLM prompt and training scripts, and automatic
evaluation code. This document should be updated if the released artifact set
differs from the inspected repository state.
