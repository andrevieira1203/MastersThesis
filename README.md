# Language Models for the Detection of Manipulative Discourse and Disinformation in Text

This is the official repository of the MSc thesis **"Language Models for the Detection of Manipulative Discourse and Disinformation in Text"**, developed at [FCUP](https://sigarra.up.pt/fcup/) / [INESC TEC](https://www.inesctec.pt/).

PersuasionLens is a two-stage framework for **automatic detection and classification of 23 persuasion techniques at the text-span level** in Portuguese news articles, based on the [CLEF-2024 CheckThat! Lab Task 3](https://checkthat.gitlab.io/clef2024/task3/) formulation. The system relies entirely on **cross-lingual transfer** from six source languages, as no Portuguese training data exists. Our best model achieves **F1-micro = 0.134**, surpassing both the competition winner UniBO (0.107) and the organizers' post-competition system PersuasionMultiSpan (0.132).

---

## 1. Project Overview

The spread of persuasion and propaganda techniques in news media poses a significant challenge to information integrity. This project addresses the automatic identification of these techniques in Portuguese, a low-resource language for this task. The system combines multilingual transformer models with a novel embedding-based language selection strategy to maximize cross-lingual transfer.

The pipeline operates in two stages:

1. **Stage 1 (Multilingual Base)** — trains on annotated data from all six available languages (EN, FR, DE, IT, PL, RU) plus machine-translated versions, learning cross-lingual representations of persuasion patterns.
2. **Stage 2 (Portuguese Fine-tuning)** — fine-tunes on Italian-to-Portuguese translated data (the linguistically closest source), adapting the model specifically for Portuguese text.

Together, these stages enable detection of 23 persuasion techniques without any native Portuguese training data.

---

## 2. Key Features

- **Multi-Label Architecture:** Custom XLM-RoBERTa with 46 sigmoid outputs per token (B+I for each of 23 techniques), handling the ~20% multi-label overlap in gold annotations that BIO-softmax approaches miss.
- **Embedding-Based Language Selection:** Novel analysis using `paraphrase-multilingual-MiniLM-L12-v2` demonstrating that Italian is the closest language to Portuguese (39.5% of nearest neighbours), informing the training strategy.
- **Two-Stage Training Pipeline:** Multilingual base training followed by targeted Portuguese fine-tuning with frozen embeddings.
- **Per-Output Class Weighting:** Individual `pos_weight` per technique (ranging 10-100) in `BCEWithLogitsLoss`, addressing the extreme class imbalance (~0.7% positive tokens).
- **State-of-the-Art Results:** F1-micro = 0.134, surpassing UniBO (+25% relative) and PersuasionMultiSpan (+1.4% absolute).

---

## 3. Results

### Main Results

| System | F1-micro | F1-macro | Notes |
|---|---|---|---|
| **PersuasionLens (Ours)** | **0.134** | **0.103** | XLM-RoBERTa, IT→PT Stage 2, global thr=0.70 |
| PersuasionMultiSpan (Organizers) | 0.132 | 0.120 | Post-competition system |
| UniBO (1st place) | 0.107 | 0.073 | Competition winner |
| Baseline (zero-shot) | 0.002 | — | XLM-RoBERTa zero-shot |

### Stage 2 Language Ablation

| Stage 2 Data | F1-micro | F1-macro |
|---|---|---|
| **IT→PT only** | **0.134** | **0.103** |
| IT→PT + FR→PT | 0.128 | 0.099 |
| IT→PT + FR→PT + RU→PT | 0.117 | 0.095 |

### Top Performing Techniques

| Technique | F1 | Gold Spans |
|---|---|---|
| Appeal_to_Time | 0.215 | 47 |
| Questioning_the_Reputation | 0.207 | 281 |
| Appeal_to_Values | 0.180 | 113 |
| Appeal_to_Fear-Prejudice | 0.153 | 102 |

---

## 4. Architecture

### Model

The core model (`XLMRMultiLabelNER`) consists of:

- **Encoder:** `xlm-roberta-base` (278M parameters)
- **Head:** `Dropout(0.15)` → `Linear(768, 46)` with sigmoid activation
- **Loss:** `BCEWithLogitsLoss` with per-output `pos_weight` (capped 10-100)
- **Inference:** Sliding window (512 tokens, 50% overlap) with probability averaging

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Multilingual Base Training                        │
│  Data: EN + FR + DE + IT + PL + RU + IT→PT + FR→PT + devs  │
│  ~10,000 windows │ LR: 3e-5 │ 10 epochs │ All params       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 2: Portuguese Fine-tuning                            │
│  Data: IT→PT (302 docs, 919 windows)                        │
│  LR: 1e-5 │ 15 epochs │ Embeddings frozen                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Inference: Sliding Window + Post-Processing                │
│  512 tokens │ 50% overlap │ Global threshold: 0.70          │
└─────────────────────────────────────────────────────────────┘
```

### Embedding Analysis

We used `paraphrase-multilingual-MiniLM-L12-v2` to embed all training documents and Portuguese gold into the same vector space. For each of the 104 Portuguese test documents, we identified the 20 nearest training documents by cosine similarity:

| Language | Nearest Neighbours (%) | Linguistic Family |
|---|---|---|
| Italian (IT) | **39.5%** | Romance |
| French (FR) | 17.0% | Romance |
| Russian (RU) | 13.2% | Slavic |
| Polish (PL) | 11.3% | Slavic |
| English (EN) | 10.8% | Germanic |
| German (DE) | 8.2% | Germanic |

**Key finding:** Linguistic proximity outweighs data quantity. English has the largest dataset (9,002 spans) but ranks 5th in proximity. Italian, with a similar-sized dataset (7,961 spans), dominates with 39.5%.

---

## 5. Technology Stack

### Language
- **Python 3.9+**

### Core Frameworks
- **PyTorch** — Deep learning backend
- **Transformers (Hugging Face)** — Pre-trained `xlm-roberta-base` encoder
- **Datasets (Hugging Face)** — Dataset management and preprocessing
- **sentence-transformers** — Multilingual document embeddings for language analysis

### Translation & Alignment
- **DeepL API** — Machine translation (RU→PT)
- **Google Translate** — Machine translation (IT→PT, FR→PT)
- **LinguAligner** — Word alignment for span reprojection after translation

### Utilities
- **NumPy** — Numerical operations
- **Weights & Biases** — Experiment tracking
- **openpyxl** — Results export to Excel

### Hardware
- Training performed on NVIDIA GTX 1050 Ti (4GB VRAM) / CPU fallback
- Translation server: Ubuntu with NVIDIA GPU

---

## 6. Repository Structure

```
├── embeddings/
│   ├── embedding_analysis.py          # Document embedding analysis
├── models_improved/
│   ├── train_model_base_v4.py         # Stage 1: multilingual base training
│   ├── train_it_pt_finetune_v4_it.py     # Stage 2: Portuguese fine-tuning
│   ├── testing.py                     # Inference + evaluation (Algorithm 1)
├── outputs_v4/
│   └── pt_finetuned_only_it/
│       └── best_model/                # Best trained model (IT→PT only Stage 2)
│           ├── model.pt               # NOT included — see Section 11 (Hugging Face)
│           ├── multilabel_config.json
│           └── tokenizer files
├── results/
│   └── results_persuasion_detection.xlsx  # All results across models
├── translation/
│   ├── translate_it_to_pt.py          # IT→PT translation (Google + LinguAligner)
│   ├── translate_fr_to_pt.py          # FR→PT translation (Google + LinguAligner)
│   ├── translate_ru_to_pt.py          # RU→PT translation (DeepL + LinguAligner)
└── .gitignore
└── README.md
└── requirements.txt
```
---

## 7. Usage

### Training

#### Stage 1: Multilingual Base

```bash
python models_improved/train_model_base_v4.py
```

Trains on all six languages plus translations and dev sets (~10,000 windows). Outputs to `outputs_v4/final_model/`.

#### Stage 2: Portuguese Fine-tuning

```bash
python models_improved/train_it_pt_finetune_v4_it.py
```

Fine-tunes on IT→PT data with frozen embeddings. Outputs to `outputs_v4/pt_finetuned_only_it/best_model/`.

### Inference & Evaluation

```bash
python models_improved/testing.py
```

Runs inference on the Portuguese test articles, performs threshold sweep, and evaluates using Algorithm 1 (partial span matching with greedy assignment).

### Embedding Analysis

```bash
pip install sentence-transformers umap-learn
python embeddings/embedding_analysis.py
```

Generates UMAP visualizations and nearest-neighbour language distribution analysis.

---

## 8. Evaluation Metric

The evaluation follows Algorithm 1 from the [CLEF-2024 Task 3 paper](https://checkthat.gitlab.io/clef2024/task3/), implementing partial span matching:

- Predictions and gold spans are matched greedily by maximum overlap score
- A prediction receives full credit if it overlaps ≥50% of the gold span and is ≤2x the gold length
- Longer predictions are penalized proportionally
- F1-micro and F1-macro are computed across all 23 techniques

---

## 9. Task & Data Description

### CLEF-2024 CheckThat! Lab Task 3

The task requires detecting spans of text that employ persuasion techniques and classifying them into one of 23 categories, organized into 6 coarse categories:

| Category | Techniques |
|---|---|
| Attack on Reputation | Name Calling, Doubt, Appeal to Hypocrisy, Guilt by Association, Questioning the Reputation |
| Justification | Appeal to Authority, Appeal to Fear/Prejudice, Appeal to Popularity, Appeal to Values, Flag Waving |
| Simplification | Causal Oversimplification, Consequential Oversimplification, False Dilemma |
| Distraction | Red Herring, Straw Man, Whataboutism |
| Call | Appeal to Time, Conversation Killer, Slogans |
| Manipulative Wording | Exaggeration/Minimisation, Loaded Language, Obfuscation, Repetition |

### Data Availability

The training data used in this project is provided by the CLEF-2024 CheckThat! Lab organizers and is not publicly redistributable due to licensing restrictions. To obtain the data, please refer to the [official CLEF-2024 CheckThat! Lab page](https://checkthat.gitlab.io/clef2024/task3/) and follow the data request procedure. The machine-translated versions (IT→PT, FR→PT, RU→PT) can be reproduced using the scripts in the `translation/` folder once the original data is obtained.

### Challenge

- **No Portuguese training data** — the system must rely entirely on cross-lingual transfer
- Training data available in 6 languages: EN (536 docs), FR (211), DE (177), IT (303), PL (194), RU (191)
- Gold test: 104 Portuguese articles, 1,727 annotated spans
- 4 techniques are absent from English data but represent 33.6% of Portuguese gold

---

## 10. Model Weights

The trained model (Stage 2, IT→PT only, F1-micro = 0.134) is hosted on the **Hugging Face Model Hub**:

- [`AndreCVieira/persuasion-lens-pt`](https://huggingface.co/AndreCVieira/persuasion-lens-pt)

```python
from huggingface_hub import hf_hub_download
import torch, json
from transformers import AutoTokenizer, AutoModel

repo_id = "AndreCVieira/persuasion-lens-pt"
config_path = hf_hub_download(repo_id, "multilabel_config.json")
model_path = hf_hub_download(repo_id, "model.pt")
tokenizer = AutoTokenizer.from_pretrained(repo_id)

with open(config_path) as f:
    config = json.load(f)
```

---

## 11. Reporting Issues

Please report any issues or bugs through the [GitHub issue tracker](https://github.com/andrevieira1203/MastersThesis/issues).

When reporting an issue, please include:
- Python version
- PyTorch version
- Complete error message and stack trace
- Steps to reproduce

---

## 12. License

This project is developed as part of an MSc thesis at FCUP / INESC TEC.

The training data is provided by the CLEF-2024 CheckThat! Lab organizers and is not included in this repository due to licensing restrictions. See Section 9 for access instructions.

---

## 13. Acknowledgments

- **Supervisors:** Nuno Guimarães, Alípio Jorge
- **Institutions:** [FCUP](https://sigarra.up.pt/fcup/), [INESC TEC](https://www.inesctec.pt/)
- **CLEF-2024 CheckThat! Lab** organizers for the task formulation and data
- **Hugging Face** for the Transformers library and pre-trained models
- **DeepL** for translation API access

---

## 14. Citation

If you use this work, please cite:

```bibtex
@mastersthesis{vieira2026persuasionlens,
  title={Language Models for the Detection of Manipulative Discourse and Disinformation in Text},
  author={Vieira, André},
  school={Faculdade de Ciências da Universidade do Porto},
  year={2026},
  type={MSc Thesis}
