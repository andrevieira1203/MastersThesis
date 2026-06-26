import os, json, numpy as np
from datasets import Dataset
import wandb
from transformers import (
    AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback,
)
import torch
import torch.nn as nn
from transformers import AutoModel

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "xlm-roberta-base"

# ALL available training data — weighted by embedding similarity to PT
TRAIN_FILES = {
    "json/train_it.json":    "IT",        # 39.5% — most similar
    "json/train_it_pt.json": "IT→PT",     # translated IT — direct PT text
    "json/train_fr.json":    "FR",        # 17.0%
    "json/train_fr_pt.json": "FR→PT",     # translated FR — direct PT text
    "json/train_ru.json":    "RU",        # 13.2%
    "json/train_po.json":    "PL",        # 11.3%
    "json/train.json":       "EN",        # 10.8%
    "json/train_ge.json":    "DE",        # 8.2%
}

# Dev sets as additional training data
DEV_AS_TRAIN = {
    "json/dev.json":    "EN_dev",
    "json/dev_fr.json": "FR_dev",
    "json/dev_ge.json": "DE_dev",
    "json/dev_it.json": "IT_dev",
    "json/dev_po.json": "PL_dev",
    "json/dev_ru.json": "RU_dev",
}

# Hold out EN dev for evaluation during training
EVAL_FILE = "json/dev.json"

OUTPUT_DIR = "outputs_v4"
MAX_LENGTH = 512
STRIDE     = 256

# ============================================================
# LABELS
# ============================================================

LABELS = [
    "Loaded_Language", "Name_Calling-Labeling", "Repetition", "Slogans",
    "Appeal_to_Fear-Prejudice", "Flag_Waving", "Causal_Oversimplification",
    "Appeal_to_Authority", "Appeal_to_Values", "Doubt", "Exaggeration-Minimisation",
    "Guilt_by_Association", "False_Dilemma-No_Choice", "Straw_Man", "Red_Herring",
    "Whataboutism", "Obfuscation-Vagueness-Confusion", "Appeal_to_Time",
    "Conversation_Killer", "Appeal_to_Popularity", "Appeal_to_Hypocrisy",
    "Consequential_Oversimplification", "Questioning_the_Reputation",
]
NUM_LABELS = len(LABELS)
NUM_OUTPUTS = NUM_LABELS * 2
label2idx = {l: i for i, l in enumerate(LABELS)}

print(f"Multi-label: {NUM_LABELS} techniques, {NUM_OUTPUTS} outputs/token")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"

# ============================================================
# DATA
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_windowed_examples(data, tokenizer, max_length=MAX_LENGTH, stride=STRIDE):
    all_examples = []
    for doc in data:
        text = doc["text"]
        spans = doc.get("spans", [])
        clean_spans = []
        for s in spans:
            st = max(0, int(s["start"]))
            en = min(len(text), int(s["end"]))
            lab = s["label"]
            if en > st and lab in label2idx:
                clean_spans.append((st, en, lab))

        full_enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
        full_ids = full_enc["input_ids"]
        full_offsets = full_enc["offset_mapping"]
        total_tokens = len(full_ids)

        if total_tokens <= max_length - 2:
            windows = [(0, total_tokens)]
        else:
            windows = []
            start = 0
            while start < total_tokens:
                end = min(start + max_length - 2, total_tokens)
                windows.append((start, end))
                if end == total_tokens:
                    break
                start += stride

        for (win_start, win_end) in windows:
            chunk_ids = full_ids[win_start:win_end]
            chunk_offsets = full_offsets[win_start:win_end]
            bos = tokenizer.bos_token_id or tokenizer.cls_token_id
            eos = tokenizer.eos_token_id or tokenizer.sep_token_id
            input_ids = [bos] + chunk_ids + [eos]
            attention_mask = [1] * len(input_ids)
            offsets_with_special = [(0, 0)] + chunk_offsets + [(0, 0)]

            seq_len = len(input_ids)
            labels = np.full((seq_len, NUM_OUTPUTS), 0, dtype=np.float32)
            special_mask = []
            for idx, (ts, te) in enumerate(offsets_with_special):
                if ts == 0 and te == 0:
                    labels[idx, :] = -100
                    special_mask.append(True)
                else:
                    special_mask.append(False)

            for (sp_start, sp_end, lab) in clean_spans:
                li = label2idx[lab]
                b_pos = 2 * li
                i_pos = 2 * li + 1
                first_token_found = False
                for idx, (ts, te) in enumerate(offsets_with_special):
                    if special_mask[idx]:
                        continue
                    if not (te <= sp_start or ts >= sp_end):
                        if not first_token_found:
                            labels[idx, b_pos] = 1.0
                            first_token_found = True
                        else:
                            labels[idx, i_pos] = 1.0

            all_examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels.tolist(),
            })
    return all_examples

# ============================================================
# COLLATOR
# ============================================================

class MultiLabelCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id or 0
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_ids, batch_mask, batch_labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [[-100.0] * NUM_OUTPUTS] * pad_len)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.float32),
        }

# ============================================================
# MODEL — per-output pos_weight
# ============================================================

class XLMRMultiLabelNER(nn.Module):
    def __init__(self, model_name, num_outputs, pos_weight=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(hidden, num_outputs)
        self.num_outputs = num_outputs

        # pos_weight: tensor of shape (num_outputs,) — one weight per binary output
        if pos_weight is not None:
            if isinstance(pos_weight, (list, np.ndarray)):
                self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
            else:
                # scalar fallback
                self.register_buffer("pos_weight", torch.tensor([pos_weight] * num_outputs, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            mask = (labels[:, :, 0] != -100).float()
            clean_labels = labels.clone()
            clean_labels[clean_labels == -100] = 0
            loss_fct = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)
            raw_loss = loss_fct(logits, clean_labels)
            mask_expanded = mask.unsqueeze(-1).expand_as(raw_loss)
            masked_loss = raw_loss * mask_expanded
            num_active = mask_expanded.sum()
            loss = masked_loss.sum() / (num_active + 1e-8)

        return {"loss": loss, "logits": logits}

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        self.encoder.config.save_pretrained(path)
        with open(os.path.join(path, "multilabel_config.json"), "w") as f:
            json.dump({
                "model_name": MODEL_NAME,
                "num_outputs": self.num_outputs,
                "labels": LABELS,
                "pos_weight": self.pos_weight.tolist() if self.pos_weight is not None else None,
            }, f)

    @classmethod
    def load_pretrained(cls, path):
        with open(os.path.join(path, "multilabel_config.json")) as f:
            config = json.load(f)
        pw = config.get("pos_weight")
        model = cls(config["model_name"], config["num_outputs"], pos_weight=pw)
        state = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model

# ============================================================
# TRAINER + METRICS
# ============================================================

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["labels"])
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["labels"])
        if prediction_loss_only:
            return (outputs["loss"], None, None)
        return (outputs["loss"], outputs["logits"], inputs["labels"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.3).astype(int)
    mask = labels[:, :, 0] != -100
    tp = fp = fn = 0
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if not mask[i, j]: continue
            for k in range(NUM_OUTPUTS):
                g = int(labels[i, j, k])
                p = int(preds[i, j, k])
                if g == 1 and p == 1: tp += 1
                elif g == 0 and p == 1: fp += 1
                elif g == 1 and p == 0: fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

# ============================================================
# PER-OUTPUT POS_WEIGHT
# ============================================================

def compute_per_output_pos_weight(train_examples, min_pw=10.0, max_pw=100.0):
    """Compute pos_weight separately for each of the 46 binary outputs."""
    pos_counts = np.zeros(NUM_OUTPUTS, dtype=np.float64)
    neg_counts = np.zeros(NUM_OUTPUTS, dtype=np.float64)

    for ex in train_examples:
        for token_labels in ex["labels"]:
            if token_labels[0] == -100:
                continue
            for k in range(NUM_OUTPUTS):
                if token_labels[k] > 0.5:
                    pos_counts[k] += 1
                else:
                    neg_counts[k] += 1

    pos_weights = np.ones(NUM_OUTPUTS, dtype=np.float64)
    for k in range(NUM_OUTPUTS):
        if pos_counts[k] > 0:
            ratio = neg_counts[k] / pos_counts[k]
            pos_weights[k] = np.clip(ratio, min_pw, max_pw)
        else:
            pos_weights[k] = max_pw  # never seen → max weight

    # Print summary
    print(f"\nPer-output pos_weight (B=begin, I=inside):")
    print(f"  {'Technique':<40} {'B-weight':>10} {'I-weight':>10} {'B-pos#':>10} {'I-pos#':>10}")
    print(f"  {'-'*80}")
    for li, label in enumerate(LABELS):
        b_pos = 2 * li
        i_pos = 2 * li + 1
        print(f"  {label:<40} {pos_weights[b_pos]:>10.1f} {pos_weights[i_pos]:>10.1f} "
              f"{pos_counts[b_pos]:>10.0f} {pos_counts[i_pos]:>10.0f}")

    global_pos = pos_counts.sum()
    global_neg = neg_counts.sum()
    print(f"\n  Total positives: {global_pos:,.0f} / {global_pos + global_neg:,.0f} "
          f"({100*global_pos/(global_pos+global_neg):.3f}%)")

    return pos_weights.tolist()

# ============================================================
# MAIN
# ============================================================

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Load ALL training data
    all_train_examples = []
    for path, lang in TRAIN_FILES.items():
        if not os.path.isfile(path):
            print(f"  SKIP {path} (not found)")
            continue
        data = load_json(path)
        print(f"Loading {lang}: {len(data)} docs from {path}")
        examples = build_windowed_examples(data, tokenizer)
        print(f"  → {len(examples)} windows")
        all_train_examples.extend(examples)

    # Add dev sets as training data (except EN dev which we use for eval)
    for path, lang in DEV_AS_TRAIN.items():
        if path == EVAL_FILE:
            continue  # keep EN dev for evaluation
        if not os.path.isfile(path):
            print(f"  SKIP {path} (not found)")
            continue
        data = load_json(path)
        print(f"Loading {lang}: {len(data)} docs from {path}")
        examples = build_windowed_examples(data, tokenizer)
        print(f"  → {len(examples)} windows")
        all_train_examples.extend(examples)

    print(f"\nTotal training windows: {len(all_train_examples)}")

    # Eval set
    data_dev = load_json(EVAL_FILE)
    ex_dev = build_windowed_examples(data_dev, tokenizer)
    print(f"Eval windows (EN dev): {len(ex_dev)}")

    # Compute per-output pos_weight
    print("\nComputing per-output pos_weight...")
    pw = compute_per_output_pos_weight(all_train_examples)

    # Create model
    model = XLMRMultiLabelNER(MODEL_NAME, NUM_OUTPUTS, pos_weight=pw)
    if device == "cuda":
        model = model.cuda()

    ds_train = Dataset.from_list(all_train_examples)
    ds_dev = Dataset.from_list(ex_dev)
    collator = MultiLabelCollator(tokenizer)

    wandb.init(
        project="bert-propaganda-improved",
        name="stage1_v4_all_langs",
        config={
            "model": MODEL_NAME,
            "approach": "multi-label-sigmoid-per-output-posweight",
            "languages": list(TRAIN_FILES.values()),
            "learning_rate": 3e-5,
            "epochs": 10,
            "batch_size": 8,
            "train_windows": len(all_train_examples),
            "per_output_posweight": True,
        }
    )

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "stage1"),
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_ratio=0.06,
        max_grad_norm=1.0,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="wandb",
        dataloader_num_workers=0,
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nBest dev metrics: {metrics}")

    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    wandb.finish()
    print(f"\nModel saved to: {final_dir}")

if __name__ == "__main__":
    train()
