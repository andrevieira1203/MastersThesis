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

MODEL_BASE    = "outputs_v4/final_model"
DATA_PT_TRAIN = "json/train_it_pt.json"
DATA_PT_DEV   = "json/checkthat24_pt.json"
OUTPUT_DIR    = "outputs_v4/pt_finetuned_only_it"

MAX_LENGTH = 512
STRIDE     = 256

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================
# MODEL (same as v4 stage 1)
# ============================================================

class XLMRMultiLabelNER(nn.Module):
    def __init__(self, model_name, num_outputs, pos_weight=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(hidden, num_outputs)
        self.num_outputs = num_outputs
        if pos_weight is not None:
            if isinstance(pos_weight, (list, np.ndarray)):
                self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
            else:
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
                "model_name": "xlm-roberta-base",
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
        model.load_state_dict(state)
        return model

# ============================================================
# LOAD MODEL
# ============================================================

model = XLMRMultiLabelNER.load_pretrained(MODEL_BASE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True)

for p in model.encoder.embeddings.parameters():
    p.requires_grad = False

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

if device == "cuda":
    model = model.cuda()

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
# COLLATOR + TRAINER + METRICS
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
# MAIN
# ============================================================

def main():
    # Stage 2: fine-tune on PT-translated data (IT→PT)
    all_pt_train = []
    for path, lang in [(DATA_PT_TRAIN, "IT→PT")]:
        if os.path.isfile(path):
            data = load_json(path)
            print(f"{lang}: {len(data)} docs")
            all_pt_train.extend(data)

    data_dev = load_json(DATA_PT_DEV)
    print(f"PT dev: {len(data_dev)} docs")
    print(f"Total PT train docs: {len(all_pt_train)}")

    train_examples = build_windowed_examples(all_pt_train, tokenizer)
    dev_examples = build_windowed_examples(data_dev, tokenizer)
    print(f"Train windows: {len(train_examples)}, Dev windows: {len(dev_examples)}")

    ds_train = Dataset.from_list(train_examples)
    ds_dev = Dataset.from_list(dev_examples)
    collator = MultiLabelCollator(tokenizer)

    wandb.init(
        project="bert-propaganda-improved",
        name="stage2_model2_IT",
        config={
            "base_model": MODEL_BASE,
            "learning_rate": 1e-5,
            "epochs": 15,
            "frozen": "embeddings_only",
            "pt_train_data": "IT→PT",
        }
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=15,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
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
    print(f"\nBest PT dev metrics: {metrics}")

    final_dir = os.path.join(OUTPUT_DIR, "best_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    wandb.finish()
    print(f"\nSaved to: {final_dir}")

if __name__ == "__main__":
    main()
