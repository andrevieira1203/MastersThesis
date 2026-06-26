import json, os
from tqdm import tqdm
import deepl
from LinguAligner import AlignmentPipeline

# ============================================================
# CONFIG 
# ============================================================

INPUT_FILE    = r"C:C:\Users\andre\Documents\Faculdade\Mestrado\Tese\cenas\json\train_ru.json"
OUTPUT_FILE   = r"C:C:\Users\andre\Documents\Faculdade\Mestrado\Tese\cenas\json\train_ru_pt.json"
PROGRESS_FILE = r"C:C:\Users\andre\Documents\Faculdade\Mestrado\Tese\cenas\json\train_ru_pt_progress.json"

DEEPL_API_KEY = "a0a8757f-efad-45ca-9705-46f65414e2be:fx"  

SRC_LANG = "RU"
TGT_LANG = "PT-PT"  # português europeu

MAX_CHARS = 4500

ALIGN_CONFIG = {
    "pipeline": ["lemma", "word_aligner", "gestalt", "levenshtein"],
    "spacy_model": "pt_core_news_lg",
    "WAligner_model": "bert-base-multilingual-uncased"
}

# ============================================================
# TRANSLATOR — usa deepl diretamente, ignora LinguAligner translator
# ============================================================

translator = deepl.Translator(DEEPL_API_KEY)

def translate_long_text(text, max_chars=MAX_CHARS):
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current += p + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            current = p + "\n\n"
    if current.strip():
        chunks.append(current.strip())

    translated_chunks = []
    for chunk in chunks:
        result = translator.translate_text(chunk, source_lang=SRC_LANG, target_lang=TGT_LANG)
        translated_chunks.append(result.text)

    return "\n\n".join(translated_chunks)

# ============================================================
# ALIGNER
# ============================================================

aligner = AlignmentPipeline(ALIGN_CONFIG)

# ============================================================
# LOAD DATA
# ============================================================

with open(INPUT_FILE, encoding="utf-8") as f:
    data = json.load(f)
print(f"Artigos carregados: {len(data)}")

# Resume from progress if exists
already_done = {}
if os.path.isfile(PROGRESS_FILE):
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        already_done = {d["id"]: d for d in json.load(f)}
    print(f"A retomar: {len(already_done)} artigos já processados")

translated_data = list(already_done.values())
total_spans = 0
aligned_spans = 0

# ============================================================
# PROCESS
# ============================================================

for article in tqdm(data, desc="RU → PT"):
    if article["id"] in already_done:
        continue

    src_text = article["text"]
    spans = article.get("spans", [])

    tgt_text = translate_long_text(src_text)

    new_spans = []
    for span in spans:
        total_spans += 1
        src_ann = src_text[span["start"]: span["end"]]
        try:
            aligned = aligner.align_annotation(src_text, src_ann, tgt_text, src_ann)
        except Exception:
            continue
        if aligned is None:
            continue
        aligned_text, (start, end) = aligned[0], aligned[1]
        if start >= end or end > len(tgt_text):
            continue
        aligned_spans += 1
        new_spans.append({"start": start, "end": end, "label": span["label"]})

    article_out = {"id": article["id"], "text": tgt_text, "spans": new_spans}
    translated_data.append(article_out)

    # Save progress after every article
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

# ============================================================
# FINAL SAVE
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)

print("\nRESUMO")
print(f"Artigos: {len(translated_data)}")
print(f"Spans originais: {total_spans}")
print(f"Spans alinhados: {aligned_spans}")
print(f"Taxa retenção: {aligned_spans / max(1, total_spans):.2%}")
print(f"Output: {OUTPUT_FILE}")