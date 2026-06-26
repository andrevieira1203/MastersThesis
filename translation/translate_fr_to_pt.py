import json
from tqdm import tqdm
from LinguAligner import translation
from LinguAligner import AlignmentPipeline

# CONFIG
INPUT_FILE = r"C:\Users\andre\Documents\Faculdade\Mestrado\Tese\cenas\json\train_fr.json"
OUTPUT_FILE = r"C:\Users\andre\Documents\Faculdade\Mestrado\Tese\cenas\json\train_fr_pt.json"

SRC_LANG = "fr"
TGT_LANG = "pt"

TRANSLATION_METHOD = "google"  # "google", "deepl", "microsoft"

ALIGN_CONFIG = {
    "pipeline": ["lemma", "word_aligner", "gestalt", "levenshtein"],
    "spacy_model": "pt_core_news_lg",
    "WAligner_model": "bert-base-multilingual-uncased"
}

# TRANSLATOR
if TRANSLATION_METHOD == "google":
    translator = translation.GoogleTranslator(
        source_lang=SRC_LANG,
        target_lang=TGT_LANG
    )
elif TRANSLATION_METHOD == "deepl":
    translator = translation.DeepLTranslator(
        source_lang=SRC_LANG,
        target_lang=TGT_LANG,
        key="DEEPL_API_KEY"
    )
elif TRANSLATION_METHOD == "microsoft":
    translator = translation.MicrosoftTranslator(
        source_lang=SRC_LANG,
        target_lang=TGT_LANG,
        key="MICROSOFT_API_KEY"
    )
else:
    raise ValueError("Método de tradução inválido")

def translate_long_text(translator, text, max_chars=4500):
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current += p + "\n\n"
        else:
            chunks.append(current.strip())
            current = p + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    translated_chunks = []
    for chunk in chunks:
        translated_chunks.append(translator.translate(chunk))

    return "\n\n".join(translated_chunks)

# ALIGNER
aligner = AlignmentPipeline(ALIGN_CONFIG)

# LOAD DATA
with open(INPUT_FILE, encoding="utf-8") as f:
    data = json.load(f)

print(f"Artigos carregados: {len(data)}")

translated_data = []
total_spans = 0
aligned_spans = 0

# PROCESS
for article in tqdm(data, desc="FR → PT"):
    src_text = article["text"]
    spans = article.get("spans", [])

    tgt_text = translate_long_text(translator, src_text)

    new_spans = []

    for span in spans:
        total_spans += 1

        src_ann = src_text[span["start"]: span["end"]]

        try:
            aligned = aligner.align_annotation(
                src_text,
                src_ann,
                tgt_text,
                src_ann
            )
        except Exception:
            continue

        if aligned is None:
            continue

        aligned_text = aligned[0]
        start, end = aligned[1]

        if start >= end or end > len(tgt_text):
            continue

        aligned_spans += 1

        new_spans.append({
            "start": start,
            "end": end,
            "label": span["label"]
        })

    translated_data.append({
        "id": article["id"],
        "text": tgt_text,
        "spans": new_spans
    })

# SAVE
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)

# STATS
print("\nRESUMO")
print(f"Artigos: {len(translated_data)}")
print(f"Spans originais: {total_spans}")
print(f"Spans alinhados: {aligned_spans}")
print(f"Taxa retenção: {aligned_spans / max(1, total_spans):.2%}")
print(f"Output: {OUTPUT_FILE}")
