"""
Document Embedding Analysis for Language Selection
Embeds all training/dev/gold documents, visualizes similarity,
and determines optimal language mix per PT document.

Usage:
  python embeddings/embedding_analysis.py

Requires: sentence-transformers, umap-learn, matplotlib, sklearn
  pip install sentence-transformers umap-learn matplotlib scikit-learn
"""

import json, os, sys
import numpy as np
from collections import Counter, defaultdict

# ============================================================
# CONFIG
# ============================================================

# All data files with language labels
DATA_FILES = {
    # Training data (original languages)
    "EN":    "json/train.json",
    "FR":    "json/train_fr.json",
    "DE":    "json/train_ge.json",
    "IT":    "json/train_it.json",
    "PL":    "json/train_po.json",
    "RU":    "json/train_ru.json",
    # Dev sets
    "EN_dev":    "json/dev.json",
    "FR_dev":    "json/dev_fr.json",
    "DE_dev":    "json/dev_ge.json",
    "IT_dev":    "json/dev_it.json",
    "PL_dev":    "json/dev_po.json",
    "RU_dev":    "json/dev_ru.json",
    # PT gold (target)
    "PT_gold": "json/checkthat24_pt.json",
}

# Sentence embedding model — multilingual, good for cross-lingual similarity
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 20  # nearest neighbours per PT doc

OUTPUT_DIR = "embeddings"

# ============================================================
# LOAD DATA
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("Loading documents...")
all_docs = []  # list of {id, text, lang, spans, techniques}

for lang, path in DATA_FILES.items():
    if not os.path.isfile(path):
        print(f"  SKIP {path} (not found)")
        continue
    data = load_json(path)
    for doc in data:
        techniques = [s["label"] for s in doc.get("spans", [])]
        tech_dist = Counter(techniques)
        all_docs.append({
            "id": doc["id"],
            "lang": lang,
            "text": doc["text"],
            "n_spans": len(doc.get("spans", [])),
            "techniques": tech_dist,
            "is_train": "dev" not in lang and lang != "PT_gold",
            "is_pt_gold": lang == "PT_gold",
        })

print(f"Total documents: {len(all_docs)}")
for lang in sorted(set(d["lang"] for d in all_docs)):
    n = sum(1 for d in all_docs if d["lang"] == lang)
    print(f"  {lang}: {n} docs")

# ============================================================
# COMPUTE EMBEDDINGS
# ============================================================

print(f"\nLoading embedding model: {EMBED_MODEL}")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(EMBED_MODEL)

# For long documents, encode first 1000 chars (captures topic/style)
# Plus technique distribution as a feature
print("Computing embeddings...")
texts = []
for doc in all_docs:
    # Use first 1000 chars for topic, last 500 for style diversity
    t = doc["text"]
    if len(t) > 1500:
        text_sample = t[:1000] + " " + t[-500:]
    else:
        text_sample = t
    texts.append(text_sample)

embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# ============================================================
# TECHNIQUE DISTRIBUTION FEATURES
# ============================================================

# All techniques
ALL_TECHNIQUES = sorted(set(
    tech for doc in all_docs for tech in doc["techniques"].keys()
))
print(f"Techniques found: {len(ALL_TECHNIQUES)}")

# Create technique distribution vectors (normalized)
tech_vectors = np.zeros((len(all_docs), len(ALL_TECHNIQUES)), dtype=np.float32)
for i, doc in enumerate(all_docs):
    total = max(doc["n_spans"], 1)
    for tech, count in doc["techniques"].items():
        if tech in ALL_TECHNIQUES:
            tech_vectors[i, ALL_TECHNIQUES.index(tech)] = count / total

# Combine text embeddings + technique distributions
# Weight technique features to be meaningful alongside text embeddings
combined = np.hstack([
    embeddings,
    tech_vectors * 5.0,  # scale up technique features
])
print(f"Combined features shape: {combined.shape}")

# ============================================================
# DIMENSIONALITY REDUCTION FOR VISUALIZATION
# ============================================================

print("\nComputing UMAP projection...")
try:
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric="cosine")
    coords_2d = reducer.fit_transform(combined)
except ImportError:
    print("  UMAP not available, using t-SNE...")
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2, random_state=42, perplexity=30, metric="cosine")
    coords_2d = reducer.fit_transform(combined)

# ============================================================
# VISUALIZATION
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color map for languages
lang_colors = {
    "EN": "#1f77b4", "FR": "#ff7f0e", "DE": "#2ca02c", "IT": "#d62728",
    "PL": "#9467bd", "RU": "#8c564b",
    "EN_dev": "#aec7e8", "FR_dev": "#ffcc99", "DE_dev": "#98df8a",
    "IT_dev": "#ff9896", "PL_dev": "#c5b0d5", "RU_dev": "#c49c94",
    "PT_gold": "#e377c2",
}

# Plot 1: All documents colored by language
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

for lang in sorted(set(d["lang"] for d in all_docs)):
    mask = [d["lang"] == lang for d in all_docs]
    idxs = [i for i, m in enumerate(mask) if m]
    if not idxs:
        continue
    x = coords_2d[idxs, 0]
    y = coords_2d[idxs, 1]
    color = lang_colors.get(lang, "#333333")
    size = 80 if lang == "PT_gold" else 20
    marker = "*" if lang == "PT_gold" else "o"
    alpha = 0.9 if lang == "PT_gold" else 0.5
    zorder = 10 if lang == "PT_gold" else 1
    ax.scatter(x, y, c=color, s=size, marker=marker, alpha=alpha,
               label=lang, zorder=zorder, edgecolors="white" if lang == "PT_gold" else "none",
               linewidths=0.5)

ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.set_title("Document Embeddings — All Languages\n(PT gold = stars)", fontsize=14)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "embeddings_all_languages.png"), dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/embeddings_all_languages.png")

# Plot 2: Only training data + PT gold (cleaner view)
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
train_langs = ["EN", "FR", "DE", "IT", "PL", "RU", "FR→PT", "PT_gold"]
for lang in train_langs:
    mask = [d["lang"] == lang for d in all_docs]
    idxs = [i for i, m in enumerate(mask) if m]
    if not idxs:
        continue
    x = coords_2d[idxs, 0]
    y = coords_2d[idxs, 1]
    color = lang_colors.get(lang, "#333333")
    size = 100 if lang == "PT_gold" else 15
    marker = "*" if lang == "PT_gold" else "o"
    alpha = 1.0 if lang == "PT_gold" else 0.4
    ax.scatter(x, y, c=color, s=size, marker=marker, alpha=alpha,
               label=lang, zorder=10 if lang == "PT_gold" else 1,
               edgecolors="black" if lang == "PT_gold" else "none", linewidths=0.3)

ax.legend(loc="upper left", fontsize=9)
ax.set_title("Training Data + PT Gold Documents", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "embeddings_train_pt.png"), dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/embeddings_train_pt.png")

# ============================================================
# NEAREST NEIGHBOUR ANALYSIS
# ============================================================

print(f"\nFinding {TOP_K} nearest training docs per PT gold doc...")

from sklearn.metrics.pairwise import cosine_similarity

# Indices
pt_idxs = [i for i, d in enumerate(all_docs) if d["is_pt_gold"]]
train_idxs = [i for i, d in enumerate(all_docs) if d["is_train"]]

pt_embeddings = combined[pt_idxs]
train_embeddings = combined[train_idxs]

# Cosine similarity: (n_pt, n_train)
sim_matrix = cosine_similarity(pt_embeddings, train_embeddings)

# For each PT doc, find top-K training docs
lang_counts_global = Counter()
lang_counts_per_pt = []

for pi, pt_idx in enumerate(pt_idxs):
    pt_doc = all_docs[pt_idx]
    sims = sim_matrix[pi]
    top_k_train = np.argsort(sims)[-TOP_K:][::-1]

    neighbours = []
    lang_counts_local = Counter()
    for ti in top_k_train:
        train_doc = all_docs[train_idxs[ti]]
        neighbours.append({
            "id": train_doc["id"],
            "lang": train_doc["lang"],
            "similarity": float(sims[ti]),
        })
        lang_counts_local[train_doc["lang"]] += 1
        lang_counts_global[train_doc["lang"]] += 1

    lang_counts_per_pt.append({
        "pt_doc": pt_doc["id"],
        "lang_distribution": dict(lang_counts_local),
        "top_neighbour": neighbours[0],
    })

# ============================================================
# AGGREGATE ANALYSIS
# ============================================================

print(f"\n{'='*60}")
print(f"GLOBAL: Language distribution in top-{TOP_K} neighbours of PT gold")
print(f"{'='*60}")
total_neighbours = sum(lang_counts_global.values())
for lang, count in lang_counts_global.most_common():
    pct = 100 * count / total_neighbours
    bar = "█" * int(pct / 2)
    print(f"  {lang:<8} {count:>5} ({pct:>5.1f}%) {bar}")

# Per-PT-doc analysis
print(f"\n{'='*60}")
print(f"Per PT document: dominant training language")
print(f"{'='*60}")
dominant_langs = Counter()
for entry in lang_counts_per_pt:
    dominant = max(entry["lang_distribution"], key=entry["lang_distribution"].get)
    dominant_langs[dominant] += 1

for lang, count in dominant_langs.most_common():
    print(f"  {lang:<8} dominant for {count:>3} PT docs ({100*count/len(pt_idxs):.1f}%)")

# ============================================================
# TECHNIQUE-AWARE ANALYSIS
# ============================================================

print(f"\n{'='*60}")
print(f"Per technique: which language's training data is closest")
print(f"{'='*60}")

# For each technique, find PT docs that have it, then see which training langs are closest
all_techs = sorted(set(tech for d in all_docs if d["is_pt_gold"] for tech in d["techniques"]))

for tech in all_techs:
    # PT docs with this technique
    pt_with_tech = [i for i, pi in enumerate(pt_idxs) if tech in all_docs[pi]["techniques"]]
    if not pt_with_tech:
        continue

    # Aggregate neighbour langs for these PT docs
    tech_lang_counts = Counter()
    for pi in pt_with_tech:
        for entry_lang, entry_count in lang_counts_per_pt[pi]["lang_distribution"].items():
            tech_lang_counts[entry_lang] += entry_count

    total = sum(tech_lang_counts.values())
    top3 = tech_lang_counts.most_common(3)
    top3_str = ", ".join(f"{l}:{100*c/total:.0f}%" for l, c in top3)
    print(f"  {tech:<40} (in {len(pt_with_tech):>3} docs) → {top3_str}")

# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results = {
    "global_lang_distribution": dict(lang_counts_global),
    "dominant_per_doc": dict(dominant_langs),
    "per_pt_doc": lang_counts_per_pt,
    "top_k": TOP_K,
    "embed_model": EMBED_MODEL,
}

with open(os.path.join(OUTPUT_DIR, "neighbour_analysis.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUTPUT_DIR}/neighbour_analysis.json")

# ============================================================
# RECOMMENDED LANGUAGE MIX
# ============================================================

print(f"\n{'='*60}")
print(f"RECOMMENDED LANGUAGE MIX FOR TRAINING")
print(f"{'='*60}")

# Normalize global counts to get recommended proportions
total = sum(lang_counts_global.values())
print("Based on embedding similarity to PT gold documents:")
print()
for lang, count in lang_counts_global.most_common():
    pct = 100 * count / total
    if pct >= 3:
        print(f"  {lang:<8} → include, weight ~{pct:.0f}% of training data")
    elif pct >= 1:
        print(f"  {lang:<8} → optional ({pct:.1f}%)")
    else:
        print(f"  {lang:<8} → skip ({pct:.1f}%)")

# Plot 3: Language distribution bar chart
fig, ax = plt.subplots(figsize=(10, 5))
langs_sorted = [l for l, _ in lang_counts_global.most_common()]
counts_sorted = [lang_counts_global[l] for l in langs_sorted]
colors = [lang_colors.get(l, "#333") for l in langs_sorted]
ax.bar(langs_sorted, counts_sorted, color=colors, edgecolor="white")
ax.set_ylabel(f"Count in top-{TOP_K} neighbours")
ax.set_title("Which training languages are closest to PT gold?")
for i, (l, c) in enumerate(zip(langs_sorted, counts_sorted)):
    ax.text(i, c + 5, f"{100*c/total:.1f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "language_distribution.png"), dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/language_distribution.png")

# Plot 4: Dominant training language per PT document (pie chart)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
dom_langs = [l for l, _ in dominant_langs.most_common()]
dom_counts = [dominant_langs[l] for l in dom_langs]
dom_colors = [lang_colors.get(l, "#333") for l in dom_langs]
wedges, texts, autotexts = ax1.pie(
    dom_counts, labels=dom_langs, colors=dom_colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.8, textprops={"fontsize": 9},
)
for t in autotexts:
    t.set_fontsize(8)
ax1.set_title(f"Dominant Training Language\nper PT Document (n={len(pt_idxs)})", fontsize=12)

# Horizontal bar chart (clearer for many languages)
dom_langs_rev = list(reversed(dom_langs))
dom_counts_rev = list(reversed(dom_counts))
dom_colors_rev = list(reversed(dom_colors))
bars = ax2.barh(dom_langs_rev, dom_counts_rev, color=dom_colors_rev, edgecolor="white")
ax2.set_xlabel("Number of PT documents")
ax2.set_title("How many PT docs have each language\nas dominant neighbour?", fontsize=12)
for bar, count in zip(bars, dom_counts_rev):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             str(count), va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dominant_language_per_doc.png"), dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/dominant_language_per_doc.png")

# Plot 5: Per-technique language heatmap
# Build matrix: rows=techniques, cols=languages, values=percentage
train_lang_list = sorted(set(d["lang"] for d in all_docs if d["is_train"]))
tech_lang_matrix = np.zeros((len(all_techs), len(train_lang_list)), dtype=np.float64)

for ti, tech in enumerate(all_techs):
    pt_with_tech = [i for i, pi in enumerate(pt_idxs) if tech in all_docs[pi]["techniques"]]
    if not pt_with_tech:
        continue
    tech_lang_counts = Counter()
    for pi in pt_with_tech:
        for entry_lang, entry_count in lang_counts_per_pt[pi]["lang_distribution"].items():
            tech_lang_counts[entry_lang] += entry_count
    total_tech = sum(tech_lang_counts.values())
    if total_tech > 0:
        for li, lang in enumerate(train_lang_list):
            tech_lang_matrix[ti, li] = 100 * tech_lang_counts.get(lang, 0) / total_tech

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(tech_lang_matrix, cmap="YlOrRd", aspect="auto", vmin=0)

ax.set_xticks(range(len(train_lang_list)))
ax.set_xticklabels(train_lang_list, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(all_techs)))
ax.set_yticklabels(all_techs, fontsize=8)

# Annotate cells with percentages
for ti in range(len(all_techs)):
    for li in range(len(train_lang_list)):
        val = tech_lang_matrix[ti, li]
        if val >= 1:
            color = "white" if val > 30 else "black"
            ax.text(li, ti, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="% of nearest neighbours")
ax.set_title("Per Technique: Language Distribution of Nearest Training Documents (%)", fontsize=12)
ax.set_xlabel("Training Language")
ax.set_ylabel("Persuasion Technique")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "technique_language_heatmap.png"), dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/technique_language_heatmap.png")

# Plot 6: Stacked bar chart - per technique, which languages contribute
fig, ax = plt.subplots(figsize=(14, 8))
bottom = np.zeros(len(all_techs))
for li, lang in enumerate(train_lang_list):
    values = tech_lang_matrix[:, li]
    color = lang_colors.get(lang, "#333")
    ax.barh(range(len(all_techs)), values, left=bottom, color=color, label=lang, edgecolor="white", linewidth=0.3)
    bottom += values

ax.set_yticks(range(len(all_techs)))
ax.set_yticklabels(all_techs, fontsize=8)
ax.set_xlabel("% of nearest neighbours from each language")
ax.set_title("Per Technique: Training Language Composition of Nearest Neighbours", fontsize=12)
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "technique_language_stacked.png"), dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/technique_language_stacked.png")

