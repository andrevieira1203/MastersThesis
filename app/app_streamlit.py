"""
PersuasionLens - Streamlit demo for persuasion technique detection in Portuguese.
Loads the trained XLM-RoBERTa model and serves an interactive interface.

Usage:
  pip install streamlit newspaper3k torch transformers pdfplumber
  streamlit run app_streamlit.py 
"""

import os, json, re, argparse, io
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from transformers import AutoTokenizer, AutoModel

# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="PersuasionLens",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="outputs_v4/pt_finetuned_only_it/best_model")
parser.add_argument("--threshold", type=float, default=0.70)
args, _ = parser.parse_known_args()

# ============================================================
# CONFIG
# ============================================================

MAX_LENGTH = 512
CHUNK_OVERLAP = 0.5

LABELS = [
    "Loaded_Language", "Name_Calling-Labeling", "Repetition", "Slogans",
    "Appeal_to_Fear-Prejudice", "Flag_Waving", "Causal_Oversimplification",
    "Appeal_to_Authority", "Appeal_to_Values", "Doubt", "Exaggeration-Minimisation",
    "Guilt_by_Association", "False_Dilemma-No_Choice", "Straw_Man", "Red_Herring",
    "Whataboutism", "Obfuscation-Vagueness-Confusion", "Appeal_to_Time",
    "Conversation_Killer", "Appeal_to_Popularity", "Appeal_to_Hypocrisy",
    "Consequential_Oversimplification", "Questioning_the_Reputation",
]

TECHNIQUE_INFO = {
    "Loaded_Language": {
        "category": "Manipulative Wording",
        "description": "Uso de palavras com forte carga emocional para influenciar o leitor, para além do significado literal.",
        "color": "#e74c3c",
    },
    "Name_Calling-Labeling": {
        "category": "Attack on Reputation",
        "description": "Atribuição de rótulos negativos a uma pessoa ou grupo para desacreditar sem argumentar.",
        "color": "#c0392b",
    },
    "Repetition": {
        "category": "Manipulative Wording",
        "description": "Repetição de uma mensagem para a tornar mais familiar e, por isso, mais credível.",
        "color": "#e67e22",
    },
    "Slogans": {
        "category": "Call",
        "description": "Uso de frases curtas e memoráveis para substituir argumentação racional.",
        "color": "#d35400",
    },
    "Appeal_to_Fear-Prejudice": {
        "category": "Justification",
        "description": "Apelo ao medo ou preconceito para justificar uma posição ou ação.",
        "color": "#9b59b6",
    },
    "Flag_Waving": {
        "category": "Justification",
        "description": "Apelo ao patriotismo ou identidade nacional para justificar uma posição.",
        "color": "#8e44ad",
    },
    "Causal_Oversimplification": {
        "category": "Simplification",
        "description": "Redução de causas complexas a uma única causa simples.",
        "color": "#2980b9",
    },
    "Appeal_to_Authority": {
        "category": "Justification",
        "description": "Uso da opinião de uma autoridade como argumento, mesmo fora da sua área de competência.",
        "color": "#3498db",
    },
    "Appeal_to_Values": {
        "category": "Justification",
        "description": "Apelo a valores morais ou éticos partilhados para persuadir.",
        "color": "#1abc9c",
    },
    "Doubt": {
        "category": "Attack on Reputation",
        "description": "Levantamento de dúvidas sobre a credibilidade de alguém sem apresentar provas.",
        "color": "#16a085",
    },
    "Exaggeration-Minimisation": {
        "category": "Manipulative Wording",
        "description": "Exagero ou minimização de factos para distorcer a perceção do leitor.",
        "color": "#f59e0b",
    },
    "Guilt_by_Association": {
        "category": "Attack on Reputation",
        "description": "Desacreditação de alguém por associação a outra pessoa ou grupo negativo.",
        "color": "#e74c3c",
    },
    "False_Dilemma-No_Choice": {
        "category": "Simplification",
        "description": "Apresentação de apenas duas opções quando existem mais alternativas.",
        "color": "#2c3e50",
    },
    "Straw_Man": {
        "category": "Distraction",
        "description": "Distorção do argumento do oponente para o tornar mais fácil de atacar.",
        "color": "#7f8c8d",
    },
    "Red_Herring": {
        "category": "Distraction",
        "description": "Introdução de um assunto irrelevante para desviar a atenção do tema principal.",
        "color": "#95a5a6",
    },
    "Whataboutism": {
        "category": "Distraction",
        "description": "Desvio da crítica apontando falhas equivalentes no oponente.",
        "color": "#bdc3c7",
    },
    "Obfuscation-Vagueness-Confusion": {
        "category": "Manipulative Wording",
        "description": "Uso deliberado de linguagem vaga ou confusa para obscurecer o significado.",
        "color": "#34495e",
    },
    "Appeal_to_Time": {
        "category": "Call",
        "description": "Criação de urgência temporal para pressionar uma decisão.",
        "color": "#27ae60",
    },
    "Conversation_Killer": {
        "category": "Call",
        "description": "Uso de frases que encerram o debate sem argumentação (ex: 'é o que é').",
        "color": "#2ecc71",
    },
    "Appeal_to_Popularity": {
        "category": "Justification",
        "description": "Argumento de que algo é verdade porque muitas pessoas acreditam nele.",
        "color": "#1abc9c",
    },
    "Appeal_to_Hypocrisy": {
        "category": "Attack on Reputation",
        "description": "Desacreditação de alguém apontando contradições entre as suas palavras e ações.",
        "color": "#e74c3c",
    },
    "Consequential_Oversimplification": {
        "category": "Simplification",
        "description": "Apresentação de consequências exageradas como inevitáveis (bola de neve).",
        "color": "#2980b9",
    },
    "Questioning_the_Reputation": {
        "category": "Attack on Reputation",
        "description": "Questionamento da reputação ou competência de alguém para desacreditar a sua posição.",
        "color": "#c0392b",
    },
}

NUM_LABELS = len(LABELS)
NUM_OUTPUTS = NUM_LABELS * 2

SAMPLES = {
    "Política": "Este governo é uma vergonha absoluta! Os políticos corruptos que nos governam estão a destruir o país enquanto enchem os bolsos. Se não agirmos agora, Portugal será um país de terceiro mundo em poucos anos. Todos os especialistas concordam que esta é a pior governação de sempre. É preciso mudar radicalmente ou vamos todos sofrer as consequências devastadoras desta incompetência criminosa.",
    "Clima": "Os alarmistas do clima querem destruir a economia com as suas teorias exageradas. Dizem que o mundo vai acabar, mas há 30 anos diziam o mesmo e aqui estamos. É tudo uma conspiração dos globalistas para controlar as nossas vidas. Se proibirmos os combustíveis fósseis, voltamos à Idade da Pedra. Ou apoiamos o progresso ou destruímos a civilização.",
    "Saúde": "O nosso maravilhoso Sistema Nacional de Saúde está a ser sabotado por interesses privados. Os médicos traidores que fogem para o privado são responsáveis pela morte de milhares de portugueses. Como disse um famoso professor, \"a saúde pública é sagrada\". Toda a gente sabe que o SNS é o melhor do mundo. Temos de agir já antes que seja tarde demais!",
}

CAT_COLORS = {
    "Attack on Reputation": "#ef4444",
    "Justification": "#6366f1",
    "Simplification": "#8b5cf6",
    "Distraction": "#64748b",
    "Call": "#22c55e",
    "Manipulative Wording": "#f59e0b",
}

# ============================================================
# MODEL
# ============================================================

class XLMRMultiLabelNER(nn.Module):
    def __init__(self, model_name, num_outputs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(hidden, num_outputs)
        self.num_outputs = num_outputs

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return {"logits": self.classifier(self.dropout(outputs.last_hidden_state))}

    @classmethod
    def load_pretrained(cls, path):
        with open(os.path.join(path, "multilabel_config.json")) as f:
            config = json.load(f)
        model = cls(config["model_name"], config["num_outputs"])
        state = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        state = {k: v for k, v in state.items() if "pos_weight" not in k}
        model.load_state_dict(state, strict=False)
        return model


@st.cache_resource(show_spinner="A carregar o modelo...")
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XLMRMultiLabelNER.load_pretrained(model_path)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return model, tokenizer, device


# ============================================================
# INFERENCE
# ============================================================

def analyze_text(text, model, tokenizer, device, threshold=0.70):
    full_enc = tokenizer(
        text, return_offsets_mapping=True, add_special_tokens=False, truncation=False
    )
    full_ids = full_enc["input_ids"]
    full_offsets = full_enc["offset_mapping"]
    total_tokens = len(full_ids)

    usable_len = MAX_LENGTH - 2
    stride = max(1, int(usable_len * (1 - CHUNK_OVERLAP)))

    if total_tokens <= usable_len:
        windows = [(0, total_tokens)]
    else:
        windows = []
        s = 0
        while s < total_tokens:
            e = min(s + usable_len, total_tokens)
            windows.append((s, e))
            if e == total_tokens:
                break
            s += stride

    prob_sum = np.zeros((total_tokens, NUM_OUTPUTS), dtype=np.float64)
    prob_count = np.zeros(total_tokens, dtype=np.float64)
    bos = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos = tokenizer.eos_token_id or tokenizer.sep_token_id

    for ws, we in windows:
        chunk_ids = full_ids[ws:we]
        input_ids = torch.tensor(
            [[bos] + chunk_ids + [eos]], dtype=torch.long, device=device
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"][0].cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        chunk_probs = probs[1:-1]
        for i, ti in enumerate(range(ws, we)):
            prob_sum[ti] += chunk_probs[i]
            prob_count[ti] += 1

    prob_count[prob_count == 0] = 1
    avg_probs = prob_sum / prob_count[:, None]

    # Extract spans
    token_preds = []
    for t_idx in range(len(full_offsets)):
        ts, te = full_offsets[t_idx]
        if ts == 0 and te == 0:
            continue
        for li, label in enumerate(LABELS):
            b_prob = avg_probs[t_idx, 2 * li]
            i_prob = avg_probs[t_idx, 2 * li + 1]
            if b_prob > threshold or i_prob > threshold:
                tag = "B" if b_prob >= i_prob else "I"
                token_preds.append((t_idx, label, tag, float(max(b_prob, i_prob))))

    by_label = {}
    for t_idx, label, tag, score in token_preds:
        by_label.setdefault(label, []).append((t_idx, tag, score))

    spans = []
    for label, tokens in by_label.items():
        tokens.sort(key=lambda x: x[0])
        cs = ce = None
        csc = []
        for t_idx, tag, score in tokens:
            if tag == "B" or cs is None or t_idx > ce + 2:
                if cs is not None:
                    spans.append(
                        {
                            "start": int(full_offsets[cs][0]),
                            "end": int(full_offsets[ce][1]),
                            "label": label,
                            "score": sum(csc) / len(csc),
                        }
                    )
                cs = ce = t_idx
                csc = [score]
            else:
                ce = t_idx
                csc.append(score)
        if cs is not None:
            spans.append(
                {
                    "start": int(full_offsets[cs][0]),
                    "end": int(full_offsets[ce][1]),
                    "label": label,
                    "score": sum(csc) / len(csc),
                }
            )

    # Post-process
    cleaned = []
    for s in spans:
        clen = s["end"] - s["start"]
        stxt = text[s["start"] : s["end"]]
        if clen < 2 or clen > 600:
            continue
        if len(re.sub(r"[^\w]", "", stxt).strip()) == 0:
            continue
        cleaned.append(s)

    # Dedup
    by_label2 = {}
    for s in cleaned:
        by_label2.setdefault(s["label"], []).append(s)
    deduped = []
    for label, lspans in by_label2.items():
        lspans.sort(key=lambda x: -x["score"])
        kept = []
        for s in lspans:
            overlap = False
            for k in kept:
                inter = max(0, min(s["end"], k["end"]) - max(s["start"], k["start"]))
                union = (s["end"] - s["start"]) + (k["end"] - k["start"]) - inter
                if union > 0 and inter / union > 0.5:
                    overlap = True
                    break
            if not overlap:
                kept.append(s)
        deduped.extend(kept)

    deduped.sort(key=lambda x: x["start"])

    # Add technique info
    for s in deduped:
        info = TECHNIQUE_INFO.get(s["label"], {})
        s["category"] = info.get("category", "")
        s["description"] = info.get("description", "")
        s["color"] = info.get("color", "#666")
        s["text"] = text[s["start"] : s["end"]]

    return deduped


# ============================================================
# HELPERS
# ============================================================

def compute_manipulation_score(spans, text_length):
    if text_length == 0 or not spans:
        return 0
    density = sum(min(s["score"], 1.0) for s in spans) / (text_length / 100)
    density_score = min(density * 10, 70)
    unique = len(set(s["label"] for s in spans))
    diversity_score = min(unique / 23 * 100, 30)
    return round(min(density_score + diversity_score, 100))


def build_summary(spans, text_length):
    technique_counts = {}
    category_counts = {}
    avg_confidence = []
    for s in spans:
        technique_counts[s["label"]] = technique_counts.get(s["label"], 0) + 1
        cat = s.get("category", "Other")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        avg_confidence.append(s["score"])

    return {
        "total_spans": len(spans),
        "techniques_found": len(technique_counts),
        "technique_counts": technique_counts,
        "category_counts": category_counts,
        "text_length": text_length,
        "manipulation_score": compute_manipulation_score(spans, text_length),
        "avg_confidence": (
            round(sum(avg_confidence) / len(avg_confidence), 3)
            if avg_confidence
            else 0
        ),
    }


def extract_from_url(url):
    try:
        from newspaper import Article

        article = Article(url, language="pt")
        article.download()
        article.parse()
        text = article.text
        title = article.title
        if text and len(text.strip()) > 50:
            return title, text.strip()
    except Exception:
        pass
    try:
        import urllib.request
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.result = []
                self.skip = False

            def handle_starttag(self, tag, attrs):
                self.skip = tag in ("script", "style", "nav", "header", "footer")

            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "header", "footer"):
                    self.skip = False
                if tag in ("p", "br", "div", "h1", "h2", "h3"):
                    self.result.append("\n")

            def handle_data(self, data):
                if not self.skip:
                    self.result.append(data)

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = (
            urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
        )
        p = TextExtractor()
        p.feed(html)
        text = re.sub(r"\n{3,}", "\n\n", "".join(p.result)).strip()
        return "", text
    except Exception:
        return "", ""


def extract_from_pdf(pdf_bytes):
    try:
        import pdfplumber

        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
        text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
        pdf.close()
        return text.strip()
    except Exception:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            return "\n\n".join(
                p.extract_text() or "" for p in reader.pages
            ).strip()
        except Exception:
            return ""


def build_annotated_html(text, spans):
    """Build HTML with colored highlights for each detected span."""
    sorted_spans = sorted(spans, key=lambda s: (s["start"], -s["end"]))
    parts = []
    pos = 0
    for s in sorted_spans:
        if s["start"] < pos:
            continue
        if s["start"] > pos:
            parts.append(escape_html(text[pos : s["start"]]))
        color = s["color"]
        label_display = s["label"].replace("_", " ").replace("-", " / ")
        parts.append(
            f'<span title="{escape_attr(label_display)} ({s["score"]:.3f})\n'
            f'{escape_attr(s["description"])}" '
            f'style="background:{color}22;border-bottom:2px solid {color};'
            f'color:{color};padding:1px 3px;border-radius:3px;cursor:help">'
            f"{escape_html(text[s['start']:s['end']])}</span>"
        )
        pos = s["end"]
    if pos < len(text):
        parts.append(escape_html(text[pos:]))
    return "".join(parts)


def escape_html(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_attr(t):
    return t.replace('"', "&quot;").replace("'", "&#39;").replace("\n", "&#10;")


def render_gauge(score):
    """Return SVG gauge for the manipulation score."""
    if score > 60:
        color = "#ef4444"
    elif score > 30:
        color = "#f59e0b"
    else:
        color = "#22c55e"
    circ = 2 * 3.14159265 * 52
    offset = circ * (1 - score / 100)
    return f"""
    <div style="text-align:center">
      <svg width="140" height="140" viewBox="0 0 120 120" style="transform:rotate(-90deg)">
        <circle cx="60" cy="60" r="52" fill="none" stroke="#1c1c20" stroke-width="8"/>
        <circle cx="60" cy="60" r="52" fill="none" stroke="{color}" stroke-width="8"
                stroke-linecap="round"
                stroke-dasharray="{circ}" stroke-dashoffset="{offset}"
                style="transition:stroke-dashoffset .8s ease"/>
      </svg>
      <div style="margin-top:-95px;margin-bottom:55px;text-align:center">
        <div style="font-size:2rem;font-weight:700;color:{color};font-family:monospace">{score}</div>
        <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em;color:#9898a4">
          Manipulação</div>
      </div>
    </div>
    """


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
    /* Dark theme overrides */
    .stApp { background-color: #0a0a0b; }

    /* Header */
    .app-header {
        text-align: center;
        padding: 1.2rem 0 0.8rem;
        border-bottom: 1px solid #2a2a30;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #e8e8ec;
        margin: 0;
    }
    .app-header h1 span { color: #6366f1; }
    .app-header p {
        font-size: 0.8rem;
        color: #9898a4;
        margin: 0.2rem 0 0;
    }

    /* Annotated text container */
    .annotated-box {
        background: #141416;
        border: 1px solid #2a2a30;
        border-radius: 10px;
        padding: 1.2rem;
        line-height: 2;
        font-size: 0.9rem;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 500px;
        overflow-y: auto;
        color: #e8e8ec;
    }

    /* Category bar */
    .cat-bar-track {
        height: 20px;
        background: #1c1c20;
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    .cat-bar-fill {
        height: 100%;
        border-radius: 4px;
        display: flex;
        align-items: center;
        padding-left: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        color: white;
    }

    /* Technique card */
    .tech-card {
        background: #141416;
        border: 1px solid #2a2a30;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .tech-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .tech-label {
        font-weight: 500;
        font-size: 0.85rem;
        color: #e8e8ec;
    }
    .tech-cat {
        font-size: 0.68rem;
        color: #5a5a66;
    }
    .tech-count {
        margin-left: auto;
        font-family: monospace;
        font-size: 0.75rem;
        color: #818cf8;
        background: #1c1c20;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
    }

    /* No results */
    .no-results {
        text-align: center;
        padding: 2.5rem;
        color: #9898a4;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0;
        font-size: 0.7rem;
        color: #3a3a44;
        border-top: 1px solid #2a2a30;
        margin-top: 2rem;
    }
    .app-footer a { color: #818cf8; text-decoration: none; }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { display: none; }
    .stDeployButton { display: none; }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #141416;
        border: 1px solid #2a2a30;
        border-radius: 10px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        justify-content: center;
        padding: 0.7rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #9898a4;
        background: transparent;
        border-right: 1px solid #2a2a30;
    }
    .stTabs [data-baseweb="tab"]:last-child {
        border-right: none;
    }
    .stTabs [aria-selected="true"] {
        background: #6366f1 !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UI
# ============================================================

# Header
st.markdown(
    """
<div class="app-header">
    <h1>Persuasion<span>Lens</span></h1>
    <p>Deteção automática de técnicas de persuasão em textos portugueses</p>
</div>
""",
    unsafe_allow_html=True,
)

# Load model
model, tokenizer, device = load_model(args.model)

# Input tabs
tab_text, tab_url, tab_pdf = st.tabs(["📝 Texto", "🔗 URL", "📄 PDF"])

with tab_text:
    # Sample buttons
    def _load_sample(text):
        st.session_state["text_area"] = text

    sample_cols = st.columns(len(SAMPLES))
    for i, (label, sample_text) in enumerate(SAMPLES.items()):
        sample_cols[i].button(
            f"Ex: {label}",
            key=f"sample_{i}",
            use_container_width=True,
            on_click=_load_sample,
            args=(sample_text,),
        )

    input_text = st.text_area(
        "Cole o texto para análise",
        height=200,
        placeholder="Cole aqui um excerto de notícia, artigo de opinião ou publicação em português...",
        key="text_area",
    )
    st.caption(f"{len(input_text)} caracteres")

with tab_url:
    input_url = st.text_input(
        "URL do artigo",
        placeholder="https://www.exemplo.pt/artigo-de-opiniao",
    )

with tab_pdf:
    uploaded_pdf = st.file_uploader(
        "Carregar PDF",
        type=["pdf"],
        help="Arraste ou selecione um ficheiro PDF",
    )

# Controls
ctrl_cols = st.columns([2, 3, 2])
with ctrl_cols[0]:
    analyze_clicked = st.button("Analisar", type="primary", use_container_width=True)
with ctrl_cols[1]:
    threshold = st.slider(
        "Limiar de deteção",
        min_value=0.30,
        max_value=0.90,
        value=args.threshold,
        step=0.05,
        format="%.2f",
        help="Valores mais baixos detetam mais técnicas (mais sensível). "
             "Valores mais altos são mais conservadores.",
    )

# ============================================================
# ANALYSIS
# ============================================================

if analyze_clicked:
    text_to_analyze = None
    source_info = ""

    # Determine input source (last active tab is tricky in Streamlit,
    # so we check which input has content)
    if uploaded_pdf is not None:
        with st.spinner("A extrair texto do PDF..."):
            pdf_text = extract_from_pdf(uploaded_pdf.read())
        if not pdf_text or len(pdf_text) < 50:
            st.error("Não foi possível extrair texto deste PDF.")
        else:
            text_to_analyze = pdf_text
            source_info = f"📄 {uploaded_pdf.name}"
    elif input_url and input_url.strip():
        with st.spinner("A extrair texto do URL..."):
            title, url_text = extract_from_url(input_url.strip())
        if not url_text or len(url_text) < 50:
            st.error(
                "Não foi possível extrair texto deste URL. Tente colar o texto diretamente."
            )
        else:
            text_to_analyze = url_text
            source_info = f"🔗 {title or input_url}"
    elif input_text and input_text.strip():
        text_to_analyze = input_text.strip()
    else:
        st.warning("Cole um texto, introduza um URL ou carregue um PDF para analisar.")

    if text_to_analyze:
        if len(text_to_analyze) > 50000:
            st.error("Texto demasiado longo (máx 50.000 caracteres).")
        else:
            with st.spinner("A analisar o texto..."):
                spans = analyze_text(
                    text_to_analyze, model, tokenizer, device, threshold=threshold
                )
                summary = build_summary(spans, len(text_to_analyze))

            if source_info:
                st.caption(source_info)

            # --- Results ---
            st.divider()

            # Score gauge + stats
            gauge_col, stats_col = st.columns([1, 2])

            with gauge_col:
                st.markdown(
                    render_gauge(summary["manipulation_score"]),
                    unsafe_allow_html=True,
                )

            with stats_col:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Técnicas", summary["total_spans"])
                m2.metric("Tipos", summary["techniques_found"])
                m3.metric("Caracteres", f"{summary['text_length'] / 1000:.1f}k")
                m4.metric(
                    "Confiança",
                    f"{summary['avg_confidence'] * 100:.0f}%",
                )

                # Category bars
                cats = summary["category_counts"]
                if cats:
                    max_cat = max(cats.values())
                    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
                        w = max(cnt / max_cat * 100, 8)
                        c = CAT_COLORS.get(cat, "#666")
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:8px;'
                            f'margin-bottom:4px;font-size:0.78rem">'
                            f'<div style="width:130px;text-align:right;color:#9898a4;'
                            f'font-size:0.72rem;flex-shrink:0">{cat}</div>'
                            f'<div class="cat-bar-track" style="flex:1">'
                            f'<div class="cat-bar-fill" style="width:{w}%;background:{c}">'
                            f"{cnt}</div></div></div>",
                            unsafe_allow_html=True,
                        )

            # Annotated text
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;'
                'letter-spacing:0.08em;color:#9898a4;margin:1.2rem 0 0.6rem;'
                'display:flex;align-items:center;gap:0.5rem">'
                "Texto anotado"
                '<div style="flex:1;height:1px;background:#2a2a30"></div></div>',
                unsafe_allow_html=True,
            )

            if not spans:
                st.markdown(
                    '<div class="no-results">'
                    '<div style="font-size:2rem;margin-bottom:0.3rem">✓</div>'
                    "Nenhuma técnica detetada.</div>",
                    unsafe_allow_html=True,
                )
            else:
                annotated_html = build_annotated_html(text_to_analyze, spans)
                st.markdown(
                    f'<div class="annotated-box">{annotated_html}</div>',
                    unsafe_allow_html=True,
                )

            # Technique list
            if spans:
                st.markdown(
                    '<div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;'
                    'letter-spacing:0.08em;color:#9898a4;margin:1.2rem 0 0.6rem;'
                    'display:flex;align-items:center;gap:0.5rem">'
                    "Técnicas identificadas"
                    '<div style="flex:1;height:1px;background:#2a2a30"></div></div>',
                    unsafe_allow_html=True,
                )

                tech_counts = summary["technique_counts"]
                for tech, count in sorted(
                    tech_counts.items(), key=lambda x: -x[1]
                ):
                    info = TECHNIQUE_INFO.get(tech, {})
                    c = info.get("color", "#666")
                    cat = info.get("category", "")
                    desc = info.get("description", "")
                    label_display = tech.replace("_", " ").replace("-", " / ")
                    st.markdown(
                        f'<div class="tech-card">'
                        f'<div class="tech-dot" style="background:{c}"></div>'
                        f"<div>"
                        f'<div class="tech-label">{label_display}</div>'
                        f'<div class="tech-cat">{cat} &mdash; {desc}</div>'
                        f"</div>"
                        f'<div class="tech-count">{count}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Export
            st.divider()
            export_data = {
                "spans": spans,
                "summary": summary,
                "text": text_to_analyze,
            }
            st.download_button(
                "📥 Exportar JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name="persuasion_analysis.json",
                mime="application/json",
            )

# Footer
st.markdown(
    """
<div class="app-footer">
    PersuasionLens &mdash; MSc Thesis, FCUP / INESC TEC &mdash;
    XLM-RoBERTa fine-tuned, CLEF-2024 Task 3 &mdash;
    <a href="https://github.com/andrevieira1203/MastersThesis">André Vieira</a>
</div>
""",
    unsafe_allow_html=True,
)