# Chat.py
# -------------------------------------------------------------------
# ZELO Chat – UI estilo ChatGPT + leitura direta de kb/ (PDF/TXT/MD)
# + Busca e exibição de imagens por nome
# + Autenticação por CPF via Excel (auth/usuarios.xlsx, coluna CPF)
#
# Requisitos:
#   pip install streamlit openai tiktoken python-dotenv pypdf pandas openpyxl
# Rode:
#   streamlit run Chat.py
# -------------------------------------------------------------------

import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import numpy as np
from pypdf import PdfReader
import pandas as pd

# =========================
# Config & Setup
# =========================
load_dotenv()
st.set_page_config(page_title="ZELO Chat", page_icon="💬", layout="wide")

OPENAI_API_KEY = st.secrets.get("openai_apikey", os.getenv("OPENAI_API_KEY", ""))
OPENAI_ORG     = st.secrets.get("openai_org",    os.getenv("OPENAI_ORG",    ""))
if not OPENAI_API_KEY:
    st.error("Configure a OPENAI_API_KEY em .streamlit/secrets.toml ou .env.")
    st.stop()

CHAT_MODEL  = st.secrets.get("chat_model",  os.getenv("CHAT_MODEL",  "gpt-4o-mini"))
EMBED_MODEL = st.secrets.get("embed_model", os.getenv("EMBED_MODEL", "text-embedding-3-large"))
client = OpenAI(api_key=OPENAI_API_KEY, organization=(OPENAI_ORG or None))

# Pastas
KB_DIR    = "kb"                     # PDFs/.txt/.md aqui
IMG_DIRS  = ["kb", "img", "assets"]  # pastas de imagens
AUTH_XLSX = "auth/usuarios.xlsx"     # base de CPFs (coluna CPF)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff")

# =========================
# Autenticação por CPF
# =========================
def _only_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")

def _valid_cpf_digits(cpf_digits: str) -> bool:
    """Valida apenas formato: 11 dígitos (sem verificar dígitos verificadores).
    Pode trocar para validação completa depois, se quiser.
    """
    return cpf_digits.isdigit() and len(cpf_digits) == 11

@st.cache_resource(show_spinner=False)
def load_allowed_cpfs(path: str) -> set:
    """Lê Excel (auth/usuarios.xlsx) e retorna um set de CPFs (apenas dígitos)."""
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_excel(path)  # requer openpyxl
        if "CPF" not in df.columns:
            return set()
        cpfs = set(_only_digits(str(x)) for x in df["CPF"].dropna().astype(str))
        cpfs = {c for c in cpfs if _valid_cpf_digits(c)}
        return cpfs
    except Exception:
        return set()

def ensure_authenticated() -> None:
    """Pede CPF e bloqueia acesso se não estiver na base."""
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if "auth_cpf" not in st.session_state:
        st.session_state.auth_cpf = ""

    allowed = load_allowed_cpfs(AUTH_XLSX)

    st.sidebar.header("🔐 Autenticação")
    if st.session_state.auth_ok:
        st.sidebar.success(f"Acesso liberado — CPF: {st.session_state.auth_cpf}")
        if st.sidebar.button("Sair"):
            st.session_state.auth_ok = False
            st.session_state.auth_cpf = ""
            st.cache_resource.clear()
            st.rerun()
        return

    st.sidebar.info("Informe seu CPF para acessar o ZELO Chat.")
    cpf_input = st.sidebar.text_input("CPF (apenas números ou com máscara)", value="", max_chars=14)
    if st.sidebar.button("Entrar"):
        cpf_norm = _only_digits(cpf_input)
        if not _valid_cpf_digits(cpf_norm):
            st.sidebar.error("CPF inválido. Digite 11 dígitos.")
        else:
            if not allowed:
                st.sidebar.error("Base de CPFs não encontrada ou sem coluna 'CPF'. Contate o administrador.")
            elif cpf_norm in allowed:
                st.session_state.auth_ok = True
                st.session_state.auth_cpf = cpf_norm
                st.sidebar.success("Acesso liberado!")
                st.rerun()
            else:
                st.sidebar.error("CPF não autorizado. Acesso bloqueado.")
    # Bloqueia o app inteiro até autenticar
    st.stop()

# Checagem de autenticação antes de mostrar o chat
ensure_authenticated()

# =========================
# Utilitários de RAG
# =========================
def chunk_text(text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text or "")
    chunks, start = [], 0
    while start < len(toks):
        end = min(start + max_tokens, len(toks))
        chunks.append(enc.decode(toks[start:end]))
        if end == len(toks):
            break
        start = end - overlap_tokens
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def read_txt_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()

def load_all_kb_texts(kb_dir: str) -> List[Dict[str, Any]]:
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir, exist_ok=True)
    docs = []
    for root, _, files in os.walk(kb_dir):
        for f in files:
            fpath = os.path.join(root, f)
            fl = f.lower()
            try:
                if fl.endswith(".pdf"):
                    text = read_pdf(fpath)
                elif fl.endswith((".txt", ".md")):
                    text = read_txt_md(fpath)
                else:
                    continue
            except Exception:
                continue
            text = (text or "").strip()
            if text:
                docs.append({"source_path": fpath, "text": text})
    return docs

@st.cache_resource(show_spinner=False)
def build_index(kb_dir: str) -> Dict[str, Any]:
    raw_docs = load_all_kb_texts(kb_dir)
    chunks = []
    for d in raw_docs:
        for i, ch in enumerate(chunk_text(d["text"])):
            chunks.append({
                "id": str(uuid.uuid4()),
                "source_path": d["source_path"],
                "chunk_id": i,
                "text": ch,
            })
    embeds = embed_texts([c["text"] for c in chunks]) if chunks else []
    for c, e in zip(chunks, embeds):
        c["embedding"] = e
    return {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "model": EMBED_MODEL,
        "docs": chunks,
        "files_total": len(raw_docs),
        "chunks_total": len(chunks),
    }

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search_index(query: str, index: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
    docs = index.get("docs", [])
    if not docs:
        return []
    q_emb = np.array(embed_texts([query])[0])
    scored = []
    for d in docs:
        sim = cosine_sim(q_emb, np.array(d["embedding"]))
        scored.append((sim, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [dict(item[1], score=item[0]) for item in scored[:k]]

# =========================
# Busca de imagens relacionadas
# =========================
def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[_\-\.]", " ", s)
    s = re.sub(r"[^a-z0-9à-úãõâêôç ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> List[str]:
    return [t for t in _norm(s).split(" ") if len(t) > 2]

def collect_images(roots: List[str]) -> List[str]:
    paths = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    paths.append(os.path.join(dirpath, f))
    return paths

@st.cache_resource(show_spinner=False)
def cached_image_list(roots: Tuple[str, ...]) -> List[str]:
    return collect_images(list(roots))

def score_image_filename(query_tokens: List[str], filename: str) -> int:
    name_tokens = set(_tokens(os.path.basename(filename)))
    score = sum(1 for t in query_tokens if t in name_tokens)
    path_tokens = set(_tokens(filename))
    score += int(sum(1 for t in query_tokens if t in path_tokens) * 0.5)
    return score

def find_related_images(query: str, roots: List[str], limit: int = 3, min_score: int = 1) -> List[str]:
    imgs = cached_image_list(tuple(roots))
    q_tokens = _tokens(query)
    if not q_tokens or not imgs:
        return []
    scored = []
    for p in imgs:
        s = score_image_filename(q_tokens, p)
        if s > 0:
            scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for s, p in scored if s >= min_score][:limit]
    return top

# =========================
# Prompt e streaming
# =========================
SYSTEM_PROMPT = (
    """
Você é um assistente do time de vendas do Grupo Zelo.
Seu papel é responder dúvidas dos vendedores sobre a venda de planos funerários, sempre com linguagem simples, como se fosse uma conversa entre colegas de trabalho.
Fale simples, com exemplos práticos e passo a passo. Se não houver contexto suficiente, diga o que falta.
Quando houver imagem com nome relacionado ao tema (plano, benefício, objeção), complemente mostrando a imagem.
    """.strip()
)

def build_user_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return f"Pergunta: {question}\n\nSem contexto disponível. Responda com o que for seguro e prático."
    ctx = []
    for i, r in enumerate(retrieved, start=1):
        src = os.path.relpath(r["source_path"], KB_DIR)
        ctx.append(f"[Fonte {i}] ({src} #chunk{r['chunk_id']} score={r['score']:.3f})\n{r['text']}\n")
    context_block = "\n\n".join(ctx)
    return (
        f"CONTEXTOS RELEVANTES (não invente fora deles):\n{context_block}\n\n"
        f"PERGUNTA: {question}\n"
        f"INSTRUÇÕES: Fale simples e objetivo, com exemplos práticos aplicáveis agora."
    )

def stream_answer(messages: List[Dict[str, str]]):
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        temperature=0.2,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content

# =========================
# UI – estilo ChatGPT + sidebar
# =========================
# Header
try:
    st.image("images.png", use_container_width=True)
except Exception:
    pass
st.title("💬 ZELO Chat")
st.caption("Chat ao vivo no estilo GPT. Acesso restrito por CPF. Lendo PDFs/TXT/MD de `kb/` e mostrando imagens de `kb/`, `img/` e `assets/`.")

# Sidebar – histórico e admin
with st.sidebar:
    st.header("📚 Histórico")
    if "history" not in st.session_state:
        st.session_state["history"] = []  # [{'role': 'user'/'assistant', 'content': str, 'sources': [...], 'images': [...]}]

    # Navegação por perguntas
    if st.session_state["history"]:
        perguntas = [
            f"{i+1:02d}. {m['content'][:60]}{'...' if len(m['content'])>60 else ''}"
            for i, m in enumerate([h for h in st.session_state["history"] if h["role"] == "user"])
        ]
        idxs = [i for i, h in enumerate(st.session_state["history"]) if h["role"] == "user"]
        sel = st.selectbox("Navegar por perguntas:", options=list(range(len(perguntas))),
                           format_func=lambda i: perguntas[i])
        if 0 <= sel < len(idxs):
            i_hist = idxs[sel]
            st.markdown("**Pergunta:**")
            st.write(st.session_state["history"][i_hist]["content"])
            if i_hist + 1 < len(st.session_state["history"]) and st.session_state["history"][i_hist+1]["role"] == "assistant":
                st.markdown("**Resposta:**")
                st.write(st.session_state["history"][i_hist+1]["content"])
    else:
        st.info("Sem perguntas ainda. Envie a primeira!")

    st.divider()
    st.header("⚙️ Base")
    kb_index = build_index(KB_DIR)  # cacheado
    img_count = len(cached_image_list(tuple(IMG_DIRS)))
    st.caption(
        f"Arquivos: {kb_index['files_total']} • Chunks: {kb_index['chunks_total']} • "
        f"Indexado: {kb_index['built_at']} • Imagens: {img_count}"
    )
    colA, colB = st.columns(2)
    with colA:
        if st.button("🔁 Recarregar base"):
            st.cache_resource.clear()
            st.rerun()
    with colB:
        if st.button("🧹 Limpar histórico"):
            st.session_state["history"] = []
            st.rerun()

    st.divider()
    if "include_images" not in st.session_state:
        st.session_state["include_images"] = True
    st.checkbox("Incluir imagens automaticamente", key="include_images", value=st.session_state["include_images"])

# Linha do tempo (mensagens)
for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            for p in msg["images"]:
                try:
                    st.image(p, caption=os.path.basename(p), use_container_width=True)
                except Exception:
                    st.write(f"(não consegui exibir a imagem: {p})")
        if msg.get("sources"):
            with st.expander("Fontes usadas"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# Entrada do usuário
user_query = st.chat_input("Digite sua pergunta…")

if user_query:
    # 1) Mostrar pergunta
    st.session_state["history"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2) Recuperação textual
    retrieved = search_index(user_query, kb_index, k=5)
    user_prompt = build_user_prompt(user_query, retrieved)
    sources_list = [
        f"[Fonte {i+1}] {os.path.relpath(r['source_path'], KB_DIR)} #chunk{r['chunk_id']} (score={r['score']:.3f})"
        for i, r in enumerate(retrieved)
    ]

    # 3) (Opcional) Busca de imagens relacionadas
    images_related: List[str] = []
    if st.session_state.get("include_images", True):
        images_related = find_related_images(user_query, IMG_DIRS, limit=3, min_score=1)

    # 4) Geração (streaming)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        for token in stream_answer(messages):
            acc += token
            placeholder.markdown(acc)
        final_answer = acc.strip() or "Desculpe, não consegui gerar uma resposta agora. Tente reformular."

        # Mostrar imagens, se houver
        if images_related:
            st.markdown("**Imagens relacionadas:**")
            for p in images_related:
                try:
                    st.image(p, caption=os.path.basename(p), use_container_width=True)
                except Exception:
                    st.write(f"(não consegui exibir a imagem: {p})")

        # Mostrar fontes
        if sources_list:
            with st.expander("Fontes usadas"):
                for s in sources_list:
                    st.markdown(f"- {s}")

    # 5) Persistir no histórico
    st.session_state["history"].append({
        "role": "assistant",
        "content": final_answer,
        "sources": sources_list,
        "images": images_related,
    })

# Rodapé
st.caption(
    "Acesso restrito por CPF (auth/usuarios.xlsx). Base: PDFs/TXT/MD em `kb/`. "
    "Imagens procuradas em `kb/`, `img/` e `assets/` por nome. Para PDFs escaneados (imagem), faça OCR antes (ex.: ocrmypdf)."
)
