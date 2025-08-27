# ingest_kb.py
# Converte arquivos em kb_raw/ para .txt em kb/
# Requisitos: pip install pypdf python-docx chardet

import pathlib
import csv
import chardet
from pypdf import PdfReader
from docx import Document

RAW_DIR = pathlib.Path("kb_raw")
OUT_DIR = pathlib.Path("kb")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_txt(name: str, text: str):
    p = OUT_DIR / (name + ".txt")
    p.write_text(text.strip(), encoding="utf-8", errors="ignore")
    print("gravado:", p)

def from_pdf(path: pathlib.Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def from_docx(path: pathlib.Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def from_csv(path: pathlib.Path) -> str:
    with open(path, "rb") as fb:
        enc = chardet.detect(fb.read()).get("encoding") or "utf-8"
    rows = []
    with open(path, "r", encoding=enc, newline="") as fh:
        rdr = csv.reader(fh)
        for r in rdr:
            rows.append(" | ".join(r))
    return "\n".join(rows)

def from_txt_md(path: pathlib.Path) -> str:
    with open(path, "rb") as fb:
        enc = chardet.detect(fb.read()).get("encoding") or "utf-8"
    return path.read_text(encoding=enc, errors="ignore")

def convert_one(p: pathlib.Path) -> None:
    name = p.stem
    if p.suffix.lower() == ".pdf":
        text = from_pdf(p)
    elif p.suffix.lower() == ".docx":
        text = from_docx(p)
    elif p.suffix.lower() in (".csv",):
        text = from_csv(p)
    elif p.suffix.lower() in (".txt", ".md"):
        text = from_txt_md(p)
    else:
        return
    # (opcional) prefixe metadados para melhorar busca:
    # header = "[TAGS: televendas, PAP]\n[TITULO: Politica de Descontos]\n\n"
    # text = header + text
    save_txt(name, text)

def main():
    if not RAW_DIR.exists():
        print("Pasta kb_raw/ não encontrada.")
        return
    files = [p for p in RAW_DIR.rglob("*") if p.is_file()]
    if not files:
        print("Nenhum arquivo em kb_raw/.")
        return
    for p in files:
        try:
            convert_one(p)
        except Exception as e:
            print("falhou:", p, e)

if __name__ == "__main__":
    main()
