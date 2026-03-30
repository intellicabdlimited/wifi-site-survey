from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KBChunk:
    chunk_id: str
    source_type: str
    source_path: str
    title: str
    text: str


class LocalKnowledgeBase:
    def __init__(self, chunks: List[KBChunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english") if chunks else None
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks]) if chunks else None

    @classmethod
    def build_from_run_folder(cls, run_dir: Path | str) -> "LocalKnowledgeBase":
        run_dir = Path(run_dir)
        chunks: List[KBChunk] = []

        for csv_path in sorted(run_dir.rglob("*_curve_tables.csv")):
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            group_cols = [c for c in ["router_key", "floor_name", "band", "scenario"] if c in df.columns]
            if not group_cols:
                continue
            y_col = "p50" if "p50" in df.columns else ("mean" if "mean" in df.columns else None)
            if not y_col:
                continue
            grouped = df.groupby(group_cols, dropna=False)
            for idx, (key, part) in enumerate(grouped):
                if not isinstance(key, tuple):
                    key = (key,)
                summary = {
                    group_cols[i]: str(key[i]) for i in range(len(group_cols))
                }
                text = (
                    f"Curve summary from {csv_path.name}. "
                    + ", ".join(f"{k}={v}" for k, v in summary.items())
                    + f". average={pd.to_numeric(part[y_col], errors='coerce').mean():.2f}. "
                    + f"min={pd.to_numeric(part[y_col], errors='coerce').min():.2f}. "
                    + f"max={pd.to_numeric(part[y_col], errors='coerce').max():.2f}."
                )
                chunks.append(KBChunk(f"curve-{csv_path.stem}-{idx}", "curve_table", str(csv_path), csv_path.name, text))

        for index_csv in sorted(run_dir.rglob("_index.csv")):
            try:
                df = pd.read_csv(index_csv)
            except Exception:
                continue
            for idx, row in df.fillna("").head(500).iterrows():
                text = "; ".join(f"{col}={row[col]}" for col in df.columns if str(row[col]).strip())
                if text:
                    chunks.append(KBChunk(f"index-{index_csv.stem}-{idx}", "index", str(index_csv), index_csv.name, text))

        for manifest in sorted(run_dir.rglob("_extract_manifest.csv")):
            try:
                df = pd.read_csv(manifest)
            except Exception:
                continue
            for idx, row in df.fillna("").iterrows():
                text = "; ".join(f"{col}={row[col]}" for col in df.columns if str(row[col]).strip())
                chunks.append(KBChunk(f"manifest-{manifest.stem}-{idx}", "extract_manifest", str(manifest), manifest.name, text))

        return cls(chunks)

    def save(self, output_path: Path | str) -> Path:
        output_path = Path(output_path)
        output_path.write_text(json.dumps([asdict(c) for c in self.chunks], indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, input_path: Path | str) -> "LocalKnowledgeBase":
        items = json.loads(Path(input_path).read_text(encoding="utf-8"))
        return cls([KBChunk(**item) for item in items])

    def search(self, query: str, top_k: int = 5) -> List[KBChunk]:
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        order = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in order if scores[i] > 0]



def answer_with_context(question: str, kb: LocalKnowledgeBase, model: str = "llama3.2", base_url: str = "http://localhost:11434") -> tuple[str, List[KBChunk]]:
    import requests

    chunks = kb.search(question, top_k=5)
    if not chunks:
        return "No relevant survey context was found in the local knowledge base.", []
    context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
    prompt = (
        "Answer the user's question using only the provided local survey context. "
        "If the answer is uncertain, say so. End with short source references like [1], [2].\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_predict": 500}},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip(), chunks
    except Exception:
        fallback = "\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
        return f"Relevant local context:\n{fallback}", chunks
