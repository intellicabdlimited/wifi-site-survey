# from __future__ import annotations

# import json
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from typing import Iterable, List

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# @dataclass
# class KBChunk:
#     chunk_id: str
#     source_type: str
#     source_path: str
#     title: str
#     text: str


# class LocalKnowledgeBase:
#     def __init__(self, chunks: List[KBChunk]):
#         self.chunks = chunks
#         self.vectorizer = TfidfVectorizer(stop_words="english") if chunks else None
#         self.matrix = self.vectorizer.fit_transform([c.text for c in chunks]) if chunks else None

#     @classmethod
#     def build_from_run_folder(cls, run_dir: Path | str) -> "LocalKnowledgeBase":
#         run_dir = Path(run_dir)
#         chunks: List[KBChunk] = []

#         for csv_path in sorted(run_dir.rglob("*_curve_tables.csv")):
#             df = pd.read_csv(csv_path)
#             if df.empty:
#                 continue
#             group_cols = [c for c in ["router_key", "floor_name", "band", "scenario"] if c in df.columns]
#             if not group_cols:
#                 continue
#             y_col = "p50" if "p50" in df.columns else ("mean" if "mean" in df.columns else None)
#             if not y_col:
#                 continue
#             grouped = df.groupby(group_cols, dropna=False)
#             for idx, (key, part) in enumerate(grouped):
#                 if not isinstance(key, tuple):
#                     key = (key,)
#                 summary = {
#                     group_cols[i]: str(key[i]) for i in range(len(group_cols))
#                 }
#                 text = (
#                     f"Curve summary from {csv_path.name}. "
#                     + ", ".join(f"{k}={v}" for k, v in summary.items())
#                     + f". average={pd.to_numeric(part[y_col], errors='coerce').mean():.2f}. "
#                     + f"min={pd.to_numeric(part[y_col], errors='coerce').min():.2f}. "
#                     + f"max={pd.to_numeric(part[y_col], errors='coerce').max():.2f}."
#                 )
#                 chunks.append(KBChunk(f"curve-{csv_path.stem}-{idx}", "curve_table", str(csv_path), csv_path.name, text))

#         for index_csv in sorted(run_dir.rglob("_index.csv")):
#             try:
#                 df = pd.read_csv(index_csv)
#             except Exception:
#                 continue
#             for idx, row in df.fillna("").head(500).iterrows():
#                 text = "; ".join(f"{col}={row[col]}" for col in df.columns if str(row[col]).strip())
#                 if text:
#                     chunks.append(KBChunk(f"index-{index_csv.stem}-{idx}", "index", str(index_csv), index_csv.name, text))

#         for manifest in sorted(run_dir.rglob("_extract_manifest.csv")):
#             try:
#                 df = pd.read_csv(manifest)
#             except Exception:
#                 continue
#             for idx, row in df.fillna("").iterrows():
#                 text = "; ".join(f"{col}={row[col]}" for col in df.columns if str(row[col]).strip())
#                 chunks.append(KBChunk(f"manifest-{manifest.stem}-{idx}", "extract_manifest", str(manifest), manifest.name, text))

#         return cls(chunks)

#     def save(self, output_path: Path | str) -> Path:
#         output_path = Path(output_path)
#         output_path.write_text(json.dumps([asdict(c) for c in self.chunks], indent=2), encoding="utf-8")
#         return output_path

#     @classmethod
#     def load(cls, input_path: Path | str) -> "LocalKnowledgeBase":
#         items = json.loads(Path(input_path).read_text(encoding="utf-8"))
#         return cls([KBChunk(**item) for item in items])

#     def search(self, query: str, top_k: int = 5) -> List[KBChunk]:
#         if not self.chunks or self.vectorizer is None or self.matrix is None:
#             return []
#         q = self.vectorizer.transform([query])
#         scores = cosine_similarity(q, self.matrix)[0]
#         order = scores.argsort()[::-1][:top_k]
#         return [self.chunks[i] for i in order if scores[i] > 0]



# def answer_with_context(question: str, kb: LocalKnowledgeBase, model: str = "llama3.2", base_url: str = "http://localhost:11434") -> tuple[str, List[KBChunk]]:
#     import requests

#     chunks = kb.search(question, top_k=5)
#     if not chunks:
#         return "No relevant survey context was found in the local knowledge base.", []
#     context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
#     prompt = (
#         "Answer the user's question using only the provided local survey context. "
#         "If the answer is uncertain, say so. End with short source references like [1], [2].\n\n"
#         f"Question: {question}\n\nContext:\n{context}"
#     )
#     try:
#         resp = requests.post(
#             f"{base_url.rstrip('/')}/api/generate",
#             json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_predict": 500}},
#             timeout=90,
#         )
#         resp.raise_for_status()
#         return resp.json().get("response", "").strip(), chunks
#     except Exception:
#         fallback = "\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
#         return f"Relevant local context:\n{fallback}", chunks


from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from metadata_utils import (
    PARAM_PRETTY,
    canonical_metric_key,
    clean_router_name,
    normalize_band_value,
    normalize_floor_name,
)


@dataclass
class KBChunk:
    chunk_id: str
    source_type: str
    source_path: str
    title: str
    text: str


_HIGHER_IS_BETTER = {
    "signal_strength": True,
    "secondary_signal_strength": True,
    "tertiary_signal_strength": True,
    "snr": True,
    "noise": False,
    "data_rate": True,
    "throughput": True,
    "channel_utilization": False,
    "channel_interference": False,
    "channel_width": True,
    "spectrum_channel_power": True,
    "network_health": True,
    "network_issues": False,
    "number_of_access_points": False,
    "number_of_access_points_or_aps": False,
}


def _prepare_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\b5ghz\b", "5 ghz", text)
    text = re.sub(r"\b2\.4ghz\b", "2.4 ghz", text)
    text = re.sub(r"\b6ghz\b", "6 ghz", text)
    text = text.replace("_", " ").replace("/", " ")
    text = re.sub(r"\bno ap\b", "without mesh", text)
    text = re.sub(r"\bnon mesh\b", "without mesh", text)
    text = re.sub(r"\bmesh setup\b", "with mesh", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _query_expansions(query: str) -> str:
    q = _prepare_text(query)
    extra: List[str] = []
    if "best" in q or "top" in q or "strongest" in q:
        extra.extend(["ranking best highest strongest winner"])
    if "worst" in q or "weakest" in q:
        extra.extend(["ranking worst lowest weakest"])
    if "compare" in q or "versus" in q or "vs" in q:
        extra.extend(["difference comparison compare versus vs"])
    if "throughput" in q:
        extra.extend(["throughput mbps speed"])
    if "signal strength" in q:
        extra.extend(["signal strength dbm rssi"])
    if "snr" in q:
        extra.extend(["snr signal to noise ratio"])
    if "noise" in q:
        extra.extend(["noise floor dbm"])
    if "mesh" in q:
        extra.extend(["with mesh without mesh scenario"])
    return " ".join([q] + extra).strip()


def _infer_metric_key(path: Path, df: pd.DataFrame | None = None) -> str:
    candidates: List[str] = []
    candidates.append(path.stem)
    candidates.append(path.parent.name)
    if path.parent.parent != path.parent:
        candidates.append(path.parent.parent.name)
    if df is not None:
        for col in ["parameter_key", "parameter_display", "metric", "metric_folder"]:
            if col in df.columns:
                vals = [str(v) for v in df[col].dropna().astype(str).unique()[:5]]
                candidates.extend(vals)
    for raw in candidates:
        raw = re.sub(r"_mesh_curve_tables$", "", str(raw))
        raw = re.sub(r"_curve_tables$", "", raw)
        key = canonical_metric_key(raw)
        if key:
            return key
    return "survey metric"


def _metric_pretty(metric_key: str) -> str:
    return PARAM_PRETTY.get(metric_key, metric_key.replace("_", " ").title())


def _fmt_value(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):.2f}"


def _group_mean(part: pd.DataFrame, col: str) -> float:
    series = pd.to_numeric(part[col], errors="coerce").dropna()
    return float(series.mean()) if not series.empty else float("nan")


def _group_min(part: pd.DataFrame, col: str) -> float:
    series = pd.to_numeric(part[col], errors="coerce").dropna()
    return float(series.min()) if not series.empty else float("nan")


def _group_max(part: pd.DataFrame, col: str) -> float:
    series = pd.to_numeric(part[col], errors="coerce").dropna()
    return float(series.max()) if not series.empty else float("nan")


def _ranking_lines(stats_rows: Sequence[dict], metric_pretty: str, floor_name: str, band: str, higher_is_better: bool) -> str:
    if not stats_rows:
        return ""
    best = stats_rows[0]
    worst = stats_rows[-1]
    gap = abs(float(best["avg"]) - float(worst["avg"]))
    parts = [f"Ranking for {metric_pretty} on {floor_name} {band}: "]
    parts.append(" > ".join(f"{row['router']} ({_fmt_value(row['avg'])})" for row in stats_rows))
    parts.append(
        f". Best router is {best['router']} with average {_fmt_value(best['avg'])}. "
        f"Worst router is {worst['router']} with average {_fmt_value(worst['avg'])}. "
        f"Best to worst gap is {_fmt_value(gap)}. "
        f"Interpretation: {'higher is better' if higher_is_better else 'lower is better'}."
    )
    if len(stats_rows) >= 3:
        mids = ", ".join(f"{row['router']} ({_fmt_value(row['avg'])})" for row in stats_rows[1:-1])
        parts.append(f" Mid-tier routers: {mids}.")
    return "".join(parts)


class LocalKnowledgeBase:
    def __init__(self, chunks: List[KBChunk]):
        self.chunks = chunks
        self.search_texts = [_prepare_text(f"{c.title}. {c.text}") for c in chunks]
        self.token_sets = [set(text.split()) for text in self.search_texts]

        if chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            self.matrix = self.vectorizer.fit_transform(self.search_texts)
            self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            self.char_matrix = self.char_vectorizer.fit_transform(self.search_texts)
        else:
            self.vectorizer = None
            self.matrix = None
            self.char_vectorizer = None
            self.char_matrix = None

    @classmethod
    def build_from_run_folder(cls, run_dir: Path | str) -> "LocalKnowledgeBase":
        run_dir = Path(run_dir)
        chunks: List[KBChunk] = []

        for csv_path in sorted(run_dir.rglob("*_curve_tables.csv")):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if df.empty:
                continue

            metric_key = _infer_metric_key(csv_path, df)
            metric_pretty = _metric_pretty(metric_key)
            higher_is_better = _HIGHER_IS_BETTER.get(metric_key, True)

            if "band" in df.columns:
                df["band"] = df["band"].astype(str).map(normalize_band_value)
            if "floor_name" in df.columns:
                df["floor_name"] = df["floor_name"].astype(str).map(normalize_floor_name)
            if "router_key" in df.columns:
                df["router_key"] = df["router_key"].astype(str).map(clean_router_name)
            if "router_display" in df.columns:
                df["router_display"] = df["router_display"].astype(str).map(clean_router_name)
            if "scenario" in df.columns:
                df["scenario"] = df["scenario"].astype(str)
            if "scenario_label" in df.columns:
                df["scenario_label"] = df["scenario_label"].astype(str)

            y_col = "p50" if "p50" in df.columns else ("mean" if "mean" in df.columns else None)
            if not y_col:
                continue

            group_cols = [c for c in ["router_key", "floor_name", "band", "scenario"] if c in df.columns]
            if group_cols:
                grouped = df.groupby(group_cols, dropna=False)
                for idx, (key, part) in enumerate(grouped):
                    if not isinstance(key, tuple):
                        key = (key,)
                    summary = {group_cols[i]: str(key[i]) for i in range(len(group_cols))}
                    router_name = clean_router_name(summary.get("router_key", summary.get("router_display", "")))
                    floor_name = normalize_floor_name(summary.get("floor_name", ""))
                    band = normalize_band_value(summary.get("band", ""))
                    scenario = summary.get("scenario", "")
                    avg = _group_mean(part, y_col)
                    mn = _group_min(part, y_col)
                    mx = _group_max(part, y_col)
                    count = int(pd.to_numeric(part[y_col], errors="coerce").dropna().shape[0])
                    scenario_text = f" scenario {scenario}." if scenario else ""
                    text = (
                        f"Curve-table summary for {metric_pretty}. Router {router_name}. Floor {floor_name}. Band {band}."
                        f"{scenario_text} Average {metric_pretty} is {_fmt_value(avg)}."
                        f" Minimum is {_fmt_value(mn)}. Maximum is {_fmt_value(mx)}. Samples {count}."
                        f" Source file {csv_path.name}."
                    )
                    title = f"{metric_pretty} | {router_name or 'all routers'} | {floor_name or 'all floors'} | {band or 'all bands'}"
                    chunks.append(KBChunk(f"curve-{csv_path.stem}-{idx}", "curve_table", str(csv_path), title, text))

            # Router ranking chunks for standard router-vs-router tables.
            if {"router_key", "floor_name", "band"}.issubset(df.columns):
                rank_group_cols = ["floor_name", "band"]
                if "scenario" in df.columns:
                    rank_group_cols.append("scenario")
                grouped = df.groupby(rank_group_cols, dropna=False)
                for idx, (key, part) in enumerate(grouped):
                    if not isinstance(key, tuple):
                        key = (key,)
                    info = {rank_group_cols[i]: str(key[i]) for i in range(len(rank_group_cols))}
                    floor_name = normalize_floor_name(info.get("floor_name", ""))
                    band = normalize_band_value(info.get("band", ""))
                    scenario = info.get("scenario", "")
                    stats_rows = []
                    for router_key, router_part in part.groupby("router_key", dropna=False):
                        router_name = clean_router_name(str(router_key))
                        series = pd.to_numeric(router_part[y_col], errors="coerce").dropna()
                        if series.empty:
                            continue
                        stats_rows.append({"router": router_name, "avg": float(series.mean())})
                    stats_rows.sort(key=lambda row: row["avg"], reverse=higher_is_better)
                    if stats_rows:
                        scenario_text = f" in scenario {scenario}" if scenario else ""
                        title = f"Ranking | {metric_pretty} | {floor_name} | {band}{scenario_text}"
                        text = _ranking_lines(stats_rows, metric_pretty, floor_name, band, higher_is_better)
                        chunks.append(KBChunk(f"ranking-{csv_path.stem}-{idx}", "curve_ranking", str(csv_path), title, text))

            # With-mesh vs without-mesh comparison chunks.
            if {"router_key", "floor_name", "band", "scenario"}.issubset(df.columns):
                grouped = df.groupby(["router_key", "floor_name", "band"], dropna=False)
                for idx, (key, part) in enumerate(grouped):
                    router_key, floor_name, band = key
                    router_name = clean_router_name(str(router_key))
                    floor_name = normalize_floor_name(str(floor_name))
                    band = normalize_band_value(str(band))
                    scenario_stats = {}
                    for scenario, scenario_part in part.groupby("scenario", dropna=False):
                        scenario_stats[str(scenario)] = _group_mean(scenario_part, y_col)
                    if scenario_stats:
                        with_val = scenario_stats.get("with_mesh")
                        without_val = scenario_stats.get("without_mesh")
                        text = (
                            f"Scenario comparison for {metric_pretty}. Router {router_name}. Floor {floor_name}. Band {band}. "
                            f"With mesh average is {_fmt_value(with_val)}. Without mesh average is {_fmt_value(without_val)}."
                        )
                        if with_val == with_val and without_val == without_val:
                            diff = with_val - without_val
                            better = "with mesh" if (diff >= 0 if higher_is_better else diff <= 0) else "without mesh"
                            text += f" Difference is {_fmt_value(abs(diff))}. Better scenario is {better}."
                        title = f"Mesh comparison | {metric_pretty} | {router_name} | {floor_name} | {band}"
                        chunks.append(KBChunk(f"meshcmp-{csv_path.stem}-{idx}", "mesh_curve_comparison", str(csv_path), title, text))

        for index_csv in sorted(run_dir.rglob("_index.csv")):
            try:
                df = pd.read_csv(index_csv)
            except Exception:
                continue
            for idx, row in df.fillna("").head(800).iterrows():
                metric_key = canonical_metric_key(row.get("parameter_key") or row.get("parameter_display") or _infer_metric_key(index_csv)) or ""
                metric_pretty = _metric_pretty(metric_key) if metric_key else "Survey metric"
                router_name = clean_router_name(row.get("router_key", ""))
                floor_name = normalize_floor_name(row.get("floor_name", ""))
                band = normalize_band_value(row.get("band", ""))
                title = f"OCR index | {metric_pretty} | {router_name} | {floor_name} | {band}"
                text = (
                    f"OCR output index row for {metric_pretty}. Router {router_name}. Floor {floor_name}. Band {band}. "
                    f"Heatmap path {row.get('heatmap_path', '')}. Scale path {row.get('scale_path', '')}. "
                    f"CSV output {row.get('csv', '')}. Caption {row.get('caption_text', '')}."
                )
                chunks.append(KBChunk(f"index-{index_csv.stem}-{idx}", "index", str(index_csv), title, text))

        for manifest in sorted(run_dir.rglob("_extract_manifest.csv")):
            try:
                df = pd.read_csv(manifest)
            except Exception:
                continue
            for idx, row in df.fillna("").head(1200).iterrows():
                metric_key = canonical_metric_key(row.get("parameter_key") or row.get("parameter_display") or "") or ""
                metric_pretty = _metric_pretty(metric_key) if metric_key else "Survey metric"
                router_name = clean_router_name(row.get("router_key", ""))
                floor_name = normalize_floor_name(row.get("floor_name", ""))
                band = normalize_band_value(row.get("band", ""))
                title = f"Extracted asset | {metric_pretty} | {router_name} | {floor_name} | {band}"
                text = (
                    f"Extracted survey asset for {metric_pretty}. Router {router_name}. Floor {floor_name}. Band {band}. "
                    f"Role {row.get('role', '')}. Caption {row.get('caption_text', '')}. Path {row.get('path', '')}. "
                    f"Source DOCX {row.get('source_docx', '')}."
                )
                chunks.append(KBChunk(f"manifest-{manifest.stem}-{idx}", "extract_manifest", str(manifest), title, text))

        return cls(chunks)

    def save(self, output_path: Path | str) -> Path:
        output_path = Path(output_path)
        output_path.write_text(json.dumps([asdict(c) for c in self.chunks], indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, input_path: Path | str) -> "LocalKnowledgeBase":
        items = json.loads(Path(input_path).read_text(encoding="utf-8"))
        return cls([KBChunk(**item) for item in items])

    def search_with_scores(self, query: str, top_k: int = 5) -> List[Tuple[KBChunk, float]]:
        if not self.chunks or self.vectorizer is None or self.matrix is None or self.char_vectorizer is None or self.char_matrix is None:
            return []

        query_text = _query_expansions(query)
        q_word = self.vectorizer.transform([query_text])
        q_char = self.char_vectorizer.transform([query_text])
        word_scores = cosine_similarity(q_word, self.matrix)[0]
        char_scores = cosine_similarity(q_char, self.char_matrix)[0]

        q_tokens = set(query_text.split())
        scored: List[Tuple[int, float]] = []
        for idx, (w_score, c_score) in enumerate(zip(word_scores, char_scores)):
            overlap = 0.0
            if q_tokens:
                overlap = len(q_tokens & self.token_sets[idx]) / max(len(q_tokens), 1)
            title_bonus = 0.10 if any(tok in _prepare_text(self.chunks[idx].title) for tok in q_tokens if len(tok) > 2) else 0.0
            score = (0.75 * float(w_score)) + (0.20 * float(c_score)) + (0.25 * float(overlap)) + title_bonus
            scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        top = scored[: max(int(top_k), 1)]
        # Return best few even for weak lexical matches so the chat does not fail closed.
        return [(self.chunks[idx], float(score)) for idx, score in top if score > 0 or bool(self.chunks)]

    def search(self, query: str, top_k: int = 5) -> List[KBChunk]:
        return [chunk for chunk, _score in self.search_with_scores(query, top_k=top_k)]


def _fallback_answer(question: str, scored_chunks: Sequence[Tuple[KBChunk, float]]) -> str:
    if not scored_chunks:
        return "No relevant survey context was found in the local knowledge base."

    top_chunks = list(scored_chunks[:3])
    lines = ["I found these survey facts in the local knowledge base:"]
    for idx, (chunk, _score) in enumerate(top_chunks, start=1):
        snippet = re.sub(r"\s+", " ", chunk.text).strip()
        if len(snippet) > 260:
            snippet = snippet[:257].rstrip() + "..."
        lines.append(f"- {snippet} [{idx}]")
    return "\n".join(lines)



def answer_with_context(question: str, kb: LocalKnowledgeBase, model: str = "gemma3:4b", base_url: str = "http://localhost:11434") -> tuple[str, List[KBChunk]]:
    import requests # model: str = "gemma3:4b" slower, model: str = "llama3.2" faster but less accurate for numeric details

    scored = kb.search_with_scores(question, top_k=5)
    chunks = [chunk for chunk, _score in scored]
    if not chunks:
        return "No relevant survey context was found in the local knowledge base.", []

    context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
    prompt = (
        "Answer the user's question using only the provided local survey context. "
        "Prefer exact router, floor, band, metric, and scenario names from the context. "
        "If the context does not fully answer the question, say what is known and what is missing. "
        "End with short source references like [1], [2].\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_predict": 500}},
            timeout=45,
        )
        resp.raise_for_status()
        answer = str(resp.json().get("response", "")).strip()
        if answer:
            return answer, chunks
    except Exception:
        pass

    return _fallback_answer(question, scored), chunks
