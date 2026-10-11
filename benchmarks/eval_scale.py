#!/usr/bin/env python3
"""Scale evaluation harness for Hermes VectorMemoryStore.

Evaluates retrieval quality (Recall@k, MRR) and search latency across synthetic
and realistic corpora (100 to 1,000 memories with realistic distractors).
"""
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hermes_improvements.vector_memory import VectorMemoryStore


def build_synthetic_corpus(n_entries: int) -> Tuple[List[Tuple[str, str, dict]], List[dict]]:
    """Build a realistic benchmark corpus with golden query pairs and distractors."""
    entries = []
    golden_queries = [
        {"query": "cum fac backup la baza de date postgres", "target": "mem_pg_backup"},
        {"query": "configuratie nginx reverse proxy ssl amsterdam", "target": "mem_nginx_ssl"},
        {"query": "detalii despre pilonul 2 si 3 de pensie BCR", "target": "mem_pensie_bcr"},
        {"query": "modul hermes improvements token savings telemetry", "target": "mem_improvements"},
        {"query": "ce model de ceas are vasile galaxy watch ultra", "target": "mem_watch_ultra"},
    ]

    # Core target memories
    entries.append(("mem_pg_backup", "Procedura de backup la baza de date Postgres 16 pe serverul Amsterdam folosind pg_dump si cron.", {"topic": "database"}))
    entries.append(("mem_nginx_ssl", "Configuratia completa Nginx reverse proxy cu Cloudflare SSL pe rednic.org cu securitate intarita.", {"topic": "infra"}))
    entries.append(("mem_pensie_bcr", "Date despre conturile de pensie privata: Pilonul 2 BCR si Pilonul 3 facultativ lunar.", {"topic": "finante"}))
    entries.append(("mem_improvements", "Pachetul hermes improvements aduce economie de 95% tokeni prin telemetrie si vector memory.", {"topic": "hermes"}))
    entries.append(("mem_watch_ultra", "Dispozitive purtate: ceas Samsung Galaxy Watch Ultra conectat la telefonul Galaxy S24 Ultra.", {"topic": "devices"}))

    # Realistic distractor topics
    distractors = [
        "Jurnal de activitate de mentenanta pe statii de lucru Dell din laboratorul de informatica.",
        "Planificare saptamanala pentru orar scolar si repartizare clase liceu tehnologic.",
        "Verificare consum resurse memorie RAM si incarcare procesor pe instanta Oracle Cloud.",
        "Retete de cafea si pauza de dimineata inainte de inceperea cursurilor.",
        "Note de contabilitate si receptie facturi NIR in sistemul primar de gestiune.",
        "Configurare router Cisco si echipamente switch pentru retea securizata.",
        "Actualizare automata pachete apt si securizare firewall UFW cu reguli restrictive.",
        "Monitorizare alerte meteo locale si statie de prognoza pentru temperatura exterioara.",
        "Stremio addon streaming debrid si cataloage de filme 4K remux de inalta calitate.",
        "Instructiuni teologice si documente istorice bisericesti despre patrimoniul local.",
    ]

    for i in range(len(entries), n_entries):
        base_text = distractors[i % len(distractors)]
        entries.append((f"distractor_{i}", f"{base_text} Document de referinta #{i} cu date suplimentare.", {"topic": "general", "idx": i}))

    return entries, golden_queries


def evaluate_retrieval(store: VectorMemoryStore, golden_queries: List[dict], k_values=(1, 3, 5)) -> dict:
    """Compute Recall@k and Mean Reciprocal Rank (MRR)."""
    recall = {f"recall@{k}": 0 for k in k_values}
    mrr_sum = 0.0
    latencies = []

    for item in golden_queries:
        query = item["query"]
        target = item["target"]

        t0 = time.perf_counter()
        results = store.search(query, top_k=max(k_values))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        retrieved_keys = [r["key"] for r in results]

        for k in k_values:
            if target in retrieved_keys[:k]:
                recall[f"recall@{k}"] += 1

        if target in retrieved_keys:
            rank = retrieved_keys.index(target) + 1
            mrr_sum += 1.0 / rank

    n = len(golden_queries)
    metrics = {k: round(v / n, 4) for k, v in recall.items()}
    metrics["MRR"] = round(mrr_sum / n, 4)
    metrics["p50_ms"] = round(sorted(latencies)[len(latencies) // 2], 2)
    metrics["p95_ms"] = round(sorted(latencies)[int(len(latencies) * 0.95)], 2)
    return metrics


def run_scale_benchmark():
    sizes = [50, 200, 500]
    print("=" * 65)
    print(" Hermes Improvements — Retrieval Scale Benchmark Harness")
    print("=" * 65)

    tmp = Path(tempfile.mkdtemp())
    try:
        for size in sizes:
            store_dir = tmp / f"scale_{size}"
            store = VectorMemoryStore(memory_dir=store_dir)
            entries, golden = build_synthetic_corpus(size)

            t0 = time.perf_counter()
            store.bulk_add(entries)
            ingest_ms = (time.perf_counter() - t0) * 1000.0

            metrics = evaluate_retrieval(store, golden)
            print(f"Corpus size: {size:4d} entries | Ingest: {ingest_ms:6.1f}ms")
            print(f"  Recall@1: {metrics['recall@1']:.2f} | Recall@3: {metrics['recall@3']:.2f} | Recall@5: {metrics['recall@5']:.2f}")
            print(f"  MRR:      {metrics['MRR']:.2f} | Latency p50: {metrics['p50_ms']}ms | p95: {metrics['p95_ms']}ms")
            print("-" * 65)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_scale_benchmark()
