# Overall Metrics

Evaluation date: 2026-05-21

All metrics use `data/train.jsonl` as the candidate/ground-truth lookup and `1591` test queries. Ranking is evaluated at duplicate-bucket level.

| Rank | Model | Recall@1 | Recall@5 | Recall@10 | MRR | Notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | Hybrid BM25 + Feature late fusion | 0.2514 | 0.4406 | 0.5387 | 0.3332 | `alpha_bm25=0.8`, `beta_feature=0.2`; fused existing top-10 BM25 and Feature predictions |
| 2 | Tuned BM25 | 0.2458 | 0.4199 | 0.5091 | 0.3223 | `summary_repeat=5`, `k1=2.0`, `b=0.5`, metadata fields enabled, `max_df=0.1` |
| 3 | Feature-based | 0.2087 | 0.3620 | 0.4293 | 0.2757 | Existing feature predictions, `max_candidates=5000` |
| 4 | TF-IDF | 0.1546 | 0.2747 | 0.3294 | 0.2067 | Existing TF-IDF predictions |

Best current result: **Hybrid BM25 + Feature late fusion** with **Recall@10 = 0.5387**.
