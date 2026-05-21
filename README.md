# Bug Report Duplicate Retrieval

Hệ thống tìm kiếm bug report trùng lặp trên dataset Mozilla (~155K bug reports). Triển khai nhiều thuật toán từ cơ bản (TF-IDF, BM25) đến nâng cao (Feature Scoring, Hybrid Fusion), kèm demo giao diện web bằng Streamlit.

## Kết quả

| Rank | Model | Recall@1 | Recall@5 | Recall@10 | MRR | Notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | Hybrid BM25 + Feature late fusion | 0.2514 | 0.4406 | 0.5387 | 0.3332 | `alpha_bm25=0.8`, `beta_feature=0.2`; fused existing top-10 BM25 and Feature predictions |
| 2 | Tuned BM25 | 0.2458 | 0.4199 | 0.5091 | 0.3223 | `summary_repeat=5`, `k1=2.0`, `b=0.5`, metadata fields enabled, `max_df=0.1` |
| 3 | Feature-based | 0.2087 | 0.3620 | 0.4293 | 0.2757 | Existing feature predictions, `max_candidates=5000` |
| 4 | TF-IDF | 0.1546 | 0.2747 | 0.3294 | 0.2067 | Existing TF-IDF predictions |

## Cấu trúc thư mục

```
bug-report-retrieval/
├── main.py                          # Entry point
├── Plan.md                          # Kế hoạch tổng thể
├── data/
│   ├── raw/mozilla.json             # Dataset gốc
│   ├── processed/processed_bugs.jsonl
│   ├── train.jsonl                  # Tập train (~155K)
│   ├── test.jsonl                   # Tập test (~1.6K)
│   └── bucket_map.json             # Ánh xạ bug_id → bucket_id
├── src/
│   ├── preprocessing/
│   │   ├── build_schema.py          # Chuẩn hóa schema
│   │   ├── clean_text.py            # Tiền xử lý văn bản
│   │   └── split_by_time.py         # Chia train/test theo thời gian
│   ├── retrieval/
│   │   ├── tfidf_retriever.py       # TF-IDF baseline
│   │   ├── bm25_retriever.py        # BM25 (Okapi)
│   │   ├── bug_feature_scorer.py    # Bộ scorer 9 features
│   │   ├── feature_retriever.py     # Feature-based retriever
│   │   ├── feature_diverse_retriever.py  # Feature mở rộng + boost
│   │   ├── hybrid_retriever.py      # Hybrid fusion (BM25 + Feature)
│   │   └── reranker.py              # Score normalization & fusion
│   ├── eval/
│   │   └── metrics.py               # Recall@K, MRR
│   └── app/
│       ├── demo.py                  # Streamlit demo
│       └── build_cache.py           # Build cache cho demo
└── reports/
    ├── bm25_metrics.json
    ├── tfidf_metrics.json
    ├── feature_metrics.json
    ├── feature_diverse_metrics.json
    ├── *_predictions.jsonl
    ├── feature_design.md
    ├── schema.md
    └── cache/                       # Cache cho demo (pkl)
```

## Pipeline

```
mozilla.json → [preprocessing] → train.jsonl + test.jsonl
  → [TF-IDF / BM25]           → baseline predictions
  → [Feature Score]            → feature predictions
  → [Feature Diverse]          → feature diverse predictions
  → [Hybrid Fusion]            → hybrid predictions (BM25 + Feature)
  → [metrics.py]               → Recall@1/5/10, MRR
```

## Chạy demo

```bash
# Build cache (chạy lần đầu)
python src/app/build_cache.py

# Chạy demo
streamlit run src/app/demo.py
```

## Yêu cầu

```bash
pip install streamlit
```
