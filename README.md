# Bug Report Duplicate Retrieval

Hệ thống tìm kiếm bug report trùng lặp trên dataset Mozilla (~155K bug reports). Triển khai nhiều thuật toán từ cơ bản (TF-IDF, BM25) đến nâng cao (Feature Scoring, Hybrid Fusion), kèm demo giao diện web bằng Streamlit.

## Kết quả

| Thuật toán | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| TF-IDF | 15.46% | 26.96% | 32.94% | 0.206 |
| BM25 | 21.31% | 35.20% | 42.61% | 0.276 |
| Feature Score | 20.99% | 35.76% | 42.93% | 0.273 |
| Feature Diverse | 20.99% | 36.52% | 43.18% | 0.284 |

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