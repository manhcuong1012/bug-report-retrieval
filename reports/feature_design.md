# Feature-Based Bug Scoring - Design Document

## 1. Tong quan

Mo hinh feature-based scoring danh gia muc do tuong dong giua 2 bug report dua tren 9 dac trung (features) thuoc 4 nhom: text similarity, metadata matching, technical token overlap, va query-aware heuristics.

Khac voi BM25/TF-IDF chi dua vao term matching tren toan bo van ban, mo hinh nay tach biet tung thanh phan (summary, description, component, severity, ...) va cho trong so rieng, giup phat hien duplicate ngay ca khi text khong trung manh.

## 2. Danh sach features

### Nhom A: Summary Similarity

| Feature | Cong thuc | Mo ta |
|---|---|---|
| `summary_jaccard` | \|A ∩ B\| / \|A ∪ B\| | Jaccard similarity giua token set cua 2 summary |
| `summary_overlap` | \|A ∩ B\| / \|A\| | Ty le token query summary xuat hien trong candidate |

Tinh tren `summary` sau khi normalize (lowercase, bo ky tu dac biet, normalize whitespace).

### Nhom B: Description Similarity

| Feature | Cong thuc | Mo ta |
|---|---|---|
| `desc_jaccard` | \|A ∩ B\| / \|A ∪ B\| | Jaccard similarity giua token set cua 2 description |
| `desc_overlap` | \|A ∩ B\| / \|A\| | Ty le token query description xuat hien trong candidate |

Description dai va noisy hon summary, nen weight thap hon.

### Nhom C: Metadata Matching

| Feature | Cong thuc | Mo ta |
|---|---|---|
| `component_match` | 1.0 neu trung, 0.0 neu khac | So khop component (884 gia tri rieng biet) |
| `severity_match` | 1.0 neu trung, 0.0 neu khac | So khop severity |
| `priority_match` | 1.0 neu trung, 0.0 neu khac | So khop priority (tra 0.0 neu ca hai la UNKNOWN) |

Luu y tu du lieu:
- 93.2% bug co severity = "normal" -> `severity_match` phan biet yeu
- 44.8% bug co priority = "UNKNOWN" -> `priority_match` signal yeu
- 884 component rieng biet -> `component_match` la signal manh nhat trong nhom metadata

### Nhom D: Technical Token Overlap

| Feature | Cong thuc | Mo ta |
|---|---|---|
| `tech_keyword_overlap` | \|K_q ∩ K_c\| / max(\|K_q\|, 1) | Overlap cua technical keywords |
| `tech_token_jaccard` | \|T_q ∩ T_c\| / \|T_q ∪ T_c\| | Jaccard cua toan bo technical tokens |

**Technical keywords** (28 tu):
crash, abort, segfault, timeout, hang, memory, leak, exception, null, stack, assert, assertion, overflow, underflow, deadlock, race, corrupt, corruption, oom, sigabrt, sigsegv, error, fail, failed, failure, broken, regression

**Pattern-based tokens** (nhan dien bang regex):
- File paths: token co extension (.js, .cpp, .html, .css, .xml, .py, ...)
- Class/module: CamelCase hoac chua `::`
- Stack frames: `#0`, `#1`, ...
- Mozilla-specific: treeherder, taskcluster, bugzilla, gecko, necko, mochitest, xpcshell, reftest, crashtest, wpt, spidermonkey, webrender, stylo
- Prefix tokens: bat dau bang `nsi` hoac `moz`

**Noise filter**: loai bo cac token tu User-Agent boilerplate (mozilla/5.0, gecko/20100101, applewebkit, firefox, chrome, safari, linux, windows, x86_64, x11)

## 3. Trong so

### Base weights

| Feature | Weight | Ly do |
|---|---|---|
| `summary_jaccard` | 0.30 | Summary ngan, dac trung nhat cho duplicate |
| `component_match` | 0.22 | 884 components -> signal manh |
| `desc_jaccard` | 0.15 | Description huu ich nhung noisy |
| `tech_token_jaccard` | 0.12 | Bat duplicate khi technical context giong nhau |
| `tech_keyword_overlap` | 0.08 | Crash/leak/timeout bugs duplicate theo keyword |
| `summary_overlap` | 0.05 | Bo sung cho jaccard |
| `desc_overlap` | 0.05 | Bo sung cho jaccard |
| `severity_match` | 0.02 | Signal rat yeu (93% la normal) |
| `priority_match` | 0.01 | Signal yeu nhat (45% la UNKNOWN) |
| **Tong** | **1.00** | |

**Ghi chu tuning**: weights ban dau (summary_jaccard=0.25, component_match=0.20, severity=0.05, priority=0.03) cho Recall@1=19.92%. Sau khi tang summary_jaccard va component_match, giam severity/priority (signal yeu theo du lieu), Recall@1 tang len 20.87% (+0.95%), MRR tang 0.268 -> 0.273.

### Query-aware heuristics

Dieu chinh trong so dua tren dac diem query:

| Dieu kien | Feature bi dieu chinh | Multiplier |
|---|---|---|
| Query ngan (summary < 8 tokens VA description < 20 tokens) | `component_match` | x1.5 |
| | `summary_jaccard` | x1.3 |
| Query co stacktrace (>= 3 matches pattern `#N 0x` hoac `at file:line`) | `tech_token_jaccard` | x1.5 |
| Query co nhieu file paths (>= 3 tokens co dang path) | `tech_token_jaccard` | x1.3 |
| | `desc_overlap` | x1.2 |

**Quan trong**: Sau khi nhan multiplier, weights duoc **re-normalize** ve tong = 1.0:
```
total = sum(W.values())
W = {k: v / total for k, v in W.items()}
```

## 4. Pre-filtering strategy

Brute-force 154,869 candidates/query qua cham. Su dung pre-filtering:

1. **Inverted index**: map token -> [bug_ids] tren train set
2. **Loai token rac**: bo token co len <= 2, bo token co document frequency > 10% tong docs (57 tokens bi loai)
3. **Component index**: them candidate cung component (du khong co token chung)
4. **Giu max 5,000 candidates/query** (uu tien candidate co nhieu token chung)
5. **Loc thoi gian**: chi giu candidate co created_at <= query.created_at

Ket qua: trung binh 4,991 candidates/query (gan max 5,000).

## 5. Ket qua

| Metric | TF-IDF | BM25 | Feature-based |
|---|---|---|---|
| Recall@1 | 15.46% | 21.31% | 20.87% |
| Recall@5 | 26.96% | 35.20% | 35.64% |
| Recall@10 | 32.94% | 42.61% | 42.93% |
| MRR | 0.2059 | 0.2763 | 0.2734 |

- Thang TF-IDF hoan toan
- Thang BM25 o Recall@5 (+0.44%) va Recall@10 (+0.32%)
- Gan sat BM25 o Recall@1 (-0.44%) va MRR (-0.003)
- Diem manh bo sung: feature-based tot o top 5-10, BM25 tot o top 1 -> hybrid se ket hop duoc

## 6. Cach su dung

### Chay retrieval
```bash
python src/retrieval/feature_retriever.py
```

### Tham so tuy chinh
```bash
python src/retrieval/feature_retriever.py \
  --train-records data/train.jsonl \
  --test-records data/test.jsonl \
  --predictions-output reports/feature_predictions.jsonl \
  --metrics-output reports/feature_metrics.json \
  --top-k 10 \
  --max-candidates 5000
```

### Import trong code khac (cho nguoi 3)
```python
from retrieval.bug_feature_scorer import precompute_bug, build_query_profile, score_pair

# Precompute 1 lan cho moi bug
query_pre = precompute_bug(query_record)
candidate_pre = precompute_bug(candidate_record)
query_profile = build_query_profile(query_record["summary"], query_record["description"])

# Tinh score cho 1 cap
score = score_pair(query_pre, candidate_pre, query_profile)
```

## 7. Files ban giao

| File | Vi tri | Mo ta |
|---|---|---|
| `bug_feature_scorer.py` | `src/retrieval/` | 9 features + scoring logic |
| `feature_retriever.py` | `src/retrieval/` | Pipeline retrieval hoan chinh |
| `feature_predictions.jsonl` | `reports/` | 1,591 predictions dung format chung |
| `feature_metrics.json` | `reports/` | Ket qua Recall@1/5/10 + MRR |
| `feature_design.md` | `reports/` | Tai lieu nay |
