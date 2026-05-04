# Kế Hoạch Tổng Thể Cho 3 Người Trên Dataset Mozilla

## Tóm tắt

Toàn bộ nhóm sẽ chỉ làm trên **dataset `mozilla.json`**. Mục tiêu của đề tài là xây dựng hệ thống **truy xuất bug report tương tự để hỗ trợ phát hiện bug trùng lặp**, với 3 lớp mô hình:

- Người 1: nền dữ liệu + baseline text retrieval (`TF-IDF`, `BM25`)
- Người 2: feature-based bug scoring
- Người 3: hybrid fusion + reranking + tổng hợp kết quả cuối

Định hướng bàn giao là **đồ án nhóm**, nên plan ưu tiên:
- chia việc rõ cho từng người
- thống nhất schema, input/output, metric
- có mốc tiến độ
- có file bàn giao cụ thể
- có phần đánh giá, demo, báo cáo, slides

Dataset duy nhất:
- nguồn chính: `mozilla.json`

## Thiết kế chung cả nhóm

### 1. Dữ liệu và schema thống nhất

Mọi người phải dùng cùng một schema chuẩn:

```json
{
  "bug_id": "...",
  "project": "mozilla",
  "created_at": "...",
  "summary": "...",
  "description": "...",
  "component": "...",
  "priority": "...",
  "severity": "...",
  "bucket_id": "...",
  "text_raw": "...",
  "text_clean": "..."
}
```

Map field từ `mozilla.json`:
- `created_at` <- `creation_ts`
- `summary` <- `short_desc`
- `description` <- `description`
- `component` <- `component`
- `priority` <- `priority`
- `severity` <- `bug_severity`
- `bucket_id` tạo từ `dup_id`

Rule duplicate bucket:
- mỗi bug dùng key nội bộ dạng `mozilla:<bug_id>`
- nếu có `dup_id`, union bug hiện tại với `mozilla:<dup_id>`
- dùng union-find để gom bucket
- `bucket_id` chuẩn là `mozilla:<canonical_bug_id>`
- canonical bug là bug id nhỏ nhất trong bucket

### 2. Train/test và candidate pool thống nhất

Split chỉ theo thời gian:
- 80% bug cũ hơn vào `train`
- 20% bug mới hơn là candidate `test`

Rule lọc test:
- chỉ giữ bug có same-bucket xuất hiện trong `train`
- chỉ giữ bug thuộc bucket có kích thước > 1

Candidate pool cho mọi thuật toán:
- query bug lấy từ `test`
- candidate bugs chỉ lấy từ `train`
- chỉ xét bug có `created_at <= query.created_at`

### 3. Metric chung

Mọi mô hình phải dùng cùng evaluator:
- `Recall@1`
- `Recall@5`
- `Recall@10`
- `MRR`

Rule chấm đúng:
- top-k đúng nếu trong kết quả có ít nhất một bug cùng `bucket_id` với query

### 4. Format output chung

Mọi thuật toán đều phải xuất prediction theo format này:

```json
{
  "query_bug_id": "...",
  "project": "mozilla",
  "query_bucket_id": "...",
  "results": [
    {"bug_id": "...", "project": "mozilla", "score": 1.234, "rank": 1}
  ]
}
```

Người 3 sẽ dùng chính format này để fusion và rerank.

## Phân công chi tiết theo 3 người

## Người 1: Nền dữ liệu + text retrieval baseline

### Mục tiêu

Người 1 chịu trách nhiệm tạo nền tảng mà cả nhóm cùng dùng:
- chuẩn hóa dữ liệu
- bucket duplicate
- chia train/test
- evaluator chung
- baseline text retrieval để làm mốc so sánh

### Phần việc bắt buộc

1. Đọc và hiểu `mozilla.json`
2. Chuẩn hóa toàn bộ bug về schema chung
3. Làm sạch text nhẹ:
   - lowercase
   - normalize whitespace
   - giữ token kỹ thuật
   - không stem mạnh trong mainline
4. Tạo:
   - `text_raw`
   - `text_clean`
5. Tạo `bucket_id`
6. Chia `train.jsonl` và `test.jsonl`
7. Viết evaluator chung
8. Cài 2 baseline:
   - `TF-IDF`
   - `BM25`
9. Thử field weighting:
   - `summary` nặng hơn `description`
   - mặc định `summary_repeat = 3`

### Thuật toán người 1 phải làm

#### A. Tokenization và preprocessing cho retrieval
- `normalize_text(text)`
- `tokenize(text)`
- `build_weighted_text(summary, description, summary_repeat=3)`

#### B. TF-IDF
- build inverted index cho `train`
- tính TF, DF, IDF
- vectorize query/document
- cosine similarity
- xuất top-k

#### C. BM25
- build inverted index
- lưu doc length, df, avgdl
- dùng tham số mặc định:
  - `k1 = 1.5`
  - `b = 0.75`
- xuất top-k

### File bàn giao

- `processed_bugs.jsonl`
- `train.jsonl`
- `test.jsonl`
- `bucket_map.json`
- `schema_summary.json`
- `split_summary.json`
- `metrics.py`
- `tfidf_retriever.py`
- `bm25_retriever.py`
- `schema.md`

### Tiêu chí hoàn thành

- pipeline dữ liệu chạy hết trên Mozilla
- evaluator chạy được
- có metric cho TF-IDF và BM25
- output đúng format chung

## Người 2: Bug-feature scoring

### Mục tiêu

Người 2 làm mô hình riêng dựa trên đặc trưng bug, không phụ thuộc hoàn toàn vào term matching. Mục tiêu là tạo một bộ điểm feature-based có thể bắt được duplicate dù text không trùng mạnh.

### Input bắt buộc

Người 2 chỉ dùng:
- `train.jsonl`
- `test.jsonl`
- schema chuẩn do người 1 bàn giao

### Phần việc bắt buộc

1. Thiết kế feature set cho bug report Mozilla
2. Viết scorer cho từng loại feature
3. Tạo hàm tổng hợp score có trọng số
4. Viết retrieval API trả về top-k
5. Xuất prediction theo format chung
6. Chạy evaluator chung để lấy metric

### Feature phải có

#### A. Summary similarity
- token overlap
- Jaccard similarity
- exact/near phrase overlap nếu hợp lý

#### B. Description similarity
- token overlap ratio
- phrase overlap nhẹ
- match các cụm lỗi đặc trưng

#### C. Metadata similarity
- cùng `component`
- cùng `priority`
- cùng `severity`

#### D. Technical token overlap
Trích các token kỹ thuật như:
- `crash`
- `abort`
- `segfault`
- `timeout`
- `hang`
- `memory`
- `leak`
- `exception`
- `null`
- `stack`
- `assert`
- file/class/module-like tokens
- test path / browser test / taskcluster / treeherder-like tokens nếu xuất hiện nhiều trong Mozilla

#### E. Query-aware heuristics
- bug ngắn thì tăng trọng số summary/component
- bug dài có stacktrace thì tăng trọng số technical overlap
- query có nhiều URL/log/test-path thì tăng trọng số token kỹ thuật/log overlap

### Hàm score tổng

Người 2 phải chốt một hàm dạng:

```python
score = (
    w1 * summary_score
    + w2 * description_score
    + w3 * component_score
    + w4 * severity_score
    + w5 * priority_score
    + w6 * technical_overlap_score
)
```

Trọng số phải được chốt rõ trong bàn giao, không để người 3 tự đoán.

### File bàn giao

- `bug_feature_scorer.py`
- `feature_retriever.py` hoặc API tương đương
- `feature_metrics.json`
- `feature_predictions.jsonl`
- `feature_design.md`

### Tiêu chí hoàn thành

- feature scorer chạy được trên Mozilla
- top-k output đúng format chung
- có metric riêng để so với BM25
- có mô tả rõ từng feature và trọng số

## Người 3: Hybrid + reranking + phần cuối

### Mục tiêu

Người 3 tạo mô hình cuối cùng bằng cách kết hợp điểm từ người 1 và người 2, rồi chịu trách nhiệm tổng hợp kết quả cuối, phân tích lỗi, demo và ghép báo cáo.

### Input bắt buộc

Người 3 dùng:
- `bm25_predictions.jsonl` hoặc score API từ người 1
- `feature_predictions.jsonl` hoặc score API từ người 2
- `train.jsonl`, `test.jsonl`
- evaluator chung

### Phần việc bắt buộc

1. Chuẩn hóa score từ các nguồn khác nhau
2. Thiết kế weighted fusion
3. Thiết kế reranking pipeline
4. Thử metadata boosting
5. Thử heuristic theo loại bug/query
6. Chạy metric cuối
7. Gom bảng kết quả tất cả mô hình
8. Phân tích lỗi
9. Làm demo
10. Ghép phần báo cáo/slides cuối

### Pipeline hybrid bắt buộc

1. Lấy top N từ BM25
2. Với top N đó, tính lại feature score
3. Chuẩn hóa score về cùng thang
4. Tính final score
5. Sort lại để lấy top-k cuối

### Thành phần phải có

#### A. Score normalization
- `min-max normalization` hoặc `rank normalization`
- phải chọn một cách cố định cho final pipeline

#### B. Score fusion
- `final_score = alpha * bm25_score + beta * feature_score`
- `alpha`, `beta` phải được tuning rõ ràng trên validation nội bộ hoặc tập dev do nhóm chốt

#### C. Metadata boost
- cùng component thì cộng boost
- cùng severity thì cộng boost nhẹ nếu có lợi

#### D. Rule-based heuristics
- query ngắn thì tăng trọng số metadata
- query technical/log-heavy thì tăng trọng số feature kỹ thuật hoặc BM25 technical token influence

### File bàn giao

- `hybrid_retriever.py`
- `reranker.py`
- `hybrid_predictions.jsonl`
- `hybrid_metrics.json`
- `comparison_table.md`
- `error_analysis.md`
- `demo_spec.md`

### Trách nhiệm phần cuối

Người 3 chịu trách nhiệm chính cho:
- bảng metric cuối cùng
- phân tích case đúng/sai
- demo input/output
- phần evaluation, comparison, hybrid, conclusion trong báo cáo

## Tiến độ đề xuất theo mốc

### Giai đoạn 1: Nền dữ liệu
Người 1 làm trước, 100% ưu tiên:
- schema
- bucket
- split
- evaluator
- baseline BM25/TF-IDF

Đầu ra giai đoạn này là điều kiện để người 2 và 3 bắt đầu đồng bộ.

### Giai đoạn 2: Code mô hình song song
Sau khi người 1 bàn giao dữ liệu chuẩn:
- Người 1 tiếp tục tinh chỉnh `TF-IDF/BM25`
- Người 2 làm feature-based scorer
- Người 3 dựng khung hybrid và reranker, chưa cần score cuối

### Giai đoạn 3: Tích hợp
Khi có output đầu tiên từ người 1 và người 2:
- Người 3 ghép fusion thật
- cả nhóm thống nhất format prediction và metric table

### Giai đoạn 4: So sánh và tối ưu
- Người 1 tuning `summary_repeat`, BM25 parameters nếu cần
- Người 2 tuning feature weights
- Người 3 tuning `alpha/beta`, rerank depth, metadata boost

### Giai đoạn 5: Chốt bài
- Người 3 gom metric cuối
- phân tích lỗi
- làm demo
- ghép báo cáo và slides
- người 1 và 2 hỗ trợ viết phần thuật toán của mình

## Kiểm thử và nghiệm thu

### Kiểm thử dữ liệu
- mọi record có đủ field schema
- mọi `bucket_id` hợp lệ
- `train` và `test` đúng logic thời gian
- mọi `test` query đều có ground truth duplicate trong `train`

### Kiểm thử thuật toán
- mọi model trả đúng format prediction
- chỉ retrieve trên `train`
- không lấy candidate mới hơn query
- evaluator chạy được cho cả 3 mô hình

### Kiểm thử tích hợp
- người 3 đọc được output của người 1 và người 2 mà không cần sửa format
- bảng metric cuối có đủ:
  - TF-IDF
  - BM25
  - Feature-based
  - Hybrid

### Tiêu chí chốt đồ án
- pipeline chạy được trên Mozilla end-to-end
- có ít nhất 3 mức mô hình:
  - baseline text
  - feature-based
  - hybrid
- có bảng metric rõ ràng
- có demo
- có phân tích lỗi
- có báo cáo/slides khớp với implementation thật

## Giả định và mặc định đã khóa

- Chỉ làm trên **Mozilla**
- Dùng **`mozilla.json`** làm nguồn chính
- Không mở rộng sang deep learning hoặc embedding model trong plan chính
- Hybrid cuối dùng **BM25 + feature-based**, không dùng TF-IDF làm nhánh chính cho fusion
- Người 3 là người giữ kết quả cuối và ghép báo cáo/demo
- Người 1 là người sở hữu chuẩn dữ liệu và evaluator chung
- Người 2 chỉ làm trên schema do người 1 bàn giao, không tự tạo format riêng


project/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ train.jsonl
│  ├─ test.jsonl
│  └─ bucket_map.json
├─ src/
│  ├─ preprocessing/
│  │  ├─ build_schema.py
│  │  ├─ clean_text.py
│  │  └─ split_by_time.py
│  ├─ retrieval/
│  │  ├─ tfidf_retriever.py
│  │  ├─ bm25_retriever.py
│  │  ├─ bug_feature_scorer.py
│  │  ├─ hybrid_retriever.py
│  │  └─ reranker.py
│  ├─ eval/
│  │  └─ metrics.py
│  └─ app/
│     └─ demo.py
└─ reports/