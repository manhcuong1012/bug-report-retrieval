# Refactor Đúng Cấu Trúc Và Hoàn Thành Người 1 Trên Mozilla

## Tóm tắt

Repo sẽ được refactor về **đúng cấu trúc bạn yêu cầu**, chỉ giữ các thư mục và file sau:

```text
data/
  raw/
  processed/
  train.jsonl
  test.jsonl
  bucket_map.json
src/
  preprocessing/
    build_schema.py
    clean_text.py
    split_by_time.py
  retrieval/
    tfidf_retriever.py
    bm25_retriever.py
  eval/
    metrics.py
reports/
```

Mọi logic của người 1 sẽ được dồn/gộp vào đúng các file trên, không giữ file phụ như `src/common.py` hay `src/run_person1.py`. Dataset duy nhất là **`mozilla.json`**.

## Thay đổi triển khai

### 1. Sắp xếp lại cấu trúc repo

- Di chuyển `mozilla.json` vào `data/raw/mozilla.json` làm nguồn chuẩn.
- Chỉ giữ output người 1 trong:
  - `data/processed/processed_bugs.jsonl`
  - `data/train.jsonl`
  - `data/test.jsonl`
  - `data/bucket_map.json`
- Xóa vai trò của các file phụ ngoài khung mẫu; helper nào đang nằm ở `src/common.py` sẽ được gộp vào các module chính.
- `reports/` chỉ dùng cho kết quả metric, ghi chú schema, và bảng kết quả nếu cần; không tạo script điều phối ở đây.

### 2. `src/preprocessing/clean_text.py`

File này chịu trách nhiệm toàn bộ text preprocessing dùng chung.

Nội dung phải có:
- `normalize_whitespace(text)`
- `normalize_text(text)`
- `build_text_raw(summary, description)`
- `build_text_clean(summary, description)`
- `build_weighted_text(summary, description, summary_repeat=3)`

Rule clean chốt:
- lowercase
- chuẩn hóa khoảng trắng
- bỏ phần lớn ký tự nhiễu
- giữ token kỹ thuật có ích cho retrieval như path, module, version, test identifiers
- không stem mạnh
- `summary_repeat=3` là mặc định cho retrieval

### 3. `src/preprocessing/build_schema.py`

File này chịu trách nhiệm đọc `data/raw/mozilla.json`, chuẩn hóa schema và tạo bucket duplicate.

Schema đầu ra cố định:
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

Mapping cố định:
- `created_at <- creation_ts`
- `summary <- short_desc`
- `severity <- bug_severity`
- `project = "mozilla"`

Logic duplicate bucket:
- key nội bộ: `mozilla:<bug_id>`
- nếu có `dup_id`, union với `mozilla:<dup_id>`
- dùng union-find để gom bucket
- canonical bug là bug id nhỏ nhất trong bucket
- `bucket_id = mozilla:<canonical_bug_id>`

Đầu ra bắt buộc:
- `data/processed/processed_bugs.jsonl`
- `data/bucket_map.json`

CLI của file này phải đủ để chạy độc lập:
```powershell
python .\src\preprocessing\build_schema.py
```

Mặc định của script:
- input: `data/raw/mozilla.json`
- outputs:
  - `data/processed/processed_bugs.jsonl`
  - `data/bucket_map.json`

### 4. `src/preprocessing/split_by_time.py`

File này chịu trách nhiệm chia train/test theo thời gian.

Logic cố định:
- đọc `data/processed/processed_bugs.jsonl`
- sort theo `created_at`
- lấy 80% đầu vào `train`
- phần còn lại là candidate `test`
- lọc `test` để chỉ giữ:
  - bug thuộc bucket có size > 1
  - bug có same-bucket đã xuất hiện trong `train`

Outputs cố định:
- `data/train.jsonl`
- `data/test.jsonl`

CLI của file này phải chạy độc lập:
```powershell
python .\src\preprocessing\split_by_time.py
```

Mặc định:
- input: `data/processed/processed_bugs.jsonl`
- outputs:
  - `data/train.jsonl`
  - `data/test.jsonl`

### 5. `src/eval/metrics.py`

File này là evaluator chung cho toàn nhóm.

Phải có:
- `hit_at_k(query_bucket, ranked_bug_ids, bugid_to_bucket, k)`
- `reciprocal_rank(query_bucket, ranked_bug_ids, bugid_to_bucket)`
- `evaluate(predictions, train_records)` hoặc tương đương

Metric cố định:
- `Recall@1`
- `Recall@5`
- `Recall@10`
- `MRR`

Format prediction chuẩn cho mọi retriever:
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

CLI evaluator phải chạy được độc lập để chấm file predictions do TF-IDF hoặc BM25 sinh ra.

### 6. `src/retrieval/tfidf_retriever.py`

File này triển khai baseline TF-IDF hoàn chỉnh cho người 1.

Phải có:
- tokenizer dùng lại logic từ `clean_text.py`
- build postings cho `train`
- tính TF, DF, IDF
- vector hóa query/document
- cosine similarity
- top-k retrieval

Rule retrieval:
- query lấy từ `data/test.jsonl`
- candidate chỉ từ `data/train.jsonl`
- candidate phải có `created_at <= query.created_at`
- text dùng retrieval là weighted text từ `summary` và `description`

Outputs mặc định:
- `reports/tfidf_predictions.jsonl`
- `reports/tfidf_metrics.json`

CLI chạy độc lập:
```powershell
python .\src\retrieval\tfidf_retriever.py
```

### 7. `src/retrieval/bm25_retriever.py`

File này triển khai baseline BM25 hoàn chỉnh cho người 1.

Phải có:
- inverted index
- doc length
- df
- avgdl
- BM25 scoring

Tham số mặc định:
- `k1 = 1.5`
- `b = 0.75`

Rule retrieval giống TF-IDF:
- query từ `test`
- candidate từ `train`
- không lấy bug mới hơn query

Outputs mặc định:
- `reports/bm25_predictions.jsonl`
- `reports/bm25_metrics.json`

CLI chạy độc lập:
```powershell
python .\src\retrieval\bm25_retriever.py
```

## Cách chạy từ đầu tới cuối

### Luồng chuẩn

1. Chép hoặc đặt `mozilla.json` vào:
```text
data/raw/mozilla.json
```

2. Chạy chuẩn hóa schema:
```powershell
python .\src\preprocessing\build_schema.py
```

3. Chạy split train/test:
```powershell
python .\src\preprocessing\split_by_time.py
```

4. Chạy TF-IDF:
```powershell
python .\src\retrieval\tfidf_retriever.py
```

5. Chạy BM25:
```powershell
python .\src\retrieval\bm25_retriever.py
```

### Kết quả mong đợi sau khi chạy

Trong `data/`:
- `raw/mozilla.json`
- `processed/processed_bugs.jsonl`
- `train.jsonl`
- `test.jsonl`
- `bucket_map.json`

Trong `reports/`:
- `tfidf_predictions.jsonl`
- `tfidf_metrics.json`
- `bm25_predictions.jsonl`
- `bm25_metrics.json`

## Kiểm thử và nghiệm thu

### Test dữ liệu

- `processed_bugs.jsonl` có đủ field schema cho mọi dòng.
- Mọi record có `project = "mozilla"`.
- Mọi `bucket_id` có prefix `mozilla:`.
- `bucket_map.json` phản ánh đúng các duplicate groups.
- `train.jsonl` và `test.jsonl` tồn tại và không rỗng.

### Test split

- Tất cả bug trong `test.jsonl` phải có same-bucket xuất hiện trong `train.jsonl`.
- Không có query nào trong `test` không có ground truth duplicate trong `train`.
- `train` phải là phần cũ hơn theo thời gian trước khi áp dụng lọc test.

### Test retrieval

- TF-IDF và BM25 đều sinh prediction đúng format chung.
- Mọi result đều nằm trong `train`.
- Không có result nào mới hơn query.
- `tfidf_metrics.json` và `bm25_metrics.json` có đủ `Recall@1/5/10` và `MRR`.

### Tiêu chí hoàn thành người 1

Người 1 chỉ hoàn thành khi:
- dữ liệu đã được chuẩn hóa xong theo schema chung
- duplicate bucket và split chạy đúng
- evaluator chạy được
- TF-IDF chạy được
- BM25 chạy được
- có metric JSON cho cả hai baseline
- cấu trúc repo khớp đúng mẫu bạn đã chốt

## Giả định đã khóa

- Chỉ làm trên `mozilla.json`
- Không dùng `mozilla_soft_clean.json` trong mainline
- Không giữ file phụ ngoài cấu trúc đã chốt
- Mọi helper phải gộp vào đúng các file trong khung mẫu
- Không có script “one-command” ngoài danh sách file bạn yêu cầu
- `reports/` dùng cho output và ghi chú kết quả, không dùng làm nơi chứa logic thực thi
