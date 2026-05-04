# Mozilla Schema For Team

Dataset source: `data/raw/mozilla.json`

Normalized record schema:

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

Field mapping:

- `created_at <- creation_ts`
- `summary <- short_desc`
- `description <- description`
- `component <- component`
- `priority <- priority`
- `severity <- bug_severity`
- `project = "mozilla"`

Duplicate bucket rules:

- Internal key format: `mozilla:<bug_id>`
- If a bug has `dup_id`, union the current bug with `mozilla:<dup_id>`
- Buckets are built with union-find
- Canonical bug is the smallest bug id inside the connected component
- Final `bucket_id` format: `mozilla:<canonical_bug_id>`

Text fields:

- `text_raw`: normalized whitespace over `summary + description`
- `text_clean`: lowercase, noise-reduced text that keeps technical tokens
- Retrieval text uses weighted text with `summary_repeat = 3`

Split rules:

- Records are split by `created_at`
- Oldest 80% go to `train`
- Remaining 20% are candidate `test`
- A test bug is kept only if its bucket size is greater than 1
- A test bug is kept only if the same bucket already appears in `train`

Evaluation rules:

- Candidate pool is always `train`
- A retrieved bug is valid only if `created_at <= query.created_at`
- Metrics: `Recall@1`, `Recall@5`, `Recall@10`, `MRR`
- A hit means at least one retrieved bug shares the same `bucket_id` as the query
