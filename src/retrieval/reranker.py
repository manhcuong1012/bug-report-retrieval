from __future__ import annotations


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score <= min_score:
        return {key: 0.0 for key in scores}

    scale = max_score - min_score
    return {key: (value - min_score) / scale for key, value in scores.items()}


def rank_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}

    denominator = max(len(ordered) - 1, 1)
    return {bug_id: 1.0 - (rank / denominator) for rank, (bug_id, _) in enumerate(ordered)}


def metadata_boost(
    query_record: dict,
    candidate_record: dict,
    query_profile: dict,
) -> float:
    boost = 0.0

    if query_record.get("component", "UNKNOWN") == candidate_record.get("component", "UNKNOWN"):
        boost += 0.05 if query_profile.get("is_short") else 0.03

    if query_record.get("severity", "UNKNOWN") == candidate_record.get("severity", "UNKNOWN"):
        boost += 0.01

    query_priority = query_record.get("priority", "UNKNOWN")
    candidate_priority = candidate_record.get("priority", "UNKNOWN")
    if query_priority != "UNKNOWN" and query_priority == candidate_priority:
        boost += 0.005

    query_tech_tokens = set(query_record.get("tech_tokens", []))
    candidate_tech_tokens = set(candidate_record.get("tech_tokens", []))
    tech_overlap = len(query_tech_tokens & candidate_tech_tokens)
    if tech_overlap > 0:
        boost += min(0.04, 0.01 * tech_overlap)

    if query_profile.get("has_stacktrace") and tech_overlap > 0:
        boost += 0.02

    if query_profile.get("has_many_paths") and tech_overlap > 0:
        boost += 0.01

    return boost


def fuse_scores(
    bm25_scores: dict[str, float],
    feature_scores: dict[str, float],
    query_record: dict,
    candidate_records: dict[str, dict],
    query_profile: dict,
    alpha: float = 0.6,
    beta: float = 0.4,
    use_rank_normalization: bool = False,
) -> dict[str, float]:
    normalizer = rank_normalize if use_rank_normalization else min_max_normalize
    bm25_norm = normalizer(bm25_scores)
    feature_norm = normalizer(feature_scores)

    candidate_ids: set[str] = set(bm25_scores) | set(feature_scores)
    fused: dict[str, float] = {}

    query_component = query_record.get("component", "UNKNOWN")
    is_short = bool(query_profile.get("is_short"))

    bm25_weight = alpha
    feature_weight = beta
    if is_short:
        bm25_weight -= 0.05
        feature_weight += 0.05
    if query_profile.get("has_stacktrace") or query_profile.get("has_many_paths"):
        bm25_weight -= 0.1
        feature_weight += 0.1

    bm25_weight = max(bm25_weight, 0.1)
    feature_weight = max(feature_weight, 0.1)
    weight_total = bm25_weight + feature_weight
    bm25_weight /= weight_total
    feature_weight /= weight_total

    for bug_id in candidate_ids:
        candidate = candidate_records.get(bug_id)
        if candidate is None:
            continue
        fused[bug_id] = (
            bm25_weight * bm25_norm.get(bug_id, 0.0)
            + feature_weight * feature_norm.get(bug_id, 0.0)
            + metadata_boost(query_record, candidate, query_profile)
        )

        if query_component != "UNKNOWN" and candidate.get("component", "UNKNOWN") == query_component:
            fused[bug_id] += 0.01 if not is_short else 0.015

    return fused


def rank_candidates(scores: dict[str, float], top_k: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
