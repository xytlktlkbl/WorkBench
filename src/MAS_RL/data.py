"""Dataset loading for MAS architecture policy training."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

import pandas as pd

from src.MAS_RL.schema import normalize_domains


DEFAULT_QUERY_PATHS = [
    "data/processed/queries_and_answers/calendar_queries_and_answers.csv",
    "data/processed/queries_and_answers/email_queries_and_answers.csv",
    "data/processed/queries_and_answers/analytics_queries_and_answers.csv",
    "data/processed/queries_and_answers/project_management_queries_and_answers.csv",
    "data/processed/queries_and_answers/customer_relationship_manager_queries_and_answers.csv",
    "data/processed/queries_and_answers/multi_domain_queries_and_answers.csv",
]


@dataclass(frozen=True)
class QueryRecord:
    query: str
    required_domains: list[str]
    answer: list[str]
    source_path: str


def _infer_domain_from_path(path: str) -> list[str]:
    name = os.path.basename(path).replace("_queries_and_answers.csv", "")
    return normalize_domains([name])


def _parse_domains(value: object, fallback: list[str]) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    if isinstance(value, list):
        return normalize_domains([str(v) for v in value])
    text = str(value).strip()
    if not text:
        return fallback
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            parsed_domains = normalize_domains([str(v) for v in parsed])
            return parsed_domains or fallback
    except Exception:
        pass
    return normalize_domains([part.strip() for part in text.split(",")]) or fallback


def _parse_answer(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def load_query_records(
    paths: list[str] | None = None,
    limit: int | None = None,
    limit_per_path: int | None = None,
) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for path in paths or DEFAULT_QUERY_PATHS:
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        fallback = _infer_domain_from_path(path)
        path_count = 0
        for _, row in frame.iterrows():
            records.append(
                QueryRecord(
                    query=str(row["query"]),
                    required_domains=_parse_domains(row.get("domains"), fallback),
                    answer=_parse_answer(row.get("answer")),
                    source_path=path,
                )
            )
            path_count += 1
            if limit_per_path is not None and path_count >= limit_per_path:
                break
            if limit is not None and len(records) >= limit:
                return records
    return records
