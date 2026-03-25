from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _json_default(value: Any) -> Any:
    """
    Fallback serializer for values that json.dumps cannot handle directly.
    """
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def log_experiment(
    *,
    log_path: str | Path,
    script: str,
    question: str,
    model: str,
    prompt_a: str | None = None,
    prompt_b: str | None = None,
    comparison_type: str | None = None,
    metric_summary: dict[str, Any] | None = None,
    result_summary: str | None = None,
    notes: str | None = None,
    artifacts: list[str | Path] | None = None,
    extra: dict[str, Any] | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """
    Append one experiment record to a JSONL log file.

    Returns the record that was written.
    """
    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
        "script": script,
        "question": question,
        "model": model,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "comparison_type": comparison_type,
        "metric_summary": metric_summary or {},
        "result_summary": result_summary,
        "notes": notes,
        "artifacts": [str(a) for a in (artifacts or [])],
        "extra": extra or {},
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default))
        f.write("\n")

    return record