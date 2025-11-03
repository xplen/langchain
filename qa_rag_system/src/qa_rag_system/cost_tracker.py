"""Cost tracking module for monitoring token usage per query type."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages.ai import UsageMetadata


@dataclass
class QueryUsage:
    """Token usage for a single query."""

    query_type: str
    """Type of query: 'regular', 'comparison', etc."""
    input_tokens: int
    """Number of input tokens used."""
    output_tokens: int
    """Number of output tokens used."""
    total_tokens: int
    """Total tokens used."""
    timestamp: str
    """ISO format timestamp of the query."""
    query_text: str | None = None
    """The query text (optional, for reference)."""
    model: str | None = None
    """The model used (optional)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class QueryTypeStats:
    """Statistics for a specific query type."""

    query_type: str
    """Type of query."""
    total_queries: int
    """Total number of queries of this type."""
    total_input_tokens: int
    """Total input tokens used."""
    total_output_tokens: int
    """Total output tokens used."""
    total_tokens: int
    """Total tokens used."""
    avg_input_tokens: float
    """Average input tokens per query."""
    avg_output_tokens: float
    """Average output tokens per query."""
    avg_total_tokens: float
    """Average total tokens per query."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CostTracker:
    """Tracks token usage per query type."""

    def __init__(self, storage_path: str = "./cost_tracking.json") -> None:
        """Initialize the cost tracker.

        Args:
            storage_path: Path to JSON file for storing usage data.
        """
        self.storage_path = Path(storage_path)
        self._usages: list[QueryUsage] = []
        self._load_usages()

    def _load_usages(self) -> None:
        """Load usage data from storage file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    data = json.load(f)
                    self._usages = [
                        QueryUsage(**usage_dict) for usage_dict in data.get("usages", [])
                    ]
            except (json.JSONDecodeError, KeyError, TypeError):
                # If file is corrupted, start fresh
                self._usages = []
        else:
            self._usages = []

    def _save_usages(self) -> None:
        """Save usage data to storage file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"usages": [usage.to_dict() for usage in self._usages]}
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def track_usage(
        self,
        query_type: str,
        usage_metadata: UsageMetadata | dict[str, Any] | None = None,
        query_text: str | None = None,
        model: str | None = None,
    ) -> None:
        """Track token usage for a query.

        Args:
            query_type: Type of query ('regular', 'comparison', etc.).
            usage_metadata: Usage metadata from LangChain or dict with token counts.
            query_text: Optional query text for reference.
            model: Optional model name.
        """
        if usage_metadata is None:
            # Default to zero if no usage metadata provided
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
        elif isinstance(usage_metadata, dict):
            input_tokens = usage_metadata.get("input_tokens", 0)
            output_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)
        else:
            input_tokens = usage_metadata.get("input_tokens", 0)
            output_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)

        usage = QueryUsage(
            query_type=query_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            timestamp=datetime.utcnow().isoformat(),
            query_text=query_text,
            model=model,
        )

        self._usages.append(usage)
        self._save_usages()

    def get_stats_by_query_type(self) -> dict[str, QueryTypeStats]:
        """Get statistics grouped by query type.

        Returns:
            Dictionary mapping query types to their statistics.
        """
        stats_by_type: dict[str, list[QueryUsage]] = {}
        for usage in self._usages:
            if usage.query_type not in stats_by_type:
                stats_by_type[usage.query_type] = []
            stats_by_type[usage.query_type].append(usage)

        result: dict[str, QueryTypeStats] = {}
        for query_type, usages in stats_by_type.items():
            total_queries = len(usages)
            total_input = sum(u.input_tokens for u in usages)
            total_output = sum(u.output_tokens for u in usages)
            total_tokens = sum(u.total_tokens for u in usages)

            result[query_type] = QueryTypeStats(
                query_type=query_type,
                total_queries=total_queries,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_tokens=total_tokens,
                avg_input_tokens=total_input / total_queries if total_queries > 0 else 0.0,
                avg_output_tokens=total_output / total_queries if total_queries > 0 else 0.0,
                avg_total_tokens=total_tokens / total_queries if total_queries > 0 else 0.0,
            )

        return result

    def get_all_stats(self) -> dict[str, Any]:
        """Get overall statistics.

        Returns:
            Dictionary with overall statistics and per-query-type breakdown.
        """
        stats_by_type = self.get_stats_by_query_type()

        total_queries = len(self._usages)
        total_input = sum(u.input_tokens for u in self._usages)
        total_output = sum(u.output_tokens for u in self._usages)
        total_tokens = sum(u.total_tokens for u in self._usages)

        return {
            "overall": {
                "total_queries": total_queries,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_tokens,
                "avg_input_tokens": total_input / total_queries if total_queries > 0 else 0.0,
                "avg_output_tokens": total_output / total_queries if total_queries > 0 else 0.0,
                "avg_total_tokens": total_tokens / total_queries if total_queries > 0 else 0.0,
            },
            "by_query_type": {
                query_type: stats.to_dict() for query_type, stats in stats_by_type.items()
            },
        }

    def get_recent_queries(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent queries.

        Args:
            limit: Maximum number of recent queries to return.

        Returns:
            List of recent query usage dictionaries, sorted by timestamp (newest first).
        """
        sorted_usages = sorted(
            self._usages, key=lambda u: u.timestamp, reverse=True
        )[:limit]
        return [usage.to_dict() for usage in sorted_usages]

    def clear_history(self) -> None:
        """Clear all usage history."""
        self._usages = []
        self._save_usages()

