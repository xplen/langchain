"""Feedback storage and management for RAG system."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class QueryFeedback(BaseModel):
    """Feedback record for a query-answer pair."""

    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer: str
    sources: list[dict[str, Any]]
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    semantic_weight: float | None = None
    keyword_weight: float | None = None


class FeedbackStore:
    """Manages storage and retrieval of user feedback."""

    def __init__(self, storage_path: str | Path = "./feedback_data.json") -> None:
        """Initialize the feedback store.

        Args:
            storage_path: Path to the JSON file for storing feedback.
        """
        self.storage_path = Path(storage_path)
        self._ensure_storage_file()

    def _ensure_storage_file(self) -> None:
        """Ensure the storage file exists."""
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump([], f)

    def _load_feedback(self) -> list[dict[str, Any]]:
        """Load all feedback from storage.

        Returns:
            List of feedback records as dictionaries.
        """
        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_feedback(self, feedback_list: list[dict[str, Any]]) -> None:
        """Save feedback list to storage.

        Args:
            feedback_list: List of feedback records to save.
        """
        with open(self.storage_path, "w") as f:
            json.dump(feedback_list, f, indent=2)

    def add_feedback(
        self,
        query: str,
        answer: str,
        sources: list[dict[str, Any]],
        rating: int | None = None,
        comment: str | None = None,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
    ) -> str:
        """Add a new feedback record.

        Args:
            query: The user's query.
            answer: The generated answer.
            sources: List of source documents used.
            rating: User rating (1-5).
            comment: Optional comment from the user.
            semantic_weight: Semantic retrieval weight used.
            keyword_weight: Keyword retrieval weight used.

        Returns:
            The feedback ID.
        """
        feedback = QueryFeedback(
            query=query,
            answer=answer,
            sources=sources,
            rating=rating,
            comment=comment,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )

        feedback_list = self._load_feedback()
        feedback_list.append(feedback.model_dump())
        self._save_feedback(feedback_list)

        return feedback.feedback_id

    def update_feedback(
        self,
        feedback_id: str,
        rating: int | None = None,
        comment: str | None = None,
    ) -> bool:
        """Update an existing feedback record.

        Args:
            feedback_id: The feedback ID to update.
            rating: New rating (1-5).
            comment: New comment.

        Returns:
            True if feedback was updated, False if not found.
        """
        feedback_list = self._load_feedback()
        for feedback in feedback_list:
            if feedback["feedback_id"] == feedback_id:
                if rating is not None:
                    feedback["rating"] = rating
                if comment is not None:
                    feedback["comment"] = comment
                feedback["timestamp"] = datetime.utcnow().isoformat()
                self._save_feedback(feedback_list)
                return True
        return False

    def get_feedback(self, feedback_id: str) -> QueryFeedback | None:
        """Get a specific feedback record.

        Args:
            feedback_id: The feedback ID to retrieve.

        Returns:
            QueryFeedback instance or None if not found.
        """
        feedback_list = self._load_feedback()
        for feedback_dict in feedback_list:
            if feedback_dict["feedback_id"] == feedback_id:
                return QueryFeedback(**feedback_dict)
        return None

    def get_all_feedback(self) -> list[QueryFeedback]:
        """Get all feedback records.

        Returns:
            List of all QueryFeedback instances.
        """
        feedback_list = self._load_feedback()
        return [QueryFeedback(**f) for f in feedback_list]

    def get_rated_feedback(self) -> list[QueryFeedback]:
        """Get all feedback records with ratings.

        Returns:
            List of QueryFeedback instances that have ratings.
        """
        all_feedback = self.get_all_feedback()
        return [f for f in all_feedback if f.rating is not None]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about stored feedback.

        Returns:
            Dictionary with feedback statistics.
        """
        all_feedback = self.get_all_feedback()
        rated_feedback = self.get_rated_feedback()

        if not rated_feedback:
            return {
                "total_queries": len(all_feedback),
                "rated_queries": 0,
                "average_rating": 0.0,
                "rating_distribution": {},
            }

        ratings = [f.rating for f in rated_feedback if f.rating is not None]
        rating_distribution = {i: ratings.count(i) for i in range(1, 6)}

        return {
            "total_queries": len(all_feedback),
            "rated_queries": len(rated_feedback),
            "average_rating": sum(ratings) / len(ratings),
            "rating_distribution": rating_distribution,
        }

