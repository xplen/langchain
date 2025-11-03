"""Feedback analyzer for learning optimal retrieval weights."""

from typing import Any

import numpy as np

from qa_rag_system.feedback_store import FeedbackStore, QueryFeedback


class FeedbackAnalyzer:
    """Analyzes user feedback to optimize retrieval weights."""

    def __init__(
        self,
        feedback_store: FeedbackStore,
        learning_rate: float = 0.1,
        min_samples: int = 5,
    ) -> None:
        """Initialize the feedback analyzer.

        Args:
            feedback_store: FeedbackStore instance to read feedback from.
            learning_rate: Learning rate for weight updates (0.0-1.0).
            min_samples: Minimum number of rated samples before adjusting weights.
        """
        self.feedback_store = feedback_store
        self.learning_rate = learning_rate
        self.min_samples = min_samples

    def analyze_feedback(self) -> dict[str, float]:
        """Analyze feedback to determine optimal retrieval weights.

        Returns:
            Dictionary with 'semantic_weight' and 'keyword_weight' recommendations.
        """
        rated_feedback = self.feedback_store.get_rated_feedback()

        if len(rated_feedback) < self.min_samples:
            # Not enough data, return default weights
            return {"semantic_weight": 0.5, "keyword_weight": 0.5}

        # Separate high-rated and low-rated queries
        high_rated = [f for f in rated_feedback if f.rating is not None and f.rating >= 4]
        low_rated = [f for f in rated_feedback if f.rating is not None and f.rating <= 2]

        if not high_rated and not low_rated:
            return {"semantic_weight": 0.5, "keyword_weight": 0.5}

        # Calculate average weights for high-rated queries
        high_semantic_weights = [
            f.semantic_weight
            for f in high_rated
            if f.semantic_weight is not None
        ]
        high_keyword_weights = [
            f.keyword_weight
            for f in high_rated
            if f.keyword_weight is not None
        ]

        # Calculate average weights for low-rated queries
        low_semantic_weights = [
            f.semantic_weight
            for f in low_rated
            if f.semantic_weight is not None
        ]
        low_keyword_weights = [
            f.keyword_weight
            for f in low_rated
            if f.keyword_weight is not None
        ]

        # Start with default weights
        semantic_weight = 0.5
        keyword_weight = 0.5

        # Adjust weights based on high-rated queries
        if high_semantic_weights and high_keyword_weights:
            avg_high_semantic = np.mean(high_semantic_weights)
            avg_high_keyword = np.mean(high_keyword_weights)

            # Move weights towards what worked well
            semantic_weight = (
                semantic_weight * (1 - self.learning_rate)
                + avg_high_semantic * self.learning_rate
            )
            keyword_weight = (
                keyword_weight * (1 - self.learning_rate)
                + avg_high_keyword * self.learning_rate
            )

        # Adjust away from low-rated queries
        if low_semantic_weights and low_keyword_weights:
            avg_low_semantic = np.mean(low_semantic_weights)
            avg_low_keyword = np.mean(low_keyword_weights)

            # Move weights away from what didn't work
            semantic_weight = (
                semantic_weight * (1 - self.learning_rate)
                + (1 - avg_low_semantic) * self.learning_rate
            )
            keyword_weight = (
                keyword_weight * (1 - self.learning_rate)
                + (1 - avg_low_keyword) * self.learning_rate
            )

        # Normalize weights to sum to 1.0
        total = semantic_weight + keyword_weight
        if total > 0:
            semantic_weight = semantic_weight / total
            keyword_weight = keyword_weight / total
        else:
            semantic_weight = 0.5
            keyword_weight = 0.5

        # Ensure weights are within reasonable bounds
        semantic_weight = max(0.1, min(0.9, semantic_weight))
        keyword_weight = max(0.1, min(0.9, keyword_weight))

        # Normalize again after clamping
        total = semantic_weight + keyword_weight
        semantic_weight = semantic_weight / total
        keyword_weight = keyword_weight / total

        return {
            "semantic_weight": float(semantic_weight),
            "keyword_weight": float(keyword_weight),
        }

    def get_weight_recommendations(
        self, current_semantic_weight: float, current_keyword_weight: float
    ) -> dict[str, Any]:
        """Get weight recommendations based on feedback analysis.

        Args:
            current_semantic_weight: Current semantic retrieval weight.
            current_keyword_weight: Current keyword retrieval weight.

        Returns:
            Dictionary with recommended weights and analysis.
        """
        optimal_weights = self.analyze_feedback()

        # Calculate recommended changes
        semantic_change = optimal_weights["semantic_weight"] - current_semantic_weight
        keyword_change = optimal_weights["keyword_weight"] - current_keyword_weight

        stats = self.feedback_store.get_statistics()

        return {
            "current_weights": {
                "semantic_weight": current_semantic_weight,
                "keyword_weight": current_keyword_weight,
            },
            "recommended_weights": optimal_weights,
            "recommended_changes": {
                "semantic_weight": semantic_change,
                "keyword_weight": keyword_change,
            },
            "feedback_stats": stats,
            "ready_for_update": stats["rated_queries"] >= self.min_samples,
        }

