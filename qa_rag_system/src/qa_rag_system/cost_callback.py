"""Callback handler for tracking token usage in RAG chains."""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, LLMResult


class CostTrackingCallbackHandler(BaseCallbackHandler):
    """Callback handler that captures token usage from LLM calls."""

    def __init__(self) -> None:
        """Initialize the callback handler."""
        super().__init__()
        self.usage_metadata: UsageMetadata | None = None
        self.model_name: str | None = None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Capture usage metadata when LLM call completes.

        Args:
            response: The LLM response containing usage information.
            **kwargs: Additional keyword arguments.
        """
        # Check for usage_metadata in ChatGeneration (preferred method)
        try:
            generation = response.generations[0][0]
        except (IndexError, AttributeError):
            generation = None

        usage_metadata = None
        model_name = None

        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                    if hasattr(message, "response_metadata"):
                        model_name = message.response_metadata.get("model_name")
            except AttributeError:
                pass

        # Fallback to llm_output token_usage if no usage_metadata found
        if usage_metadata is None and response.llm_output:
            if "token_usage" in response.llm_output:
                token_usage = response.llm_output["token_usage"]
                usage_metadata = UsageMetadata(
                    input_tokens=token_usage.get("prompt_tokens", 0),
                    output_tokens=token_usage.get("completion_tokens", 0),
                    total_tokens=token_usage.get("total_tokens", 0),
                )

            # Try to get model name from response
            if "model_name" in response.llm_output:
                model_name = response.llm_output["model_name"]

        if usage_metadata:
            self.usage_metadata = usage_metadata
        if model_name:
            self.model_name = model_name

    def get_usage_metadata(self) -> UsageMetadata | None:
        """Get the captured usage metadata.

        Returns:
            UsageMetadata if available, None otherwise.
        """
        return self.usage_metadata

    def get_model_name(self) -> str | None:
        """Get the captured model name.

        Returns:
            Model name if available, None otherwise.
        """
        return self.model_name

    def reset(self) -> None:
        """Reset the captured usage data."""
        self.usage_metadata = None
        self.model_name = None

