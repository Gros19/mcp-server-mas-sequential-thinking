"""Response formatting service for thought processing.

This service handles formatting and synthesizing responses from various processing
modes, extracting content from Agno RunOutput objects, and preparing final responses.
"""

from mcp_server_mas_sequential_thinking.core import ThoughtData
from mcp_server_mas_sequential_thinking.services.response_processor import (
    ResponseExtractor,
)
from mcp_server_mas_sequential_thinking.utils import setup_logging

logger = setup_logging()


class ResponseFormatter:
    """Service responsible for formatting and synthesizing responses."""

    def __init__(self) -> None:
        """Initialize the response formatter."""

    def format_response(self, content: str, thought_data: ThoughtData) -> str:
        """Format response for MCP - clean content without additional guidance.

        MCP servers should return clean content and let the AI decide next steps.

        Args:
            content: The raw response content to format
            thought_data: The thought data context

        Returns:
            Formatted response string
        """
        # MCP servers should return clean content, let AI decide next steps
        final_response = content

        # Log response formatting details
        self._log_response_formatting(content, thought_data, final_response)

        return final_response

    def extract_response_content(self, response) -> str:
        """Extract clean content from Agno RunOutput objects.

        Handles various response types and extracts text content properly.

        Args:
            response: The response object from Agno processing

        Returns:
            Extracted text content
        """
        # Import ResponseExtractor to handle the extraction

        return ResponseExtractor.extract_content(response)

    def _log_response_formatting(
        self, content: str, thought_data: ThoughtData, final_response: str
    ) -> None:
        """Log response formatting details for debugging and monitoring.

        Args:
            content: The original response content
            thought_data: The thought data context
            final_response: The final formatted response
        """
        logger.info("📤 RESPONSE FORMATTING:")
        logger.info(f"  Original content length: {len(content)} chars")
        logger.info(f"  Next needed: {thought_data.nextThoughtNeeded}")
        logger.info(f"  Final response length: {len(final_response)} chars")
        logger.info(f"  Final response:\n{final_response}")

        # Use field length limits constant if available
        separator_length = 60  # Default fallback
        try:
            from mcp_server_mas_sequential_thinking.config import FieldLengthLimits

            separator_length = FieldLengthLimits.SEPARATOR_LENGTH
        except ImportError:
            pass

        logger.info(f"  {'=' * separator_length}")
