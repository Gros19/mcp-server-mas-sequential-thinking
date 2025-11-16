"""Refactored server core with separated concerns and reduced complexity."""

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Lazy import to break circular dependency
from pydantic import ValidationError

from mcp_server_mas_sequential_thinking.config import (
    DefaultTimeouts,
    DefaultValues,
    PerformanceMetrics,
)
from mcp_server_mas_sequential_thinking.core import (
    ConfigurationError,
    SessionMemory,
    ThoughtData,
)
from mcp_server_mas_sequential_thinking.utils import setup_logging

logger = setup_logging()


class LoggingMixin:
    """Mixin class providing common logging utilities with reduced duplication."""

    @staticmethod
    def _log_section_header(
        title: str, separator_length: int = PerformanceMetrics.SEPARATOR_LENGTH
    ) -> None:
        """Log a formatted section header."""
        logger.info(f"{title}")

    @staticmethod
    def _log_metrics_block(title: str, metrics: dict[str, Any]) -> None:
        """Log a formatted metrics block."""
        logger.info(f"{title}")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            elif isinstance(value, (int, str)):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value}")

    @staticmethod
    def _log_separator(length: int = PerformanceMetrics.SEPARATOR_LENGTH) -> None:
        """Log a separator line."""
        logger.info(f"  {'=' * length}")

    @staticmethod
    def _calculate_efficiency_score(processing_time: float) -> float:
        """Calculate efficiency score using standard metrics."""
        return (
            PerformanceMetrics.PERFECT_EFFICIENCY_SCORE
            if processing_time < PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD
            else max(
                PerformanceMetrics.MINIMUM_EFFICIENCY_SCORE,
                PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD / processing_time,
            )
        )

    @staticmethod
    def _calculate_execution_consistency(success_indicator: bool) -> float:
        """Calculate execution consistency using standard metrics."""
        return (
            PerformanceMetrics.PERFECT_EXECUTION_CONSISTENCY
            if success_indicator
            else PerformanceMetrics.DEFAULT_EXECUTION_CONSISTENCY
        )


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Immutable server configuration with clear defaults."""

    provider: str
    team_mode: str = DefaultValues.DEFAULT_TEAM_MODE
    log_level: str = DefaultValues.DEFAULT_LOG_LEVEL
    max_retries: int = DefaultValues.DEFAULT_MAX_RETRIES
    timeout: float = DefaultTimeouts.PROCESSING_TIMEOUT

    @classmethod
    def from_environment(cls) -> "ServerConfig":
        """Create config from environment with sensible defaults."""
        return cls(
            provider=os.environ.get("LLM_PROVIDER", DefaultValues.DEFAULT_LLM_PROVIDER),
            team_mode=os.environ.get(
                "TEAM_MODE", DefaultValues.DEFAULT_TEAM_MODE
            ).lower(),
            log_level=os.environ.get("LOG_LEVEL", DefaultValues.DEFAULT_LOG_LEVEL),
            max_retries=int(
                os.environ.get("MAX_RETRIES", str(DefaultValues.DEFAULT_MAX_RETRIES))
            ),
            timeout=float(
                os.environ.get("TIMEOUT", str(DefaultValues.DEFAULT_TIMEOUT))
            ),
        )


class ServerInitializer(ABC):
    """Abstract initializer for server components."""

    @abstractmethod
    async def initialize(self, config: ServerConfig) -> None:
        """Initialize server component."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up server component."""


class EnvironmentInitializer(ServerInitializer):
    """Handles environment validation and setup with comprehensive early validation."""

    async def initialize(self, config: ServerConfig) -> None:
        """Validate environment requirements with enhanced error handling and early failure detection."""
        logger.info(f"Initializing environment with {config.provider} provider")

        try:
            # Early comprehensive validation
            await self._validate_provider_configuration(config)
            await self._validate_system_requirements()
            await self._setup_directories()

        except Exception as e:
            if not isinstance(e, ConfigurationError):
                raise ConfigurationError(
                    f"Environment initialization failed: {e}"
                ) from e
            raise

    async def _validate_provider_configuration(self, config: ServerConfig) -> None:
        """Validate provider-specific configuration with actionable error messages."""
        try:
            # Import here to avoid circular dependency
            from mcp_server_mas_sequential_thinking.config.modernized_config import (
                get_available_providers,
                get_model_config,
            )

            # Check if provider is supported
            available_providers = get_available_providers()
            if config.provider not in available_providers:
                raise ConfigurationError(
                    f"Unsupported provider '{config.provider}'. "
                    f"Available providers: {', '.join(available_providers)}. "
                    f"Set LLM_PROVIDER environment variable to one of the supported providers."
                )

            # Validate provider configuration
            try:
                model_config = get_model_config(config.provider)
                logger.info(f"✓ Provider '{config.provider}' configuration validated")
                logger.info(f"  Enhanced model: {model_config.enhanced_model_id}")
                logger.info(f"  Standard model: {model_config.standard_model_id}")

            except Exception as e:
                provider_specific_help = self._get_provider_setup_help(config.provider)
                raise ConfigurationError(
                    f"Failed to configure provider '{config.provider}': {e}\n\n"
                    f"Setup instructions:\n{provider_specific_help}"
                ) from e

        except ImportError as e:
            raise ConfigurationError(
                f"Failed to import configuration modules: {e}. "
                "This indicates a broken installation. Please reinstall the package."
            ) from e

    async def _validate_system_requirements(self) -> None:
        """Validate system-level requirements."""
        # Python version check

        # Check required packages are available
        required_packages = [
            ("agno", "2.0.5"),
            ("mcp", None),
            ("pydantic", None),
            ("asyncio", None),
        ]

        for package_name, min_version in required_packages:
            try:
                __import__(package_name)
                if min_version and package_name == "agno":
                    import agno

                    if hasattr(agno, "__version__"):
                        from packaging import version

                        if version.parse(agno.__version__) < version.parse(min_version):
                            raise ConfigurationError(
                                f"Agno {min_version}+ required, but found {agno.__version__}. "
                                "Run: uv pip install --upgrade agno"
                            )
            except ImportError as e:
                raise ConfigurationError(
                    f"Required package '{package_name}' not found: {e}. "
                    "Run: uv pip install -e '.[dev]' to install all dependencies."
                ) from e

        logger.info("✓ System requirements validated")

    async def _setup_directories(self) -> None:
        """Setup required directories with proper error handling."""
        # Ensure log directory exists
        log_dir = Path.home() / ".sequential_thinking" / "logs"
        if not log_dir.exists():
            logger.info(f"Creating log directory: {log_dir}")
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                logger.info("✓ Log directory created successfully")
            except OSError as e:
                raise ConfigurationError(
                    f"Failed to create log directory {log_dir}: {e}. "
                    "Please check file system permissions."
                ) from e
        else:
            logger.info("✓ Log directory already exists")

        # Validate write permissions
        test_file = log_dir / "test_write.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
            logger.info("✓ Log directory write permissions validated")
        except OSError as e:
            raise ConfigurationError(
                f"Cannot write to log directory {log_dir}: {e}. "
                "Please check file system permissions."
            ) from e

    def _get_provider_setup_help(self, provider: str) -> str:
        """Get provider-specific setup instructions."""
        help_text = {
            "deepseek": (
                "1. Get API key from https://platform.deepseek.com/api_keys\n"
                "2. Set environment variable: export DEEPSEEK_API_KEY='your-key'\n"
                "3. (Optional) Set models: DEEPSEEK_ENHANCED_MODEL_ID, DEEPSEEK_STANDARD_MODEL_ID"
            ),
            "groq": (
                "1. Get API key from https://console.groq.com/keys\n"
                "2. Set environment variable: export GROQ_API_KEY='your-key'\n"
                "3. (Optional) Set models: GROQ_ENHANCED_MODEL_ID, GROQ_STANDARD_MODEL_ID"
            ),
            "github": (
                "1. Get GitHub token from https://github.com/settings/tokens\n"
                "2. Set environment variable: export GITHUB_TOKEN='your-token'\n"
                "3. (Optional) Set models: GITHUB_ENHANCED_MODEL_ID, GITHUB_STANDARD_MODEL_ID\n"
                "4. Token must have appropriate permissions for GitHub Models API"
            ),
            "anthropic": (
                "1. Get API key from https://console.anthropic.com/\n"
                "2. Set environment variable: export ANTHROPIC_API_KEY='your-key'\n"
                "3. (Optional) Set models: ANTHROPIC_ENHANCED_MODEL_ID, ANTHROPIC_STANDARD_MODEL_ID"
            ),
            "openrouter": (
                "1. Get API key from https://openrouter.ai/keys\n"
                "2. Set environment variable: export OPENROUTER_API_KEY='your-key'\n"
                "3. (Optional) Set models: OPENROUTER_ENHANCED_MODEL_ID, OPENROUTER_STANDARD_MODEL_ID"
            ),
            "ollama": (
                "1. Install Ollama from https://ollama.ai/\n"
                "2. Start Ollama service: ollama serve\n"
                "3. Pull required models: ollama pull llama2\n"
                "4. (Optional) Set models: OLLAMA_ENHANCED_MODEL_ID, OLLAMA_STANDARD_MODEL_ID"
            ),
        }

        return help_text.get(
            provider, f"Please check documentation for '{provider}' provider setup."
        )

    async def cleanup(self) -> None:
        """No cleanup needed for environment."""
        logger.info("Environment cleanup completed")


class ServerState:
    """Manages server state with proper lifecycle and separation of concerns."""

    def __init__(self) -> None:
        self._config: ServerConfig | None = None
        self._session: SessionMemory | None = None
        self._initializers = [
            EnvironmentInitializer(),
        ]

    async def initialize(self, config: ServerConfig) -> None:
        """Initialize all server components."""
        self._config = config

        # Ordered initialization prevents dependency conflicts
        for initializer in self._initializers:
            await initializer.initialize(config)

        # Session-based architecture simplifies state management
        self._session = SessionMemory()

        logger.info(
            "Server state initialized successfully with multi-thinking workflow"
        )

    async def cleanup(self) -> None:
        """Clean up all server components."""
        # Clean up in reverse order
        for initializer in reversed(self._initializers):
            await initializer.cleanup()

        self._config = None
        self._session = None

        logger.info("Server state cleaned up")

    @property
    def config(self) -> ServerConfig:
        """Get current configuration."""
        if self._config is None:
            raise RuntimeError("Server not initialized - config unavailable")
        return self._config

    @property
    def session(self) -> SessionMemory:
        """Get current session."""
        if self._session is None:
            raise RuntimeError("Server not initialized - session unavailable")
        return self._session


# Remove redundant exception definition as it's now in types.py


class ThoughtProcessor:
    """Thought processor using dependency injection and clean architecture.

    This class orchestrates thought processing by delegating specific responsibilities
    to specialized services, maintaining a clean separation of concerns.
    """

    __slots__ = (
        "_context_builder",
        "_metrics_logger",
        "_processing_orchestrator",
        "_response_formatter",
        "_session",
        "_workflow_executor",
    )

    def __init__(self, session: SessionMemory) -> None:
        """Initialize the thought processor with dependency injection.

        Args:
            session: The session memory instance for accessing team and context
        """
        from mcp_server_mas_sequential_thinking.infrastructure import MetricsLogger

        from .context_builder import ContextBuilder
        from .processing_orchestrator import ProcessingOrchestrator
        from .response_formatter import ResponseFormatter
        from .response_processor import ResponseProcessor
        from .retry_handler import TeamProcessingRetryHandler
        from .workflow_executor import WorkflowExecutor

        self._session = session

        # Initialize core services with dependency injection
        self._context_builder = ContextBuilder(session)
        self._workflow_executor = WorkflowExecutor(session)
        self._response_formatter = ResponseFormatter()

        # Initialize supporting services
        response_processor = ResponseProcessor()
        retry_handler = TeamProcessingRetryHandler()
        self._processing_orchestrator = ProcessingOrchestrator(
            session, response_processor, retry_handler
        )

        # Initialize monitoring services
        self._metrics_logger = MetricsLogger()

        logger.info("ThoughtProcessor initialized with specialized services")

    async def process_thought(self, thought_data: ThoughtData) -> str:
        """Process a thought through the appropriate workflow with comprehensive error handling.

        This is the main public API method that orchestrates the complete thought processing
        workflow using specialized services.

        Args:
            thought_data: The thought data to process

        Returns:
            Processed thought response

        Raises:
            ThoughtProcessingError: If processing fails
        """
        import time

        from mcp_server_mas_sequential_thinking.config import ProcessingDefaults
        from mcp_server_mas_sequential_thinking.core import (
            ProcessingMetadata,
            ThoughtProcessingError,
        )

        try:
            start_time = time.time()

            # Log thought data and add to session (async for thread safety)
            self._log_thought_data(thought_data)
            await self._session.add_thought(thought_data)

            # Build context using specialized service (async for thread safety)
            input_prompt = await self._context_builder.build_context_prompt(
                thought_data
            )
            await self._context_builder.log_context_building(thought_data, input_prompt)

            # Execute Multi-Thinking workflow using specialized service
            (
                content,
                workflow_result,
                total_time,
            ) = await self._workflow_executor.execute_workflow(
                thought_data, input_prompt, start_time
            )

            # Format response using specialized service
            final_response = self._response_formatter.format_response(
                content, thought_data
            )

            # Log workflow completion
            self._workflow_executor.log_workflow_completion(
                thought_data, workflow_result, total_time, final_response
            )

            return final_response

        except Exception as e:
            error_msg = f"Failed to process {thought_data.thought_type.value} thought #{thought_data.thoughtNumber}: {e}"
            logger.error(error_msg, exc_info=True)
            metadata: ProcessingMetadata = {
                "error_count": ProcessingDefaults.ERROR_COUNT_INITIAL,
                "retry_count": ProcessingDefaults.RETRY_COUNT_INITIAL,
                "processing_time": ProcessingDefaults.PROCESSING_TIME_INITIAL,
            }
            raise ThoughtProcessingError(error_msg, metadata) from e

    def _log_thought_data(self, thought_data: ThoughtData) -> None:
        """Log comprehensive thought data information using centralized logger.

        Args:
            thought_data: The thought data to log
        """
        basic_info = {
            f"Thought #{thought_data.thoughtNumber}": f"{thought_data.thoughtNumber}/{thought_data.totalThoughts}",
            "Type": thought_data.thought_type.value,
            "Content": thought_data.thought,
            "Next needed": thought_data.nextThoughtNeeded,
            "Needs more": thought_data.needsMoreThoughts,
        }

        logger.info(f"Processing thought data: {basic_info}")


@asynccontextmanager
async def create_server_lifespan() -> AsyncIterator[ServerState]:
    """Create server lifespan context manager with proper resource management."""
    config = ServerConfig.from_environment()
    server_state = ServerState()

    try:
        await server_state.initialize(config)
        logger.info("Server started successfully")
        yield server_state

    except Exception as e:
        logger.error(f"Server initialization failed: {e}", exc_info=True)
        raise ServerInitializationError(f"Failed to initialize server: {e}") from e

    finally:
        await server_state.cleanup()
        logger.info("Server shutdown complete")


class ServerInitializationError(Exception):
    """Custom exception for server initialization failures."""


def create_validated_thought_data(
    thought: str,
    thoughtNumber: int,
    totalThoughts: int,
    nextThoughtNeeded: bool,
    isRevision: bool,
    branchFromThought: int | None,
    branchId: str | None,
    needsMoreThoughts: bool,
) -> ThoughtData:
    """Create and validate thought data with enhanced error reporting."""
    try:
        return ThoughtData(
            thought=thought.strip(),
            thoughtNumber=thoughtNumber,
            totalThoughts=totalThoughts,
            nextThoughtNeeded=nextThoughtNeeded,
            isRevision=isRevision,
            branchFromThought=branchFromThought,
            branchId=branchId.strip() if branchId else None,
            needsMoreThoughts=needsMoreThoughts,
        )
    except ValidationError as e:
        raise ValueError(f"Invalid thought data: {e}") from e


# Legacy global state removed - use dependency injection through ApplicationContainer
