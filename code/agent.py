import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import Optional, List, Any, Dict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from config import Config

import openai

# ==========================
# Validation Config Path
# ==========================
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# ==========================
# Constants
# ==========================
SYSTEM_PROMPT = (
    "You are a professional Python developer and code generation assistant. "
    "Your primary responsibility is to generate clear, efficient, and well-documented Python code that fulfills the user's specified requirement. "
    "Always follow best practices, including PEP 8 style guidelines, and provide concise explanations or comments within the code when appropriate. "
    "If the requirement is unclear or incomplete, politely request clarification before proceeding. "
    "Do not execute or test code; only provide the code as text output. "
    "If you cannot fulfill the request, respond with an appropriate fallback message."
)
OUTPUT_FORMAT = (
    "- Output only valid Python code in a properly formatted code block.\n\n"
    "- Include inline comments or a brief docstring if necessary for clarity.\n\n"
    "- If clarification is needed, ask the user for more details before generating code.\n\n"
    "- Do not include any extraneous text outside the code block unless clarification is requested."
)
FALLBACK_RESPONSE = (
    "I'm unable to generate Python code for the given requirement. Please provide more details or clarify your request."
)
FEW_SHOT_EXAMPLES = [
    "Write a Python function that returns the factorial of a number.",
    "Create a script that reads a CSV file and prints the number of rows."
]

# ==========================
# Input Models
# ==========================
class GenerateCodeRequest(BaseModel):
    requirement_description: str = Field(..., description="Natural language Python code requirement")
    user_context: Optional[str] = Field(None, description="Additional context or constraints for the code")

    @model_validator(mode="after")
    def validate_content(self):
        if not self.requirement_description or not self.requirement_description.strip():
            raise ValueError("Requirement description must be non-empty.")
        if len(self.requirement_description.strip()) > 50000:
            raise ValueError("Requirement description exceeds maximum length.")
        self.requirement_description = self.requirement_description.strip()
        if self.user_context:
            self.user_context = self.user_context.strip()
        return self

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation succeeded")
    code: Optional[str] = Field(None, description="Generated Python code")
    clarification: Optional[str] = Field(None, description="Clarification request if requirement is unclear")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    tool_calls_made: Optional[List[str]] = Field(None, description="List of tool calls made (none for this agent)")

# ==========================
# LLM Output Sanitizer
# ==========================
import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# ==========================
# Logger Utility
# ==========================
class Logger:
    """Audit and error logger."""
    def __init__(self):
        self.logger = logging.getLogger("PythonCodeGenerationAgent")
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, details: Any):
        self.logger.info(f"[{event_type}] {details}")

    def log_error(self, error_code: str, context: Any):
        self.logger.error(f"[{error_code}] {context}")

# ==========================
# Error Handler Utility
# ==========================
class ErrorHandler:
    """Centralized error handling and retry logic."""
    def __init__(self, logger: Logger):
        self.logger = logger

    def handle_error(self, error_code: str, context: Any) -> str:
        self.logger.log_error(error_code, context)
        if error_code == "INVALID_REQUIREMENT":
            return "The requirement is unclear or incomplete. Please clarify your request."
        elif error_code == "CODE_GENERATION_ERROR":
            return FALLBACK_RESPONSE
        else:
            return "An unexpected error occurred. Please try again."

    async def retry(self, operation, attempts: int = 3, *args, **kwargs):
        last_exc = None
        for attempt in range(attempts):
            try:
                return await operation(*args, **kwargs)
            except Exception as exc:
                self.logger.log_error("RETRY_ATTEMPT", f"Attempt {attempt+1} failed: {exc}")
                last_exc = exc
                await self._backoff(attempt)
        self.logger.log_error("RETRY_FAILED", f"All {attempts} attempts failed.")
        raise last_exc

    async def _backoff(self, attempt: int):
        delay = min(2 ** attempt, 8)
        await _asyncio.sleep(delay)

# ==========================
# Requirement Validator
# ==========================
class RequirementValidator:
    """Validates requirement clarity and actionability."""
    def validate(self, requirement_description: str) -> str:
        if not requirement_description or not requirement_description.strip():
            return "invalid"
        # Simple heuristic: must contain at least one verb and noun
        tokens = requirement_description.lower().split()
        verbs = {"create", "write", "generate", "build", "implement", "return", "print", "read", "extract", "calculate", "find", "display"}
        has_verb = any(token in verbs for token in tokens)
        has_noun = any(token for token in tokens if token not in verbs)
        if has_verb and has_noun and len(tokens) >= 4:
            return "valid"
        return "invalid"

# ==========================
# Response Formatter
# ==========================
class ResponseFormatter:
    """Formats Python code output, ensures code block formatting, adds comments/docstrings."""
    def format_code(self, code_text: str) -> str:
        code = sanitize_llm_output(code_text, content_type="code")
        # Ensure code block formatting
        if not code.strip().startswith("```"):
            code = f"```python\n{code.strip()}\n```"
        return code

# ==========================
# Security Manager
# ==========================
class SecurityManager:
    """Manages authentication, encryption, and session management."""
    def authenticate(self, token: Optional[str]) -> bool:
        # Token-based authentication stub (not enforced in agent code)
        return True

    def encrypt(self, data: str) -> str:
        # AES-256 encryption stub (not used, as agent does not persist data)
        return data

    def manage_session(self, session_id: Optional[str]) -> None:
        # Stateless session management stub
        pass

# ==========================
# LLM Service
# ==========================
class LLMService:
    """Handles interaction with Azure OpenAI GPT-4.1."""
    def __init__(self):
        self.model = Config.LLM_MODEL or "gpt-4.1"
        self.temperature = Config.LLM_TEMPERATURE or 0.7
        self.max_tokens = Config.LLM_MAX_TOKENS or 2000
        self.client = None

    def _get_client(self):
        if self.client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self.client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self.client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_python_code(self, prompt: str, user_context: Optional[str] = None, few_shot_examples: Optional[List[str]] = None) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT},
            {"role": "user", "content": prompt},
        ]
        if user_context:
            messages.append({"role": "user", "content": f"Additional context: {user_context}"})
        if few_shot_examples:
            for example in few_shot_examples:
                messages.append({"role": "user", "content": f"Example: {example}"})
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=self.model,
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return content
        except Exception as exc:
            try:
                trace_model_call(
                    provider="azure",
                    model_name=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=str(exc),
                    status="error",
                    error=exc,
                )
            except Exception:
                pass
            raise

    def get_few_shot_examples(self) -> List[str]:
        return FEW_SHOT_EXAMPLES

# ==========================
# Main Agent Class
# ==========================
class PythonCodeGenerationAgent:
    """Orchestrates input validation, LLM interaction, response formatting, and error handling."""

    def __init__(self):
        self.llm_service = LLMService()
        self.validator = RequirementValidator()
        self.formatter = ResponseFormatter()
        self.logger = Logger()
        self.error_handler = ErrorHandler(self.logger)
        self.security_manager = SecurityManager()

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process(self, requirement_description: str, user_context: Optional[str] = None) -> Dict[str, Any]:
        async with trace_step(
            "validate_requirement",
            step_type="parse",
            decision_summary="Validate requirement clarity and actionability",
            output_fn=lambda r: f"validation_status={r}",
        ) as step:
            validation_status = self.validator.validate(requirement_description)
            step.capture(validation_status)
        if validation_status != "valid":
            clarification = self.clarify_requirement(requirement_description)
            self.logger.log_event("CLARIFICATION_REQUEST", clarification)
            return {
                "success": False,
                "clarification": clarification,
                "code": None,
                "error": self.error_handler.handle_error("INVALID_REQUIREMENT", requirement_description),
                "tool_calls_made": [],
            }
        async with trace_step(
            "generate_code",
            step_type="llm_call",
            decision_summary="Generate Python code using LLM",
            output_fn=lambda r: f"code_length={len(r) if r else 0}",
        ) as step:
            try:
                code_raw = await self.error_handler.retry(
                    self.llm_service.generate_python_code,
                    attempts=3,
                    prompt=requirement_description,
                    user_context=user_context,
                    few_shot_examples=self.llm_service.get_few_shot_examples(),
                )
                step.capture(code_raw)
                code = self.formatter.format_code(code_raw)
                self.logger.log_event("CODE_GENERATED", {"length": len(code)})
                return {
                    "success": True,
                    "code": code,
                    "clarification": None,
                    "error": None,
                    "tool_calls_made": [],
                }
            except Exception as exc:
                error_msg = self.error_handler.handle_error("CODE_GENERATION_ERROR", str(exc))
                self.logger.log_event("CODE_GENERATION_ERROR", error_msg)
                return {
                    "success": False,
                    "code": None,
                    "clarification": None,
                    "error": error_msg,
                    "tool_calls_made": [],
                }

    def clarify_requirement(self, requirement_description: str) -> str:
        return (
            "Your requirement is unclear or incomplete. "
            "Please provide more details about the desired functionality, inputs, and expected outputs."
        )

# ==========================
# Observability Lifespan
# ==========================
@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

# ==========================
# FastAPI App
# ==========================
app = FastAPI(
    title="Python Code Generation Assistant",
    description="Generates clear, efficient, and well-documented Python code based on user requirements.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

# ==========================
# Exception Handlers
# ==========================
@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error. Please try again.",
            "tips": "Check your input format and ensure all required fields are present."
        }
    )

@app.exception_handler(ValueError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": str(exc),
            "tips": "Check your input for missing or invalid values."
        }
    )

@app.exception_handler(json.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_error_handler(request: Request, exc: json.JSONDecodeError):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "Malformed JSON request.",
            "tips": "Ensure your JSON is properly formatted with correct quotes, commas, and braces."
        }
    )

# ==========================
# Health Check Endpoint
# ==========================
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# ==========================
# Main Business Endpoint
# ==========================
_agent_instance = PythonCodeGenerationAgent()

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: GenerateCodeRequest):
    async with trace_step(
        "process_query",
        step_type="process",
        decision_summary="Process user requirement and generate Python code",
        output_fn=lambda r: f"success={r.get('success', False)}",
    ) as step:
        result = await _agent_instance.process(
            requirement_description=req.requirement_description,
            user_context=req.user_context,
        )
        step.capture(result)
    # Sanitize output
    if result.get("code"):
        result["code"] = sanitize_llm_output(result["code"], content_type="code")
    return QueryResponse(**result)

# ==========================
# Entrypoint
# ==========================
async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())