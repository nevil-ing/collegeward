import asyncio
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any, Union
from datetime import datetime, timedelta

import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.utils.exceptions import AIServiceError

logger = logging.getLogger(__name__)


class GroqMessage(BaseModel):
    """Message format for Groq API"""
    role: str  # "system", "user", "assistant"
    content: str


class GroqChatRequest(BaseModel):
    """Request format for Groq chat completions"""
    model: str
    messages: List[GroqMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: float = 1.0


class GroqChatResponse(BaseModel):
    """Response format from Groq API"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, Union[int, float]]] = None  # Accept both int and float for timing fields


class GroqRateLimiter:
    """Simple rate limiter for Groq API calls"""

    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests
                         if now - req_time < timedelta(minutes=1)]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until the oldest request is more than 1 minute old
            oldest_request = min(self.requests)
            wait_time = 60 - (now - oldest_request).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)

        self.requests.append(now)


class GroqClient:
    """Async client for Groq API with error handling and rate limiting"""

    # Available models (updated to current Groq models)
    LLAMA_3_1_70B = "llama-3.3-70b-versatile"  # Current recommended model
    LLAMA_3_1_8B = "llama-3.1-8b-instant"  # Faster alternative
    MIXTRAL_8X7B = "mixtral-8x7b-32768"  # Deprecated - kept for reference
    LLAMA3_70B = "llama3-70b-8192"  # Deprecated - kept for reference
    LLAMA3_8B = "llama3-8b-8192"  # Deprecated - kept for reference

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.base_url = base_url or settings.GROQ_BASE_URL
        self.rate_limiter = GroqRateLimiter()

        if not self.api_key:
            raise ValueError("Groq API key is required")

        # Configure HTTP client with retries
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def chat_completion(
            self,
            messages: List[Dict[str, str]],
            model: str = LLAMA_3_1_70B,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stream: bool = False
    ) -> GroqChatResponse:
        """
        Create a chat completion using Groq API

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use for completion
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            GroqChatResponse object

        Raises:
            AIServiceError: If the API call fails
        """
        await self.rate_limiter.wait_if_needed()

        # Convert messages to Groq format
        groq_messages = [GroqMessage(role=msg["role"], content=msg["content"])
                         for msg in messages]

        request_data = GroqChatRequest(
            model=model,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )

        try:
            logger.info(f"Making Groq API request with model {model}")
            response = await self._make_request(
                "POST",
                "/chat/completions",
                json=request_data.model_dump(exclude_none=True)
            )

            return GroqChatResponse(**response)

        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise AIServiceError(f"Groq API request failed: {str(e)}")

    async def chat_completion_stream(
            self,
            messages: List[Dict[str, str]],
            model: str = LLAMA_3_1_70B,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion response

        Args:
            messages: List of message dictionaries
            model: Model to use for completion
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Content chunks as they arrive
        """
        await self.rate_limiter.wait_if_needed()

        groq_messages = [GroqMessage(role=msg["role"], content=msg["content"])
                         for msg in messages]

        request_data = GroqChatRequest(
            model=model,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        try:
            logger.info(f"Starting Groq API stream with model {model}")

            async with self.client.stream(
                    "POST",
                    "/chat/completions",
                    json=request_data.model_dump(exclude_none=True)
            ) as response:
                # Check status before processing
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = error_text.decode('utf-8') if error_text else f"HTTP {response.status_code}"
                    logger.error(f"Groq API returned {response.status_code}: {error_msg}")
                    raise AIServiceError(f"Groq API error {response.status_code}: {error_msg}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            if chunk.get("choices") and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            error_text = e.response.text if hasattr(e.response, 'text') else str(e)
            logger.error(f"Groq streaming HTTP error: {e.response.status_code} - {error_text}")
            raise AIServiceError(f"Groq streaming failed: HTTP {e.response.status_code} - {error_text}")
        except Exception as e:
            logger.error(f"Groq streaming error: {str(e)}")
            raise AIServiceError(f"Groq streaming failed: {str(e)}")

    async def _make_request(
            self,
            method: str,
            endpoint: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with exponential backoff retry

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters

        Returns:
            Response JSON data
        """
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = await self.client.request(method, endpoint, **kwargs)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise AIServiceError("Rate limit exceeded, max retries reached")

                elif e.response.status_code >= 500:  # Server error
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Server error, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise AIServiceError(f"Server error: {e.response.status_code}")

                else:
                    # Client error, don't retry
                    error_detail = ""
                    try:
                        error_data = e.response.json()
                        error_detail = error_data.get("error", {}).get("message", "")
                    except:
                        pass

                    raise AIServiceError(f"API error {e.response.status_code}: {error_detail}")

            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Request error, retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise AIServiceError(f"Request failed: {str(e)}")

    def get_recommended_model(self, task_type: str = "reasoning") -> str:
        """
        Get recommended model based on task type

        Args:
            task_type: Type of task ("reasoning", "speed", "balanced")

        Returns:
            Model name
        """
        if task_type == "reasoning":
            return self.LLAMA_3_1_70B  # Best for complex reasoning
        elif task_type == "speed":
            return self.LLAMA_3_1_8B  # Fastest responses
        else:
            return self.LLAMA_3_1_70B  # Balanced performance