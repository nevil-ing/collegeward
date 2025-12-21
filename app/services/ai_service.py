import asyncio
import logging
from app.ai.clients.groq_client import GroqClient
from app.ai.clients.medical_sources import MedicalSourcesClient, MedicalSource
from app.core.config import settings
from app.utils.exceptions import AIServiceError
from app.models.ai_response import AIResponse, ConversationMessage
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime

logger = logging.getLogger(__name__)


class AIServiceManager:
    """
    Manages all AI services and provides unified interface
    """

    def __init__(self):
        self.groq_client = None
        self.medical_sources_client = None
        self._initialized = False

    async def initialize(self):
        """Initialize AI service clients"""
        if self._initialized:
            return

        try:
            self.groq_client = GroqClient()
            self.medical_sources_client = MedicalSourcesClient()
            await self.medical_sources_client.__aenter__()
            self._initialized = True
            logger.info("AI services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI services: {str(e)}")
            raise AIServiceError(f"AI service initialization failed: {str(e)}")

    async def cleanup(self):
        """Cleanup AI service clients"""
        if self.medical_sources_client:
            await self.medical_sources_client.__aexit__(None, None, None)
        if self.groq_client:
            await self.groq_client.__aexit__(None, None, None)
        self._initialized = False

    async def generate_response(
            self,
            messages: List[ConversationMessage],
            mode: str = "ai_mode",
            context: Optional[str] = None,
            medical_query: Optional[str] = None,
            temperature: float = None,
            max_tokens: Optional[int] = None
    ) -> AIResponse:
        """
        Generate AI response with optional medical source integration

        Args:
            messages: Conversation history
            mode: "ai_mode" (fast) or "verified_mode" (with medical sources)
            context: Additional context from user notes
            medical_query: Query for medical sources (if different from last message)
            temperature: AI temperature setting
            max_tokens: Maximum tokens to generate

        Returns:
            AIResponse with content and sources
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        try:
            # Prepare AI messages
            ai_messages = self._prepare_ai_messages(messages, context)

            # Get medical sources if in verified mode
            sources = []
            citations = []

            if mode == "verified_mode" and medical_query:
                logger.info(f"Fetching medical sources for: {medical_query}")
                sources_result = await self.medical_sources_client.search_medical_information(
                    query=medical_query,
                    max_pubmed_results=3,
                    max_medlineplus_results=2
                )
                sources = sources_result.sources
                citations = sources_result.sources and [s.citation for s in sources] or []

                # Add medical sources to AI context
                if sources:
                    medical_context = self.medical_sources_client.format_sources_for_ai(sources)
                    ai_messages.append({
                        "role": "system",
                        "content": f"Use the following verified medical sources to inform your response:\n\n{medical_context}"
                    })

            # Generate AI response
            model = self._select_model(mode)
            temp = temperature if temperature is not None else settings.AI_TEMPERATURE
            max_tok = max_tokens if max_tokens is not None else settings.AI_MAX_TOKENS

            response = await self.groq_client.chat_completion(
                messages=ai_messages,
                model=model,
                temperature=temp,
                max_tokens=max_tok
            )

            content = response.choices[0]["message"]["content"] if response.choices else ""
            response_time = (datetime.now() - start_time).total_seconds()

            return AIResponse(
                content=content,
                model_used=model,
                response_time=response_time,
                sources=sources,
                citations=citations,
                mode=mode
            )

        except Exception as e:
            logger.error(f"AI response generation failed: {str(e)}")
            raise AIServiceError(f"Failed to generate AI response: {str(e)}")

    async def generate_response_stream(
            self,
            messages: List[ConversationMessage],
            mode: str = "ai_mode",
            context: Optional[str] = None,
            medical_query: Optional[str] = None,
            temperature: float = None,
            max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming AI response

        Args:
            messages: Conversation history
            mode: "ai_mode" or "verified_mode"
            context: Additional context from user notes
            medical_query: Query for medical sources
            temperature: AI temperature setting
            max_tokens: Maximum tokens to generate

        Yields:
            Content chunks as they arrive
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Prepare AI messages
            ai_messages = self._prepare_ai_messages(messages, context)

            # Get medical sources if in verified mode (but don't wait for streaming)
            if mode == "verified_mode" and medical_query:
                # Run medical sources search in background
                asyncio.create_task(self._log_medical_sources(medical_query))

                # Add a note about verified sources
                ai_messages.append({
                    "role": "system",
                    "content": "You are providing information in verified mode. Base your response on established medical knowledge and indicate when citing external sources."
                })

            # Generate streaming response
            model = self._select_model(mode)
            temp = temperature if temperature is not None else settings.AI_TEMPERATURE
            max_tok = max_tokens if max_tokens is not None else settings.AI_MAX_TOKENS

            async for chunk in self.groq_client.chat_completion_stream(
                    messages=ai_messages,
                    model=model,
                    temperature=temp,
                    max_tokens=max_tok
            ):
                yield chunk

        except Exception as e:
            logger.error(f"AI streaming response failed: {str(e)}")
            raise AIServiceError(f"Failed to generate streaming response: {str(e)}")

    async def search_medical_sources(
            self,
            query: str,
            max_results: int = 10,
            include_research: bool = True,
            include_patient_info: bool = True
    ) -> List[MedicalSource]:
        """
        Search medical sources independently

        Args:
            query: Medical search query
            max_results: Maximum number of results
            include_research: Include PubMed research papers
            include_patient_info: Include MedlinePlus patient information

        Returns:
            List of medical sources
        """
        if not self._initialized:
            await self.initialize()

        try:
            pubmed_results = max_results // 2 if include_research else 0
            medlineplus_results = max_results - pubmed_results if include_patient_info else max_results

            result = await self.medical_sources_client.search_medical_information(
                query=query,
                max_pubmed_results=pubmed_results,
                max_medlineplus_results=medlineplus_results,
                include_research=include_research,
                include_patient_info=include_patient_info
            )

            return result.sources

        except Exception as e:
            logger.error(f"Medical sources search failed: {str(e)}")
            raise AIServiceError(f"Medical sources search failed: {str(e)}")

    def _prepare_ai_messages(
            self,
            messages: List[ConversationMessage],
            context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for AI API"""
        ai_messages = []

        # Add system message with context
        system_content = self._get_system_prompt()
        if context:
            system_content += f"\n\nUser's study materials context:\n{context}"

        ai_messages.append({
            "role": "system",
            "content": system_content
        })

        # Add conversation messages
        for msg in messages:
            ai_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        return ai_messages

    def _get_system_prompt(self) -> str:
        """Get system prompt for AI tutor"""
        return """You are CollegeWard, an expert medical tutor designed to help medical students learn complex concepts through step-by-step reasoning and clinical examples.

Your teaching approach:
1. Break down complex medical concepts into digestible steps
2. Use clinical scenarios and real-world examples
3. Encourage critical thinking with guided questions
4. Provide clear explanations of pathophysiology, diagnosis, and treatment
5. Connect theoretical knowledge to practical applications

Guidelines:
- Always prioritize accuracy and evidence-based information
- Use medical terminology appropriately but explain complex terms
- Encourage active learning through questions and case-based reasoning
- When uncertain, acknowledge limitations and suggest consulting authoritative sources
- Focus on helping students understand underlying principles, not just memorization

Remember: You are an educational tool. Always encourage students to verify information with their instructors and authoritative medical sources for clinical decision-making."""

    def _select_model(self, mode: str) -> str:
        """Select appropriate AI model based on mode"""
        if mode == "verified_mode":
            # Use more capable model for verified mode
            return GroqClient.LLAMA_3_1_70B
        else:
            # Use capable model for AI mode
            return GroqClient.LLAMA_3_1_70B

    async def _log_medical_sources(self, query: str):
        """Log medical sources search in background"""
        try:
            result = await self.medical_sources_client.search_medical_information(
                query=query,
                max_pubmed_results=2,
                max_medlineplus_results=2
            )
            logger.info(f"Found {result.total_results} medical sources for: {query}")
        except Exception as e:
            logger.warning(f"Background medical sources search failed: {str(e)}")


ai_service_manager = AIServiceManager()