import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from pydantic import BaseModel

from .pubmed_client import PubMedClient, PubMedArticle, PubMedSearchResult
from .medlineplus_client import MedlinePlusClient, MedlinePlusArticle, MedlinePlusSearchResult
from app.utils.exceptions import AIServiceError

logger = logging.getLogger(__name__)


class MedicalSource(BaseModel):
    """Unified medical source information"""
    title: str
    url: str
    summary: str
    source_type: str  # "pubmed" or "medlineplus"
    citation: str
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = {}


class MedicalSourcesResult(BaseModel):
    """Combined medical sources search result"""
    query: str
    pubmed_results: int
    medlineplus_results: int
    total_results: int
    sources: List[MedicalSource]
    search_time: float


class ContentFilter:
    """Filter and validate medical content"""

    # Keywords that indicate high-quality medical content
    QUALITY_INDICATORS = [
        "systematic review", "meta-analysis", "randomized controlled trial",
        "clinical trial", "evidence-based", "peer-reviewed", "cochrane",
        "guidelines", "consensus", "practice parameters"
    ]

    # Keywords that might indicate lower quality or non-medical content
    EXCLUSION_KEYWORDS = [
        "advertisement", "sponsored", "commercial", "blog", "opinion",
        "personal experience", "testimonial", "unverified"
    ]

    @classmethod
    def calculate_quality_score(cls, content: str, title: str = "") -> float:
        """
        Calculate content quality score based on indicators

        Args:
            content: Article content/abstract
            title: Article title

        Returns:
            Quality score (0.0 to 1.0)
        """
        content_lower = content.lower()
        title_lower = title.lower()
        combined_text = f"{title_lower} {content_lower}"

        quality_score = 0.5  # Base score

        # Boost for quality indicators
        for indicator in cls.QUALITY_INDICATORS:
            if indicator in combined_text:
                quality_score += 0.1

        # Penalize for exclusion keywords
        for keyword in cls.EXCLUSION_KEYWORDS:
            if keyword in combined_text:
                quality_score -= 0.2

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, quality_score))

    @classmethod
    def is_medical_content(cls, content: str, title: str = "") -> bool:
        """
        Check if content is medical/health related

        Args:
            content: Article content
            title: Article title

        Returns:
            True if content appears to be medical
        """
        medical_keywords = [
            "medical", "health", "disease", "treatment", "therapy", "diagnosis",
            "clinical", "patient", "medicine", "healthcare", "symptom", "condition",
            "drug", "medication", "pharmaceutical", "pathology", "anatomy",
            "physiology", "surgery", "hospital", "doctor", "physician", "nurse"
        ]

        combined_text = f"{title} {content}".lower()

        # Count medical keywords
        medical_count = sum(1 for keyword in medical_keywords if keyword in combined_text)

        # Consider it medical if it has at least 2 medical keywords
        return medical_count >= 2


class MedicalSourcesClient:
    """
    Unified client for medical information sources

    Combines PubMed (research papers) and MedlinePlus (patient information)
    with content filtering and citation formatting
    """

    def __init__(self):
        self.pubmed_client = PubMedClient()
        self.medlineplus_client = MedlinePlusClient()
        self.content_filter = ContentFilter()

    async def __aenter__(self):
        await self.pubmed_client.__aenter__()
        await self.medlineplus_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.pubmed_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.medlineplus_client.__aexit__(exc_type, exc_val, exc_tb)

    async def search_medical_information(
            self,
            query: str,
            max_pubmed_results: int = 5,
            max_medlineplus_results: int = 5,
            include_research: bool = True,
            include_patient_info: bool = True,
            quality_threshold: float = 0.3
    ) -> MedicalSourcesResult:
        """
        Search both PubMed and MedlinePlus for medical information

        Args:
            query: Medical search query
            max_pubmed_results: Maximum PubMed articles
            max_medlineplus_results: Maximum MedlinePlus articles
            include_research: Whether to include PubMed research papers
            include_patient_info: Whether to include MedlinePlus patient info
            quality_threshold: Minimum quality score for inclusion

        Returns:
            Combined search results with filtered and scored sources
        """
        start_time = datetime.now()

        try:
            # Run searches concurrently
            tasks = []

            if include_research:
                tasks.append(self._search_pubmed_safe(query, max_pubmed_results))
            else:
                tasks.append(asyncio.create_task(self._empty_pubmed_result(query)))

            if include_patient_info:
                tasks.append(self._search_medlineplus_safe(query, max_medlineplus_results))
            else:
                tasks.append(asyncio.create_task(self._empty_medlineplus_result(query)))

            pubmed_result, medlineplus_result = await asyncio.gather(*tasks)

            # Convert to unified format
            sources = []

            # Process PubMed results
            for article in pubmed_result.articles:
                source = self._convert_pubmed_to_source(article)
                if self._should_include_source(source, quality_threshold):
                    sources.append(source)

            # Process MedlinePlus results
            for article in medlineplus_result.articles:
                source = self._convert_medlineplus_to_source(article)
                if self._should_include_source(source, quality_threshold):
                    sources.append(source)

            # Sort by relevance score
            sources.sort(key=lambda x: x.relevance_score, reverse=True)

            search_time = (datetime.now() - start_time).total_seconds()

            return MedicalSourcesResult(
                query=query,
                pubmed_results=len(pubmed_result.articles),
                medlineplus_results=len(medlineplus_result.articles),
                total_results=len(sources),
                sources=sources,
                search_time=search_time
            )

        except Exception as e:
            logger.error(f"Medical sources search error: {str(e)}")
            raise AIServiceError(f"Medical sources search failed: {str(e)}")

    async def _search_pubmed_safe(self, query: str, max_results: int) -> PubMedSearchResult:
        """Safely search PubMed with error handling"""
        try:
            return await self.pubmed_client.search_articles(query, max_results)
        except Exception as e:
            logger.warning(f"PubMed search failed: {str(e)}")
            return PubMedSearchResult(query=query, total_results=0, articles=[], search_time=0.0)

    async def _search_medlineplus_safe(self, query: str, max_results: int) -> MedlinePlusSearchResult:
        """Safely search MedlinePlus with error handling"""
        try:
            return await self.medlineplus_client.search_health_topics(query, max_results)
        except Exception as e:
            logger.warning(f"MedlinePlus search failed: {str(e)}")
            return MedlinePlusSearchResult(query=query, total_results=0, articles=[], search_time=0.0)

    async def _empty_pubmed_result(self, query: str) -> PubMedSearchResult:
        """Return empty PubMed result"""
        return PubMedSearchResult(query=query, total_results=0, articles=[], search_time=0.0)

    async def _empty_medlineplus_result(self, query: str) -> MedlinePlusSearchResult:
        """Return empty MedlinePlus result"""
        return MedlinePlusSearchResult(query=query, total_results=0, articles=[], search_time=0.0)

    def _convert_pubmed_to_source(self, article: PubMedArticle) -> MedicalSource:
        """Convert PubMed article to unified source format"""
        return MedicalSource(
            title=article.title,
            url=article.url,
            summary=article.abstract[:500] + "..." if len(article.abstract) > 500 else article.abstract,
            source_type="pubmed",
            citation=self.pubmed_client.format_citation(article),
            relevance_score=article.relevance_score,
            metadata={
                "pmid": article.pmid,
                "authors": article.authors,
                "journal": article.journal,
                "publication_date": article.publication_date,
                "doi": article.doi
            }
        )

    def _convert_medlineplus_to_source(self, article: MedlinePlusArticle) -> MedicalSource:
        """Convert MedlinePlus article to unified source format"""
        return MedicalSource(
            title=article.title,
            url=article.url,
            summary=article.summary,
            source_type="medlineplus",
            citation=self.medlineplus_client.format_citation(article),
            relevance_score=article.relevance_score,
            metadata={
                "categories": article.categories,
                "date_created": article.date_created,
                "date_revised": article.date_revised,
                "language": article.language
            }
        )

    def _should_include_source(self, source: MedicalSource, quality_threshold: float) -> bool:
        """Determine if source should be included based on quality filters"""
        # Check if content is medical
        if not self.content_filter.is_medical_content(source.summary, source.title):
            return False

        # Calculate quality score
        quality_score = self.content_filter.calculate_quality_score(source.summary, source.title)

        # Update source metadata with quality score
        source.metadata["quality_score"] = quality_score

        return quality_score >= quality_threshold

    def format_sources_for_ai(self, sources: List[MedicalSource], max_sources: int = 5) -> str:
        """
        Format sources for inclusion in AI prompts

        Args:
            sources: List of medical sources
            max_sources: Maximum number of sources to include

        Returns:
            Formatted string for AI context
        """
        if not sources:
            return "No relevant medical sources found."

        formatted_sources = []

        for i, source in enumerate(sources[:max_sources], 1):
            source_text = f"""
Source {i} ({source.source_type.upper()}):
Title: {source.title}
Summary: {source.summary}
Citation: {source.citation}
"""
            formatted_sources.append(source_text.strip())

        return "\n\n".join(formatted_sources)

    def get_citations_list(self, sources: List[MedicalSource]) -> List[str]:
        """
        Get list of formatted citations

        Args:
            sources: List of medical sources

        Returns:
            List of citation strings
        """
        return [source.citation for source in sources]