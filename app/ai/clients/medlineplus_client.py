import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.utils.exceptions import AIServiceError

logger = logging.getLogger(__name__)


class MedlinePlusArticle(BaseModel):
    """MedlinePlus article information"""
    title: str
    url: str
    summary: str
    full_summary: str
    language: str = "en"
    categories: List[str] = []
    date_created: Optional[str] = None
    date_revised: Optional[str] = None
    relevance_score: float = 0.0


class MedlinePlusSearchResult(BaseModel):
    """MedlinePlus search results"""
    query: str
    total_results: int
    articles: List[MedlinePlusArticle]
    search_time: float


class MedlinePlusClient:
    """
    Client for MedlinePlus Connect API

    Provides access to patient-friendly health information from NIH/NLM
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.MEDLINEPLUS_API_BASE_URL

        # Configure HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        # Rate limiting: Be respectful to NIH servers
        self.min_request_interval = 0.5  # 2 requests per second
        self.last_request_time = 0.0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def search_health_topics(
            self,
            query: str,
            max_results: int = 10,
            language: str = "en"
    ) -> MedlinePlusSearchResult:
        """
        Search MedlinePlus for health topics matching the query

        Args:
            query: Search query (health topics, conditions, treatments)
            max_results: Maximum number of articles to return
            language: Language code (en, es)

        Returns:
            MedlinePlusSearchResult with article information
        """
        start_time = datetime.now()

        try:
            await self._rate_limit()

            # MedlinePlus Connect API parameters
            params = {
                "db": "healthTopics",
                "term": query,
                "knowledgeResponseType": "application/json",
                "informationRecipient": "PROV",
                "informationRecipient.languageCode": language
            }

            response = await self.client.get(self.base_url, params=params)
            response.raise_for_status()

            # Parse JSON response
            data = response.json()
            articles = self._parse_search_response(data, query)

            # Limit results
            articles = articles[:max_results]

            search_time = (datetime.now() - start_time).total_seconds()

            return MedlinePlusSearchResult(
                query=query,
                total_results=len(articles),
                articles=articles,
                search_time=search_time
            )

        except Exception as e:
            logger.error(f"MedlinePlus search error: {str(e)}")
            raise AIServiceError(f"MedlinePlus search failed: {str(e)}")

    async def get_health_topic_details(self, topic_id: str) -> Optional[MedlinePlusArticle]:
        """
        Get detailed information for a specific health topic

        Args:
            topic_id: MedlinePlus topic identifier

        Returns:
            Detailed MedlinePlusArticle or None if not found
        """
        try:
            await self._rate_limit()

            params = {
                "db": "healthTopics",
                "id": topic_id,
                "knowledgeResponseType": "application/json",
                "informationRecipient": "PROV"
            }

            response = await self.client.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()
            articles = self._parse_search_response(data, "")

            return articles[0] if articles else None

        except Exception as e:
            logger.error(f"MedlinePlus topic details error: {str(e)}")
            return None

    def _parse_search_response(
            self,
            data: Dict[str, Any],
            query: str
    ) -> List[MedlinePlusArticle]:
        """Parse MedlinePlus API response"""
        articles = []

        try:
            # Navigate the nested JSON structure
            feed = data.get("feed", {})
            entries = feed.get("entry", [])

            if not isinstance(entries, list):
                entries = [entries] if entries else []

            for entry in entries:
                article = self._parse_entry(entry)
                if article:
                    # Calculate relevance score
                    article.relevance_score = self._calculate_relevance_score(article, query)
                    articles.append(article)

            # Sort by relevance
            articles.sort(key=lambda x: x.relevance_score, reverse=True)

        except Exception as e:
            logger.warning(f"Failed to parse MedlinePlus response: {str(e)}")

        return articles

    def _parse_entry(self, entry: Dict[str, Any]) -> Optional[MedlinePlusArticle]:
        """Parse a single entry from MedlinePlus response"""
        try:
            # Extract title
            title = entry.get("title", {}).get("_value", "No title")

            # Extract URL
            links = entry.get("link", [])
            if not isinstance(links, list):
                links = [links] if links else []

            url = ""
            for link in links:
                if isinstance(link, dict) and link.get("href"):
                    url = link["href"]
                    break

            # Extract summary
            summary_elem = entry.get("summary", {})
            summary = summary_elem.get("_value", "") if isinstance(summary_elem, dict) else str(summary_elem)

            # For MedlinePlus, summary and full_summary are often the same
            full_summary = summary

            # Extract categories
            categories = []
            category_elem = entry.get("category", [])
            if not isinstance(category_elem, list):
                category_elem = [category_elem] if category_elem else []

            for cat in category_elem:
                if isinstance(cat, dict) and cat.get("term"):
                    categories.append(cat["term"])

            # Extract dates
            date_created = None
            date_revised = None

            updated_elem = entry.get("updated")
            if updated_elem:
                date_revised = str(updated_elem)

            published_elem = entry.get("published")
            if published_elem:
                date_created = str(published_elem)

            return MedlinePlusArticle(
                title=title,
                url=url,
                summary=summary[:500] + "..." if len(summary) > 500 else summary,
                full_summary=full_summary,
                categories=categories,
                date_created=date_created,
                date_revised=date_revised
            )

        except Exception as e:
            logger.warning(f"Failed to parse MedlinePlus entry: {str(e)}")
            return None

    def _calculate_relevance_score(
            self,
            article: MedlinePlusArticle,
            query: str
    ) -> float:
        """Calculate relevance score based on query terms"""
        if not query:
            return 1.0

        query_terms = query.lower().split()
        score = 0.0

        # Check title (higher weight)
        title_lower = article.title.lower()
        for term in query_terms:
            if term in title_lower:
                score += 3.0

        # Check summary (medium weight)
        summary_lower = article.summary.lower()
        for term in query_terms:
            if term in summary_lower:
                score += 2.0

        # Check categories (lower weight)
        categories_lower = " ".join(article.categories).lower()
        for term in query_terms:
            if term in categories_lower:
                score += 1.0

        # Normalize by number of query terms
        if query_terms:
            score = score / len(query_terms)

        return score

    async def _rate_limit(self):
        """Implement rate limiting for MedlinePlus API"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = asyncio.get_event_loop().time()

    def format_citation(self, article: MedlinePlusArticle) -> str:
        """
        Format article as a citation

        Args:
            article: MedlinePlusArticle to format

        Returns:
            Formatted citation string
        """
        citation = f"{article.title}. MedlinePlus."

        if article.date_revised:
            citation += f" Updated {article.date_revised}."
        elif article.date_created:
            citation += f" Created {article.date_created}."

        citation += f" Available at: {article.url}"

        return citation

    async def search_drug_information(
            self,
            drug_name: str,
            max_results: int = 5
    ) -> MedlinePlusSearchResult:
        """
        Search for drug information specifically

        Args:
            drug_name: Name of the drug/medication
            max_results: Maximum number of results

        Returns:
            MedlinePlusSearchResult with drug information
        """
        # Add drug-specific search terms
        query = f"{drug_name} medication drug"
        return await self.search_health_topics(query, max_results)

    async def search_condition_information(
            self,
            condition: str,
            max_results: int = 5
    ) -> MedlinePlusSearchResult:
        """
        Search for medical condition information

        Args:
            condition: Medical condition or disease name
            max_results: Maximum number of results

        Returns:
            MedlinePlusSearchResult with condition information
        """
        # Add condition-specific search terms
        query = f"{condition} disease condition symptoms treatment"
        return await self.search_health_topics(query, max_results)