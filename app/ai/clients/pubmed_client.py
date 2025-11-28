import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.utils.exceptions import AIServiceError

logger = logging.getLogger(__name__)


class PubMedArticle(BaseModel):
    """PubMed article information"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: str
    doi: Optional[str] = None
    url: str
    relevance_score: float = 0.0


class PubMedSearchResult(BaseModel):
    """PubMed search results"""
    query: str
    total_results: int
    articles: List[PubMedArticle]
    search_time: float


class PubMedClient:
    """
    Client for PubMed E-utilities API

    Provides access to biomedical literature from MEDLINE/PubMed database
    """

    def __init__(self, base_url: Optional[str] = None, email: Optional[str] = None):
        self.base_url = base_url or settings.PUBMED_API_BASE_URL
        self.email = email or "studyblitzai@example.com"  # Required by NCBI

        # Configure HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        # Rate limiting: NCBI allows 3 requests per second without API key
        self.min_request_interval = 0.34  # ~3 requests per second
        self.last_request_time = 0.0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def search_articles(
            self,
            query: str,
            max_results: int = 10,
            sort_by: str = "relevance",
            publication_years: Optional[int] = 5
    ) -> PubMedSearchResult:
        """
        Search PubMed for articles matching the query

        Args:
            query: Search query (medical terms, keywords)
            max_results: Maximum number of articles to return
            sort_by: Sort order ("relevance", "date", "author")
            publication_years: Limit to articles from last N years

        Returns:
            PubMedSearchResult with article information
        """
        start_time = datetime.now()

        try:
            # Step 1: Search for article IDs
            pmids = await self._search_pmids(query, max_results, publication_years)

            if not pmids:
                return PubMedSearchResult(
                    query=query,
                    total_results=0,
                    articles=[],
                    search_time=(datetime.now() - start_time).total_seconds()
                )

            # Step 2: Fetch detailed article information
            articles = await self._fetch_article_details(pmids)

            # Step 3: Calculate relevance scores
            articles = self._calculate_relevance_scores(articles, query)

            # Step 4: Sort articles
            if sort_by == "relevance":
                articles.sort(key=lambda x: x.relevance_score, reverse=True)
            elif sort_by == "date":
                articles.sort(key=lambda x: x.publication_date, reverse=True)

            search_time = (datetime.now() - start_time).total_seconds()

            return PubMedSearchResult(
                query=query,
                total_results=len(articles),
                articles=articles,
                search_time=search_time
            )

        except Exception as e:
            logger.error(f"PubMed search error: {str(e)}")
            raise AIServiceError(f"PubMed search failed: {str(e)}")

    async def _search_pmids(
            self,
            query: str,
            max_results: int,
            publication_years: Optional[int]
    ) -> List[str]:
        """Search for PubMed IDs matching the query"""
        await self._rate_limit()

        # Build search query
        search_query = query
        if publication_years:
            current_year = datetime.now().year
            start_year = current_year - publication_years
            search_query += f" AND {start_year}:{current_year}[pdat]"

        params = {
            "db": "pubmed",
            "term": search_query,
            "retmax": str(max_results),
            "retmode": "xml",
            "sort": "relevance",
            "tool": "StudyBlitzAI",
            "email": self.email
        }

        try:
            response = await self.client.get(f"{self.base_url}/esearch.fcgi", params=params)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.text)
            pmids = []

            for id_elem in root.findall(".//Id"):
                if id_elem.text:
                    pmids.append(id_elem.text)

            logger.info(f"Found {len(pmids)} PubMed articles for query: {query}")
            return pmids

        except Exception as e:
            logger.error(f"PubMed ID search error: {str(e)}")
            raise AIServiceError(f"Failed to search PubMed IDs: {str(e)}")

    async def _fetch_article_details(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch detailed information for articles by PMID"""
        if not pmids:
            return []

        await self._rate_limit()

        # Fetch details in batches to avoid URL length limits
        batch_size = 20
        all_articles = []

        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            batch_articles = await self._fetch_batch_details(batch_pmids)
            all_articles.extend(batch_articles)

        return all_articles

    async def _fetch_batch_details(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch details for a batch of PMIDs"""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "tool": "StudyBlitzAI",
            "email": self.email
        }

        try:
            response = await self.client.get(f"{self.base_url}/efetch.fcgi", params=params)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.text)
            articles = []

            for article_elem in root.findall(".//PubmedArticle"):
                article = self._parse_article_xml(article_elem)
                if article:
                    articles.append(article)

            return articles

        except Exception as e:
            logger.error(f"PubMed details fetch error: {str(e)}")
            return []  # Return empty list instead of raising to be more resilient

    def _parse_article_xml(self, article_elem: ET.Element) -> Optional[PubMedArticle]:
        """Parse article information from XML element"""
        try:
            # Extract PMID
            pmid_elem = article_elem.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                return None
            pmid = pmid_elem.text

            # Extract title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None and title_elem.text else "No title"

            # Extract abstract
            abstract_elems = article_elem.findall(".//AbstractText")
            abstract_parts = []
            for abs_elem in abstract_elems:
                if abs_elem.text:
                    label = abs_elem.get("Label", "")
                    text = abs_elem.text
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)

            abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available"

            # Extract authors
            authors = []
            for author_elem in article_elem.findall(".//Author"):
                last_name = author_elem.find("LastName")
                first_name = author_elem.find("ForeName")

                if last_name is not None and last_name.text:
                    author_name = last_name.text
                    if first_name is not None and first_name.text:
                        author_name += f", {first_name.text}"
                    authors.append(author_name)

            # Extract journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None and journal_elem.text else "Unknown journal"

            # Extract publication date
            pub_date_elem = article_elem.find(".//PubDate")
            publication_date = "Unknown date"
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find("Year")
                month_elem = pub_date_elem.find("Month")
                if year_elem is not None and year_elem.text:
                    publication_date = year_elem.text
                    if month_elem is not None and month_elem.text:
                        publication_date += f"-{month_elem.text}"

            # Extract DOI
            doi = None
            for id_elem in article_elem.findall(".//ArticleId"):
                if id_elem.get("IdType") == "doi" and id_elem.text:
                    doi = id_elem.text
                    break

            # Create PubMed URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=publication_date,
                doi=doi,
                url=url
            )

        except Exception as e:
            logger.warning(f"Failed to parse article XML: {str(e)}")
            return None

    def _calculate_relevance_scores(
            self,
            articles: List[PubMedArticle],
            query: str
    ) -> List[PubMedArticle]:
        """Calculate relevance scores based on query terms"""
        query_terms = query.lower().split()

        for article in articles:
            score = 0.0

            # Check title (higher weight)
            title_lower = article.title.lower()
            for term in query_terms:
                if term in title_lower:
                    score += 2.0

            # Check abstract (lower weight)
            abstract_lower = article.abstract.lower()
            for term in query_terms:
                if term in abstract_lower:
                    score += 1.0

            # Normalize by number of query terms
            if query_terms:
                score = score / len(query_terms)

            article.relevance_score = score

        return articles

    async def _rate_limit(self):
        """Implement rate limiting for NCBI API"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = asyncio.get_event_loop().time()

    def format_citation(self, article: PubMedArticle) -> str:
        """
        Format article as a citation

        Args:
            article: PubMedArticle to format

        Returns:
            Formatted citation string
        """
        authors_str = ", ".join(article.authors[:3])  # First 3 authors
        if len(article.authors) > 3:
            authors_str += " et al."

        citation = f"{authors_str}. {article.title}. {article.journal}. {article.publication_date}."

        if article.doi:
            citation += f" doi: {article.doi}"

        citation += f" PMID: {article.pmid}. Available at: {article.url}"

        return citation