from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
import re
import math

from app.services.embedding_service import embedding_service
from app.rag.qdrant_client import qdrant_manager

from app.core.logging import get_logger
from app.utils.exceptions import ProcessingError


logger = get_logger(__name__)

class RAGRetriever:

    def __init__(self):
        self.max_context_tokens = 4000
        self.min_relevance_score = 0.6
        self.max_chunks_per_query = 20
        self.diversity_threshold = 0.85

    async def retrieve_context(
            self,
            query: str,
            user_id: str,
            subject_tags: Optional[List[str]] = None,
            max_tokens: Optional[int] = None,
            include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve and rank relevant context for a query

        Args:
            query: User's question or search query
            user_id: ID of the user making the query
            subject_tags: Optional subject filters
            max_tokens: Maximum tokens for context window
            include_metadata: Whether to include chunk metadata

        Returns:
            Dictionary containing ranked context chunks and metadata
        """
        try:
            # Use provided max_tokens or default
            token_limit = max_tokens or self.max_context_tokens

            # Generate query embedding
            query_embedding = await embedding_service.generate_single_embedding(query)

            if not query_embedding or all(x == 0.0 for x in query_embedding):
                logger.warning(f"Failed to generate valid embedding for query: {query}")
                return self._empty_context_response()

            # Retrieve similar chunks from vector database
            similar_chunks = await qdrant_manager.search_similar_chunks(
                user_id=user_id,
                query_vector=query_embedding,
                limit=self.max_chunks_per_query,
                score_threshold=self.min_relevance_score,
                subject_tags=subject_tags
            )

            if not similar_chunks:
                logger.info(f"No relevant chunks found for user {user_id}")
                return self._empty_context_response()

            # Rank and score chunks
            ranked_chunks = await self._rank_chunks(query, similar_chunks)

            # Apply diversity filtering
            diverse_chunks = self._apply_diversity_filtering(ranked_chunks)

            # Optimize for token limit
            optimized_chunks = self._optimize_context_window(
                diverse_chunks,
                token_limit,
                include_metadata
            )

            # Calculate context statistics
            context_stats = self._calculate_context_stats(optimized_chunks, similar_chunks)

            logger.info(
                f"Retrieved {len(optimized_chunks)} chunks for user {user_id}, "
                f"total tokens: {context_stats['total_tokens']}"
            )

            return {
                "chunks": optimized_chunks,
                "query": query,
                "user_id": user_id,
                "subject_tags": subject_tags,
                "stats": context_stats,
                "retrieved_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Context retrieval failed for user {user_id}: {e}")
            raise ProcessingError(f"Failed to retrieve context: {str(e)}")

    async def _rank_chunks(
            self,
            query: str,
            chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank chunks using multiple scoring factors

        Combines:
        - Vector similarity score
        - Query term overlap
        - Chunk length penalty
        - Recency bonus (if applicable)
        """
        try:
            query_terms = self._extract_query_terms(query)

            for chunk in chunks:
                # Base similarity score from vector search
                similarity_score = chunk.get("score", 0.0)

                # Calculate term overlap score
                term_overlap_score = self._calculate_term_overlap(
                    query_terms,
                    chunk.get("text", "")
                )

                # Calculate length penalty (prefer moderate length chunks)
                length_penalty = self._calculate_length_penalty(chunk.get("text", ""))

                # Calculate subject relevance bonus
                subject_bonus = self._calculate_subject_bonus(
                    query_terms,
                    chunk.get("subject_tags", [])
                )

                # Combine scores with weights
                final_score = (
                        similarity_score * 0.5 +
                        term_overlap_score * 0.25 +
                        subject_bonus * 0.15 +
                        (1.0 - length_penalty) * 0.1
                )

                chunk["relevance_score"] = final_score
                chunk["scoring_details"] = {
                    "similarity_score": similarity_score,
                    "term_overlap_score": term_overlap_score,
                    "length_penalty": length_penalty,
                    "subject_bonus": subject_bonus,
                    "final_score": final_score
                }

            # Sort by final relevance score
            ranked_chunks = sorted(chunks, key=lambda x: x["relevance_score"], reverse=True)

            logger.debug(f"Ranked {len(ranked_chunks)} chunks by relevance")
            return ranked_chunks

        except Exception as e:
            logger.error(f"Chunk ranking failed: {e}")
            # Return original chunks if ranking fails
            return chunks

    def _extract_query_terms(self, query: str) -> List[str]:
            """Extract meaningful terms from query"""

            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why'
            }


            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            meaningful_terms = [word for word in words if word not in stop_words]

            return meaningful_terms

    def _calculate_term_overlap(self, query_terms: List[str], text: str) -> float:
            """Calculate overlap between query terms and chunk text"""
            if not query_terms or not text:
                return 0.0

            text_lower = text.lower()
            matches = sum(1 for term in query_terms if term in text_lower)

            # Normalize by query length
            overlap_score = matches / len(query_terms)
            return min(overlap_score, 1.0)

    def _calculate_length_penalty(self, text: str) -> float:
            """Calculate penalty for chunks that are too short or too long"""
            if not text:
                return 1.0

            text_length = len(text)
            optimal_length = 800  # Optimal chunk length

            if text_length < 100:
                # Penalty for very short chunks
                return 0.8
            elif text_length > 1500:
                # Penalty for very long chunks
                return 0.3
            else:
                # Gaussian-like penalty around optimal length
                deviation = abs(text_length - optimal_length) / optimal_length
                penalty = min(deviation * 0.5, 0.5)
                return penalty

    def _calculate_subject_bonus(self, query_terms: List[str], subject_tags: List[str]) -> float:
            """Calculate bonus for subject tag relevance"""
            if not query_terms or not subject_tags:
                return 0.0

            # Check if any query terms match subject tags
            query_text = " ".join(query_terms)
            tag_matches = sum(
                1 for tag in subject_tags
                if tag.lower() in query_text
            )

            if tag_matches > 0:
                return min(tag_matches * 0.2, 0.5)  # Max 0.5 bonus

            return 0.0

    def _apply_diversity_filtering(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            Apply diversity filtering to avoid redundant chunks

            Uses text similarity to filter out near-duplicate content
            """
            if len(chunks) <= 1:
                return chunks

            diverse_chunks = [chunks[0]]  # Always include the top chunk

            for chunk in chunks[1:]:
                is_diverse = True
                chunk_text = chunk.get("text", "").lower()

                for selected_chunk in diverse_chunks:
                    selected_text = selected_chunk.get("text", "").lower()

                    # Simple text similarity check
                    similarity = self._calculate_text_similarity(chunk_text, selected_text)

                    if similarity > self.diversity_threshold:
                        is_diverse = False
                        break

                if is_diverse:
                    diverse_chunks.append(chunk)

            logger.debug(f"Diversity filtering: {len(chunks)} -> {len(diverse_chunks)} chunks")
            return diverse_chunks

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
            """Calculate simple text similarity using word overlap"""
            if not text1 or not text2:
                return 0.0

            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))

            if not words1 or not words2:
                return 0.0

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            return intersection / union if union > 0 else 0.0

    def _optimize_context_window(
                self,
                chunks: List[Dict[str, Any]],
                token_limit: int,
                include_metadata: bool
        ) -> List[Dict[str, Any]]:
            """
            Optimize chunks to fit within token limit while maximizing relevance

            Uses greedy selection based on relevance score and token efficiency
            """
            if not chunks:
                return []

            optimized_chunks = []
            total_tokens = 0

            for chunk in chunks:
                # Estimate tokens for this chunk
                chunk_tokens = self._estimate_tokens(chunk, include_metadata)

                # Check if adding this chunk would exceed limit
                if total_tokens + chunk_tokens > token_limit:
                    # Try to fit a smaller portion if it's highly relevant
                    if chunk.get("relevance_score", 0) > 0.8 and len(optimized_chunks) < 3:
                        # Truncate chunk to fit remaining space
                        remaining_tokens = token_limit - total_tokens
                        if remaining_tokens > 100:  # Minimum useful chunk size
                            truncated_chunk = self._truncate_chunk(chunk, remaining_tokens, include_metadata)
                            if truncated_chunk:
                                optimized_chunks.append(truncated_chunk)
                                total_tokens += self._estimate_tokens(truncated_chunk, include_metadata)
                    break

                optimized_chunks.append(chunk)
                total_tokens += chunk_tokens

            logger.debug(f"Context optimization: {total_tokens} tokens from {len(optimized_chunks)} chunks")
            return optimized_chunks

    def _estimate_tokens(self, chunk: Dict[str, Any], include_metadata: bool) -> int:
            """
            Estimate token count for a chunk

            Uses rough approximation: 1 token ≈ 4 characters for English text
            """
            text = chunk.get("text", "")
            base_tokens = len(text) // 4

            if include_metadata:
                # Add tokens for metadata
                metadata_text = ""
                if chunk.get("subject_tags"):
                    metadata_text += " ".join(chunk["subject_tags"])
                if chunk.get("file_type"):
                    metadata_text += f" {chunk['file_type']}"

                metadata_tokens = len(metadata_text) // 4
                return base_tokens + metadata_tokens + 10  # Buffer for formatting

            return base_tokens

    def _truncate_chunk(
                self,
                chunk: Dict[str, Any],
                max_tokens: int,
                include_metadata: bool
        ) -> Optional[Dict[str, Any]]:
            """Truncate chunk to fit within token limit"""
            text = chunk.get("text", "")
            if not text:
                return None

            # Reserve tokens for metadata
            metadata_tokens = 20 if include_metadata else 0
            available_tokens = max_tokens - metadata_tokens

            if available_tokens < 50:  # Minimum useful text
                return None

            # Estimate character limit
            char_limit = available_tokens * 4

            if len(text) <= char_limit:
                return chunk

            # Truncate at sentence boundary if possible
            truncated_text = text[:char_limit]
            last_sentence_end = max(
                truncated_text.rfind('.'),
                truncated_text.rfind('!'),
                truncated_text.rfind('?')
            )

            if last_sentence_end > char_limit * 0.7:  # Keep if we retain >70% of text
                truncated_text = truncated_text[:last_sentence_end + 1]

            # Create truncated chunk
            truncated_chunk = chunk.copy()
            truncated_chunk["text"] = truncated_text
            truncated_chunk["truncated"] = True
            truncated_chunk["original_length"] = len(text)

            return truncated_chunk

    def _calculate_context_stats(
                self,
                optimized_chunks: List[Dict[str, Any]],
                all_chunks: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Calculate statistics about the retrieved context"""
            if not optimized_chunks:
                return {
                    "total_chunks": 0,
                    "total_tokens": 0,
                    "avg_relevance_score": 0.0,
                    "subject_coverage": [],
                    "chunks_filtered": len(all_chunks)
                }

            total_tokens = sum(
                self._estimate_tokens(chunk, True)
                for chunk in optimized_chunks
            )

            avg_score = sum(
                chunk.get("relevance_score", 0.0)
                for chunk in optimized_chunks
            ) / len(optimized_chunks)

            # Collect unique subject tags
            all_subjects = set()
            for chunk in optimized_chunks:
                all_subjects.update(chunk.get("subject_tags", []))

            return {
                "total_chunks": len(optimized_chunks),
                "total_tokens": total_tokens,
                "avg_relevance_score": round(avg_score, 3),
                "subject_coverage": list(all_subjects),
                "chunks_filtered": len(all_chunks) - len(optimized_chunks),
                "token_efficiency": round(total_tokens / self.max_context_tokens, 3)
            }

    def _empty_context_response(self) -> Dict[str, Any]:
            """Return empty context response"""
            return {
                "chunks": [],
                "query": "",
                "user_id": "",
                "subject_tags": None,
                "stats": {
                    "total_chunks": 0,
                    "total_tokens": 0,
                    "avg_relevance_score": 0.0,
                    "subject_coverage": [],
                    "chunks_filtered": 0,
                    "token_efficiency": 0.0
                },
                "retrieved_at": datetime.utcnow().isoformat()
            }

    async def get_context_for_conversation(
                self,
                messages: List[Dict[str, str]],
                user_id: str,
                subject_tags: Optional[List[str]] = None,
                max_tokens: Optional[int] = None
        ) -> Dict[str, Any]:
            """
            Retrieve context based on conversation history

            Analyzes recent messages to build a comprehensive query
            """
            try:
                if not messages:
                    return self._empty_context_response()

                # Extract recent user messages for context
                recent_messages = messages[-3:]  # Last 3 messages for context
                user_messages = [
                    msg["content"] for msg in recent_messages
                    if msg.get("role") == "user"
                ]

                if not user_messages:
                    return self._empty_context_response()

                # Combine messages into a comprehensive query
                combined_query = " ".join(user_messages)

                # Retrieve context using the combined query
                return await self.retrieve_context(
                    query=combined_query,
                    user_id=user_id,
                    subject_tags=subject_tags,
                    max_tokens=max_tokens
                )

            except Exception as e:
                logger.error(f"Conversation context retrieval failed: {e}")
                return self._empty_context_response()

    def format_context_for_ai(self, context_data: Dict[str, Any]) -> str:
            """
            Format retrieved context for AI model consumption

            Creates a structured context string optimized for AI understanding
            """
            chunks = context_data.get("chunks", [])
            if not chunks:
                return ""

            formatted_context = "RELEVANT CONTEXT FROM USER'S STUDY MATERIALS:\n\n"

            for i, chunk in enumerate(chunks, 1):
                text = chunk.get("text", "")
                subject_tags = chunk.get("subject_tags", [])
                relevance_score = chunk.get("relevance_score", 0.0)

                # Add chunk header with metadata
                formatted_context += f"[Context {i}]"
                if subject_tags:
                    formatted_context += f" (Topics: {', '.join(subject_tags)})"
                formatted_context += f" (Relevance: {relevance_score:.2f})\n"

                # Add chunk text
                formatted_context += f"{text}\n\n"

            # Add context statistics
            stats = context_data.get("stats", {})
            formatted_context += f"Context Summary: {stats.get('total_chunks', 0)} chunks, "
            formatted_context += f"{stats.get('total_tokens', 0)} tokens, "
            formatted_context += f"avg relevance: {stats.get('avg_relevance_score', 0.0):.2f}\n"

            return formatted_context

    # Global RAG retriever instance
rag_retriever = RAGRetriever()