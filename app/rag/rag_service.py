from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.rag.retriever import *
from app.models.user import  User
from app.models.note import Note
from app.core.logging import get_logger
from app.utils.exceptions import ProcessingError, DatabaseError


logger = get_logger(__name__)

class RAGService:
    def __init__(self):
        self.default_context_tokens = 3500
        self.max_context_tokens = 6000
        self.min_context_tokens = 1000

    async def get_context_for_query(
            self,
            query: str,
            user_id: str,
            db: AsyncSession,
            subject_filters: Optional[List[str]] = None,
            max_tokens: Optional[int] = None,
            mode: str = "balanced"
    ) -> Dict[str, Any]:
        try:
            # Validate user access
            await self._validate_user_access(user_id, db)

            # Determine token limit based on mode
            token_limit = self._get_token_limit(max_tokens, mode)

            # Get user's available subject tags for filtering
            available_subjects = await self._get_user_subjects(user_id, db)

            # Filter subjects if requested
            filtered_subjects = self._filter_subjects(subject_filters, available_subjects)

            # Retrieve context using RAG retriever
            context_data = await rag_retriever.retrieve_context(
                query=query,
                user_id=user_id,
                subject_tags=filtered_subjects,
                max_tokens=token_limit,
                include_metadata=True
            )

            # Enhance context with user-specific metadata
            enhanced_context = await self._enhance_context_metadata(
                context_data, user_id, db
            )

            # Log retrieval for analytics
            await self._log_retrieval_analytics(
                user_id, query, enhanced_context, db
            )

            logger.info(
                f"Retrieved context for user {user_id}: "
                f"{enhanced_context['stats']['total_chunks']} chunks, "
                f"{enhanced_context['stats']['total_tokens']} tokens"
            )

            return enhanced_context

        except Exception as e:
            logger.error(f"Context retrieval failed for user {user_id}: {e}")
            raise ProcessingError(f"Failed to retrieve context: {str(e)}")

    async def get_context_for_conversation(
                self,
                messages: List[Dict[str, str]],
                user_id: str,
                db: AsyncSession,
                subject_filters: Optional[List[str]] = None,
                max_tokens: Optional[int] = None
        ) -> Dict[str, Any]:
            """
            Get context based on conversation history

            Analyzes conversation flow to provide relevant context
            """
            try:
                # Validate user access
                await self._validate_user_access(user_id, db)

                # Use default token limit for conversations
                token_limit = max_tokens or self.default_context_tokens

                # Get context using conversation-aware retrieval
                context_data = await rag_retriever.get_context_for_conversation(
                    messages=messages,
                    user_id=user_id,
                    subject_tags=subject_filters,
                    max_tokens=token_limit
                )

                # Enhance with metadata
                enhanced_context = await self._enhance_context_metadata(
                    context_data, user_id, db
                )

                return enhanced_context

            except Exception as e:
                logger.error(f"Conversation context retrieval failed: {e}")
                raise ProcessingError(f"Failed to retrieve conversation context: {str(e)}")

    async def get_formatted_context(
            self,
            query: str,
            user_id: str,
            db: AsyncSession,
            format_type: str = "ai_prompt",
            **kwargs
       ) -> str:
        """
        Get context formatted for specific use cases

        Args:
            query: User query
            user_id: User ID
            db: Database session
            format_type: Format type ('ai_prompt', 'summary', 'citations')
            **kwargs: Additional formatting options

        Returns:
            Formatted context string
        """
        try:
            # Get raw context data
            context_data = await self.get_context_for_query(
                query=query,
                user_id=user_id,
                db=db,
                **kwargs
            )

            # Format based on type
            if format_type == "ai_prompt":
                return rag_retriever.format_context_for_ai(context_data)
            elif format_type == "summary":
                return self._format_context_summary(context_data)
            elif format_type == "citations":
                return self._format_context_citations(context_data)
            else:
                raise ValueError(f"Unknown format type: {format_type}")

        except Exception as e:
            logger.error(f"Context formatting failed: {e}")
            raise ProcessingError(f"Failed to format context: {str(e)}")

    async def check_context_availability(
            self,
            user_id: str,
            db: AsyncSession,
            subject_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check what context is available for a user

        Args:
            user_id: Firebase UID
            db: Database session
            subject_tags: Optional subject filters

        Returns statistics about available documents and subjects
        """
        try:
            # Get User from Firebase UID first
            from app.services.user_service import user_service
            user = await user_service.get_user_by_firebase_uid(db, user_id)
            if not user:
                return {
                    "has_context": False,
                    "total_notes": 0,
                    "available_subjects": [],
                    "total_chunks_estimate": 0,
                    "message": "User not found"
                }

            # Get user's notes using user.id
            stmt = select(Note).where(
                Note.user_id == user.id,
                Note.processing_status == "completed"
            )
            result = await db.execute(stmt)
            notes = result.scalars().all()

            if not notes:
                return {
                    "has_context": False,
                    "total_notes": 0,
                    "available_subjects": [],
                    "total_chunks_estimate": 0,
                    "message": "No processed documents available for context retrieval"
                }

            # Collect subject statistics
            all_subjects = set()
            total_text_length = 0

            for note in notes:
                if note.subject_tags:
                    all_subjects.update(note.subject_tags)
                if note.extracted_text:
                    total_text_length += len(note.extracted_text)

            # Estimate chunk count (rough approximation)
            estimated_chunks = total_text_length // 800  # Average chunk size

            # Filter subjects if requested
            available_subjects = list(all_subjects)
            if subject_tags:
                available_subjects = [
                    subject for subject in available_subjects
                    if subject in subject_tags
                ]

            return {
                "has_context": True,
                "total_notes": len(notes),
                "available_subjects": available_subjects,
                "total_chunks_estimate": estimated_chunks,
                "total_text_length": total_text_length,
                "message": f"Context available from {len(notes)} documents"
            }

        except Exception as e:
            logger.error(f"Context availability check failed: {e}")
            raise DatabaseError(f"Failed to check context availability: {str(e)}")


    def _get_token_limit(self, max_tokens: Optional[int], mode: str) -> int:
            """Determine appropriate token limit based on mode and constraints"""
            if max_tokens:
                return min(max_tokens, self.max_context_tokens)

            mode_limits = {
                "fast": self.min_context_tokens,
                "balanced": self.default_context_tokens,
                "comprehensive": self.max_context_tokens
            }

            return mode_limits.get(mode, self.default_context_tokens)

    async def _get_user_subjects(self, user_id: str, db: AsyncSession) -> List[str]:
            """Get all available subject tags for a user

            Args:
                user_id: Firebase UID (string), not UUID
                db: Database session
            """
            try:
                # Get User from Firebase UID first
                from app.services.user_service import user_service
                user = await user_service.get_user_by_firebase_uid(db, user_id)
                if not user:
                    return []

                # Query notes using user.id (UUID)
                stmt = select(Note.subject_tags).where(
                    Note.user_id == user.id,
                    Note.processing_status == "completed",
                    Note.subject_tags.isnot(None)
                )
                result = await db.execute(stmt)

                all_subjects = set()
                for row in result:
                    if row.subject_tags:
                        all_subjects.update(row.subject_tags)

                return list(all_subjects)

            except Exception as e:
                logger.error(f"Failed to get user subjects: {e}")
                return []

    def _filter_subjects(
                self,
                requested_subjects: Optional[List[str]],
                available_subjects: List[str]
        ) -> Optional[List[str]]:
            """Filter requested subjects against available subjects"""
            if not requested_subjects:
                return None

            # Return intersection of requested and available subjects
            filtered = [
                subject for subject in requested_subjects
                if subject in available_subjects
            ]

            return filtered if filtered else None

    async def _enhance_context_metadata(
                self,
                context_data: Dict[str, Any],
                user_id: str,
                db: AsyncSession
        ) -> Dict[str, Any]:
            """Enhance context with additional metadata from database"""
            try:
                chunks = context_data.get("chunks", [])
                if not chunks:
                    return context_data

                # Get note IDs from chunks
                note_ids = list(set(chunk.get("note_id") for chunk in chunks if chunk.get("note_id")))

                if not note_ids:
                    return context_data

                # Fetch note metadata
                stmt = select(Note).where(Note.id.in_(note_ids))
                result = await db.execute(stmt)
                notes_by_id = {str(note.id): note for note in result.scalars().all()}

                # Enhance chunks with note metadata
                for chunk in chunks:
                    note_id = chunk.get("note_id")
                    if note_id and note_id in notes_by_id:
                        note = notes_by_id[note_id]
                        chunk["note_metadata"] = {
                            "filename": note.filename,
                            "file_type": note.file_type,
                            "upload_date": note.upload_date.isoformat() if note.upload_date else None,
                            "processed_date": note.processed_date.isoformat() if note.processed_date else None
                        }

                # Add source summary to stats
                context_data["stats"]["source_files"] = [
                    {
                        "note_id": note_id,
                        "filename": note.filename,
                        "file_type": note.file_type
                    }
                    for note_id, note in notes_by_id.items()
                ]

                return context_data

            except Exception as e:
                logger.error(f"Context metadata enhancement failed: {e}")
                # Return original context if enhancement fails
                return context_data

    async def _log_retrieval_analytics(
                self,
                user_id: str,
                query: str,
                context_data: Dict[str, Any],
                db: AsyncSession
        ):
            """Log retrieval analytics for performance monitoring"""
            try:
                # This could be expanded to store analytics in database

                stats = context_data.get("stats", {})

                logger.info(
                    f"RAG Analytics - User: {user_id}, "
                    f"Query length: {len(query)}, "
                    f"Chunks retrieved: {stats.get('total_chunks', 0)}, "
                    f"Tokens used: {stats.get('total_tokens', 0)}, "
                    f"Avg relevance: {stats.get('avg_relevance_score', 0.0)}, "
                    f"Token efficiency: {stats.get('token_efficiency', 0.0)}"
                )

            except Exception as e:
                logger.error(f"Analytics logging failed: {e}")
                # Don't raise exception for analytics failures

    def _format_context_summary(self, context_data: Dict[str, Any]) -> str:
            """Format context as a summary"""
            chunks = context_data.get("chunks", [])
            stats = context_data.get("stats", {})

            if not chunks:
                return "No relevant context found in your study materials."

            summary = f"Found {len(chunks)} relevant sections from your study materials:\n\n"

            for i, chunk in enumerate(chunks[:3], 1):  # Show top 3 chunks
                text = chunk.get("text", "")[:200]  # First 200 chars
                subjects = chunk.get("subject_tags", [])
                score = chunk.get("relevance_score", 0.0)

                summary += f"{i}. "
                if subjects:
                    summary += f"[{', '.join(subjects)}] "
                summary += f"{text}... (Relevance: {score:.2f})\n\n"

            if len(chunks) > 3:
                summary += f"... and {len(chunks) - 3} more sections.\n\n"

            summary += f"Total context: {stats.get('total_tokens', 0)} tokens from "
            summary += f"{stats.get('total_chunks', 0)} sections."

            return summary

    def _format_context_citations(self, context_data: Dict[str, Any]) -> str:
            """Format context with citations"""
            chunks = context_data.get("chunks", [])

            if not chunks:
                return "No sources found."

            citations = "Sources from your study materials:\n\n"

            # Group by source file
            sources_by_file = {}
            for chunk in chunks:
                note_metadata = chunk.get("note_metadata", {})
                filename = note_metadata.get("filename", "Unknown file")

                if filename not in sources_by_file:
                    sources_by_file[filename] = []
                sources_by_file[filename].append(chunk)

            for i, (filename, file_chunks) in enumerate(sources_by_file.items(), 1):
                citations += f"{i}. {filename}\n"
                citations += f"   - {len(file_chunks)} relevant sections found\n"

                # Show the highest relevance score
                max_score = max(chunk.get("relevance_score", 0.0) for chunk in file_chunks)
                citations += f"   - Max relevance: {max_score:.2f}\n\n"

            return citations


rag_service = RAGService()