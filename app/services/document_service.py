from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import uuid


from app.models.user import User
from app.models.note import Note
from app.services.file_processor import file_processor
from app.services.embedding_service import embedding_service
from app.services.local_storage import local_storage
from app.rag.qdrant_client import qdrant_manager
from app.core.logging import get_logger
from app.utils.exceptions import ProcessingError, StorageError, DatabaseError

logger = get_logger(__name__)


class DocumentService:
    """Service for complete document processing pipeline"""

    async def process_uploaded_file(
            self,
            file_content: bytes,
            filename: str,
            file_type: str,
            user_id: str,
            db: AsyncSession,
            subject_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process uploaded file through complete pipeline"""
        note_id = None

        try:
            # Validate file
            if not local_storage.validate_file_type(filename):
                raise ProcessingError(f"Unsupported file type: {file_type}")

            if not local_storage.validate_file_size(len(file_content)):
                raise ProcessingError("File size exceeds maximum limit")

            # Get User by Firebase UID first
            from app.services.user_service import user_service
            user = await user_service.get_user_by_firebase_uid(db, user_id)
            if not user:
                raise ProcessingError(f"User not found for Firebase UID: {user_id}")

            # Create initial note record
            note = Note(
                user_id=user.id,  # Use user.id (UUID) instead of converting Firebase UID
                filename=filename,
                file_type=file_type,
                file_size=len(file_content),
                storage_path="",  # Will be updated after upload
                processing_status="pending"
            )

            db.add(note)
            await db.commit()
            await db.refresh(note)
            note_id = str(note.id)

            logger.info(f"Created note record {note_id} for file {filename}")

            # Upload file to local storage
            await self._update_processing_status(db, note.id, "uploading")

            content_type = local_storage.get_content_type(filename)
            # Use Firebase UID for storage directory structure (user_id is already Firebase UID)
            upload_result = await local_storage.upload_file(
                file_content, filename, user_id, content_type
            )

            # Update note with storage path
            note.storage_path = upload_result['storage_path']
            await db.commit()

            # Extract text from file
            await self._update_processing_status(db, note.id, "extracting_text")

            extracted_text = await file_processor.extract_text(file_content, file_type)

            if not extracted_text.strip():
                raise ProcessingError("No text could be extracted from the file")

            # Detect subject tags if not provided
            if not subject_tags:
                subject_tags = file_processor.detect_subject_tags(extracted_text)

            # Update note with extracted text and tags
            note.extracted_text = extracted_text
            note.subject_tags = subject_tags
            await db.commit()

            # Chunk the text
            await self._update_processing_status(db, note.id, "chunking_text")

            chunks = file_processor.chunk_text(
                extracted_text,
                chunk_size=1000,
                overlap=200,
                preserve_sentences=True
            )

            if not chunks:
                raise ProcessingError("Failed to create text chunks")

            # Generate embeddings
            await self._update_processing_status(db, note.id, "gen_embeddings")

            chunks_with_embeddings = await embedding_service.process_document_chunks(chunks)

            # Add metadata to chunks
            for chunk in chunks_with_embeddings:
                chunk.update({
                    "file_type": file_type,
                    "subject_tags": subject_tags,
                    "created_at": datetime.utcnow().isoformat()
                })

            # Store embeddings in Qdrant
            await self._update_processing_status(db, note.id, "storing_embeddings")

            try:
                qdrant_success = await qdrant_manager.add_document_chunks(
                    user_id=user_id,
                    note_id=note_id,
                    chunks=chunks_with_embeddings
                )
                if not qdrant_success:
                    logger.warning(f"Qdrant storage failed for note {note_id}, but continuing with file processing")
            except Exception as e:
                logger.warning(
                    f"Failed to store embeddings in Qdrant for note {note_id}: {e}. Continuing with file processing.")

            # Mark processing as completed
            await self._update_processing_status(db, note.id, "completed")
            note.processed_date = datetime.utcnow()
            await db.commit()

            logger.info(f"Successfully processed document {filename} with {len(chunks)} chunks")

            return {
                "note_id": note_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": len(file_content),
                "processing_status": "completed",
                "extracted_text_length": len(extracted_text),
                "chunks_created": len(chunks),
                "subject_tags": subject_tags,
                "processed_date": note.processed_date
            }

        except Exception as e:
            logger.error(f"Document processing failed for {filename}: {e}")

            # Update status to failed if note was created
            if note_id:
                try:
                    await self._update_processing_status(db, uuid.UUID(note_id), "failed")
                except Exception as update_error:
                    logger.error(f"Failed to update status to failed: {update_error}")

            # Clean up on failure
            await self._cleanup_failed_processing(user_id, note_id, db)

            raise ProcessingError(f"Document processing failed: {str(e)}")

    async def _update_processing_status(
                self,
                db: AsyncSession,
                note_id: uuid.UUID,
                status: str
        ):
            """Update note processing status"""
            try:
                stmt = (
                    update(Note)
                    .where(Note.id == note_id)
                    .values(processing_status=status)
                )
                await db.execute(stmt)
                await db.commit()

                logger.debug(f"Updated note {note_id} status to {status}")

            except Exception as e:
                logger.error(f"Failed to update processing status: {e}")
                raise DatabaseError(f"Status update failed: {e}")

    async def _cleanup_failed_processing(
                self,
                user_id: str,
                note_id: Optional[str],
                db: AsyncSession
        ):
            """Clean up resources after failed processing"""
            try:
                if note_id:
                    # Remove embeddings from Qdrant
                    try:
                        await qdrant_manager.delete_note_chunks(user_id, note_id)
                    except Exception as e:
                        logger.error(f"Failed to clean up Qdrant chunks: {e}")

                    # Get note to find storage path
                    try:
                        result = await db.execute(
                            select(Note).where(Note.id == uuid.UUID(note_id))
                        )
                        note = result.scalar_one_or_none()

                        if note and note.storage_path:
                            # Delete file from local storage
                            try:
                                await local_storage.delete_file(note.storage_path)
                            except Exception as e:
                                logger.error(f"Failed to clean up file: {e}")

                            # Delete note record
                            await db.delete(note)
                            await db.commit()

                    except Exception as e:
                        logger.error(f"Failed to clean up note record: {e}")

            except Exception as e:
                logger.error(f"Cleanup failed: {e}")

    async def get_user_notes(
                self,
                user_id: str,  # This is Firebase UID, not UUID
                db: AsyncSession,
                skip: int = 0,
                limit: int = 50
        ) -> List[Dict[str, Any]]:
            """Get all notes for a user"""
            try:
                # Get User by Firebase UID first
                from app.services.user_service import user_service
                user = await user_service.get_user_by_firebase_uid(db, user_id)
                if not user:
                    logger.warning(f"User not found for Firebase UID: {user_id}")
                    return []

                stmt = (
                    select(Note)
                    .where(Note.user_id == user.id)  # Use user.id (UUID) instead
                    .order_by(Note.created_at.desc())
                    .offset(skip)
                    .limit(limit)
                )

                result = await db.execute(stmt)
                notes = result.scalars().all()

                return [
                    {
                        "id": str(note.id),
                        "user_id": str(note.user_id),
                        "filename": note.filename,
                        "file_type": note.file_type,
                        "file_size": note.file_size,
                        "storage_path": note.storage_path or "",
                        "processing_status": note.processing_status,
                        "extracted_text": note.extracted_text,
                        "subject_tags": note.subject_tags or [],
                        "processed_date": note.processed_date.isoformat() if note.processed_date else None,
                        "upload_date": note.upload_date.isoformat() if note.upload_date else None,
                        "created_at": note.created_at.isoformat() if note.created_at else None,
                        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
                    }
                    for note in notes
                ]

            except Exception as e:
                logger.error(f"Failed to get user notes: {e}")
                raise DatabaseError(f"Failed to retrieve notes: {e}")

    async def delete_note(
                self,
                note_id: str,
                user_id: str,
                db: AsyncSession
        ) -> bool:
            """Delete a note and all associated data"""
            try:
                # Get User by Firebase UID first (user_id is Firebase UID, not UUID)
                from app.services.user_service import user_service
                user = await user_service.get_user_by_firebase_uid(db, user_id)
                if not user:
                    logger.warning(f"User not found for Firebase UID: {user_id}")
                    return False

                # Get note using user.id (UUID) instead of user_id (Firebase UID)
                result = await db.execute(
                    select(Note).where(
                        Note.id == uuid.UUID(note_id),
                        Note.user_id == user.id  # Use user.id (UUID) instead of converting Firebase UID
                    )
                )
                note = result.scalar_one_or_none()

                if not note:
                    return False

                # Delete embeddings from Qdrant (using Firebase UID for Qdrant)
                try:
                    await qdrant_manager.delete_note_chunks(user_id, note_id)
                except Exception as e:
                    logger.error(f"Failed to delete Qdrant chunks: {e}")

                # Delete file from local storage
                if note.storage_path:
                    try:
                        await local_storage.delete_file(note.storage_path)
                    except Exception as e:
                        logger.error(f"Failed to delete file: {e}")

                # Delete note record
                await db.delete(note)
                await db.commit()

                logger.info(f"Deleted note {note_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete note {note_id}: {e}")
                raise DatabaseError(f"Note deletion failed: {e}")
    async def get_note_details(
                self,
                note_id: str,
                user_id: str,
                db: AsyncSession
        ) -> Optional[Dict[str, Any]]:
            """Get detailed information about a specific note"""
            try:
                # Get User by Firebase UID first
                from app.services.user_service import user_service
                user = await user_service.get_user_by_firebase_uid(db, user_id)
                if not user:
                    return None

                result = await db.execute(
                    select(Note).where(
                        Note.id == uuid.UUID(note_id),
                        Note.user_id == user.id  # Use user.id (UUID) instead
                    )
                )
                note = result.scalar_one_or_none()

                if not note:
                    return None

                return {
                    "id": str(note.id),
                    "filename": note.filename,
                    "file_type": note.file_type,
                    "file_size": note.file_size,
                    "storage_path": note.storage_path,
                    "processing_status": note.processing_status,
                    "extracted_text": note.extracted_text,
                    "subject_tags": note.subject_tags or [],
                    "processed_date": note.processed_date,
                    "upload_date": note.upload_date,
                    "created_at": note.created_at,
                    "updated_at": note.updated_at
                }

            except Exception as e:
                logger.error(f"Failed to get note details: {e}")
                raise DatabaseError(f"Failed to retrieve note details: {e}")

    async def store_file_for_later_processing(
            self,
            file_content: bytes,
            filename: str,
            file_type: str,
            user_id: str,
            db: AsyncSession,
            subject_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Store file for later processing when main processing is unavailable"""
        try:
            # Validate file
            if not local_storage.validate_file_type(filename):
                raise ProcessingError(f"Unsupported file type: {file_type}")

            if not local_storage.validate_file_size(len(file_content)):
                raise ProcessingError("File size exceeds maximum limit")

            # Get User by Firebase UID first
            from app.services.user_service import user_service
            user = await user_service.get_user_by_firebase_uid(db, user_id)
            if not user:
                raise StorageError(f"User not found for Firebase UID: {user_id}")

            # Create note record with pending status
            note = Note(
                user_id=user.id,  # Use user.id (UUID) instead
                filename=filename,
                file_type=file_type,
                file_size=len(file_content),
                storage_path="",
                processing_status="pending_retry",  # Special status for degraded service
                subject_tags=subject_tags
            )

            db.add(note)
            await db.commit()
            await db.refresh(note)

            # Upload file to local storage only
            content_type = local_storage.get_content_type(filename)
            # Use Firebase UID for storage directory structure (user_id is already Firebase UID)
            upload_result = await local_storage.upload_file(
                file_content, filename, user_id, content_type
            )

            # Update note with storage path
            note.storage_path = upload_result['storage_path']
            await db.commit()

            logger.info(f"Stored file {filename} for later processing (degraded service)")

            return {
                "note_id": str(note.id),
                "filename": filename,
                "file_type": file_type,
                "file_size": len(file_content),
                "processing_status": "pending_retry",
                "chunks_created": 0,
                "subject_tags": subject_tags or [],
                "message": "File stored successfully. Processing will be completed when service is restored."
            }

        except Exception as e:
            logger.error(f"Failed to store file for later processing: {e}")
            raise StorageError(f"File storage failed: {str(e)}")


document_service = DocumentService()