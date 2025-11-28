from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional

from app.core.security import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.core.logging import get_logger
from app.utils.exceptions import ProcessingError, StorageError, DatabaseError, ServiceDegradationError
from app.services.document_service import document_service
from app.schemas.note_schema import NoteResponse, NoteProcessingStatus

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=List[NoteResponse])
async def get_user_notes(
        skip: int = Query(0, ge=0, description="Number of notes to skip"),
        limit: int = Query(50, ge=1, le=100, description="Number of notes to return"),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get all notes for the current user"""
    try:
        user_id = current_user.get("uid")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user authentication")

        notes = await document_service.get_user_notes(
            user_id=user_id,
            db=db,
            skip=skip,
            limit=limit
        )

        return notes

    except DatabaseError as e:
        logger.error(f"Database error getting user notes: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notes")
    except Exception as e:
        logger.error(f"Unexpected error getting user notes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/upload")
async def upload_file(
        request: Request,
        file: UploadFile = File(..., description="File to upload (PDF, DOCX, or image)"),
        subject_tags: Optional[str] = Form(None, description="Comma-separated subject tags"),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Upload and process a study material file with comprehensive validation"""
    from app.core.validation import InputValidator, InputSanitizer
    from app.core.exceptions import InputValidationError, SecurityValidationError

    try:
        # Validate user authentication
        user_id = current_user.get("uid")
        if not user_id:
            raise InputValidationError("Invalid user authentication", "user_id")

        # Validate request size
        content_length = int(request.headers.get("content-length", 0))
        if content_length > settings.MAX_FILE_SIZE:
            raise InputValidationError(
                f"Request size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes",
                "content_length",
                content_length
            )

        # Validate file presence
        if not file.filename:
            raise InputValidationError("No file provided", "filename")

        # Sanitize filename
        sanitized_filename = InputSanitizer.sanitize_filename(file.filename)

        # Read and validate file content
        file_content = await file.read()
        if not file_content:
            raise InputValidationError("File is empty", "file_content")

        # Validate file type and content
        file_extension = InputValidator.validate_file_type(sanitized_filename, file_content)

        # Sanitize and validate subject tags
        parsed_subject_tags = None
        if subject_tags:
            try:
                parsed_subject_tags = InputSanitizer.sanitize_subject_tags(subject_tags)
            except SecurityValidationError:
                raise
            except Exception as e:
                raise InputValidationError(f"Invalid subject tags: {str(e)}", "subject_tags")

        # Process the file with circuit breaker protection
        from app.core.circuit_breaker import service_registry, CircuitBreakerConfig

        # Configure circuit breaker for file processing
        # File processing can take 2-5 minutes for large files (extraction, chunking, embeddings)
        processing_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            timeout=300  # 5 minutes for file processing (extraction, chunking, embeddings)
        )

        async def process_file():
            return await document_service.process_uploaded_file(
                file_content=file_content,
                filename=sanitized_filename,
                file_type=file_extension,
                user_id=user_id,
                db=db,
                subject_tags=parsed_subject_tags
            )

        # Fallback function for degraded service
        async def process_file_fallback():
            # Store file without processing for later
            return await document_service.store_file_for_later_processing(
                file_content=file_content,
                filename=sanitized_filename,
                file_type=file_extension,
                user_id=user_id,
                db=db,
                subject_tags=parsed_subject_tags
            )

        try:
            result = await service_registry.call_service(
                "file_processing",
                process_file,
                config=processing_config,
                fallback=process_file_fallback
            )
        except ServiceDegradationError as e:
            # Service is degraded but functional
            logger.warning(f"File processing degraded: {e}")
            # Continue with degraded result
            result = e.details.get("result", {})

        return {
            "message": "File uploaded and processed successfully",
            "note_id": result["note_id"],
            "filename": result["filename"],
            "processing_status": result["processing_status"],
            "chunks_created": result.get("chunks_created", 0),
            "subject_tags": result["subject_tags"]
        }

    except (InputValidationError, SecurityValidationError) as e:
        logger.warning(f"Validation error uploading file: {e}")
        raise e
    except ProcessingError as e:
        logger.error(f"Processing error uploading file: {e}")
        raise e
    except StorageError as e:
        logger.error(f"Storage error uploading file: {e}")
        raise e
    except DatabaseError as e:
        logger.error(f"Database error uploading file: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error uploading file: {e}")
        raise ProcessingError(f"File upload failed: {str(e)}")


@router.get("/{note_id}", response_model=NoteResponse)
async def get_note_details(
        note_id: str,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific note"""
    try:
        user_id = current_user.get("uid")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user authentication")

        note = await document_service.get_note_details(
            note_id=note_id,
            user_id=user_id,
            db=db
        )

        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        return note

    except DatabaseError as e:
        logger.error(f"Database error getting note details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve note")
    except Exception as e:
        logger.error(f"Unexpected error getting note details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{note_id}/status", response_model=NoteProcessingStatus)
async def get_note_processing_status(
        note_id: str,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get processing status of a specific note"""
    try:
        user_id = current_user.get("uid")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user authentication")

        note = await document_service.get_note_details(
            note_id=note_id,
            user_id=user_id,
            db=db
        )

        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        return {
            "id": note["id"],
            "processing_status": note["processing_status"],
            "extracted_text": note["extracted_text"],
            "processed_date": note["processed_date"]
        }

    except DatabaseError as e:
        logger.error(f"Database error getting note status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve note status")
    except Exception as e:
        logger.error(f"Unexpected error getting note status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{note_id}")
async def delete_note(
        note_id: str,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Delete a specific note and all associated data"""
    try:
        user_id = current_user.get("uid")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user authentication")

        deleted = await document_service.delete_note(
            note_id=note_id,
            user_id=user_id,
            db=db
        )

        if not deleted:
            raise HTTPException(status_code=404, detail="Note not found")

        return {"message": "Note deleted successfully", "note_id": note_id}

    except DatabaseError as e:
        logger.error(f"Database error deleting note: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete note")
    except Exception as e:
        logger.error(f"Unexpected error deleting note: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")