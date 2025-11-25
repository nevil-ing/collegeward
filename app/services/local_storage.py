import os
import shutil
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import uuid

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.exceptions import StorageError

logger = get_logger(__name__)

class LocalStorageService:
    """service for managing file uploads to local system"""

    def __init__(self):

        storage_path = getattr(settings, 'LOCAL_STORAGE_ROOT', './storage')
        self.storage_root = Path(storage_path).resolve()  # Use absolute path
        self.storage_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local storage initialized at: {self.storage_root.absolute()}")

    async def upload_file(
            self,
            file_content: bytes,
            filename: str,
            user_id: str,
            content_type: str
    ) -> Dict[str, Any]:
        """Upload file to local filesystem"""
        try:
            # Generate unique file path
            file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
            unique_filename = f"{uuid.uuid4()}.{file_extension}" if file_extension else str(uuid.uuid4())

            # Creates user directory structure: storage/users/{user_id}/documents/{filename}
            user_dir = self.storage_root / "users" / user_id / "documents"
            user_dir.mkdir(parents=True, exist_ok=True)

            storage_path = user_dir / unique_filename

            # Write file to disk
            with open(storage_path, 'wb') as f:
                f.write(file_content)

            # Store relative path for database
            relative_path = str(storage_path.relative_to(self.storage_root))

            logger.info(f"Uploaded file {filename} to {relative_path}")

            return {
                'storage_path': relative_path,
                'file_size': len(file_content),
                'content_type': content_type,
                'upload_date': datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Failed to upload file {filename}: {e}")
            raise StorageError(f"File upload failed: {e}")

    async def download_file(self, storage_path: str) -> bytes:
        """Download file from local filesystem"""
        try:
            full_path = self.storage_root / storage_path

            if not full_path.exists():
                raise StorageError(f"File not found: {storage_path}")

            with open(full_path, 'rb') as f:
                return f.read()

        except Exception as e:
            logger.error(f"Failed to download file {storage_path}: {e}")
            raise StorageError(f"File download failed: {e}")

    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from local filesystem"""
        try:
            full_path = self.storage_root / storage_path

            if full_path.exists():
                full_path.unlink()
                logger.info(f"Deleted file: {storage_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {storage_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete file {storage_path}: {e}")
            raise StorageError(f"File deletion failed: {e}")

    async def file_exists(self, storage_path: str) -> bool:
        """Check if file exists"""
        full_path = self.storage_root / storage_path
        return full_path.exists()

    async def generate_signed_url(
            self,
            storage_path: str,
            expiration_hours: int = 1
    ) -> str:
        """Generate a URL for accessing the file

        For local storage, we'll return a simple path that can be served
        by a static file endpoint. In production, you might want to use
        a proper file serving endpoint.
        """
        # Return a relative path that can be accessed via /storage/{path}
        return f"/storage/{storage_path}"

    async def generate_signed_url(
            self,
            storage_path: str,
            expiration_hours: int = 1
    ) -> str:
        """Generate a URL for accessing the file

        For local storage, we'll return a simple path that can be served
        by a static file endpoint. In production, you might want to use
        a proper file serving endpoint.
        """
        # Return a relative path that can be accessed via /storage/{path}
        return f"/storage/{storage_path}"

    def validate_file_type(self, filename: str) -> bool:
        """Validate if file type is allowed"""
        if not filename:
            return False

        file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
        allowed_types = getattr(settings, 'ALLOWED_FILE_TYPES', ['pdf', 'docx', 'png', 'jpg', 'jpeg'])
        return file_extension in allowed_types

    def validate_file_size(self, file_size: int) -> bool:
        """Validate if file size is within limits"""
        max_size = getattr(settings, 'MAX_FILE_SIZE', 10 * 1024 * 1024)  # 10MB default
        return 0 < file_size <= max_size

    def get_content_type(self, filename: str) -> str:
        """Get content type based on file extension"""
        if not filename:
            return "application/octet-stream"

        file_extension = filename.split('.')[-1].lower() if '.' in filename else ''

        content_types = {
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg'
        }

        return content_types.get(file_extension, "application/octet-stream")

    def get_file_size(self, storage_path: str) -> int:
        """Get file size in bytes"""
        try:
            full_path = self.storage_root / storage_path
            if full_path.exists():
                return full_path.stat().st_size
            return 0
        except Exception as e:
            logger.error(f"Failed to get file size for {storage_path}: {e}")
            return 0

local_storage = LocalStorageService()










