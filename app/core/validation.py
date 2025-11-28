import re
import html
import bleach
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from email_validator import validate_email, EmailNotValidError
from pydantic import BaseModel, ValidationError as PydanticValidationError

from app.utils.exceptions import InputValidationError, SecurityValidationError
from app.core.config import settings


class InputSanitizer:
    """Input sanitization utilities"""

    # Allowed HTML tags for rich text content
    ALLOWED_HTML_TAGS = [
        'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
    ]

    # Allowed HTML attributes
    ALLOWED_HTML_ATTRIBUTES = {
        '*': ['class'],
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'title']
    }

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\bUNION\s+SELECT\b)",
        r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\b)"
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
        r"<embed[^>]*>.*?</embed>"
    ]

    @classmethod
    def sanitize_text(cls, text: str, allow_html: bool = False) -> str:
        """Sanitize text input"""
        if not isinstance(text, str):
            return str(text)

        # Check for security violations
        cls._check_security_violations(text)

        if allow_html:
            # Clean HTML but preserve allowed tags
            return bleach.clean(
                text,
                tags=cls.ALLOWED_HTML_TAGS,
                attributes=cls.ALLOWED_HTML_ATTRIBUTES,
                strip=True
            )
        else:
            # Escape HTML entities
            return html.escape(text.strip())

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename for safe storage"""
        if not filename:
            raise InputValidationError("Filename cannot be empty", "filename")

        # Remove path traversal attempts
        filename = filename.replace("../", "").replace("..\\", "")

        # Remove or replace dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')

        return filename.strip()

    @classmethod
    def sanitize_subject_tags(cls, tags: Union[str, List[str]]) -> List[str]:
        """Sanitize and validate subject tags"""
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(',') if tag.strip()]

        if not isinstance(tags, list):
            raise InputValidationError("Tags must be a string or list", "subject_tags")

        sanitized_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                continue

            # Sanitize tag
            clean_tag = cls.sanitize_text(tag).lower()

            # Validate tag format
            if not re.match(r'^[a-zA-Z0-9\s\-_]+$', clean_tag):
                continue

            # Limit tag length
            if len(clean_tag) > 50:
                clean_tag = clean_tag[:50]

            if clean_tag and clean_tag not in sanitized_tags:
                sanitized_tags.append(clean_tag)

        # Limit number of tags
        return sanitized_tags[:10]

    @classmethod
    def _check_security_violations(cls, text: str) -> None:
        """Check for potential security violations"""
        text_lower = text.lower()

        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                raise SecurityValidationError(
                    "Potential SQL injection detected",
                    "sql_injection"
                )

        # Check for XSS patterns
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                raise SecurityValidationError(
                    "Potential XSS attack detected",
                    "xss_attempt"
                )


class InputValidator:
    """Input validation utilities"""

    @staticmethod
    def validate_email_address(email: str) -> str:
        """Validate and normalize email address"""
        try:
            validated_email = validate_email(email)
            return validated_email.email
        except EmailNotValidError as e:
            raise InputValidationError(f"Invalid email address: {str(e)}", "email")

    @staticmethod
    def validate_file_type(filename: str, file_content: bytes) -> str:
        """Validate file type based on extension and content"""
        if not filename:
            raise InputValidationError("Filename is required", "filename")

        # Get file extension
        file_extension = filename.split('.')[-1].lower() if '.' in filename else ''

        if not file_extension:
            raise InputValidationError("File must have an extension", "filename")

        if file_extension not in settings.ALLOWED_FILE_TYPES:
            raise InputValidationError(
                f"File type '{file_extension}' not allowed",
                "file_type",
                file_extension,
                settings.ALLOWED_FILE_TYPES
            )

        # Validate file size
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise InputValidationError(
                f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes",
                "file_size",
                len(file_content)
            )

        # Basic file signature validation
        if not InputValidator._validate_file_signature(file_content, file_extension):
            raise InputValidationError(
                "File content doesn't match the file extension",
                "file_signature"
            )

        return file_extension

    @staticmethod
    def _validate_file_signature(content: bytes, extension: str) -> bool:
        """Validate file signature matches extension"""
        if not content:
            return False

        # File signatures (magic numbers)
        signatures = {
            'pdf': [b'%PDF'],
            'png': [b'\x89PNG\r\n\x1a\n'],
            'jpg': [b'\xff\xd8\xff'],
            'jpeg': [b'\xff\xd8\xff'],
            'docx': [b'PK\x03\x04']  # ZIP-based format
        }

        if extension not in signatures:
            return True  # Allow unknown extensions to pass

        return any(content.startswith(sig) for sig in signatures[extension])

    @staticmethod
    def validate_pagination_params(skip: int, limit: int) -> tuple[int, int]:
        """Validate pagination parameters"""
        if skip < 0:
            raise InputValidationError("Skip parameter must be non-negative", "skip", skip)

        if limit < 1:
            raise InputValidationError("Limit parameter must be positive", "limit", limit)

        if limit > 100:
            raise InputValidationError("Limit parameter cannot exceed 100", "limit", limit)

        return skip, limit

    @staticmethod
    def validate_uuid(uuid_str: str, field_name: str = "id") -> str:
        """Validate UUID format"""
        import uuid
        try:
            uuid.UUID(uuid_str)
            return uuid_str
        except ValueError:
            raise InputValidationError(f"Invalid UUID format", field_name, uuid_str)

    @staticmethod
    def validate_text_length(text: str, field_name: str, min_length: int = 0, max_length: int = 1000) -> str:
        """Validate text length constraints"""
        if len(text) < min_length:
            raise InputValidationError(
                f"{field_name} must be at least {min_length} characters",
                field_name,
                len(text)
            )

        if len(text) > max_length:
            raise InputValidationError(
                f"{field_name} cannot exceed {max_length} characters",
                field_name,
                len(text)
            )

        return text

    @staticmethod
    def validate_url(url: str, field_name: str = "url") -> str:
        """Validate URL format"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL format")

            # Only allow HTTP/HTTPS
            if result.scheme not in ['http', 'https']:
                raise ValueError("Only HTTP/HTTPS URLs are allowed")

            return url
        except Exception:
            raise InputValidationError("Invalid URL format", field_name, url)


class RequestValidator:
    """Request-level validation utilities"""

    @staticmethod
    def validate_content_type(content_type: str, allowed_types: List[str]) -> None:
        """Validate request content type"""
        if content_type not in allowed_types:
            raise InputValidationError(
                "Invalid content type",
                "content_type",
                content_type,
                allowed_types
            )

    @staticmethod
    def validate_request_size(content_length: int, max_size: int) -> None:
        """Validate request size"""
        if content_length > max_size:
            raise InputValidationError(
                f"Request size exceeds maximum allowed size of {max_size} bytes",
                "content_length",
                content_length
            )

    @staticmethod
    def validate_pydantic_model(model_class: BaseModel, data: Dict[str, Any]) -> BaseModel:
        """Validate data against Pydantic model with enhanced error handling"""
        try:
            return model_class(**data)
        except PydanticValidationError as e:
            # Convert Pydantic validation errors to our custom format
            errors = []
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                errors.append({
                    "field": field,
                    "message": error['msg'],
                    "type": error['type']
                })

            raise InputValidationError(
                "Validation failed",
                "request_body",
                details={"validation_errors": errors}
            )


def sanitize_and_validate_input(
        data: Dict[str, Any],
        text_fields: Optional[List[str]] = None,
        html_fields: Optional[List[str]] = None,
        required_fields: Optional[List[str]] = None,
        max_lengths: Optional[Dict[str, int]] = None,
        allowed_values: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Comprehensive input sanitization and validation

    Args:
        data: Input data dictionary
        text_fields: Fields to sanitize as plain text
        html_fields: Fields to sanitize as HTML (allowing safe tags)
        required_fields: Fields that are required
        max_lengths: Maximum lengths for specific fields
        allowed_values: Allowed values for specific fields

    Returns:
        Sanitized and validated data dictionary
    """
    sanitized_data = {}

    # Check required fields
    if required_fields:
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                raise InputValidationError(f"Field '{field}' is required", field)

    # Sanitize text fields
    if text_fields:
        for field in text_fields:
            if field in data and data[field] is not None:
                sanitized_value = InputSanitizer.sanitize_text(str(data[field]))

                # Check length constraints
                if max_lengths and field in max_lengths:
                    InputValidator.validate_text_length(
                        sanitized_value, field, 0, max_lengths[field]
                    )

                sanitized_data[field] = sanitized_value

    # Sanitize HTML fields
    if html_fields:
        for field in html_fields:
            if field in data and data[field] is not None:
                sanitized_value = InputSanitizer.sanitize_text(str(data[field]), allow_html=True)

                # Check length constraints
                if max_lengths and field in max_lengths:
                    InputValidator.validate_text_length(
                        sanitized_value, field, 0, max_lengths[field]
                    )

                sanitized_data[field] = sanitized_value

    # Validate allowed values
    if allowed_values:
        for field, allowed_list in allowed_values.items():
            if field in data and data[field] is not None:
                if str(data[field]) not in allowed_list:
                    raise InputValidationError(
                        f"Invalid value for field '{field}'",
                        field,
                        data[field],
                        allowed_list
                    )

    # Copy other fields as-is
    for key, value in data.items():
        if key not in sanitized_data:
            sanitized_data[key] = value

    return sanitized_data


class ComprehensiveValidator:
    """Comprehensive validation system for API requests"""

    @staticmethod
    def validate_file_upload(
            filename: str,
            file_content: bytes,
            max_size: Optional[int] = None,
            allowed_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Comprehensive file upload validation"""

        max_size = max_size or settings.MAX_FILE_SIZE
        allowed_types = allowed_types or settings.ALLOWED_FILE_TYPES

        # Validate filename
        if not filename or len(filename.strip()) == 0:
            raise InputValidationError("Filename is required", "filename")

        # Sanitize filename
        clean_filename = InputSanitizer.sanitize_filename(filename)

        # Validate file type
        file_extension = InputValidator.validate_file_type(clean_filename, file_content)

        # Additional security checks
        if len(file_content) == 0:
            raise InputValidationError("File cannot be empty", "file_content")

        # Check for suspicious file patterns
        if any(pattern in filename.lower() for pattern in ['.exe', '.bat', '.cmd', '.scr', '.vbs']):
            raise SecurityValidationError(
                "Potentially dangerous file type detected",
                "dangerous_file_type"
            )

        return {
            "original_filename": filename,
            "clean_filename": clean_filename,
            "file_extension": file_extension,
            "file_size": len(file_content),
            "is_valid": True
        }

    @staticmethod
    def validate_chat_message(
            message: str,
            max_length: int = 5000,
            min_length: int = 1
    ) -> str:
        """Validate chat message input"""

        if not message or not isinstance(message, str):
            raise InputValidationError("Message is required", "message")

        # Sanitize message
        clean_message = InputSanitizer.sanitize_text(message.strip())

        # Validate length
        InputValidator.validate_text_length(clean_message, "message", min_length, max_length)

        # Check for spam patterns
        if len(clean_message.split()) < 1:
            raise InputValidationError("Message must contain at least one word", "message")

        # Check for excessive repetition
        words = clean_message.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
                raise SecurityValidationError(
                    "Message contains excessive repetition",
                    "spam_detection"
                )

        return clean_message

    @staticmethod
    def validate_subject_tags(
            tags: Union[str, List[str]],
            max_tags: int = 10,
            max_tag_length: int = 50
    ) -> List[str]:
        """Validate and sanitize subject tags"""

        # Convert to list if string
        if isinstance(tags, str):
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        elif isinstance(tags, list):
            tag_list = [str(tag).strip() for tag in tags if str(tag).strip()]
        else:
            raise InputValidationError("Tags must be a string or list", "subject_tags")

        # Validate number of tags
        if len(tag_list) > max_tags:
            raise InputValidationError(
                f"Maximum {max_tags} tags allowed",
                "subject_tags",
                len(tag_list)
            )

        validated_tags = []
        for tag in tag_list:
            # Sanitize tag
            clean_tag = InputSanitizer.sanitize_text(tag).lower()

            # Validate tag format
            if not re.match(r'^[a-zA-Z0-9\s\-_]+$', clean_tag):
                raise InputValidationError(
                    f"Tag '{tag}' contains invalid characters",
                    "subject_tags"
                )

            # Validate tag length
            if len(clean_tag) > max_tag_length:
                raise InputValidationError(
                    f"Tag '{tag}' exceeds maximum length of {max_tag_length}",
                    "subject_tags"
                )

            if clean_tag and clean_tag not in validated_tags:
                validated_tags.append(clean_tag)

        return validated_tags

    @staticmethod
    def validate_pagination(
            skip: Optional[int] = None,
            limit: Optional[int] = None,
            max_limit: int = 100
    ) -> Tuple[int, int]:
        """Validate pagination parameters"""

        skip = skip or 0
        limit = limit or 20

        if skip < 0:
            raise InputValidationError("Skip parameter must be non-negative", "skip", skip)

        if limit < 1:
            raise InputValidationError("Limit parameter must be positive", "limit", limit)

        if limit > max_limit:
            raise InputValidationError(
                f"Limit parameter cannot exceed {max_limit}",
                "limit",
                limit
            )

        return skip, limit