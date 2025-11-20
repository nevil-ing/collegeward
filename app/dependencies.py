from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user, get_current_user_optional
from app.db.session import get_db
from app.models.user import User
from app.services.user_service import user_service
from app.utils.exceptions import AuthenticationError, NotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)


async def get_current_db_user(
        db: AsyncSession = Depends(get_db),
        current_user_claims: Dict[str, Any] = Depends(get_current_user)
) -> User:
    try:
        firebase_uid = current_user_claims.get("uid")
        email = current_user_claims.get("email")

        if not firebase_uid or not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing required fields"
            )

        # Get user from database
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)

        # Auto-register user if they don't exist
        if not user:
            from app.schemas.user_schema import UserCreate

            user_data = UserCreate(
                firebase_uid=firebase_uid,
                email=email,
                display_name=current_user_claims.get("name")
            )

            user = await user_service.user_repo.create_user(db, user_data)
            await db.commit()

            logger.info(f"Auto registered user during authentication: {email}")

        # Check if user account is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated"
            )

        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current database user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to authenticate user"
        )


async def get_current_db_user_optional(
        db: AsyncSession = Depends(get_db),
        current_user_claims: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Optional[User]:
    """
    Optional dependency to get the current authenticated user from the database

    Returns None if no authentication token is provided or if authentication fails.
    """
    if not current_user_claims:
        return None

    try:
        firebase_uid = current_user_claims.get("uid")
        if not firebase_uid:
            return None

        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)

        # Return None if user doesn't exist or is inactive
        if not user or not user.is_active:
            return None

        return user

    except Exception as e:
        logger.warning(f"Optional authentication failed: {e}")
        return None


async def get_user_by_id(
        user_id: str,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_db_user)
) -> User:
    """
    Dependency to get a user by ID with access control

    Users can only access their own profile unless they have admin privileges.
    """
    try:
        from uuid import UUID
        user_uuid = UUID(user_id)

        # Users can only access their own profile
        if current_user.id != user_uuid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: can only access your own profile"
            )

        return current_user

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user by ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


class UserAccessControl:
    """Access control utilities for user-related operations"""

    @staticmethod
    def check_user_access(current_user: User, target_user_id: str) -> bool:
        """Check if current user can access target user's data"""
        try:
            from uuid import UUID
            target_uuid = UUID(target_user_id)
            return current_user.id == target_uuid
        except (ValueError, TypeError):
            return False

    @staticmethod
    def ensure_user_access(current_user: User, target_user_id: str) -> None:
        if not UserAccessControl.check_user_access(current_user, target_user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: insufficient permissions"
            )