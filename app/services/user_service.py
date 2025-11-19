from typing import Optional, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from firebase_admin import auth as firebase_auth

from app.models.user import User
from app.db.utils import BaseRepository
from app.schemas.user_schema import UserCreate, UserUpdate
from app.utils.exceptions import NotFoundError, DatabaseError, AuthenticationError
from app.core.logging import get_logger

logger = get_logger(__name__)

class UserRepository(BaseRepository[User]):
    def __init__(self):
        super().__init__(User)

    async def get_by_firebase_uid(self, db: AsyncSession, firebase_uid: str) -> Optional[User]:
        try:
            result = await db.execute(
                select(User).where(User.firebase_uid == firebase_uid)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by Firebase UID {firebase_uid}: {e}")
            raise DatabaseError("Failed to get user by Firebase UID")

    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        try:
            result = await db.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by Email {email}: {e}")
            raise DatabaseError("Failed to get user by Email")

    async def create_user(self, db: AsyncSession, user_data: UserCreate) -> User:
        """Create a new user"""
        try:
            # Check if user already exists
            existing_user = await self.get_by_firebase_uid(db, user_data.firebase_uid)
            if existing_user:
                logger.warning(f"User with Firebase UID {user_data.firebase_uid} already exists")
                return existing_user

            # Create new user
            user = await self.create(
                db,
                firebase_uid=user_data.firebase_uid,
                email=user_data.email,
                display_name=user_data.display_name,
                study_level=user_data.study_level
            )

            logger.info(f"Created new user: {user.id} ({user.email})")
            return user

        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise DatabaseError("Failed to create user")

    async def update_user(self, db: AsyncSession, user_id: UUID, user_data: UserUpdate) -> User:
        """Update user profile"""
        try:
            user = await self.get_by_id_or_404(db, user_id)

            # Update fields
            update_data = {}
            if user_data.display_name is not None:
                update_data["display_name"] = user_data.display_name
            if user_data.study_level is not None:
                update_data["study_level"] = user_data.study_level
            if user_data.is_active is not None:
                update_data["is_active"] = user_data.is_active

            if update_data:
                updated_user = await self.update(db, user_id, **update_data)
                logger.info(f"Updated user profile: {user_id}")
                return updated_user

            return user

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
            raise DatabaseError("Failed to update user")


class UserService:
    """Service for user authentication and profile management"""

    def __init__(self):
        self.user_repo = UserRepository()

    async def authenticate_user(self, db: AsyncSession, firebase_token: str) -> Dict[str, Any]:
        """Authenticate user with Firebase token and return user data"""
        try:
            # Verify Firebase token
            decoded_token = firebase_auth.verify_id_token(firebase_token)
            firebase_uid = decoded_token.get("uid")
            email = decoded_token.get("email")

            if not firebase_uid or not email:
                raise AuthenticationError("Invalid token: missing required fields")

            # Get or create user in database
            user = await self.user_repo.get_by_firebase_uid(db, firebase_uid)

            if not user:
                user_data = UserCreate(
                    firebase_uid=firebase_uid,
                    email=email,
                    display_name=decoded_token.get("name")
                )
                user = await self.user_repo.create_user(db, user_data)
                await db.commit()
                logger.info(f"Auto-registered new user: {user.email}")

            # Check if user is active
            if not user.is_active:
                raise AuthenticationError("User account is deactivated")

            return {
                "user": user,
                "firebase_claims": decoded_token
            }

        except firebase_auth.InvalidIdTokenError:
            logger.warning("Invalid Firebase ID token")
            raise AuthenticationError("Invalid authentication token")
        except firebase_auth.ExpiredIdTokenError:
            logger.warning("Expired Firebase ID token")
            raise AuthenticationError("Authentication token has expired")
        except firebase_auth.RevokedIdTokenError:
            logger.warning("Revoked Firebase ID token")
            raise AuthenticationError("Authentication token has been revoked")
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise AuthenticationError("Authentication failed")

    async def get_user_profile(self, db: AsyncSession, user_id: UUID) -> User:
        """Get user profile by ID"""
        try:
            user = await self.user_repo.get_by_id_or_404(db, user_id)
            if not user.is_active:
                raise AuthenticationError("User account is deactivated")
            return user
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting user profile {user_id}: {e}")
            raise DatabaseError("Failed to get user profile")

    async def update_user_profile(self, db: AsyncSession, user_id: UUID, user_data: UserUpdate) -> User:
        """Update user profile"""
        try:
            user = await self.user_repo.update_user(db, user_id, user_data)
            await db.commit()
            return user
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating user profile {user_id}: {e}")
            raise

    async def deactivate_user(self, db: AsyncSession, user_id: UUID) -> User:
        """Deactivate user account"""
        try:
            user_data = UserUpdate(is_active=False)
            user = await self.user_repo.update_user(db, user_id, user_data)
            await db.commit()
            logger.info(f"Deactivated user account: {user_id}")
            return user
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deactivating user {user_id}: {e}")
            raise

    async def get_user_by_firebase_uid(self, db: AsyncSession, firebase_uid: str) -> Optional[User]:
        """Get user by Firebase UID"""
        return await self.user_repo.get_by_firebase_uid(db, firebase_uid)



user_service = UserService()
