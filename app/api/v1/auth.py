from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
from uuid import UUID
from datetime import datetime

from app.core.security import (
    get_current_user,
    get_current_user_optional,
    SessionManager,
    RateLimiter
)
from app.db.session import get_db
from app.services.user_service import user_service
from app.schemas.user_schema import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserProfile
)
from app.utils.exceptions import (
    AuthenticationError,
    NotFoundError,
    DatabaseError,
    ValidationError
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
        user_data: UserCreate,
        db: AsyncSession = Depends(get_db),
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Register a new user with Firebase authentication
    """
    try:
        # Verify that the Firebase UID matches the token
        token_uid = current_user.get("uid")
        if token_uid != user_data.firebase_uid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Firebase UID mismatch"
            )

        # Verify email matches token
        token_email = current_user.get("email")
        if token_email != user_data.email:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Email mismatch with authentication token"
            )

        # Create user
        user = await user_service.user_repo.create_user(db, user_data)
        await db.commit()

        logger.info(f"User registered successfully: {user.email}")
        return user

    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"Database error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed due to database error"
        )
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Dict[str, Any])
async def login_user(
        request: Request,
        db: AsyncSession = Depends(get_db),
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Login user and return user profile with session information
"""
    try:
        # Authenticate and get/create user
        auth_result = await user_service.authenticate_user(
            db,
            request.headers.get("Authorization", "").replace("Bearer ", "")
        )

        user = auth_result["user"]
        firebase_claims = auth_result["firebase_claims"]

        # Check if token needs refresh
        should_refresh = SessionManager.should_refresh_token(firebase_claims)

        # Get client info for logging
        client_ip = RateLimiter.get_client_ip(request)
        logger.info(f"User logged in: {user.email} from {client_ip}")

        return {
            "user": UserProfile.model_validate(user),
            "session": {
                "should_refresh_token": should_refresh,
                "token_expires_at": SessionManager.get_token_expiry(firebase_claims),
                "email_verified": firebase_claims.get("email_verified", False)
            },
            "message": "Login successful"
        }

    except AuthenticationError as e:
        logger.warning(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
        db: AsyncSession = Depends(get_db),
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current authenticated user profile"""
    try:
        firebase_uid = current_user.get("uid")
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        return UserProfile.model_validate(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )


@router.put("/me", response_model=UserProfile)
async def update_current_user_profile(
        user_update: UserUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update current user profile"""
    try:
        firebase_uid = current_user.get("uid")
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        updated_user = await user_service.update_user_profile(db, user.id, user_update)

        logger.info(f"User profile updated: {user.email}")
        return UserProfile.model_validate(updated_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


@router.post("/verify-token")
async def verify_token(
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Verify Firebase authentication token and return token information
    """
    try:
        token_expiry = SessionManager.get_token_expiry(current_user)
        should_refresh = SessionManager.should_refresh_token(current_user)

        return {
            "valid": True,
            "uid": current_user.get("uid"),
            "email": current_user.get("email"),
            "email_verified": current_user.get("email_verified", False),
            "token_expires_at": token_expiry,
            "should_refresh": should_refresh
        }

    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token verification failed"
        )


@router.post("/refresh-check")
async def check_token_refresh(
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Check if the current token needs to be refreshed
    """
    try:
        should_refresh = SessionManager.should_refresh_token(current_user)
        token_expiry = SessionManager.get_token_expiry(current_user)

        return {
            "should_refresh": should_refresh,
            "token_expires_at": token_expiry,
            "current_time": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"Token refresh check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh check failed"
        )


@router.post("/logout")
async def logout_user(
        request: Request,
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Logout user

   """
    try:
        client_ip = RateLimiter.get_client_ip(request)
        user_email = current_user.get("email", "unknown")

        logger.info(f"User logged out: {user_email} from {client_ip}")

        return {
            "message": "Logout successful",
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"Logout error: {e}")
        # Don't fail logout on errors
        return {"message": "Logout completed"}


@router.delete("/me")
async def deactivate_current_user(
        db: AsyncSession = Depends(get_db),
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Deactivate current user account

    """
    try:
        firebase_uid = current_user.get("uid")
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        await user_service.deactivate_user(db, user.id)

        logger.info(f"User account deactivated: {user.email}")

        return {
            "message": "Account deactivated successfully",
            "deactivated_at": user.updated_at
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating user account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate account"
        )


@router.get("/health")
async def auth_health_check(
        user_claims: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    Health check endpoint for authentication service
    """
    return {
        "status": "healthy",
        "authenticated": user_claims is not None,
        "timestamp": SessionManager.get_token_expiry({"exp": int(user_claims.get("iat", 0))}) if user_claims else None
    }