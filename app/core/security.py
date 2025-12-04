import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import firebase_admin
from firebase_admin import auth, credentials
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.logging import get_logger
from app.utils.exceptions import AuthenticationError

logger = get_logger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

security = HTTPBearer(auto_error=False)

def initialize_firebase():
    try:
        if not firebase_admin._apps:
            if not settings.FIREBASE_PROJECT_ID:
                logger.warning("Firebase project ID not provided, skipping Firebase initialization")
                return
            firebase_config = {
                "type": "service_account",
                "project_id": settings.FIREBASE_PROJECT_ID,
                "private_key_id": settings.FIREBASE_PRIVATE_KEY_ID,
                "private_key": settings.FIREBASE_PRIVATE_KEY.replace("\\n", "\n"),
                "client_email": settings.FIREBASE_CLIENT_EMAIL,
                "client_id": settings.FIREBASE_CLIENT_ID,
                "auth_uri": settings.FIREBASE_AUTH_URI,
                "token_uri": settings.FIREBASE_TOKEN_URI,
            }

            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")

        if settings.ENVIRONMENT != "development":
            raise

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


async def verify_firebase_token(token: str) -> Dict[str, Any]:
    """Verify Firebase ID token and return user claims"""
    try:
        # Check if Firebase is initialized
        if not firebase_admin._apps:
            raise AuthenticationError("Firebase not initialized")

        # Verify the ID token with check_revoked=True for security
        decoded_token = auth.verify_id_token(token, check_revoked=True)

        # Validate required fields
        if not decoded_token.get("uid") or not decoded_token.get("email"):
            raise AuthenticationError("Invalid token: missing required fields")

        return decoded_token

    except auth.ExpiredIdTokenError:
        logger.warning("Expired Firebase ID token")
        raise AuthenticationError("Authentication token has expired")
    except auth.RevokedIdTokenError:
        logger.warning("Revoked Firebase ID token")
        raise AuthenticationError("Authentication token has been revoked")
    except auth.InvalidIdTokenError:
        logger.warning("Invalid Firebase ID token")
        raise AuthenticationError("Invalid authentication token")
    except auth.CertificateFetchError:
        logger.error("Firebase certificate fetch error")
        raise AuthenticationError("Authentication service temporarily unavailable")
    except Exception as e:
        logger.error(f"Firebase token verification failed: {e}")
        raise AuthenticationError("Authentication failed")


class AuthenticationMiddleware:
    """Middleware for handling authentication and session management"""

    @staticmethod
    async def get_current_user_claims(
            credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
    ) -> Dict[str, Any]:
        """
        Dependency to get current authenticated user claims from Firebase token
        """
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )

        try:
            token = credentials.credentials
            user_claims = await verify_firebase_token(token)
            return user_claims
        except AuthenticationError as e:
            raise HTTPException(
                status_code=401,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )

    @staticmethod
    async def get_current_user_optional(
            credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
    ) -> Optional[Dict[str, Any]]:


        if not credentials:
            return None

        try:
            token = credentials.credentials
            user_claims = await verify_firebase_token(token)
            return user_claims
        except Exception as e:
            logger.warning(f"Optional authentication failed: {e}")
            return None


async def get_current_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user from Firebase token
    """
    return await AuthenticationMiddleware.get_current_user_claims(credentials)


async def get_current_user_optional(
        credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[Dict[str, Any]]:
    """
    Optional authentication dependency
    """
    return await AuthenticationMiddleware.get_current_user_optional(credentials)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token (for internal use)"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def verify_access_token(token: str) -> Dict[str, Any]:
    """Verify JWT access token (for internal use)"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=401,
            detail="Could not validate token"
        )


class SessionManager:
    """Manages user sessions and token refresh"""

    @staticmethod
    def get_token_expiry(token_claims: Dict[str, Any]) -> Optional[datetime]:
        """Get token expiry time from claims"""
        exp = token_claims.get("exp")
        if exp:
            return datetime.fromtimestamp(exp)
        return None

    @staticmethod
    def is_token_expiring_soon(token_claims: Dict[str, Any], threshold_minutes: int = 5) -> bool:
        """Check if token is expiring within threshold minutes"""
        expiry = SessionManager.get_token_expiry(token_claims)
        if not expiry:
            return False

        threshold = datetime.utcnow() + timedelta(minutes=threshold_minutes)
        return expiry <= threshold

    @staticmethod
    def should_refresh_token(token_claims: Dict[str, Any]) -> bool:
        """Determine if token should be refreshed"""
        return SessionManager.is_token_expiring_soon(token_claims, threshold_minutes=10)


class RateLimiter:

    @staticmethod
    def get_client_ip(request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"



initialize_firebase()