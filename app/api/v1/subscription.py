from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
from datetime import datetime
import hmac
import hashlib

from app.core.security import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.core.logging import get_logger
from app.services.subscription_service import subscription_service
from app.schemas.subscription_schema import SubscriptionResponse

logger = get_logger(__name__)
router = APIRouter()


@router.get("/status", response_model=SubscriptionResponse)
async def get_subscription_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's subscription status and usage limits"""
    firebase_uid = current_user.get("uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid authentication")

    status = await subscription_service.get_subscription_status(db, firebase_uid)
    return status


@router.get("/can-upload")
async def check_can_upload(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Check if user can upload a note"""
    firebase_uid = current_user.get("uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid authentication")

    can_upload, message = await subscription_service.check_can_upload_note(db, firebase_uid)
    return {"allowed": can_upload, "message": message}


@router.get("/can-ask-ai")
async def check_can_ask_ai(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Check if user can ask AI tutor"""
    firebase_uid = current_user.get("uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid authentication")

    can_ask, message = await subscription_service.check_can_ask_ai(db, firebase_uid)
    return {"allowed": can_ask, "message": message}


@router.post("/webhook/revenuecat")
async def revenuecat_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
    x_revenuecat_signature: Optional[str] = Header(None, alias="X-RevenueCat-Signature")
):
    """
    Handle RevenueCat webhooks for subscription events.
    
    Configure this endpoint in RevenueCat Dashboard:
    https://app.revenuecat.com/projects/{project}/integrations/webhooks
    """
    try:
        body = await request.body()
        payload = await request.json()

        # Verify webhook signature (optional but recommended)
        # if settings.REVENUECAT_WEBHOOK_SECRET:
        #     if not _verify_revenuecat_signature(body, x_revenuecat_signature):
        #         logger.warning("Invalid RevenueCat webhook signature")
        #         raise HTTPException(status_code=401, detail="Invalid signature")

        event = payload.get("event", {})
        event_type = event.get("type")
        app_user_id = event.get("app_user_id")  # This is the Firebase UID we passed

        if not app_user_id:
            logger.warning("Webhook received without app_user_id")
            return {"status": "ignored", "reason": "no_user_id"}

        logger.info(f"RevenueCat webhook: {event_type} for user {app_user_id}")

        # Extract subscription info
        subscriber_info = event.get("subscriber_info", {})
        entitlements = subscriber_info.get("entitlements", {})
        premium_entitlement = entitlements.get("premium", {})

        is_active = premium_entitlement.get("is_active", False)
        product_id = premium_entitlement.get("product_identifier", "")
        expires_date_str = premium_entitlement.get("expires_date")
        is_trial = premium_entitlement.get("period_type") == "trial"
        will_renew = premium_entitlement.get("will_renew", True)

        expires_date = None
        if expires_date_str:
            try:
                expires_date = datetime.fromisoformat(expires_date_str.replace("Z", "+00:00"))
            except:
                pass

        # Handle different event types
        if event_type in ["INITIAL_PURCHASE", "RENEWAL", "PRODUCT_CHANGE"]:
            await subscription_service.update_subscription_from_revenuecat(
                db=db,
                firebase_uid=app_user_id,
                product_id=product_id,
                expires_date=expires_date,
                is_active=is_active,
                is_trial=is_trial,
                will_renew=will_renew,
            )
            return {"status": "processed", "event": event_type}

        elif event_type == "CANCELLATION":
            await subscription_service.cancel_subscription(db, app_user_id)
            return {"status": "processed", "event": event_type}

        elif event_type == "EXPIRATION":
            await subscription_service.expire_subscription(db, app_user_id)
            return {"status": "processed", "event": event_type}

        elif event_type == "BILLING_ISSUE":
            logger.warning(f"Billing issue for user {app_user_id}")
            return {"status": "noted", "event": event_type}

        else:
            logger.info(f"Unhandled event type: {event_type}")
            return {"status": "ignored", "event": event_type}

    except Exception as e:
        logger.error(f"Error processing RevenueCat webhook: {e}")
        # Return 200 to prevent retries for bad data
        return {"status": "error", "message": str(e)}


def _verify_revenuecat_signature(body: bytes, signature: Optional[str]) -> bool:
    """Verify RevenueCat webhook signature"""
    if not signature or not hasattr(settings, 'REVENUECAT_WEBHOOK_SECRET'):
        return True  # Skip if not configured

    expected = hmac.new(
        settings.REVENUECAT_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)
