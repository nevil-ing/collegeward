from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from typing import Dict, Any, List, Optional, AsyncGenerator
from uuid import UUID
import uuid
import json
from datetime import datetime

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.services.user_service import user_service
from app.schemas.chat_schema import (
    ChatRequest, ChatResponse, ConversationCreate, ConversationResponse,
    MessageResponse, ConversationUpdate
)
from app.rag.rag_service import rag_service
from app.core.logging import get_logger
from app.utils.exceptions import ProcessingError

logger = get_logger(__name__)
router = APIRouter()


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get all conversations for the current user"""
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Eagerly load messages relationship to avoid lazy loading issues
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.user_id == user.id)
            .order_by(desc(Conversation.updated_at))
            .offset(skip)
            .limit(limit)
        )

        result = await db.execute(stmt)
        conversations = result.scalars().all()

        # Convert to response format - build dicts to avoid lazy loading during validation
        conversation_responses = []
        for conv in conversations:
            conv_dict = {
                "id": conv.id,
                "user_id": conv.user_id,
                "title": conv.title,
                "mode": conv.mode,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "messages": [MessageResponse.model_validate(msg) for msg in conv.messages] if conv.messages else []
            }
            conversation_responses.append(ConversationResponse.model_validate(conv_dict))

        return conversation_responses

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversations for user {firebase_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
        conversation_data: ConversationCreate,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Create a new conversation"""
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        conversation = Conversation(
            user_id=user.id,
            title=conversation_data.title,
            mode=conversation_data.mode
        )

        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)

        logger.info(f"Created conversation {conversation.id} for user {user.id}")

        # Convert to dict to avoid lazy loading during validation
        conv_dict = {
            "id": conversation.id,
            "user_id": conversation.user_id,
            "title": conversation.title,
            "mode": conversation.mode,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": []
        }
        return ConversationResponse.model_validate(conv_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
        conversation_id: UUID,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get a specific conversation with messages"""
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Eagerly load messages relationship
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id
            )
        )

        result = await db.execute(stmt)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Convert to response format - messages are already loaded
        conversation_dict = {
            "id": conversation.id,
            "user_id": conversation.user_id,
            "title": conversation.title,
            "mode": conversation.mode,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": [MessageResponse.model_validate(msg) for msg in
                         conversation.messages] if conversation.messages else []
        }

        return ConversationResponse.model_validate(conversation_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")


@router.put("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
        conversation_id: UUID,
        update_data: ConversationUpdate,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Update a conversation"""
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Eagerly load messages relationship
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id
            )
        )

        result = await db.execute(stmt)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Update fields
        if update_data.title is not None:
            conversation.title = update_data.title

        conversation.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(conversation)

        # Convert to dict to avoid lazy loading during validation
        conv_dict = {
            "id": conversation.id,
            "user_id": conversation.user_id,
            "title": conversation.title,
            "mode": conversation.mode,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": [MessageResponse.model_validate(msg) for msg in
                         conversation.messages] if conversation.messages else []
        }
        return ConversationResponse.model_validate(conv_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update conversation")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
        conversation_id: UUID,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Delete a conversation and all its messages"""
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Find conversation belonging to user
        stmt = (
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id
            )
        )

        result = await db.execute(stmt)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Delete conversation (messages will be cascade deleted due to relationship)
        await db.delete(conversation)
        await db.commit()

        logger.info(f"Deleted conversation {conversation_id} for user {user.id}")

        return {"message": "Conversation deleted successfully", "conversation_id": str(conversation_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
        chat_request: ChatRequest,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Send a message to the AI tutor with RAG context retrieval"""
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get or create conversation
        conversation = await _get_or_create_conversation(
            chat_request.conversation_id,
            chat_request.mode,
            user.id,
            db
        )

        # Get conversation history for context
        conversation_messages = await _get_conversation_messages(conversation.id, db)

        # Retrieve relevant context using RAG
        context_data = await rag_service.get_context_for_conversation(
            messages=conversation_messages,
            user_id=firebase_uid,  # RAG service expects Firebase UID
            db=db,
            max_tokens=3500  # Leave room for AI response
        )

        # Create user message
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=chat_request.message
        )

        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)

        # Generate AI response (placeholder - will be implemented in AI integration task)
        ai_response_content = await _generate_ai_response(
            user_message=chat_request.message,
            context_data=context_data,
            conversation_mode=chat_request.mode,
            conversation_history=conversation_messages
        )

        # Create AI message with sources
        ai_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=ai_response_content,
            sources=_format_sources_for_storage(context_data)
        )

        db.add(ai_message)

        # Update conversation timestamp and title if needed
        conversation.updated_at = datetime.utcnow()
        if not conversation.title:
            conversation.title = _generate_conversation_title(chat_request.message)

        await db.commit()
        await db.refresh(ai_message)

        # Award XP and update streak for chat session
        try:
            from app.services.gamification_service import GamificationService
            gamification_service = GamificationService(db)

            # Award XP for chat session
            await gamification_service.award_xp(
                user_id=user.id,
                activity_type="chat_session",
                activity_id=conversation.id,
                reason="Completed AI tutor chat session"
            )

            # Update study streak
            await gamification_service.update_study_streak(
                user_id=user.id,
                study_time_seconds=60  # Estimate 1 minute per chat
            )

            # Check for new achievements
            await gamification_service.check_and_award_achievements(user.id)

        except Exception as e:
            logger.warning(f"Failed to update gamification for user {user.id}: {e}")

        logger.info(
            f"Chat completed for user {user.id}, conversation {conversation.id}, "
            f"context chunks: {context_data['stats']['total_chunks']}"
        )

        # Convert conversation to dict to avoid lazy loading during validation
        # Messages are already loaded via selectinload in _get_or_create_conversation
        conv_dict = {
            "id": conversation.id,
            "user_id": conversation.user_id,
            "title": conversation.title,
            "mode": conversation.mode,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": [MessageResponse.model_validate(msg) for msg in conversation.messages] if hasattr(conversation,
                                                                                                          'messages') and conversation.messages else []
        }

        return ChatResponse(
            message=MessageResponse.model_validate(ai_message),
            conversation=ConversationResponse.model_validate(conv_dict)
        )

    except HTTPException:
        raise
    except Exception as e:
        # Use current_user parameter which is always available
        uid = current_user.get("uid", "unknown") if current_user else "unknown"
        logger.error(f"Chat failed for user {uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")


@router.post("/chat/stream")
async def chat_with_ai_stream(
        chat_request: ChatRequest,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Stream AI tutor responses in real-time for better user experience"""
    try:
        firebase_uid = current_user.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Get user from database by Firebase UID
        user = await user_service.get_user_by_firebase_uid(db, firebase_uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get or create conversation
        conversation = await _get_or_create_conversation(
            chat_request.conversation_id,
            chat_request.mode,
            user.id,
            db
        )

        # Get conversation history for context
        conversation_messages = await _get_conversation_messages(conversation.id, db)

        # Retrieve relevant context using RAG
        context_data = await rag_service.get_context_for_conversation(
            messages=conversation_messages,
            user_id=firebase_uid,  # RAG service expects Firebase UID
            db=db,
            max_tokens=3500
        )

        # Create user message
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=chat_request.message
        )

        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)

        # Update conversation timestamp and title if needed
        conversation.updated_at = datetime.utcnow()
        if not conversation.title:
            conversation.title = _generate_conversation_title(chat_request.message)

        await db.commit()

        # Stream AI response
        return StreamingResponse(
            _stream_ai_response(
                user_message=chat_request.message,
                context_data=context_data,
                conversation_mode=chat_request.mode,
                conversation_history=conversation_messages,
                conversation_id=conversation.id,
                user_id=user.id,
                db=db
            ),
            media_type="text/plain"
        )

    except HTTPException:
        raise
    except Exception as e:
        # Use current_user parameter which is always available
        uid = current_user.get("uid", "unknown") if current_user else "unknown"
        logger.error(f"Streaming chat failed for user {uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process streaming chat")


@router.get("/context/check")
async def check_context_availability(
        subject_tags: Optional[List[str]] = Query(None),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Check what context is available for the user"""
    try:
        user_id = current_user.get("uid")

        availability = await rag_service.check_context_availability(
            user_id=user_id,
            db=db,
            subject_tags=subject_tags
        )

        return availability

    except Exception as e:
        logger.error(f"Context availability check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to check context availability")


@router.post("/context/search")
async def search_context(
        query: str,
        subject_tags: Optional[List[str]] = None,
        max_tokens: Optional[int] = Query(None, ge=500, le=6000),
        mode: str = Query("balanced", regex="^(fast|balanced|comprehensive)$"),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Search for relevant context in user's documents"""
    try:
        user_id = current_user.get("uid")

        context_data = await rag_service.get_context_for_query(
            query=query,
            user_id=user_id,
            db=db,
            subject_filters=subject_tags,
            max_tokens=max_tokens,
            mode=mode
        )

        return context_data

    except Exception as e:
        logger.error(f"Context search failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to search context")


@router.post("/tutor/clinical-scenario")
async def generate_clinical_scenario(
        topic: str,
        difficulty_level: str = Query("intermediate", regex="^(beginner|intermediate|advanced)$"),
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Generate clinical scenario for active learning"""
    try:
        from app.services.ai_tutor_service import ai_tutor_service

        user_id = current_user.get("uid")

        # Get user's context for the topic
        context_data = await rag_service.get_context_for_query(
            query=topic,
            user_id=user_id,
            db=db,
            max_tokens=1000,
            mode="fast"
        )

        formatted_context = None
        if context_data.get("chunks"):
            formatted_context = rag_service.rag_retriever.format_context_for_ai(context_data)

        # Generate clinical scenario
        scenario = await ai_tutor_service.generate_clinical_scenario(
            topic=topic,
            difficulty_level=difficulty_level,
            context=formatted_context
        )

        logger.info(f"Generated clinical scenario for user {user_id}, topic: {topic}")

        return {
            "topic": topic,
            "difficulty_level": difficulty_level,
            "scenario": scenario,
            "context_used": bool(formatted_context)
        }

    except Exception as e:
        logger.error(f"Clinical scenario generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate clinical scenario")


@router.post("/tutor/assess-understanding")
async def assess_student_understanding(
        student_response: str,
        expected_concepts: List[str],
        conversation_id: Optional[UUID] = None,
        current_user: Dict[str, Any] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Assess student's understanding and provide feedback"""
    try:
        from app.services.ai_tutor_service import ai_tutor_service
        from app.services.ai_service import ConversationMessage

        user_id = current_user.get("uid")

        # Get conversation history if provided
        conversation_messages = []
        if conversation_id:
            conversation_messages = await _get_conversation_messages(conversation_id, db)

        # Convert to AI service format
        messages = [
            ConversationMessage(role=msg["role"], content=msg["content"])
            for msg in conversation_messages
        ]

        # Assess understanding
        assessment = await ai_tutor_service.assess_student_understanding(
            student_response=student_response,
            expected_concepts=expected_concepts,
            conversation_history=messages
        )

        logger.info(f"Assessed understanding for user {user_id}")

        return {
            "student_response": student_response,
            "expected_concepts": expected_concepts,
            "assessment": assessment
        }

    except Exception as e:
        logger.error(f"Understanding assessment failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to assess understanding")


@router.get("/tutor/modes")
async def get_tutor_modes():
    """Get available tutor modes and their descriptions"""
    return {
        "modes": [
            {
                "id": "ai_mode",
                "name": "AI Mode",
                "description": "Fast responses using AI with your study materials",
                "features": ["Quick responses", "Context from notes", "Step-by-step reasoning"],
                "response_time": "~3 seconds"
            },
            {
                "id": "verified_mode",
                "name": "Verified Mode",
                "description": "Comprehensive responses with verified medical sources",
                "features": ["Medical database search", "Citations", "Evidence-based answers", "Clinical guidelines"],
                "response_time": "~8 seconds"
            }
        ]
    }


async def _get_or_create_conversation(
        conversation_id: Optional[UUID],
        mode: str,
        user_id: UUID,  # Now expects UUID, not Firebase UID string
        db: AsyncSession
) -> Conversation:
    """Get existing conversation or create new one"""
    if conversation_id:
        # Eagerly load messages relationship
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
        )
        result = await db.execute(stmt)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return conversation
    else:
        # Create new conversation
        conversation = Conversation(
            user_id=user_id,
            mode=mode
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return conversation


async def _get_conversation_messages(conversation_id: UUID, db: AsyncSession) -> List[Dict[str, str]]:
    """Get conversation messages formatted for RAG context"""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )

    result = await db.execute(stmt)
    messages = result.scalars().all()

    return [
        {
            "role": msg.role,
            "content": msg.content
        }
        for msg in messages
    ]


async def _generate_ai_response(
        user_message: str,
        context_data: Dict[str, Any],
        conversation_mode: str,
        conversation_history: List[Dict[str, str]]
) -> str:
    """
    Generate AI response using specialized tutor service with step-by-step reasoning
    """
    from app.services.ai_tutor_service import ai_tutor_service, TutorMode
    from app.services.ai_service import ConversationMessage

    try:
        # Convert conversation history to AI service format
        messages = [
            ConversationMessage(role=msg["role"], content=msg["content"])
            for msg in conversation_history
        ]

        # Format context for AI
        formatted_context = None
        if context_data.get("chunks"):
            formatted_context = rag_service.rag_retriever.format_context_for_ai(context_data)

        # Convert mode to TutorMode enum
        tutor_mode = TutorMode.VERIFIED_MODE if conversation_mode == "verified_mode" else TutorMode.AI_MODE

        # Generate comprehensive tutor response
        tutor_response = await ai_tutor_service.generate_tutor_response(
            user_message=user_message,
            conversation_history=messages,
            context=formatted_context,
            mode=tutor_mode,
            enable_step_by_step=True,  # Could use chat_request.enable_step_by_step
            enable_guided_questions=True  # Could use chat_request.enable_guided_questions
        )

        # Format comprehensive response
        formatted_response = _format_comprehensive_tutor_response(
            tutor_response,
            context_data,
            conversation_mode
        )

        return formatted_response

    except Exception as e:
        logger.error(f"AI tutor response generation failed: {e}")
        # Fallback to basic response
        return _generate_fallback_response(user_message, context_data, conversation_mode)


def _format_sources_for_storage(context_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format context sources for database storage"""
    if not context_data.get("chunks"):
        return None

    sources = {
        "context_chunks": len(context_data["chunks"]),
        "total_tokens": context_data["stats"]["total_tokens"],
        "avg_relevance": context_data["stats"]["avg_relevance_score"],
        "source_files": []
    }

    # Add source file information
    for chunk in context_data["chunks"][:5]:  # Store top 5 sources
        note_metadata = chunk.get("note_metadata", {})
        if note_metadata:
            sources["source_files"].append({
                "filename": note_metadata.get("filename"),
                "file_type": note_metadata.get("file_type"),
                "relevance_score": chunk.get("relevance_score", 0.0)
            })

    return sources


def _extract_medical_query(user_message: str, conversation_history: List[Dict[str, str]]) -> str:
    """Extract medical query for external source search"""
    # For now, use the user message directly
    # Could be enhanced to analyze conversation context for better queries
    return user_message


def _format_comprehensive_tutor_response(
        tutor_response,  # TutorResponse object
        context_data: Dict[str, Any],
        mode: str
) -> str:
    """Format comprehensive tutor response with all educational components"""
    response_parts = []

    # Add main answer
    response_parts.append(tutor_response.main_answer)

    # Add step-by-step reasoning if available
    if tutor_response.reasoning_steps:
        response_parts.append("\n\n🧠 **Step-by-Step Reasoning:**")
        for step in tutor_response.reasoning_steps:
            response_parts.append(f"\n**{step.title}**")
            response_parts.append(f"\n{step.content}")

            if step.key_concepts:
                response_parts.append(f"\n*Key concepts: {', '.join(step.key_concepts)}*")

    # Add guided questions for active learning
    if tutor_response.guided_questions:
        response_parts.append("\n\n🤔 **Think About This:**")
        for i, question in enumerate(tutor_response.guided_questions, 1):
            response_parts.append(f"\n{i}. {question}")

    # Add key takeaways
    if tutor_response.key_takeaways:
        response_parts.append("\n\n💡 **Key Takeaways:**")
        for takeaway in tutor_response.key_takeaways:
            response_parts.append(f"\n• {takeaway}")

    # Add context information if available
    if context_data.get("chunks"):
        stats = context_data.get("stats", {})
        response_parts.append(
            f"\n\n📚 **From Your Study Materials:**\n"
            f"Found {stats.get('total_chunks', 0)} relevant sections "
            f"({stats.get('total_tokens', 0)} tokens used)"
        )

    # Add verified sources for verified mode
    if mode == "verified_mode" and tutor_response.sources_used:
        response_parts.append("\n\n🔬 **Verified Medical Sources:**")
        for i, source in enumerate(tutor_response.sources_used[:3], 1):
            response_parts.append(f"\n{i}. {source}")

    # Add follow-up topics
    if tutor_response.follow_up_topics:
        response_parts.append("\n\n📖 **Related Topics to Explore:**")
        for topic in tutor_response.follow_up_topics:
            response_parts.append(f"\n• {topic}")

    # Add confidence indicator
    confidence_emoji = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🟠"
    }
    emoji = confidence_emoji.get(tutor_response.confidence_level, "🟡")
    response_parts.append(f"\n\n{emoji} *Confidence: {tutor_response.confidence_level}*")

    return "".join(response_parts)


def _format_tutor_response(
        ai_content: str,
        context_data: Dict[str, Any],
        medical_sources: List[Any],
        mode: str
) -> str:
    """Format AI response with step-by-step reasoning structure (legacy)"""
    response_parts = []

    # Add main AI response
    response_parts.append(ai_content)

    # Add context information if available
    if context_data.get("chunks"):
        stats = context_data.get("stats", {})
        response_parts.append(
            f"\n\n📚 **Context from your study materials:**\n"
            f"Found {stats.get('total_chunks', 0)} relevant sections "
            f"({stats.get('total_tokens', 0)} tokens)"
        )

    # Add medical sources for verified mode
    if mode == "verified_mode" and medical_sources:
        response_parts.append("\n\n🔬 **Verified Medical Sources:**")
        for i, source in enumerate(medical_sources[:3], 1):
            response_parts.append(f"\n{i}. {source.title}")
            response_parts.append(f"   Source: {source.source_type.upper()}")
            if hasattr(source, 'citation'):
                response_parts.append(f"   Citation: {source.citation}")

    return "".join(response_parts)


def _generate_fallback_response(
        user_message: str,
        context_data: Dict[str, Any],
        mode: str
) -> str:
    """Generate fallback response when AI service fails"""
    response_parts = []

    response_parts.append(
        "I'm having trouble processing your question right now, but let me help based on what I can access.\n\n"
    )

    if context_data.get("chunks"):
        stats = context_data.get("stats", {})
        response_parts.append(
            f"I found {stats.get('total_chunks', 0)} relevant sections in your study materials. "
        )

        # Show a snippet from the most relevant chunk
        chunks = context_data.get("chunks", [])
        if chunks:
            top_chunk = chunks[0]
            text_snippet = top_chunk.get("text", "")[:300]
            response_parts.append(f"Here's the most relevant content:\n\n{text_snippet}...")
    else:
        response_parts.append(
            "I don't have specific information from your study materials about this topic. "
        )

    if mode == "verified_mode":
        response_parts.append(
            "\n\nIn verified mode, I would normally search medical databases for additional information. "
            "Please try your question again, or consider consulting your textbooks and instructors."
        )

    response_parts.append(
        "\n\nPlease try rephrasing your question or check your connection and try again."
    )

    return "".join(response_parts)


async def _stream_ai_response(
        user_message: str,
        context_data: Dict[str, Any],
        conversation_mode: str,
        conversation_history: List[Dict[str, str]],
        conversation_id: UUID,
        user_id: UUID,
        db: AsyncSession
) -> AsyncGenerator[str, None]:
    """Stream AI response with real-time updates"""
    from app.services.ai_service import ai_service_manager
    from app.services.ai_service import ConversationMessage

    try:
        # Convert conversation history to AI service format
        messages = [
            ConversationMessage(role=msg["role"], content=msg["content"])
            for msg in conversation_history
        ]

        # Add current user message
        messages.append(ConversationMessage(role="user", content=user_message))

        # Format context for AI
        formatted_context = None
        if context_data.get("chunks"):
            formatted_context = rag_service.rag_retriever.format_context_for_ai(context_data)

        # Determine medical query for verified mode
        medical_query = None
        if conversation_mode == "verified_mode":
            medical_query = _extract_medical_query(user_message, conversation_history)

        # Send initial context information
        if context_data.get("chunks"):
            stats = context_data.get("stats", {})
            context_info = f"📚 Found {stats.get('total_chunks', 0)} relevant sections from your study materials.\n\n"
            yield f"data: {json.dumps({'type': 'context', 'content': context_info})}\n\n"

        if conversation_mode == "verified_mode":
            yield f"data: {json.dumps({'type': 'status', 'content': '🔬 Searching verified medical sources...'})}\n\n"

        # Collect full response for database storage
        full_response = ""

        # Stream AI response using tutor service
        from app.services.ai_tutor_service import TutorMode
        tutor_mode = TutorMode.VERIFIED_MODE if conversation_mode == "verified_mode" else TutorMode.AI_MODE

        # For streaming, we'll use the base AI service but with tutor prompts
        # Full tutor response with reasoning steps will be generated after streaming
        async for chunk in ai_service_manager.generate_response_stream(
                messages=messages,
                mode=conversation_mode,
                context=formatted_context,
                medical_query=medical_query,
                temperature=0.7,
                max_tokens=1500
        ):
            full_response += chunk
            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete', 'content': ''})}\n\n"

        # Store AI message in database
        ai_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_response,
            sources=_format_sources_for_storage(context_data)
        )

        db.add(ai_message)
        await db.commit()

        # Award XP and update streak for chat session
        try:
            from app.services.gamification_service import GamificationService
            gamification_service = GamificationService(db)

            # Award XP for chat session
            await gamification_service.award_xp(
                user_id=user_id,
                activity_type="chat_session",
                activity_id=conversation_id,
                reason="Completed AI tutor chat session"
            )

            # Update study streak
            await gamification_service.update_study_streak(
                user_id=user_id,
                study_time_seconds=60  # Estimate 1 minute per chat
            )

            # Check for new achievements
            await gamification_service.check_and_award_achievements(user_id)

        except Exception as e:
            logger.warning(f"Failed to update gamification for user {user_id}: {e}")

        logger.info(f"Streaming chat completed for conversation {conversation_id}")

    except Exception as e:
        logger.error(f"AI streaming response failed: {e}")
        error_message = "I'm having trouble processing your question. Please try again."
        yield f"data: {json.dumps({'type': 'error', 'content': error_message})}\n\n"


def _generate_conversation_title(first_message: str) -> str:
    """Generate a conversation title from the first message"""
    # Simple title generation - take first few words
    words = first_message.split()[:6]
    title = " ".join(words)

    if len(title) > 50:
        title = title[:47] + "..."

    return title or "New Conversation"