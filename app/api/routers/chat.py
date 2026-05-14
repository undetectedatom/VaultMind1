from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from app.api.dependencies import get_cur_user, get_session
from app.database.models import (
    ChatMessage,
    ChatMessagePublic,
    ChatRequest,
    ChatSession,
    ChatSessionPublic,
    ChatResponse,
    NewChatResponse,
    User,
)
from app.services.chat_service import generate_title, knowledge_query

router = APIRouter()


def check_session(
    db_session: Session, chat_session_id: uuid.UUID | str, user_id: uuid.UUID | str
):
    # Authorization
    # We must authorize both session id and user id, preventing anomalous user query
    session_uuid = (
        chat_session_id
        if isinstance(chat_session_id, uuid.UUID)
        else uuid.UUID(chat_session_id)
    )
    user_uuid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(user_id)
    statement = select(ChatSession).where(
        (ChatSession.id == session_uuid) & (ChatSession.user_id == user_uuid)
    )
    chat_session = db_session.exec(statement).first()
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found, please open a new session.",
        )
    return chat_session


async def get_history(
    db_session: Session, chat_session_id: uuid.UUID | str, limit: int = 10
):
    session_uuid = (
        chat_session_id
        if isinstance(chat_session_id, uuid.UUID)
        else uuid.UUID(chat_session_id)
    )
    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_uuid)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    return db_session.exec(statement).all()


@router.post("")
async def create_new_session(
    chat_request: ChatRequest,
    user: Annotated[User, Depends(get_cur_user)],
    db_session: Annotated[Session, Depends(get_session)],
):
    chat_session = ChatSession(user_id=user.id, title="")
    chat_session.title = await generate_title(chat_request.query)
    db_session.add(chat_session)
    user_msg = ChatMessage(
        session_id=chat_session.id, role="user", content=chat_request.query
    )
    db_session.add(user_msg)
    answer, sources, meta = await knowledge_query([user_msg], user.id)
    assistant_msg = ChatMessage(
        session_id=chat_session.id,
        role="assistant",
        content=answer,
        sources=sources,
    )
    db_session.add(assistant_msg)
    db_session.commit()
    return NewChatResponse(
        answer=answer,
        sources=sources,
        session_id=chat_session.id,
        title=chat_session.title,
        meta=meta,
    )


@router.post("/{chat_session_id}")
async def chat_in_session(
    chat_request: ChatRequest,
    chat_session_id: str,
    user: Annotated[User, Depends(get_cur_user)],
    db_session: Annotated[Session, Depends(get_session)],
):

    check_session(db_session, chat_session_id, user.id)

    user_msg = ChatMessage(
        session_id=uuid.UUID(chat_session_id), role="user", content=chat_request.query
    )
    db_session.add(user_msg)
    db_session.commit()

    chat_history = await get_history(db_session, chat_session_id)
    answer, sources, meta = await knowledge_query(chat_history, user.id)

    assistant_msg = ChatMessage(
        session_id=uuid.UUID(chat_session_id),
        role="assistant",
        content=answer,
        sources=sources,
    )
    db_session.add(assistant_msg)
    db_session.commit()

    return ChatResponse(answer=answer, sources=sources, meta=meta)


@router.get("", response_model=list[ChatSessionPublic])
async def get_chat_sessions(
    user: Annotated[User, Depends(get_cur_user)],
    db_session: Annotated[Session, Depends(get_session)],
):
    statement = (
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.id.desc())
    )
    return db_session.exec(statement).all()


@router.get("/{session_id}", response_model=list[ChatMessagePublic])
async def get_chat_history(
    session_id: str,
    user: Annotated[User, Depends(get_cur_user)],
    db_session: Annotated[Session, Depends(get_session)],
):
    check_session(db_session, session_id, user.id)
    return await get_history(db_session, session_id)
