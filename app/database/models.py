import enum
from datetime import datetime, timezone
from typing import Annotated, Any
import uuid

from sqlmodel import JSON, Column, Field, Relationship, SQLModel


class UserBase(SQLModel):
    username: Annotated[str, Field(index=True, unique=True)]
    email: Annotated[str, Field(index=True, unique=True)]


class User(UserBase, table=True):
    id: Annotated[uuid.UUID, Field(default_factory=uuid.uuid4, primary_key=True)]
    documents: list["Document"] = Relationship(back_populates="user")
    chat_sessions: list["ChatSession"] = Relationship(back_populates="user")
    password: str


class UserPublic(UserBase):
    id: uuid.UUID


class UserCreate(UserBase):
    password: str


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentBase(SQLModel):
    filename: Annotated[str, Field(index=True)]
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    upload_status: DocumentStatus = DocumentStatus.PENDING


class Document(DocumentBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    file_path: str
    user_id: Annotated[uuid.UUID, Field(foreign_key="user.id")]
    user: User = Relationship(back_populates="documents")
    comment: str = ""


class DocumentPublic(DocumentBase):
    id: uuid.UUID
    user_id: uuid.UUID


class ChatSession(SQLModel, table=True):
    id: Annotated[uuid.UUID, Field(default_factory=uuid.uuid4, primary_key=True)]
    user_id: Annotated[uuid.UUID, Field(foreign_key="user.id")]
    user: User = Relationship(back_populates="chat_sessions")
    messages: list["ChatMessage"] = Relationship(back_populates="session")
    title: str


class ChatMessage(SQLModel, table=True):
    id: Annotated[uuid.UUID, Field(default_factory=uuid.uuid4, primary_key=True)]
    session_id: Annotated[uuid.UUID, Field(foreign_key="chatsession.id")]
    session: ChatSession = Relationship(back_populates="messages")
    role: str
    content: str
    sources: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(SQLModel):
    query: str


class ChatResponse(SQLModel):
    answer: str
    sources: list[dict]
    meta: dict[str, Any] = Field(default_factory=dict)


class NewChatResponse(SQLModel):
    answer: str
    sources: list[dict]
    session_id: uuid.UUID
    title: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ChatSessionPublic(SQLModel):
    id: uuid.UUID
    title: str


class ChatMessagePublic(SQLModel):
    id: uuid.UUID
    role: str
    content: str
    sources: list[dict]
    created_at: datetime
