from fastapi import APIRouter

from app.api.routers import auth, users, documents, chat

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
v1_router = APIRouter()
v1_router.include_router(users.router, prefix="/users", tags=["Users"])
v1_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
v1_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(v1_router, prefix="/v1")
