from typing import Annotated
import uuid

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select

from app.database.engine import sql_engine
from app.database.models import User
from app.settings import settings


def get_session():
    with Session(sql_engine) as session:
        yield session


# Extract token from HTTP header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


async def get_cur_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[Session, Depends(get_session)],
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            jwt=token,
            key=settings.token_secret_key,
            algorithms=settings.token_encryption_algorithm,
        )
        user_id = uuid.UUID(payload["sub"])
        statement = select(User).where(User.id == user_id)
        user = session.exec(statement).first()
        return user

    except Exception:
        raise credentials_exception
