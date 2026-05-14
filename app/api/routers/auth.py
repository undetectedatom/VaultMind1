from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pwdlib import PasswordHash
from sqlmodel import Session, select

from app.api.dependencies import get_session
from app.database.models import User
from app.schemas.token import Token
from app.settings import settings

router = APIRouter()


password_hash = PasswordHash.recommended()


def authenticate_user(
    session: Session,
    username_or_email: str,
    password: str,
):
    statement = select(User).where(
        (User.username == username_or_email) | (User.email == username_or_email)
    )
    user = session.exec(statement).first()
    if user:
        if password_hash.verify(password, user.password):
            return user
    else:
        try:
            password_hash.verify(password, settings.dummy_pwd)
        except Exception:
            pass
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username, email, or password.",
        headers={"WWW-Authenticate": "bearer"},
    )


def create_token(data_dict: dict, expires_delta: timedelta = timedelta(days=365)):
    data_dict = data_dict.copy()
    expires = datetime.now(timezone.utc) + expires_delta
    data_dict.update({"exp": expires})
    encoded_jwt = jwt.encode(
        payload=data_dict,
        key=settings.token_secret_key,
        algorithm=settings.token_encryption_algorithm,
    )
    return encoded_jwt


@router.post("/token")
def users_login(
    session: Annotated[Session, Depends(get_session)],
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    user = authenticate_user(session, form_data.username, form_data.password)
    access_token = create_token({"sub": str(user.id)})
    return Token(access_token=access_token, token_type="bearer")
