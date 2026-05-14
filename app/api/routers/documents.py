import logging
import os
import uuid
from typing import Annotated
from app.database.engine import sql_engine

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    UploadFile,
    status,
)
from sqlmodel import Session, select

from app.api.dependencies import get_cur_user, get_session
from app.database.models import Document, DocumentPublic, DocumentStatus, User
from app.services.document_service import (
    delete_file_from_disk,
    save_file_to_disk,
)
from app.services.vector_database import delete_vectors, vectors_embedding
from app.settings import supported_files

router = APIRouter()


def verify_file(file: UploadFile):
    if (
        file.size == 0
        or os.path.splitext(file.filename)[-1].lower() not in supported_files
    ):
        return False
    return True


def embed_document(doc_id: uuid.UUID):
    with Session(sql_engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            return

        try:
            vectors_embedding(doc.file_path, doc.id, doc.user_id)
            session.add(doc)
            doc.upload_status = DocumentStatus.COMPLETED
        except Exception:
            logging.exception(
                "Document embedding failed: doc_id=%s, file_path=%s",
                doc.id,
                doc.file_path,
            )
            doc.upload_status = DocumentStatus.FAILED
            doc.comment = "File Conversion Failure"

        session.commit()


@router.post("/upload")
async def file_upload(
    files: list[UploadFile],
    background_task: BackgroundTasks,
    session: Annotated[Session, Depends(get_session)],
    user: Annotated[User, Depends(get_cur_user)],
):
    processing_docs = {}
    prohibited_files = []
    for file in files:
        if not verify_file(file):
            prohibited_files.append(file.filename)
            continue
        doc = Document(filename=file.filename, file_path="", user_id=user.id)
        session.add(doc)
        session.commit()
        try:
            doc.file_path = await save_file_to_disk(file, doc.id)
            doc.upload_status = DocumentStatus.PROCESSING
        except Exception as e:
            prohibited_files.append(file.filename)
            doc.upload_status = DocumentStatus.FAILED
            session.delete(doc)
            session.commit()
            logging.error(e)
            continue
        session.commit()
        processing_docs[doc.filename] = doc.id
        background_task.add_task(embed_document, doc.id)
    if not prohibited_files:
        return {
            "message": "Files upload successfully.",
            "processing_docs": processing_docs,
        }
    return {
        "message": "One or several files upload failed.",
        "processing_docs": processing_docs,
        "failed_files": prohibited_files,
    }


@router.get("")
async def get_document_list(
    session: Annotated[Session, Depends(get_session)],
    user: Annotated[User, Depends(get_cur_user)],
):
    statement = select(Document).where(Document.user_id == user.id)
    docs = session.exec(statement).all()
    return [DocumentPublic.model_validate(doc) for doc in docs]


@router.get("/{doc_id}")
async def get_document_status(
    doc_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
    user: Annotated[User, Depends(get_cur_user)],
):
    statement = select(Document).where(Document.id == doc_id)
    doc = session.exec(statement).first()
    if not doc or doc.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document can not be found."
        )
    return {"id": doc.id, "status": doc.upload_status}


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
    user: Annotated[User, Depends(get_cur_user)],
):
    statement = select(Document).where(Document.id == doc_id)
    doc = session.exec(statement).first()
    if not doc or doc.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document can not be found."
        )
    delete_vectors(user.id, doc.id)
    delete_file_from_disk(document=doc)
    session.delete(doc)
    session.commit()
    return {"message": "Document has been deleted.", "status": status.HTTP_200_OK}
