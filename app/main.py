from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.settings import settings
from app.database.engine import create_db_and_tables, init_chroma
from app.api.routers import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    init_chroma(settings.vector_database_path)
    yield


app = FastAPI(dependencies=[], lifespan=lifespan)
app.include_router(api_router, prefix="/api", dependencies=[])
