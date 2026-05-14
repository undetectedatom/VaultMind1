import chromadb
from sqlmodel import SQLModel, create_engine

# --- SQL Database ---
sqlite_file_name = "database.db"
connect_args = {"check_same_thread": False}
sql_engine = create_engine(f"sqlite:///{sqlite_file_name}", connect_args=connect_args)


def create_db_and_tables():
    # Models must be imported so SQLModel registers them in metadata
    import app.database.models  # noqa: F401

    SQLModel.metadata.create_all(sql_engine)


# --- Vector Database Singleton ---
chroma_client = None


def init_chroma(db_path: str):
    global chroma_client
    chroma_client = chromadb.PersistentClient(path=db_path)
