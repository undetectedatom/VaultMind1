from pydantic import BaseModel, Field

# ------- RAG Data Schema -------
# Wrappers for use in global settings and downstream RAG pipelines


class ServiceModel(BaseModel):
    provider: str
    base_url: str
    api_key: str
    model: str
    context_window: int | None = None
    dimension: int | None = None


class Service(BaseModel):
    embedding_model: ServiceModel
    router_model: ServiceModel
    generation_models: list[ServiceModel]


# ------- Configuration Schema -------
# Utilities for convenient JSON-based configuration loading and validation


class ModelSpec(BaseModel):
    context_window: int | None = None
    dimension: int | None = None


class ServiceProvider(BaseModel):
    base_url: str
    api_key: str | None = None
    models: dict[str, ModelSpec]


class Role(BaseModel):
    provider: str
    model: str


class Activities(BaseModel):
    embedding: Role
    router: Role
    generation: list[Role]


class HyDEConfig(BaseModel):
    enabled: bool = True
    temperature: float = 0.0


class RoutingConfig(BaseModel):
    enabled: bool = True
    confidence_threshold: float = 0.5


class RetrievalConfig(BaseModel):
    top_k: int = 4
    min_relevance: float = 0.0


class AppConfig(BaseModel):
    providers: dict[str, ServiceProvider]
    activities: Activities
    hyde: HyDEConfig = HyDEConfig()
    routing: RoutingConfig = RoutingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()


# LLM responding format


class CitedSource(BaseModel):
    document_id: str = Field(description="The exact UUID of the document used.")
    filename: str = Field(description="The human-readable name of the file.")


class RAGResponseTemplate(BaseModel):
    answer: str = Field(
        description="The detailed, professional answer to the user's query formatted in Markdown."
    )
    sources: list[CitedSource] = Field(
        description="A list of the specific documents you used to generate the answer. If you used internal knowledge and no documents, return an empty list."
    )
