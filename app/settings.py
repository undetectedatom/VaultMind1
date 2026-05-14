import json
import os
import secrets
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.external_model import (
    AppConfig,
    HyDEConfig,
    RetrievalConfig,
    Role,
    RoutingConfig,
    Service,
    ServiceModel,
)


class Settings(BaseSettings):
    vector_database_path: str
    user_documents_path: str
    token_secret_key: str
    token_encryption_algorithm: str
    dummy_pwd: str = secrets.token_urlsafe(32)
    rag_config_path: str = "rag_config.json"

    # API keys are auto-loaded from .env by provider name (e.g. doubao_api_key, openai_api_key)
    # via the dynamic getattr loop in model_post_init

    services: Service | None = None
    hyde: HyDEConfig = HyDEConfig()
    routing: RoutingConfig = RoutingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.getcwd(), ".env"), extra="allow"
    )

    def model_post_init(self, __context):
        if not os.path.exists(self.rag_config_path):
            return

        with open(self.rag_config_path, "r") as f:
            raw_config = AppConfig(**json.load(f))

        # Map provider names to Settings attribute names (hyphens -> underscores)
        for provider_name, provider_data in raw_config.providers.items():
            attr_name = f"{provider_name.replace('-', '_')}_api_key"
            env_api_key = getattr(self, attr_name, None)
            if env_api_key:
                provider_data.api_key = env_api_key

        def resolve_model(role: Role) -> ServiceModel:
            provider = raw_config.providers[role.provider]
            specs = provider.models[role.model]
            return ServiceModel(
                provider=role.provider,
                base_url=provider.base_url,
                api_key=provider.api_key or "",
                model=role.model,
                context_window=specs.context_window,
                dimension=specs.dimension,
            )

        self.services = Service(
            embedding_model=resolve_model(raw_config.activities.embedding),
            router_model=resolve_model(raw_config.activities.router),
            generation_models=[
                resolve_model(role) for role in raw_config.activities.generation
            ],
        )

        self.hyde = raw_config.hyde
        self.routing = raw_config.routing
        self.retrieval = raw_config.retrieval


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


supported_files = set((".md", ".html", ".pdf", ".txt", ".doc", ".docx", ".png", ".jpg"))
