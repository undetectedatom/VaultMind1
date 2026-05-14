from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings


def get_embeddings(
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> Embeddings:
    if provider.startswith("doubao"):
        from external_models.doubao import DoubaoMultimodalEmbeddings

        return DoubaoMultimodalEmbeddings(
            api_key, base_url, endpoint_id=model, dimensions=kwargs.get("dimension")
        )

    elif provider.startswith("openai"):
        from langchain_openai import OpenAIEmbeddings

        if not api_key:
            raise ValueError("OpenAI provider requires an 'api_key'.")
        return OpenAIEmbeddings(api_key=api_key, model=model)

    elif provider.startswith("ollama"):
        from langchain_community.embeddings import OllamaEmbeddings

        host = base_url or "http://localhost:11434"
        return OllamaEmbeddings(base_url=host, model=model)

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'"
            "We only accept provider names as suffixes in your configuration."
        )


def get_chat_model(
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> BaseChatModel:
    if provider.startswith("openai"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=kwargs.get("temperature", 0.2),
        )

    elif provider.startswith("doubao"):
        from langchain_openai import ChatOpenAI

        if not api_key:
            raise ValueError("Doubao provider requires an 'api_key'.")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=kwargs.get("temperature", 0.2),
        )

    elif provider.startswith("ollama"):
        from langchain_community.chat_models import ChatOllama

        host = base_url or "http://localhost:11434"
        return ChatOllama(
            model=model, base_url=host, temperature=kwargs.get("temperature", 0.2)
        )

    else:
        raise ValueError(f"Unsupported chat provider: '{provider}'")
