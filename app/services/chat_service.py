import asyncio
import uuid

from fastapi import HTTPException, status
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlmodel import Session, select

from app.database import engine
from app.database.models import ChatMessage, Document
from app.schemas.external_model import RAGResponseTemplate
from app.services.complexity_service import evaluate_complexity_with_llm
from app.services.hyde_service import generate_hypothetical_document
from app.services.vector_database import get_bm25_retriever
from app.settings import settings
from external_models import get_chat_model, get_embeddings


async def generate_title(query: str) -> str:
    """Uses the fast 'router' model to generate a short chat title."""
    router_config = settings.services.router_model.model_dump()
    llm = get_chat_model(**router_config, temperature=0.5)

    instruction = SystemMessage(
        "Generate a very concise title for user's query subject. Do not use quotes or punctuation. Use the same language as query."
    )
    prompt = HumanMessage(query)
    response = llm.invoke([instruction, prompt])
    return response.content.strip()


async def knowledge_query(
    chat_history: list[ChatMessage], user_id: uuid.UUID
) -> tuple[str, list[str], dict]:
    """Executes the RAG pipeline with Async Parallelization and Caching"""
    metadata = {}

    chat_history = sorted(chat_history, key=lambda x: x.created_at)
    latest_query = chat_history[-1].content

    embedding_config = settings.services.embedding_model.model_dump()
    embeddings_model = get_embeddings(**embedding_config)

    vectorstore = Chroma(
        client=engine.chroma_client,
        collection_name=f"user-{user_id}",
        embedding_function=embeddings_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # --- 1. ASYNC PARALLEL EXECUTION (HyDE + Complexity) ---
    search_text = latest_query
    query_complexity = 0.5

    if settings.routing.enabled or settings.hyde.enabled:
        router_config = settings.services.router_model.model_dump()
        router_llm = get_chat_model(
            **router_config, temperature=settings.hyde.temperature
        )

        # Define tasks (but don't wait for them yet)
        task_hyde = (
            generate_hypothetical_document(chat_history, latest_query, router_llm)
            if settings.hyde.enabled
            else None
        )
        task_complexity = (
            evaluate_complexity_with_llm(latest_query, router_llm)
            if settings.routing.enabled
            else None
        )

        # Filter out None tasks and run them concurrently!
        tasks = [t for t in [task_hyde, task_complexity] if t is not None]

        if tasks:
            results = await asyncio.gather(*tasks)

            # Map results back based on what was enabled
            idx = 0
            if settings.hyde.enabled:
                search_text = f"{latest_query}\n{results[idx]}"
                metadata["hyde_document"] = results[idx]
                idx += 1
            if settings.routing.enabled:
                query_complexity = results[idx]
                metadata["llm_complexity_score"] = query_complexity

    # --- 2. HYBRID RETRIEVAL (Using Cache for BM25) ---
    vector_docs = []
    retrieval_confidence = 1.0

    configured_top_k = settings.retrieval.top_k

    if settings.routing.enabled:
        results = vectorstore.similarity_search_with_relevance_scores(
            search_text, k=configured_top_k
        )
        vector_docs = [doc for doc, score in results]
        scores = [score for doc, score in results]

        retrieval_confidence = max(scores) if scores else 0.0
        metadata["retrieval_confidence"] = retrieval_confidence
    else:
        vector_docs = vectorstore.similarity_search(search_text, k=configured_top_k)

    bm25_retriever = get_bm25_retriever(
        vectorstore=vectorstore,
        user_id=str(user_id),
        k=configured_top_k,
    )
    bm25_docs = bm25_retriever.invoke(search_text) if bm25_retriever else []

    metadata["vector_retrieved_count"] = len(vector_docs)
    metadata["bm25_retrieved_count"] = len(bm25_docs)

    # Apply Reciprocal Rank Fusion (RRF)
    doc_scores = {}

    def add_to_fusion(docs, weight=60):
        for rank, doc in enumerate(docs):
            doc_hash = doc.page_content
            if doc_hash not in doc_scores:
                doc_scores[doc_hash] = {"doc": doc, "score": 0.0}
            doc_scores[doc_hash]["score"] += 1.0 / (weight + rank)

    add_to_fusion(vector_docs)
    add_to_fusion(bm25_docs)

    reranked_results = sorted(
        doc_scores.values(), key=lambda x: x["score"], reverse=True
    )
    retrieved_docs = [item["doc"] for item in reranked_results[:configured_top_k]]

    # --- 3. FETCH REAL FILENAMES FROM SQL DATABASE FIRST ---
    doc_id_strings = list(
        set(
            [
                doc.metadata.get("document_id")
                for doc in retrieved_docs
                if doc.metadata.get("document_id")
            ]
        )
    )

    id_to_filename = {}  # Dictionary to map UUIDs to actual filenames
    if doc_id_strings:
        valid_uuids = []
        for d_id in doc_id_strings:
            try:
                valid_uuids.append(uuid.UUID(d_id))
            except ValueError:
                pass

        if valid_uuids:
            with Session(engine.sql_engine) as session:
                statement = select(Document.id, Document.filename).where(
                    Document.id.in_(valid_uuids)
                )
                db_results = session.exec(statement).all()
                for doc_id, filename in db_results:
                    id_to_filename[str(doc_id)] = filename

    # --- 4. BUILD TAGGED CONTEXT FOR THE LLM ---
    context_chunks = []
    retrieved_filenames = []
    retrieved_document_ids = []

    for doc in retrieved_docs:
        doc_id = doc.metadata.get("document_id")
        filename = id_to_filename.get(str(doc_id), "Unknown Document")

        # Tag each chunk with BOTH filename and ID
        context_chunks.append(
            f"--- [File: {filename} | ID: {doc_id}] ---\n{doc.page_content}"
        )

        retrieved_filenames.append(filename)
        retrieved_document_ids.append(str(doc_id))

    metadata["retrieved_filenames"] = retrieved_filenames
    metadata["retrieved_document_ids"] = retrieved_document_ids
    metadata["retrieved_context_count"] = len(retrieved_docs)
    context_text = "\n\n".join(context_chunks)

    # --- 5. DYNAMIC ROUTING ---
    if settings.routing.enabled:
        from app.services.routing_service import select_generation_model

        selected_model, route_reason = select_generation_model(
            query_complexity=query_complexity,
            retrieval_confidence=retrieval_confidence,
            generation_models=settings.services.generation_models,
        )
        metadata["route_reason"] = route_reason
        metadata["selected_model"] = selected_model.model
        generation_config = selected_model.model_dump()
    else:
        generation_config = settings.services.generation_models[-1].model_dump()

    llm = get_chat_model(**generation_config)

    # --- 6. BIND THE TEMPLATE TO THE LLM ---
    # try:
    #     # Change method to "json_mode"
    #     structured_llm = llm.with_structured_output(
    #         RAGResponseTemplate, method="json_mode", include_raw=True
    #     )
    # except TypeError:
    #     structured_llm = llm.with_structured_output(
    #         RAGResponseTemplate, method="json_mode"
    #     )

    # --- 7. UPDATED SYSTEM INSTRUCTION ---
    system_instruction = (
        "You are a professional technical assistant. "
        "Review the provided context snippets. Only use the snippets that are relevant to the user's query. "
        "CRITICAL RULES:\n"
        "1. BE PROFESSIONAL: Answer directly and concisely. Do not exceed 3 paragraphs. Do not output conversational filler.\n"
        "2. CITATION RIGOR: If the context snippets do NOT contain the exact information needed to answer the prompt, YOU MUST leave the sources list completely empty.\n"
        "3. NO FORCED MATCHES: Do not force an answer if the provided context is only tangentially related.\n\n"
        "If no snippets are relevant, answer from your internal knowledge and leave the sources list empty.\n\n"
        "IMPORTANT: You MUST respond purely in valid JSON format. "
        "Your output must perfectly match this exact JSON structure:\n"
        "{\n"
        '  "answer": "Your detailed markdown answer here...",\n'
        '  "sources": [\n'
        "    {\n"
        '      "document_id": "extract the ID from the context header",\n'
        '      "filename": "extract the File from the context header"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"CONTEXT:\n{context_text}"
    )

    messages = [SystemMessage(content=system_instruction)]
    for msg in chat_history[:-1]:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=latest_query))

    # --- 8. GENERATE STRUCTURED OUTPUT ---
    import json
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.exceptions import OutputParserException

    raw_response = await llm.ainvoke(messages)

    parser = PydanticOutputParser(pydantic_object=RAGResponseTemplate)
    parsed_response = None

    try:
        parsed_response = parser.parse(raw_response.content)
    except OutputParserException:
        clean_content = raw_response.content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]

        try:
            parsed_data = json.loads(clean_content.strip())
            parsed_response = RAGResponseTemplate(**parsed_data)
        except Exception:
            metadata["parse_error"] = "json_decode_failed"
            print(f"Failed to parse LLM output. Raw content:\n{raw_response.content}")
            raise HTTPException(status_code=status.HTTP_417_EXPECTATION_FAILED)

    usage = {}
    if hasattr(raw_response, "response_metadata"):
        usage = raw_response.response_metadata.get("token_usage", {})
    if not usage and hasattr(raw_response, "usage_metadata"):
        usage = raw_response.usage_metadata

    if usage:
        metadata.setdefault("prompt_tokens", usage.get("prompt_tokens", 0))
        metadata.setdefault("completion_tokens", usage.get("completion_tokens", 0))

    answer_content = parsed_response.answer
    final_sources = [
        {"document_id": src.document_id, "filename": src.filename}
        for src in parsed_response.sources
    ]

    return answer_content, final_sources, metadata
