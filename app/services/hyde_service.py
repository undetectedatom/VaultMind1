from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.database.models import ChatMessage

HYDE_SYSTEM_PROMPT = (
    "You are an expert technical document writer. "
    "First, read the provided chat history to understand the context of the user's latest query. "
    "Then, write a detailed, factual answer paragraph (3-5 sentences) to the latest query "
    "as if it were excerpted from a technical document. "
    "Do not speculate or add disclaimers. Do not mention that this is hypothetical."
)


async def generate_hypothetical_document(
    chat_history: list[ChatMessage],
    latest_query: str,
    llm: BaseChatModel,
) -> str:
    """Generate a hypothetical answer document for HyDE retrieval using chat history."""

    # Format the past history into a readable string
    history_text = "\n".join(
        [f"{msg.role}: {msg.content}" for msg in chat_history[:-1]]
    )
    if not history_text:
        history_text = "No prior history."

    # Construct the final prompt with context
    contextual_query = (
        f"CHAT HISTORY:\n{history_text}\n\n"
        f"LATEST QUERY:\n{latest_query}\n\n"
        "Based on the history above, write the hypothetical document for the latest query."
    )

    messages = [
        SystemMessage(content=HYDE_SYSTEM_PROMPT),
        HumanMessage(content=contextual_query),
    ]
    response = llm.invoke(messages)
    return response.content.strip()
