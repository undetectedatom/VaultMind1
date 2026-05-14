from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage


async def evaluate_complexity_with_llm(query: str, llm: BaseChatModel) -> float:
    """
    Uses the router LLM to evaluate query complexity asynchronously.
    Returns a float between 0.0 (simple lookup) and 1.0 (complex reasoning).
    """
    system_prompt = (
        "You are a routing metric calculator for a RAG system. "
        "Analyze the user's technical query and determine its complexity. "
        "If it is a simple lookup (e.g., exact error code, syntax rule), output a low score (0.0 to 0.4). "
        "If it requires deep analytical reasoning or comparison, output a high score (0.7 to 1.0). "
        "Output ONLY the raw float value, nothing else."
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]

    try:
        response = await llm.ainvoke(messages)  # Ensure we use ainvoke for async!
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))  # Clamp between 0.0 and 1.0
    except Exception:
        return 0.5  # Default to medium complexity if the LLM fails to format correctly
