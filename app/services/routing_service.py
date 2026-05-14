from app.schemas.external_model import ServiceModel


def select_generation_model(
    query_complexity: float,
    retrieval_confidence: float,
    generation_models: list[
        ServiceModel
    ],  # Array input, sorted from simplest [0] to smartest [-1]
) -> tuple[ServiceModel, str]:

    num_models = len(generation_models)

    # Fallback if config is empty
    if num_models == 0:
        raise ValueError("No generation models configured.")
    if num_models == 1:
        return generation_models[0], "fallback_single_model"

    # 1. Calculate Unified Difficulty (0.0 to 1.0)
    # High complexity OR low confidence drives this score closer to 1.0
    task_difficulty = max(query_complexity, 1.0 - retrieval_confidence)

    # 2. Map the difficulty to an array index
    # We multiply by num_models and floor it (int).
    # Example for 3 models: D=0.1 -> index 0. D=0.5 -> index 1. D=0.9 -> index 2.
    target_index = int(task_difficulty * num_models)

    # Edge case: if difficulty is exactly 1.0, it would equal num_models (out of bounds)
    target_index = min(target_index, num_models - 1)

    # 3. Select the model
    selected_model = generation_models[target_index]

    reason = (
        f"routed_to_index_{target_index} "
        f"(difficulty: {task_difficulty:.2f} | "
        f"c={query_complexity:.2f}, conf={retrieval_confidence:.2f})"
    )

    return selected_model, reason
