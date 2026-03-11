from config import BEAR_COST_PER_1M_TOKENS_REMOVED


def compute_cost(model_config: dict, input_tokens: int, output_tokens: int,
                 tokens_removed: int = 0) -> dict:
    """Compute USD cost breakdown for a single request."""
    input_cost = input_tokens * model_config["cost_per_1m_input"] / 1_000_000
    output_cost = output_tokens * model_config["cost_per_1m_output"] / 1_000_000
    bear_cost = tokens_removed * BEAR_COST_PER_1M_TOKENS_REMOVED / 1_000_000

    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_llm_cost_usd": input_cost + output_cost,
        "bear_cost_usd": bear_cost,
        "total_cost_usd": input_cost + output_cost + bear_cost,
    }
