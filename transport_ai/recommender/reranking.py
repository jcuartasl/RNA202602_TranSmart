from .coldstart import parse_preferences

def rerank_by_preferences(scores, catalog, preferences, alpha=0.02):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha debe estar entre 0 y 1")
    preferred = parse_preferences(preferences)
    return {
        destination_id: float(score) + alpha * (
            catalog[destination_id]["type"].strip().lower() in preferred
        )
        for destination_id, score in scores.items()
    }
