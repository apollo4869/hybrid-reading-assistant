from typing import Dict, List, Tuple


def get_top_distances(results: Dict) -> List[float]:
    distances = results.get("distances", [[]])[0]
    return distances if distances else []


def route_answer_mode(results: Dict) -> Tuple[str, str]:
    """
    Return:
        mode: one of ["Library-grounded", "Hybrid", "General"]
        reason: short explanation
    """

    distances = get_top_distances(results)

    if not distances:
        return "General", "No retrieval results found."

    top1 = distances[0]
    avg_top3 = sum(distances[:3]) / min(3, len(distances))

    # Lower distance usually means more similar in Chroma's default behavior.
    if top1 < 1.35 and avg_top3 < 1.42:
        return "Library-grounded", (
            f"Top retrieval is strong (top1={top1:.4f}, avg_top3={avg_top3:.4f})."
        )
    elif top1 < 1.55 and avg_top3 < 1.65:
        return "Hybrid", (
            f"Retrieval is moderately relevant (top1={top1:.4f}, avg_top3={avg_top3:.4f})."
        )
    else:
        return "General", (
            f"Retrieval is weak (top1={top1:.4f}, avg_top3={avg_top3:.4f})."
        )