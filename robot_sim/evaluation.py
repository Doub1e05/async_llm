"""Task scoring independent of the model and navigation stack."""
import math


def score_task(expected, selected):
    if expected is None:
        return None
    return list(expected) == list(selected)


def task_request(task, completed):
    return {"instruction": task["instruction"], "completed_markers": list(completed)}


def validate_decision(response):
    probabilities = response.get("probabilities", {})
    if not probabilities or not all(isinstance(value, (int, float)) and math.isfinite(value)
                                    and 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("Non-finite or missing policy probabilities")
    if abs(sum(probabilities.values()) - 1) > 1e-4:
        raise ValueError("Policy probabilities must sum to one")
