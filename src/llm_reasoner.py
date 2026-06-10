def generate_llm_reasoning(requirement_text, ml_label, ml_confidence):
    explanations = {
        "ambiguous": (
            "The requirement appears ambiguous because it contains wording "
            "that may be interpreted differently by different stakeholders."
        ),
        "incomplete": (
            "The requirement appears incomplete because it may be missing "
            "important details such as actors, conditions, constraints, or measurable outcomes."
        ),
        "unverifiable": (
            "The requirement appears unverifiable because it includes subjective wording "
            "that is difficult to test objectively."
        ),
        "well-defined": (
            "The requirement appears well-defined because it is specific, measurable, "
            "and can be objectively evaluated."
        )
    }

    revision_suggestions = {
        "ambiguous": "Replace vague terms with measurable criteria.",
        "incomplete": "Add missing actors, conditions, constraints, or expected outcomes.",
        "unverifiable": "Rewrite the requirement using objective and testable measures.",
        "well-defined": "No major revision is required."
    }

    return {
        "llm_label": ml_label,
        "llm_confidence": round(float(ml_confidence), 2),
        "llm_explanation": explanations.get(ml_label, "No explanation available."),
        "revision_suggestion": revision_suggestions.get(ml_label, "No revision suggestion available.")
    }