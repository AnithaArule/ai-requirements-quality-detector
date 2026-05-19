ambiguous_words = ["fast", "easy", "flexible", "efficient", "simple", "robust"]

unverifiable_words = ["user-friendly", "attractive", "smooth","intuitive","convenient"]


def classify_requirement(requirement_text):
    requirement_text = requirement_text.lower()

    for word in ambiguous_words:
        if word in requirement_text:
            return "ambiguous"
        
    for word in unverifiable_words:
        if word in requirement_text:
            return "unverifiable"
        
    if(len(requirement_text.split()) <6):
        return "incomplete"
    
    return "well-defined"


def explain_classification(label):
    label = label.lower().replace(":", "").strip()

    explanations = {
        "ambiguous": "The requirement contains vague terms that can be interpreted in multiple ways.",
        "unverifiable": "The requirement includes subjective terms that cannot be objectively measured.",
        "incomplete": "The requirement is too short and lacks necessary details.",
        "well-defined": "The requirement is clear, specific, and can be objectively evaluated."
    }
    return explanations.get(label, "\n No explanation available for this classification.")

