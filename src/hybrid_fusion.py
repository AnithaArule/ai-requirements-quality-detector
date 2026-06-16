def fuse_outputs(rule_label, ml_label, ml_confidence, llm_label, llm_confidence):
    """
    Hybrid fusion logic combining:
    - rule-based transparency
    - ML confidence
    - LLM semantic reasoning
    """

    # 1. All three agree
    if rule_label == ml_label == llm_label:
        final_label = llm_label
        final_confidence = round((ml_confidence + llm_confidence + 1.0) / 3, 2)
        decision_reason = "All three models agree."

    # 2. Rule-based and LLM agree
    elif rule_label == llm_label:
        final_label = llm_label
        final_confidence = round((llm_confidence + 0.85) / 2, 2)
        decision_reason = "Rule-based and LLM outputs agree, so semantic and rule evidence support the decision."

    # 3. ML and LLM agree
    elif ml_label == llm_label:
        final_label = llm_label
        final_confidence = round((ml_confidence + llm_confidence) / 2, 2)
        decision_reason = "ML and LLM outputs agree."

    # 4. Rule-based and ML agree
    elif rule_label == ml_label and ml_confidence >= 0.60:
        final_label = ml_label
        final_confidence = round((ml_confidence + 0.80) / 2, 2)
        decision_reason = "Rule-based and ML outputs agree with acceptable ML confidence."

    # 5. Trust LLM for semantic categories
    elif llm_label in ["incomplete", "unverifiable"] and llm_confidence >= 0.70:
        final_label = llm_label
        final_confidence = round(llm_confidence, 2)
        decision_reason = "LLM selected because incomplete and unverifiable defects require semantic judgement."

    # 6. Trust high-confidence ML
    elif ml_confidence >= 0.80:
        final_label = ml_label
        final_confidence = round(ml_confidence, 2)
        decision_reason = "ML prediction selected due to high confidence."

    # 7. Trust high-confidence LLM
    elif llm_confidence >= 0.75:
        final_label = llm_label
        final_confidence = round(llm_confidence, 2)
        decision_reason = "LLM prediction selected due to high confidence."

    # 8. Fallback to rule-based
    else:
        final_label = rule_label
        final_confidence = 0.50
        decision_reason = "Rule-based output used as fallback due to uncertainty."

    return final_label, final_confidence, decision_reason