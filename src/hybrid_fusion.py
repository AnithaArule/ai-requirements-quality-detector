def fuse_outputs(rule_label, ml_label, ml_confidence, llm_label, llm_confidence):
    """
    Confidence-aware fusion logic.
    """

    if ml_label == llm_label and ml_confidence >= 0.60:
        final_label = ml_label
        final_confidence = round((ml_confidence + llm_confidence) / 2, 2)
        decision_reason = "ML and LLM outputs agree with acceptable confidence."

    elif ml_confidence >= 0.75:
        final_label = ml_label
        final_confidence = round(ml_confidence, 2)
        decision_reason = "ML prediction selected due to high confidence."

    elif llm_confidence >= 0.70:
        final_label = llm_label
        final_confidence = round(llm_confidence, 2)
        decision_reason = "LLM reasoning selected due to higher confidence."

    else:
        final_label = rule_label
        final_confidence = 0.50
        decision_reason = "Rule-based label used as fallback due to low confidence."

    return final_label, final_confidence, decision_reason

