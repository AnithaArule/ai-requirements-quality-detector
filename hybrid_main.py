import os
import pandas as pd

from src.rule_based_classifier import classify_requirement
from src.ml_classifier import train_ml_model, predict_with_ml, save_model
from src.llm_reasoner import generate_llm_reasoning
from src.hybrid_fusion import fuse_outputs
from src.evaluation import evaluate_model


DATASET_PATH = "data/processed/requirements_dataset.csv"
OUTPUT_PATH = "outputs/predictions/hybrid_predictions.csv"


def main():
    os.makedirs("outputs/predictions", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    df = pd.read_csv(DATASET_PATH)

    ml_model = train_ml_model(df)
    save_model(ml_model)

    final_labels = []
    final_confidences = []
    explanations = []
    revision_suggestions = []
    decision_reasons = []

    for requirement in df["requirement_sentence"]:
        rule_label = classify_requirement(requirement)

        ml_label, ml_confidence = predict_with_ml(ml_model, requirement)

        llm_output = generate_llm_reasoning(
            requirement,
            ml_label,
            ml_confidence
        )

        final_label, final_confidence, decision_reason = fuse_outputs(
            rule_label=rule_label,
            ml_label=ml_label,
            ml_confidence=ml_confidence,
            llm_label=llm_output["llm_label"],
            llm_confidence=llm_output["llm_confidence"]
        )

        final_labels.append(final_label)
        final_confidences.append(final_confidence)
        explanations.append(llm_output["llm_explanation"])
        revision_suggestions.append(llm_output["revision_suggestion"])
        decision_reasons.append(decision_reason)

    df["hybrid_label"] = final_labels
    df["hybrid_confidence"] = final_confidences
    df["hybrid_explanation"] = explanations
    df["revision_suggestion"] = revision_suggestions
    df["fusion_decision_reason"] = decision_reasons

    df.to_csv(OUTPUT_PATH, index=False)

    print("Hybrid ML-LLM Evaluation:")
    evaluate_model(df["true_label"], df["hybrid_label"])

    print(f"\nHybrid predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()