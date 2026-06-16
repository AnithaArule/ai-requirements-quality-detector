import os
import pandas as pd

from src.evaluation import evaluate_model
from src.llm_classifier import classify_requirement_with_llm

DATASET_PATH = "data/processed/requirements_dataset.csv"
OUTPUT_PATH = "outputs/predictions/llm_predictions.csv"


def main():
    os.makedirs("outputs/predictions", exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    categories = []
    confidences = []
    explanations = []
    suggestions = []

    for requirement in df["requirement_sentence"]:
        llm_output = classify_requirement_with_llm(requirement)
        print(f"Requirement: {requirement}")
        print(f"LLM Output: {llm_output}")
        categories.append(llm_output["category"])
        confidences.append(round(llm_output["confidence"], 3))
        explanations.append(llm_output["explanation"])
        suggestions.append(llm_output["suggestion"])

    df["llm_category"] = categories
    df["llm_confidence"] = confidences
    df["llm_explanation"] = explanations
    df["llm_suggestion"] = suggestions

    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\nLLM only Evaluation saved to: {OUTPUT_PATH}")
    evaluate_model(df["true_label"], df["llm_category"])

    print(f"\nPredictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
