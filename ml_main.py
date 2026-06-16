import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


from src.evaluation import evaluate_model
from src.ml_classifier import (
     train_ml_model, predict_with_ml, save_model )


DATASET_PATH = "data/processed/requirements_dataset.csv"
OUTPUT_PATH = "outputs/predictions/ml_predictions.csv"

def main():
    os.makedirs("outputs/predictions", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    X = df["requirement_sentence"]
    y = df["true_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_df = pd.DataFrame({"requirement_sentence": X_train, "true_label": y_train})

    ml_model = train_ml_model(train_df)
    save_model(ml_model)

    predictions = []
    confidences = []
    for requirement in X_test:
        label, confidence = predict_with_ml(ml_model, requirement)
        predictions.append(label)
        confidences.append(round(confidence, 3))
    
    results_df = pd.DataFrame(
        {
            "requirement_sentence": X_test,
            "true_label": y_test,
            "ml_prediction": predictions,
            "ml_confidence": confidences
        }
    )
    results_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\nML only Evaluation saved to: {OUTPUT_PATH}")
    evaluate_model(results_df["true_label"], results_df["ml_prediction"])

    print(f"\nPredictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()



