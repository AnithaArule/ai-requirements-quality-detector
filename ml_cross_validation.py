import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.evaluation import evaluate_model


DATASET_PATH = "data/processed/requirements_dataset.csv"


def main():
    df = pd.read_csv(DATASET_PATH)

    X = df["requirement_sentence"]
    y = df["true_label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )

    print("\nML Cross-Validation Evaluation")
    evaluate_model(y, predictions)


if __name__ == "__main__":
    main()