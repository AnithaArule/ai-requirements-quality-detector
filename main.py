import pandas as pd

from src.evaluation import evaluate_model
from src.rule_based_classifier import classify_requirement, explain_classification
from src.visualization import plot_dataset_distribution, plot_confusion_matrix
# Load the dataset


# Load the dataset
DATASET_PATH = 'data/processed/requirements_dataset.csv'
OUTPUT_PATH = 'outputs/predictions/rule_based_predictions.csv'
DISTRIBUTION_FIGURE_PATH = 'outputs/visualizations/dataset_distribution.png'
CONFUSION_MATRIX_FIGURE_PATH = 'outputs/visualizations/confusion_matrix.png'

def main():
    df = pd.read_csv(DATASET_PATH)
    
    df['predicted_label'] = df['requirement_sentence'].apply(classify_requirement)
    df['explanation'] = df['predicted_label'].apply(explain_classification)

    df.to_csv(OUTPUT_PATH, index=False)
    evaluate_model(df['true_label'], df['predicted_label'])

    plot_dataset_distribution(df, DISTRIBUTION_FIGURE_PATH)
    plot_confusion_matrix(df['true_label'], df['predicted_label'], CONFUSION_MATRIX_FIGURE_PATH)

    print(f"Dataset distribution plot saved to: {DISTRIBUTION_FIGURE_PATH}")
    print(f"Confusion matrix plot saved to: {CONFUSION_MATRIX_FIGURE_PATH}")

    print("\n Predictions saved successfully!")
    print(f"\nPredictions saved to: {OUTPUT_PATH}")
    print(df[["requirement_sentence", "predicted_label", "explanation"]])

if __name__ == "__main__":
    main()


