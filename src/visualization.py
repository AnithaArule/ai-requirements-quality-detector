import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

LABELS = ["ambiguous", "unverifiable", "incomplete", "well-defined"]

def plot_dataset_distribution(df, output_path):
    counts = df["true_label"].value_counts().reindex(LABELS, fill_value=0)

    plt.figure(figsize=(6, 5))
    counts.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen', 'orange'])
    plt.title("Distribution of Requirement labels in the Dataset")
    plt.xlabel("Requirement quality label")
    plt.ylabel("Number of requirements")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(true_labels, predicted_labels, output_path):
    cm = confusion_matrix(true_labels, predicted_labels, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax,values_format='d', colorbar=False)
    ax.set_title("Rule based baseline Confusion Matrix")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)    



