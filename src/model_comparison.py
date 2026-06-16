import matplotlib.pyplot as plt
import pandas as pd

def plot_model_comparison():

    models = ["Rule-Based", "ML Only", "Hybrid"]

    accuracies = [83.3, 93.3, 83.3]

    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig("outputs/visualizations/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.close()


