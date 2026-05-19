from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(true_labels, predicted_labels):
    LABELS = ["ambiguous", "unverifiable", "incomplete", "well-defined"]

    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, labels=LABELS, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels, labels=LABELS))
    
    print("Accuracy Score:")
    print(accuracy_score(true_labels, predicted_labels))

