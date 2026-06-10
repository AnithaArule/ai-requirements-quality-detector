import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


MODEL_PATH = "outputs/models/ml_classifier_model.joblib"


def train_ml_model(df):
    X = df["requirement_sentence"]
    y = df["true_label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model


def predict_with_ml(model, requirement_text):
    prediction = model.predict([requirement_text])[0]
    probabilities = model.predict_proba([requirement_text])[0]

    confidence_score = max(probabilities)
    return prediction, confidence_score


def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    return joblib.load(MODEL_PATH)


   