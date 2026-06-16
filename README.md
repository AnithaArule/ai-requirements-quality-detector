# AI-Assisted Requirements Quality Detector

## Overview

This project investigates the use of Artificial Intelligence for detecting quality defects in natural-language software requirements.

The research focuses on identifying four requirement quality categories:

- Ambiguous
- Incomplete
- Unverifiable
- Well-Defined

The project evaluates and compares multiple approaches:

1. Rule-Based Classification
2. Machine Learning Classification (TF-IDF + Logistic Regression)
3. Large Language Model (OpenAI GPT)
4. Hybrid ML–LLM Classification

The objective is to determine whether combining machine learning and large language models can improve automated requirements quality assessment.

---

## Research Objectives

The project addresses the following research questions:

1. Can a hybrid ML–LLM system detect requirement quality issues more effectively than standalone approaches?

2. Can LLM-based reasoning improve explanation quality and requirement analysis?

3. How can explainable AI techniques support software requirements engineering?

---

## Project Structure

```text
ai-requirements-quality-detector/
│
├── data/
│   └── processed/
│       └── requirements_dataset.csv
│
├── outputs/
│   ├── predictions/
│   ├── models/
│   └── visualizations/
│
├── src/
│   ├── evaluation.py
│   ├── rule_based_classifier.py
│   ├── ml_classifier.py
│   ├── llm_classifier.py
│   ├── hybrid_fusion.py
│   └── visualization.py
│
├── main.py
├── ml_main.py
├── ml_cross_validation.py
├── llm_main.py
├── hybrid_main.py
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Dataset

The dataset consists of labelled software requirements classified into:

| Label | Description |
|---------|---------|
| Ambiguous | Requirement contains vague wording |
| Incomplete | Requirement lacks necessary information |
| Unverifiable | Requirement contains subjective language |
| Well-Defined | Requirement is clear, measurable and testable |

Example:

```text
The system should be fast. → Ambiguous

The system generates reports. → Incomplete

The interface must be user-friendly. → Unverifiable

The system shall respond within 2 seconds. → Well-Defined
```

---

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd ai-requirements-quality-detector
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate Virtual Environment

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## OpenAI Configuration

Create a `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

---

## Running the Project

### Rule-Based Baseline

```bash
python main.py
```

Output:

```text
outputs/predictions/rule_based_predictions.csv
```

---

### Machine Learning Model

```bash
python ml_main.py
```

Output:

```text
outputs/predictions/ml_predictions.csv
```

---

### ML Cross-Validation

```bash
python ml_cross_validation.py
```

Used to evaluate model generalisation on small datasets.

---

### LLM Classification

```bash
python llm_main.py
```

Output:

```text
outputs/predictions/llm_predictions.csv
```

---

### Hybrid ML–LLM Model

```bash
python hybrid_main.py
```

Output:

```text
outputs/predictions/hybrid_predictions.csv
```

---

## Evaluation Metrics

The project evaluates model performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Current Results

| Model | Accuracy |
|---------|---------:|
| Rule-Based | 83.3% |
| ML (Cross Validation) | 50.0% |
| LLM Only | 83.3% |
| Hybrid ML–LLM | 86.7% |

The Hybrid ML–LLM model currently achieves the best performance.

---

## Visualisations

Generated visualisations include:

- Dataset Distribution
- Confusion Matrix
- Model Accuracy Comparison

Output location:

```text
outputs/visualizations/
```

---

## Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib
- OpenAI GPT
- TF-IDF
- Logistic Regression

---

## Dissertation Context

This project was developed as part of the MSc Data Science and Artificial Intelligence programme.

The research follows a Design Science Research methodology and investigates explainable AI techniques for software requirements engineering.

---

## Future Work

Potential future enhancements include:

- Larger labelled datasets
- Transformer-based classifiers
- Retrieval-Augmented Generation (RAG)
- Explainability evaluation
- Multi-LLM comparison
- Real-time requirements quality assistant

---

## Author

**Anitha Arulelango**

MSc Advanced Computer Science

University of Liverpool

---

## License

This project is provided for academic and research purposes.
