# ml-pipeline

# Breast Cancer Classification Using Machine Learning

## Project Overview
This project implements a machine learning classification pipeline on the Breast Cancer dataset.
Multiple models were trained and compared using cross-validation, and the best model was evaluated
on unseen test data.

## Dataset
Breast Cancer Dataset (Scikit-learn)  
https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset

## Models Used
- K-Nearest Neighbors (KNN)
- Linear Discriminant Analysis (LDA)
- Decision Tree (CART)
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)

## Evaluation
- 10-fold cross-validation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Visual comparison using box plots

## Final Model
Logistic Regression was selected as the final model based on accuracy and stability.

## How to Run the Code

### Prerequisites
- Python 3.11+ (Python 3.13 also works)
- pip (Python package manager)

### Steps to Run

1. Clone the repository or download the files:
   ```bash
   git clone https://github.com/YOUR_USERNAME/breast-cancer-ml-assignment.git
   ```
   ```bash
   cd breast-cancer-ml-assignment
   ```
   ```bash
   python -m venv ml_env
   ```
   ```bash
   ml_env\Scripts\activate
   ```
   ```bash
   pip install pandas matplotlib scikit-learn
   ```
   ```bash
   python ml.py
   ```





## Tools
- Python
- scikit-learn
- pandas
- matplotlib
