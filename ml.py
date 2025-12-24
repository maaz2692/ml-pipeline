import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
print(df.shape)
print(df.groupby('target').size())

# Visualize data
df.iloc[:, :4].plot(kind='box', subplots=True, layout=(2,2))
plt.show(block=False)
plt.pause(5)
plt.close()
# Split data
X = df.drop('target', axis=1)
Y = df['target']

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.2, random_state=1
)

# Define models
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LR', LogisticRegression(max_iter=200)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate models
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    print(f"{name} - mean: {cv_results.mean():.3f}, std: {cv_results.std():.3f}")

# Boxplot
plt.boxplot(results, labels=names)
plt.title("Algorithm Comparison")
plt.ylabel("Accuracy")
plt.show()

# Train best model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Final evaluation
print("Accuracy:", accuracy_score(Y_validation, predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_validation, predictions))
print("Classification Report:\n", classification_report(Y_validation, predictions))


