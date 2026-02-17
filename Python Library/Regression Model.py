# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
data = pd.read_csv('Data.csv')

# Split data into independent and dependent variables
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

#Create training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Train logistic regression classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

# Predict probabilities or make predictions on the whole dataset if needed
# (not strictly required for plotting, but useful for final model assessment)
preds = classifier.predict(X)


# Evaluate model performance

# Optionally show some predictions
print("Some test predictions:", Y_pred[:10])

from sklearn.metrics import confusion_matrix, accuracy_score
Y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print(accuracy_score(Y_test, Y_pred))

# Visualize results
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X_set, Y_set, title):
    X1, X2 = X_set[:, 0], X_set[:, 1]
    x1_min, x1_max = X1.min() - 1, X1.max() + 1
    x2_min, x2_max = X2.min() - 1, X2.max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                           np.arange(x2_min, x2_max, 0.01))
    Z = classifier.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.figure(figsize=(10, 10))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=ListedColormap(('red', 'green')))

    plt.scatter(X_set[Y_set == 0, 0], X_set[Y_set == 0, 1],
                color='red', edgecolor='k', label='Class 0')
    plt.scatter(X_set[Y_set == 1, 0], X_set[Y_set == 1, 1],
                color='green', edgecolor='k', label='Class 1')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# plot training set
plot_decision_boundary(X_train, Y_train, 'Logistic Regression (Training set)')
# plot test set
plot_decision_boundary(X_test, Y_test, 'Logistic Regression (Test set)')

