import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

income = np.array([40, 25, 60, 30, 50])
credit = np.array([700, 500, 800, 550, 750])

actual = np.array([1, 0, 1, 0, 1])

w0 = -15
w1 = 0.08
w2 = 0.02

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

predicted = []

for i in range(len(income)):

    z = w0 + w1 * income[i] + w2 * credit[i]

    probability = sigmoid(z)

    if probability >= 0.5:
        predicted.append(1)
    else:
        predicted.append(0)

print("Predictions:", predicted)

print("\nConfusion Matrix")
print(confusion_matrix(actual, predicted))

print("\nAccuracy :", accuracy_score(actual, predicted))
print("Precision:", precision_score(actual, predicted))
print("Recall   :", recall_score(actual, predicted))
print("F1 Score :", f1_score(actual, predicted))