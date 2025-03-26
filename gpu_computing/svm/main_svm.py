import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load MNIST
digits = datasets.load_digits()

# Split it in train test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Create a SVM classifier
clf = svm.SVC(kernel='rbf', gamma=0.001, C=100)

# Train it
clf.fit(X_train, y_train)

# Evaluate it
y_pred = clf.predict(X_test)

# Afficher le rapport de classification
print("Rapport de classification :\n", metrics.classification_report(y_test, y_pred))

# Afficher la matrice de confusion
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
disp.figure_.suptitle("Matrice de confusion")
plt.show()

# Visualiser quelques prédictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, y_pred):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f"Prediction: {prediction}")
plt.show()
