import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelBinarizer


# Load MNIST
digits = datasets.load_digits()

# Split it in train test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Create a SVM classifier
baseline_clf = svm.SVC(kernel='rbf', gamma=0.001, C=100)

# Measure training time
start_time = time.time()
baseline_clf.fit(X_train, y_train)
baseline_train_time = time.time() - start_time

# Measure inference time
start_time = time.time()
baseline_pred = baseline_clf.predict(X_test)
baseline_inference_time = time.time() - start_time

# Compute classification report
baseline_report = metrics.classification_report(y_test, baseline_pred)

# Compute confusion matrix
baseline_cm = metrics.confusion_matrix(y_test, baseline_pred)

# Afficher le rapport de classification
# print("Rapport de classification :\n", metrics.classification_report(y_test, y_pred))

### ===== Optimized Model (GridSearchCV) ===== ###
param_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001]}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1)

# Measure training time
start_time = time.time()
grid_search.fit(X_train, y_train)
optimized_train_time = time.time() - start_time

# Get best model from grid search
optimized_clf = grid_search.best_estimator_

# Measure inference time
start_time = time.time()
optimized_pred = optimized_clf.predict(X_test)
optimized_inference_time = time.time() - start_time

# Compute classification report
optimized_report = metrics.classification_report(y_test, optimized_pred)

# Compute confusion matrix
optimized_cm = metrics.confusion_matrix(y_test, optimized_pred)


### ===== Results Comparison ===== ###
print("\n========= Baseline Model =========")
print(f"Training Time: {baseline_train_time:.4f} sec")
print(f"Inference Time: {baseline_inference_time:.4f} sec")
print("Classification Report:\n", baseline_report)

print("\n========= Optimized Model =========")
print(f"Training Time: {optimized_train_time:.4f} sec")
print(f"Inference Time: {optimized_inference_time:.4f} sec")
print("Classification Report:\n", optimized_report)


# Plot confusion matrices side by side
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Baseline model confusion matrix
# metrics.ConfusionMatrixDisplay(baseline_cm).plot(ax=axes[0])
# axes[0].set_title("Baseline Model Confusion Matrix")

# # Optimized model confusion matrix
# metrics.ConfusionMatrixDisplay(optimized_cm).plot(ax=axes[1])
# axes[1].set_title("Optimized Model Confusion Matrix")

# plt.show()

# Afficher la matrice de confusion
# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
# disp.figure_.suptitle("Matrice de confusion")
# plt.show()

# Visualiser quelques prédictions
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, y_pred):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title(f"Prediction: {prediction}")
# plt.show()
