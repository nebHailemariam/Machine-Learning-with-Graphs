import sys
from sklearn.metrics import confusion_matrix

if len(sys.argv) < 2:
    print("Usage: ./hw2.sh ctf or ./hw2.sh df")
    sys.exit(1)

argument = sys.argv[1]
print("#########################")
print("Support Vector Machines")
print("Argument received:", argument)

from utils import *  # Import necessary utility functions

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
if argument == "df":
    vocab = build_vocab_df(train_data)
    train_features, train_targets = preprocess_dataset_df(train_data, vocab)
    train_features, train_targets = np.array(train_features), np.array(train_targets)
else:
    vocab = build_vocab_ctf(train_data)
    train_features, train_targets = preprocess_dataset_ctf(train_data, vocab)
    train_features, train_targets = np.array(train_features), np.array(train_targets)

# Apply Min-Max Normalization
scaler = MinMaxScaler()
train_features_normalized = scaler.fit_transform(train_features)

# Train SVM model
svm_model = SVC(kernel="linear")
svm_model.fit(train_features_normalized, train_targets)

# Evaluate on training data
train_accuracy = svm_model.score(train_features_normalized, train_targets) * 100
print("Training Accuracy: {:.2f}%".format(train_accuracy))

# Load development data
dev_data = load_json_file(YELP_DEV_DIR)

# Load and preprocess development data
dev_features, dev_targets = preprocess_dataset_ctf(dev_data, vocab)
dev_features_normalized = scaler.transform(
    dev_features
)  # Apply same normalization as on training data
dev_features_normalized = np.array(dev_features_normalized)

# Evaluate on development data
dev_accuracy = svm_model.score(dev_features_normalized, dev_targets) * 100
print("Dev Set Accuracy: {:.2f}%".format(dev_accuracy))

# Predict labels for training data
train_predictions = svm_model.predict(train_features_normalized)

# Misclassification Patterns for training data
train_misclassified_indices = np.where(train_predictions != train_targets)[0]
train_misclassified_examples = [train_data[i] for i in train_misclassified_indices]

# Confusion Matrix for training data
train_confusion_matrix = confusion_matrix(train_targets, train_predictions)

print("Misclassification Patterns for Training Data:")
print(train_misclassified_examples)
print("Confusion Matrix for Training Data:")
print(train_confusion_matrix)

# Predict labels for development data
dev_predictions = svm_model.predict(dev_features_normalized)

# Misclassification Patterns for development data
dev_misclassified_indices = np.where(dev_predictions != dev_targets)[0]
dev_misclassified_examples = [dev_data[i] for i in dev_misclassified_indices]

# Confusion Matrix for development data
dev_confusion_matrix = confusion_matrix(dev_targets, dev_predictions)

print("Misclassification Patterns for Development Data:")
print(dev_misclassified_examples)
print("Confusion Matrix for Development Data:")
print(dev_confusion_matrix)
