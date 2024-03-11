import sys

if len(sys.argv) < 2:
    print("Usage: ./hw2.sh ctf or ./hw2.sh df")
    sys.exit(1)

argument = sys.argv[1]
print("#########################")
print("Logistic Regression")
print("Argument received:", argument)

from utils import *

import numpy as np


def softmax(pred):
    return np.exp(pred) / np.sum(np.exp(pred), axis=0, keepdims=True)


class LogisticRegression:
    def __init__(self, dim, classes, lr, lambda_reg=1):
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.classes = classes
        self.W = np.zeros((dim, self.classes))
        self.X = None

    def forward(self, X):
        self.X = X
        return softmax(np.matmul(self.W.T, X))

    def backward(self, output, label):
        dl_dw = np.matmul(label - output, self.X.T).T - self.lambda_reg * self.W
        self.W = self.W + self.lr * (dl_dw / self.X.shape[1])


def get_hard_prediction(model, test):
    return np.argmax(model.forward(test.T), axis=0) + 1


def get_soft_prediction(model, test):
    return np.sum(model.forward(test.T) * np.array([[1], [2], [3], [4], [5]]), axis=0)


# Process document frequency
if argument == "df":
    vocab = build_vocab_df(train_data)
    train_features, train_targets = preprocess_dataset_df(train_data, vocab)
    train_features, train_targets = np.array(train_features), np.array(train_targets)

    X_with_intercept = np.hstack((np.ones((len(train_data), 1)), train_features))
    num_classes = len(np.unique(train_targets))
    label = np.eye(num_classes)[np.array(train_targets) - 1]

    min_max_normalize_inplace(X_with_intercept)

# Process Collection term frequency
else:
    vocab = build_vocab_ctf(train_data)
    train_features, train_targets = preprocess_dataset_ctf(train_data, vocab)
    train_features, train_targets = np.array(train_features), np.array(train_targets)
    X_with_intercept = np.hstack((np.ones((len(train_data), 1)), train_features))
    num_classes = len(np.unique(train_targets))
    label = np.eye(num_classes)[np.array(train_targets) - 1]

model = LogisticRegression(
    dim=X_with_intercept.shape[1], classes=num_classes, lr=0.1, lambda_reg=1
)
batch_size = 512

epochs = 0
while epochs < 10:
    for i in range(0, len(X_with_intercept), batch_size):
        X_batch = X_with_intercept[i : i + batch_size].T
        Y_batch = label[i : i + batch_size].T
        pred = model.forward(X_batch)
        model.backward(output=pred, label=Y_batch)
    epochs += 1
    print(f"Epoch #{epochs}")

test_pred = np.argmax(model.forward(X_with_intercept.T), axis=0)
accuracy = np.mean(test_pred == np.argmax(label.T, axis=0)) * 100
print("Accuracy: {:.2f}%".format(accuracy))

y_soft_pred = get_soft_prediction(model, X_with_intercept)
mse = mean_squared_error(train_targets, y_soft_pred)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error: {rmse}")

"""
    Predict on Dev Data
"""

dev_data = load_json_file(YELP_DEV_DIR)
dev_features = preprocess_dataset_ctf_test(dev_data, vocab)
dev_features = np.array(dev_features)
intercept_column = np.ones((len(dev_features), 1))
dev_features = np.hstack((intercept_column, dev_features))

merge_and_save_vectors(
    get_hard_prediction(model, dev_features),
    get_soft_prediction(model, dev_features),
    "../dev-predictions.txt",
)

"""
    Predict on Test Data
"""
test_data = load_json_file(YELP_TEST_DIR)
test_features = preprocess_dataset_ctf_test(test_data, vocab)
test_features = np.array(test_features)
intercept_column = np.ones((len(test_features), 1))
test_features = np.hstack((intercept_column, test_features))

merge_and_save_vectors(
    get_hard_prediction(model, test_features),
    get_soft_prediction(model, test_features),
    "../test-predictions.txt",
)
