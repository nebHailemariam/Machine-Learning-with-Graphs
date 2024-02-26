import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def softmax(pred):
    return np.exp(pred) / np.sum(np.exp(pred), axis=0, keepdims=True)


class LogisticRegressionIris:
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


# Load iris dataset
iris = datasets.load_iris()
X_iris = iris.data
Y_iris = iris.target

# Normalize the input data
scaler = StandardScaler()
X_iris = scaler.fit_transform(X_iris)

# Add a column of ones as the intercept term
intercept_column = np.ones((X_iris.shape[0], 1))
X_with_intercept = np.hstack((intercept_column, X_iris))

print(X_with_intercept[0])
# One-hot encode the labels
num_classes = len(np.unique(Y_iris))
label_iris = np.eye(num_classes)[Y_iris]

# Initialize the model
model_iris = LogisticRegressionIris(
    dim=X_with_intercept.shape[1], classes=num_classes, lr=0.0001
)

# Split data into training and testing sets
X_train_iris, X_test_iris, Y_train_iris, Y_test_iris = train_test_split(
    X_with_intercept, label_iris, test_size=0.2, random_state=42
)

# Set batch size
batch_size = 32

# Training loop with batching
epochs = 0
while epochs < 10000:
    for i in range(0, len(X_train_iris), batch_size):
        X_batch = X_train_iris[i : i + batch_size].T
        Y_batch = Y_train_iris[i : i + batch_size].T
        pred = model_iris.forward(X_batch)
        model_iris.backward(output=pred, label=Y_batch)
    epochs += 1

# Test the trained model on the test set
test_pred = np.argmax(model_iris.forward(X_test_iris.T), axis=0)
print("Test Predictions:\n", test_pred)
accuracy = np.mean(test_pred == np.argmax(Y_test_iris.T, axis=0)) * 100
print("Accuracy: {:.2f}%".format(accuracy))
