import sys

if len(sys.argv) < 2:
    print("Usage: ./hw2.sh ctf or ./hw2.sh df")
    sys.exit(1)

argument = sys.argv[1]

print("#########################")
print("SVM")
print("Argument received:", argument)

from utils import *
import numpy as np
from sklearn import svm

vocab = build_vocab_df(train_data)
train_features, train_targets = preprocess_dataset_df(train_data, vocab)
train_features, train_targets = np.array(train_features), np.array(train_targets)


clf = svm.LinearSVC(penalty="l2", loss="squared_hinge", random_state=0)
clf.fit(train_features, train_targets)

from sklearn.metrics import accuracy_score

y_pred = clf.predict(train_features)
accuracy = accuracy_score(train_targets, y_pred)

print("Accuracy:", accuracy)

dev_data = load_json_file(YELP_DEV_DIR)
dev_features = preprocess_dataset_ctf_test(dev_data, vocab)
dev_features = np.array(dev_features)

test_data = load_json_file(YELP_TEST_DIR)
test_features = preprocess_dataset_ctf_test(test_data, vocab)
test_features = np.array(test_features)

merge_and_save_vectors(
    clf.predict(dev_features), clf.predict(dev_features), "../dev-predictions.txt"
)

merge_and_save_vectors(
    clf.predict(test_features),
    clf.predict(test_features),
    "../test-predictions.txt",
)
