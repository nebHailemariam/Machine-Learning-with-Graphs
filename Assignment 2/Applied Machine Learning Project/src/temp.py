import numpy as np

confusion_matrix = np.array(
    [
        [142, 31, 15, 6, 6],
        [64, 70, 40, 20, 6],
        [20, 51, 80, 29, 20],
        [13, 21, 84, 44, 38],
        [5, 7, 12, 102, 74],
    ]
)

# Calculate accuracy per class
accuracy_per_class = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)

print("Accuracy per class:")
for i, acc in enumerate(accuracy_per_class):
    print(f"Class {i+1}: {acc * 100:.2f}%")
