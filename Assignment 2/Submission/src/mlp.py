import sys

if len(sys.argv) < 2:
    print("Usage: ./hw2.sh ctf or ./hw2.sh df")
    sys.exit(1)

argument = sys.argv[1]
print("#########################")
print("MLP")
print("Argument received:", argument)

from utils import *

import numpy as np
import torch
from torch.utils.data import Dataset


# Process document frequency
if argument == "df":
    vocab = build_vocab_df(train_data)
    train_features, train_targets = preprocess_dataset_df(train_data, vocab)
    train_features, train_targets = np.array(train_features), np.array(train_targets)

# Process Collection term frequency
else:
    vocab = build_vocab_ctf(train_data)
    train_features, train_targets = preprocess_dataset_ctf(train_data, vocab)
    train_features, train_targets = np.array(train_features), np.array(train_targets)


class YelpDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


class YelpTestDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


if 5 in train_targets:
    train_targets -= 1
train_dataset = YelpDataset(train_features, train_targets)

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, vocab_size):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(vocab_size, 1024)
        self.fc_2 = nn.Linear(1024, 512)
        self.fc_3 = nn.Linear(512, 5)

    def forward(self, input_batch):
        out = self.fc_1(input_batch)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.relu(out)
        out = self.fc_3(out)
        return out


import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = MLP(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)


def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(nn.functional.softmax(outputs.data, dim=1), 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    average_loss = total_loss / len(train_loader)
    accuracy_train = correct_train / total_train
    return average_loss, accuracy_train


def get_hard_predictions(model, test_loader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.to(device)
            outputs = model(inputs)
            outputs.data = nn.functional.softmax(outputs.data, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy() + 1)

    return all_predictions


def get_soft_predictions(model, test_loader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.to(device)
            outputs = model(inputs)
            outputs.data = nn.functional.softmax(outputs.data, dim=1)
            outputs.data = outputs.data * torch.tensor(
                [[1, 2, 3, 4, 5]], dtype=torch.float32
            ).to(device)

            all_predictions.extend(torch.sum(outputs.data, 1).cpu().numpy())

    return all_predictions


epochs = 5

for epoch in range(epochs):
    train_loss, train_accuracy = train_model(
        model, train_loader, optimizer, criterion, device
    )
    print(
        f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_accuracy * 100:.2f}%"
    )

dev_data = load_json_file(YELP_DEV_DIR)
dev_features = preprocess_dataset_ctf_test(dev_data, vocab)
dev_features = np.array(dev_features)
dev_dataset = YelpTestDataset(dev_features)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

merge_and_save_vectors(
    get_hard_predictions(model, dev_loader, device),
    get_soft_predictions(model, dev_loader, device),
    "../dev-predictions.txt",
)

test_data = load_json_file(YELP_TEST_DIR)
test_features = preprocess_dataset_ctf_test(test_data, vocab)
test_features = np.array(test_features)
test_dataset = YelpTestDataset(test_features)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

merge_and_save_vectors(
    get_hard_predictions(model, test_loader, device),
    get_soft_predictions(model, test_loader, device),
    "../test-predictions.txt",
)
