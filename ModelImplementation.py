import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import pandas as pd

class Model(torch.nn.Module):

    def __init__(self, input, output):
        super(Model, self).__init__()

        self.linear1 = torch.nn.Linear(input, 100)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, 100)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(100, output)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

# # Data Collection
df = pd.read_csv("heart.csv")
df.drop('Timestamp', axis=1, inplace=True)

print(df.HeartDisease.unique())

# # Data Preprocessing
# Handling Categorical Data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in df.columns:
    if df[i].dtype == "object":
        df[i] = label_encoder.fit_transform(df[i])

# Normalise the data
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(df.iloc[:, 1:11])
df.iloc[:, 1:11] = pd.DataFrame(scaler.transform(df.iloc[:, 1:11]), index=df.index, columns=df.iloc[:, 1:11].columns)

print(df.info())

# Creating np arrays
X = df.loc[:, df.columns != 'HeartDisease']
y = df['HeartDisease']

X = torch.FloatTensor(X.values)
y = torch.LongTensor(y.values)

# Data Splitting for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Passing to DataLoader
train = data_utils.TensorDataset(X_train, y_train)
test = data_utils.TensorDataset(X_test, y_test)

input = 11
output = 2

model = Model(input, output)

print('The model:', model)

learning_rate = 0.001
epochs = 500
batch_size = 64

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(train, batch_size=batch_size)
test_dataloader = DataLoader(test, batch_size=batch_size)

train_losses = []
test_losses = []
accuracies = []

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            train_losses.append(loss)

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_losses.append(test_loss)
    accuracies.append(100*correct)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

import matplotlib.pyplot as plt

round = [i for i in range(epochs)]

plt.grid(visible=True, which='major', axis='both', c='0.95', ls='-', linewidth=1.0, zorder=0)
plt.title("Heart Failure Prediction training/testing loss")
plt.plot(round, train_losses, '--', label='Train', color="darkgreen", alpha=0.5, linewidth=1.0)
plt.plot(round, test_losses, '--', label='Test', color="maroon", alpha=0.5, linewidth=1.0)
plt.xticks(rotation=45, fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.xlabel('Times', fontsize=10)
plt.legend(fontsize=12, loc='lower right')
plt.show()

plt.grid(visible=True, which='major', axis='both', c='0.95', ls='-', linewidth=1.0, zorder=0)
plt.title("Heart Failure Prediction accuracy")
plt.plot(round, accuracies, '--', label='Accuracy', color="darkgreen", alpha=0.5, linewidth=1.0)
plt.xticks(rotation=45, fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlabel('Times', fontsize=10)
plt.legend(fontsize=12, loc='lower right')
plt.show()

print("Done!")
