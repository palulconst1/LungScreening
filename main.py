# imports
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
import csv
from sklearn import metrics

# train function
def train(args, model, device, train_loader, optimizer, epoch, weight):

    model.train()

    all_losses = []
# for each sample in the batch
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device).float(), target.to(device).long()

        optimizer.zero_grad()

        output = model(data)
        target = target.view(-1)
# squeeze the data and compute loss
        loss = F.cross_entropy(output, target)

        all_losses.append(loss.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
# print batch info
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return np.array(all_losses).mean()

# test function
def test(args, model, device, test_loader, weight):

    model.eval()
    test_loss = 0
    correct = 0
    conf = None

    with torch.no_grad():

        num_iter = 0
        for data, target in test_loader:

            data, target = data.to(device).float(), target.to(device).long()

            output = model(data)

            target = target.view(-1)
# compute loss
            test_loss += F.cross_entropy(output, target)

            output = np.e ** output

            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).float().mean().item()

            num_iter += 1

    print(conf)

    test_loss /= num_iter
    test_accuracy = 100. * correct / num_iter

    print('\nValidation set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
        test_loss, test_accuracy))

    return test_loss, test_accuracy

# prediction funtion
def evaluate(model, device, data):
    with torch.no_grad():
        data = torch.tensor(data)
        data = data.to(device).float()

        output = model(data)

        output = np.e ** output

        pred = output.argmax(dim=1, keepdim=True)

    return pred

# vgg-16 16 model
class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 channel input, since we are using grayscale images
        self.layer1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.layer1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer1_mp = nn.MaxPool2d(2, 2)

        self.layer2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.layer2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.layer2_mp = nn.MaxPool2d(2, 2)

        self.layer3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.layer3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.layer3_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.layer3_mp = nn.MaxPool2d(2, 2)

        self.layer4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.layer4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer4_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer4_mp = nn.MaxPool2d(2, 2)

        self.layer5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.layer5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer5_mp = nn.MaxPool2d(2, 2)

        # conv layers done, time for linear
        self.fc1 = nn.Linear(512 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 3)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = F.relu(self.layer1_1(x))
        x = F.relu(self.layer1_2(x))
        x = self.layer1_mp(x)

        x = F.relu(self.layer2_1(x))
        x = F.relu(self.layer2_2(x))
        x = self.layer2_mp(x)

        x = F.relu(self.layer3_1(x))
        x = F.relu(self.layer3_2(x))
        x = F.relu(self.layer3_3(x))
        x = self.layer3_mp(x)

        x = F.relu(self.layer4_1(x))
        x = F.relu(self.layer4_2(x))
        x = F.relu(self.layer4_3(x))
        x = self.layer4_mp(x)

        x = F.relu(self.layer5_1(x))
        x = F.relu(self.layer5_2(x))
        x = F.relu(self.layer5_3(x))
        x = self.layer5_mp(x)

        # we need to flatten x for the linear layers
        x = x.view(-1, 512 * 1 * 1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.25)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.25)
        x = self.fc3(x)

        return x


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# train on GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# read data
train_data = pd.read_csv("train.txt", header=None)
validation_data = pd.read_csv("validation.txt", header=None)
test_data = pd.read_csv("test.txt", header=None)

train_data_files = train_data[[0]].to_numpy()
train_data_y = train_data[[1]].to_numpy()

validation_data_files = validation_data[[0]].to_numpy()
validation_data_y = validation_data[[1]].to_numpy()

test_data_files = test_data[[0]].to_numpy()
# initialize data arrays
train_data_X = np.zeros((len(train_data_files), 50, 50))
validation_data_X = np.zeros((len(validation_data_files), 50, 50))
test_data_X = np.zeros((len(test_data_files), 50, 50))

print("Loading train images...")
#  open images and append them to vectors
for i in tqdm(range(len(train_data_files))):
    img = np.asarray(Image.open('train/' + train_data_files[i][0]))
    train_data_X[i] = img

print("Loading validation images...")

for i in tqdm(range(len(validation_data_files))):
    img = np.asarray(Image.open('validation/' + validation_data_files[i][0]))
    validation_data_X[i] = img


print("Loading test images...")

for i in tqdm(range(len(test_data_files))):
    img = np.asarray(Image.open('test/' + test_data_files[i][0]))
    test_data_X[i] = img
# map arrays to tensors
train_dataset_X, train_dataset_y = map(
    torch.tensor, (train_data_X, train_data_y))

validation_dataset_X, validation_dataset_y = map(
    torch.tensor, (validation_data_X, validation_data_y))

train_dataset = TensorDataset(train_dataset_X, train_dataset_y)
validation_dataset = TensorDataset(validation_dataset_X, validation_dataset_y)
# data loaders for training
train_loader = DataLoader(train_dataset, batch_size=512,
                          shuffle=True, drop_last=True)
validation_loader = DataLoader(
    validation_dataset, batch_size=512, shuffle=True, drop_last=True)

# move model to GPU
model = VGG().to(device)
# load wheights
model.load_state_dict(torch.load("./model.pt"))

# create optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-5)

losses_train = []
losses_test = []
accuracy_test = []
# train
for epoch in range(1, 15 + 1):

    train_loss = train(None, model, device, train_loader,
                       optimizer, epoch, None)

    test_loss, test_accuracy = test(
        None, model, device, validation_loader, None)

    losses_train.append(train_loss)
    losses_test.append(test_loss)
    accuracy_test.append(test_accuracy)

torch.save(model.state_dict(), "model.pt")


def plot_loss(loss, label, color='blue'):
    plt.plot(loss, label=label, color=color)
    plt.legend()

# compute predictions for competition
# predictions1 = evaluate(model, device, test_data_X[0:1300],
#                         ).cpu().detach().numpy()
#
# predictions2 = evaluate(model, device, test_data_X[1300:2600],
#                         ).cpu().detach().numpy()
#
# predictions3 = evaluate(model, device, test_data_X[2600:],
#                         ).cpu().detach().numpy()
#
# predictions = np.concatenate((predictions1, predictions2, predictions3))
#
# print(predictions)
# compute predictions for model summary
predictions1 = evaluate(model, device, validation_data_X[0:1500],
                        ).cpu().detach().numpy()

predictions2 = evaluate(model, device, validation_data_X[1500:3000],
                        ).cpu().detach().numpy()

predictions3 = evaluate(model, device, validation_data_X[3000:],
                        ).cpu().detach().numpy()

predictions = np.concatenate((predictions1, predictions2, predictions3))

print(metrics.confusion_matrix(validation_data_y, predictions))
print(metrics.classification_report(validation_data_y, predictions))
# write predictions to file
# with open("predictions.csv", mode="w") as csv_file:
#     writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
#
#     writer.writerow(['id', 'label'])
#
#     for i in range(predictions.shape[0]):
#         writer.writerow([test_data_files[i][0], predictions[i][0]])

