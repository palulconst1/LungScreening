# import
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# load data
train = pd.read_csv("train.txt", header=None)
test = pd.read_csv("test.txt", header=None)
validation = pd.read_csv("validation.txt", header=None)
# initialize arrays
train_X = []
train_y = []

test_X = []

validation_X = []
validation_y = []
# read image data and labels
for i, row in tqdm(train.iterrows()):
    img = plt.imread("./train/" + row[0])
    train_X.append(img)
    train_y.append(row[1])

train_X = np.array(train_X)
train_y = np.array(train_y)

for i, row in tqdm(test.iterrows()):
    img = plt.imread("./test/" + row[0])
    test_X.append(img)

test_X = np.array(test_X)

for i, row in tqdm(validation.iterrows()):
    img = plt.imread("./validation/" + row[0])
    validation_X.append(img)
    validation_y.append(row[1])

validation_X = np.array(validation_X)
validation_y = np.array(validation_y)
# reshape data and scale it
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2])

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)

validation_X = validation_X.reshape(validation_X.shape[0], validation_X.shape[1] * validation_X.shape[2])

validation_X = scaler.transform(validation_X)
# multi layer perceptron
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, verbose=True, activation = 'relu', solver='adam', random_state=1)
# XGBOOST Classifier
model = XGBClassifier()
# train model
model.fit(train_X, train_y)
model.score(validation_X, validation_y)

test_X = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2])

test_X = scaler.transform(test_X)
# write predictions
# pred = model.predict(test_X)
pred = model.predict(validation_X)
print(metrics.confusion_matrix(validation_y, pred))
print(metrics.classification_report(validation_y, pred))

# with open('pred_01.txt', 'w', newline='\n') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["id", "label"])
#     for i, row in test.iterrows():
#         writer.writerow([row[0], pred[i]])