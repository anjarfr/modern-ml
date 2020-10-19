import pandas as pd
import numpy as np
import re
from pandas.api.types import is_string_dtype, is_numeric_dtype
import warnings
from pdb import set_trace

from torch import nn, optim, as_tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.init import *

import sklearn
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

train = pd.read_csv("data/imputed_train_data.csv")
test = pd.read_csv("data/imputed_test_data.csv")
y = np.loadtxt('data/target_train_data.csv')

train_tensor = torch.tensor(train.drop('target', axis=1).values)
test_tensor = torch.tensor(test.values)
target_tensor = torch.tensor(y.astype(int))

X_train, X_test, y_train, y_test = train_test_split(train_tensor, target_tensor, test_size=0.33, random_state=7)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Linear(2851588, 256)
      self.conv2 = nn.Linear(256, 128)
      self.dropout1 = nn.Dropout(0.25)
      self.dropout2 = nn.Dropout(0.5)
      self.fc1 = nn.Linear(128, 128)
      self.fc2 = nn.Linear(128, 33158)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Pass data through dropout1
      x = self.dropout1(x)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.softmax(x, dim=0)
      return output


net = Net()
net = net.double()

import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_tensor, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = X_train.flatten().double()
        labels = y_train.double()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.double())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
