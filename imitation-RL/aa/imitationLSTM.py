import numpy as np
import pickle

with open("ds", "rb") as fp:   # Unpickling
    ballPathList = pickle.load(fp)

bpL = []
for traj in ballPathList:
    path = []
    for point in traj:
        x = np.asarray(point)
        path.append(x)
    
    y = np.asarray(path)
    bpL.append(y)



trajectories = np.asarray(bpL)

print(trajectories)


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



X = trajectories[:, :-1, :]
y = trajectories[:, 1:, :]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import torch
import torch.nn as nn

class LSTMCellModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCellModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, horizon): # FIRST CHANGE FROM WHEN WAS WORKING
        hx = torch.zeros(x.size(0), self.hidden_size)
        cx = torch.zeros(x.size(0), self.hidden_size)

        outputs = []

        for i in range(x.size(1)):
            if(i < 5):
                hx, cx = self.lstm_cell(x[:, i, :], (hx, cx))
                outputs.append(self.fc(hx))
            else:
                if(i % horizon == 0):
                    hx, cx = self.lstm_cell(x[:, i, :], (hx, cx))
                    outputs.append(self.fc(hx))
                else:
                    state = x[:, i, :]
                    c = np.concatenate((outputs[-1], state[3:]))
                    hx, cx = self.lstm_cell(c, (hx, cx))
                    outputs.append(self.fc(hx))
        return torch.stack(outputs, dim=1)

input_size = 9
hidden_size = 512
output_size = 3

model = LSTMCellModel(input_size, hidden_size, output_size)
from torch.optim import Adam

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 50
num_horizon_lengths = 15

import matplotlib.pyplot as plt

def plot_trajectories(actual, predicted, title):
    plt.plot(actual[:, 0], actual[:, 1], 'ro-', label='Actual')
    plt.plot(predicted[:, 0], predicted[:, 1], 'bo-', label='Predicted')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()



for h in range(1,num_horizon_lengths + 1, 2):
    print("horizon = ", h)
    for epoch in range(num_epochs):
        model.train()

        train_loss = 0
        for i, (inputs, targets) in enumerate(zip(X_train, y_train)):
            inputs = torch.tensor(inputs[np.newaxis, :, :], dtype=torch.float32)
            new_targets = np.zeros((len(targets), 3))
            for q in range(len(targets)):
                new_targets[q] = targets[q][:3]
            targets = torch.tensor(new_targets[np.newaxis, :, :], dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs,h)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(X_train)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.6f}')

    with torch.no_grad():
        train_loss = 0
        for i, (inputs, targets) in enumerate(zip(X_test, y_test)):
            inputs = torch.tensor(inputs[np.newaxis, :, :], dtype=torch.float32)
            new_targets = np.zeros((len(targets), 3))
            for q in range(len(targets)):
                new_targets[q] = targets[q][:3]
            targets = torch.tensor(new_targets[np.newaxis, :, :], dtype=torch.float32)            
            outputs = model(inputs,10000)
            loss = criterion(outputs, targets)

            train_loss += loss.item()

        train_loss /= len(X_test)
        print("Current test loss: ", train_loss)
