import numpy as np

def generate_ball_trajectory(initial_position, initial_velocity, time_steps, dt, gravity=-9.81):
    position_data = [initial_position]
    velocity_data = [initial_velocity]

    for _ in range(1, time_steps):
        new_velocity = velocity_data[-1] + np.array([0, gravity]) * dt
        new_position = position_data[-1] + new_velocity * dt

        if new_position[1] < 0:
            break

        position_data.append(new_position)
        velocity_data.append(new_velocity)

    return np.array(position_data)

# Generate dataset
num_trajectories = 1000
time_steps = 100
dt = 0.1
trajectories = []

for _ in range(num_trajectories):
    initial_position = np.random.uniform(low=0, high=10, size=2)
    initial_velocity = np.random.uniform(low=0, high=20, size=2)
    trajectory = generate_ball_trajectory(initial_position, initial_velocity, time_steps, dt)
    trajectories.append(trajectory)


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Scale data
scaler = MinMaxScaler()
scaled_trajectories = [scaler.fit_transform(t) for t in trajectories]


# Pad trajectories to have equal length
padded_trajectories = np.zeros((num_trajectories, time_steps, 2))
for i, t in enumerate(scaled_trajectories):
    padded_trajectories[i, :t.shape[0], :] = t

# Create input and target data
X = padded_trajectories[:, :-1, :]
y = padded_trajectories[:, 1:, :]

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
            if(i % horizon == 0):
                hx, cx = self.lstm_cell(x[:, i, :], (hx, cx))
                outputs.append(self.fc(hx))
            else:
                hx, cx = self.lstm_cell(outputs[-1], (hx, cx))
                outputs.append(self.fc(hx))
        return torch.stack(outputs, dim=1)

input_size = 2
hidden_size = 64
output_size = 2

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
            targets = torch.tensor(targets[np.newaxis, :, :], dtype=torch.float32)

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
            targets = torch.tensor(targets[np.newaxis, :, :], dtype=torch.float32)
            
            outputs = model(inputs,10000)
            loss = criterion(outputs, targets)

            train_loss += loss.item()

        train_loss /= len(X_test)
        print("Current test loss: ", train_loss)
