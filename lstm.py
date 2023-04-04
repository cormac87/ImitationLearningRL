import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pickle

with open("bp2", "rb") as fp:   # Unpickling
    ballPosList = pickle.load(fp)


class LSTMPredictor(nn.Module):
    def  __init__(self, n_hidden_dim=128): # need to figure out n_hidden
        super(LSTMPredictor, self).__init__()
        self.n_hidden_dim = n_hidden_dim
        self.lstm = nn.LSTMCell(3,self.n_hidden_dim)
        self.linear = nn.Linear(n_hidden_dim, 3)

    def forward(self, input1, num_frames):
        outputs = []
        n_input_dim = 1 #dunno if this is right

        h_t = torch.zeros(n_input_dim, self.n_hidden_dim, dtype=torch.float32)
        c_t = torch.zeros(n_input_dim, self.n_hidden_dim, dtype=torch.float32)



        for input_t in input1.split(1): #need to figure out how split works
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs.append(output)
        
        for i in range(num_frames):
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs.append(output)

        outputs = torch.cat(outputs)
        return(outputs)

inputSize = 10
bpList = ballPosList[0]


input1 = bpList[:inputSize]
training1 = torch.tensor(bpList[:(inputSize + 1)])
numFrames = len(training1)
print(numFrames)
input1 = torch.tensor(input1)
print("input1")
print(input1)
print("training1")
print(training1)
for input_t in input1.split(1):
    print(input_t)


model = LSTMPredictor()
out = model(input1, numFrames - 10)
print(training1.size())
print(out.size())
print(out)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

n_steps = 6000
for i in range(n_steps):
    print("Step", i)

    def closure():
        optimizer.zero_grad()
        out = model(input1, numFrames - 10)
        loss = loss_function(out,training1)
        print("loss", loss.item())
        loss.backward()
        return loss
    optimizer.step(closure)

print(out)
print(training1)


        
