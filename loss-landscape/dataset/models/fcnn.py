import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def get_total_params_function(depth, units):
    total_params = 0
    
    for layer in range(1, depth + 2):
        params = units[layer] * ( units[layer - 1] + 1 )
        total_params += params
    
    return total_params

class CustomFCNN(nn.Module):
    def __init__(self, depth, units, input_size=40, output_size = 10):
        super(CustomFCNN, self).__init__()
        self.depth = depth
        self.units = units
        self.input_size = input_size
        self.output_size = output_size
        
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(units[i], units[i+1]))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(units[-2], units[-1]))
        self.layers = nn.Sequential(*layers)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        return self.layers(x)

    def get_total_params(self): # units : list (각 층의 유닛 수 저장되어 있는 리스트)
        return get_total_params_function(self.depth, self.units)