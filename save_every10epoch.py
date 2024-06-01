import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import copy
import os

from mnist1d.data import get_dataset_args, get_dataset
from mnist1d.utils import set_seed, ObjectView

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=int, required=True, help="Number of layers")
parser.add_argument("--unit_equal", action="store_false", help="Units per layer are the same")
parser.add_argument("--param_limit", type=int, default=15000)
parser.add_argument("--input_size", type=int, default=40)
parser.add_argument("--output_size", type=int, default=10)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--learning_rate", default=0.0025)
parser.add_argument("--weight_decay", type=int, default=0)
parser.add_argument("--epochs", type=int, default=500000)
parser.add_argument("--print_every", type=int, default=1)
parser.add_argument("--eval_every", type=int, default=1)
parser.add_argument("--checkpoint_every", type=int, default=1000)
parser.add_argument("--device", default='cuda')
parser.add_argument("--seed", default=42)
args = parser.parse_args()

# project
PROJECT_NAME = "Final_results_TRUE"
CHECKPOINT_PATH = f"./checkpoint/checkpoint_depth_{args.depth}.tar"
N_EPOCHS = 100

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = get_dataset(get_dataset_args(), path='./mnist1d_data.pkl', download=True)
all_cases = []

def get_total_params_function(depth, units):
    total_params = 0
    for layer in range(1, depth + 2):
        params = units[layer] * (units[layer - 1] + 1)
        total_params += params
    return total_params

class CustomFCNN(nn.Module):
    def __init__(self, depth, units, input_size=40, output_size=10):
        super(CustomFCNN, self).__init__()
        self.depth = depth
        self.units = units
        self.input_size = input_size
        self.output_size = output_size
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(units[i], units[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(units[-2], units[-1]))
        self.layers = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        return self.layers(x)

    def get_total_params(self):
        return get_total_params_function(self.depth, self.units)

def get_units(input_size, output_size, depth, params_limit=15000, error_ratio=0.001):
    global all_cases

    def brute_force(index_current_layer, num_units_of_previous_layers, acc_count_weigths, arr):
        if index_current_layer > depth:
            arr.append(output_size)
            acc_count_weigths += (num_units_of_previous_layers + 1) * output_size
            if params_limit - params_limit * error_ratio <= acc_count_weigths <= params_limit + params_limit * error_ratio:
                all_cases.append([arr.copy(), acc_count_weigths])
            arr.pop()
            return

        if acc_count_weigths > params_limit + params_limit * error_ratio:
            return

        num_units = 0
        while True:
            num_units += 1
            acc_count_weigths += num_units_of_previous_layers + 1
            if acc_count_weigths <= params_limit + params_limit * error_ratio:
                arr.append(num_units)
                brute_force(index_current_layer + 1, num_units, acc_count_weigths, arr)
                arr.pop()
            else:
                return

    brute_force(1, input_size, 0, [input_size])

def get_units_equal(input_size, output_size, depth, params_limit=15500):
    units = [input_size]
    possible_num_of_unit = 0
    test = 1
    while True:
        test_units = [input_size] + [test] * depth + [output_size]
        test_params = get_total_params_function(depth, test_units)
        if test_params > params_limit:
            break
        else:
            possible_num_of_unit = test
            test += 1
    units.extend([possible_num_of_unit] * depth)
    units.append(output_size)
    return units

def accuracy(model, inputs, targets):
    preds = model(inputs).argmax(-1).cpu().numpy()
    targets = targets.cpu().numpy().astype(np.float32)
    return 100 * sum(preds == targets) / len(targets)

def save_checkpoint(state, filename=CHECKPOINT_PATH):
    torch.save(state, filename)
    print(f"Checkpoint saved at {filename}")

def load_checkpoint(filename=CHECKPOINT_PATH):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        print(f"No checkpoint found at {filename}, starting from scratch")
        return None

def train_model(dataset, model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    x_train, x_test = torch.Tensor(dataset['x']), torch.Tensor(dataset['x_test'])
    y_train, y_test = torch.LongTensor(dataset['y']), torch.LongTensor(dataset['y_test'])

    # Move model and data to the device (GPU/CPU)
    model = model.to(device)
    x_train, x_test, y_train, y_test = [v.to(device) for v in [x_train, x_test, y_train, y_test]]

    # Load checkpoint if exists
    start_epoch = 0
    checkpoint = load_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    results = {'checkpoints': [], 'train_losses': [], 'test_losses': [], 'train_acc': [], 'test_acc': []}
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        results['train_losses'].append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        results['checkpoints'].append(copy.deepcopy(model))

        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test)
            test_loss = criterion(y_test_pred, y_test)
            results['test_losses'].append(test_loss.item())
        results['train_acc'].append(accuracy(model, x_train, y_train))
        results['test_acc'].append(accuracy(model, x_test, y_test))

        # Print progress to terminal
        if epoch % args.print_every == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {results['train_losses'][-1]:.4f}, "
                  f"Test Loss: {results['test_losses'][-1]:.4f}, "
                  f"Train Acc: {results['train_acc'][-1]:.2f}%, Test Acc: {results['test_acc'][-1]:.2f}%")
        
        # Save every 10 epoch
        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), f'./trajectory/depth_{args.depth}/model_{epoch}.pth')
            
        if int(results['train_acc'][-1]) == 100: # stop when 100 reached
            return results

    return results

set_seed(args.seed)

if args.unit_equal:
    os.makedirs(f"./trajectory/depth_{args.depth}", exist_ok=True)
    units = get_units_equal(args.input_size, args.output_size, args.depth, args.param_limit + 500)
    model = CustomFCNN(args.depth, units, args.input_size, args.output_size).to(device)
    print(f"{args.depth} layers: each with {units} units ({model.get_total_params()})")
    print("Training ...")
    results = train_model(data, model, args) 
else:
    get_units(args.input_size, args.output_size, args.depth, args.param_limit)
    num_samples = 10
    indices = np.linspace(0, len(all_cases) - 1, num_samples, dtype=int)
    sampled_cases = [all_cases[index] for index in indices]
    print(f"----> # of total models: {len(sampled_cases)}")
    for i in range(len(sampled_cases)):
        current_units = sampled_cases[i][0]
        model = CustomFCNN(args.depth, current_units, args.input_size, args.output_size)
        print(f"{args.depth} layers: {current_units} ({model.get_total_params()})")
        print("Training …")
        results = train_model(data, model, args) 