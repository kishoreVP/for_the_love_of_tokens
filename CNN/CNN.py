#importing necessary library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from NN import Custom_NN
import torch
import torch.nn as nn

class Custom_CNN(nn.Module):
    def __init__(self, model_arc, input_shape):
        super().__init__()
        self.arc = model_arc
        self.input_shape = input_shape
        self.layer_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'softmax': lambda: nn.Softmax(dim=1),
            'dropout': lambda: nn.Dropout(p=0.5),
        }
        self.network, self.flattened_size = self.build_cnn()
        self.dense_network = self.build_dense()

    def build_cnn(self):
        layers = []
        x = torch.zeros(1, *self.input_shape)  # Dummy input to trace shape
        for i in range(1, len(self.arc)):
            current = self.arc[i]
            if isinstance(current, tuple):
                layer_type = current[0].lower()
                args = current[1:]
                if layer_type == 'conv':
                    layer = nn.Conv2d(*args)
                elif layer_type == 'maxpool2d':
                    layer = nn.MaxPool2d(*args)
                elif layer_type == 'dropout':
                    layer = nn.Dropout(args[0])
                else:
                    raise ValueError(f"Unknown layer type: {layer_type}")
                layers.append(layer)
                x = layer(x)  # Update shape using dummy input
            elif isinstance(current, str) and current.lower() in self.layer_map:
                act_layer = self.layer_map[current.lower()]()
                layers.append(act_layer)
                x = act_layer(x)
            elif isinstance(current, int):
                break  # Stop at first int (dense part begins)

        flattened_size = x.view(x.size(0), -1).shape[1]
        return nn.Sequential(*layers), flattened_size

    def build_dense(self):
        # Extract dense part of the architecture
        dense_part = [self.flattened_size] + [layer for layer in self.arc if isinstance(layer, int)]
        return Custom_NN(dense_part)

    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.dense_network(x)
        return x

    def train_one_epoch(self, data_loader, loss_function, optimizer):
        self.train()
        total_loss = 0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"  Training Loss: {avg_loss:.4f}")

    def test_model(self, data_loader, loss_function):
        self.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        avg_loss = total_loss / len(data_loader)
        accuracy = (correct_predictions / total_samples) * 100
        print(f"  Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    def train_model(self, train_loader, test_loader, loss_fn, optimizer, epochs):
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            self.train_one_epoch(train_loader, loss_fn, optimizer)
            self.test_model(test_loader, loss_fn)