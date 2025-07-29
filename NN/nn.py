#importing necessary library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Fully Connected Network
class Custom_NN(nn.Module):
  def __init__(self,model_arc,return_layer_list=False):
    super().__init__()
    self.arc = model_arc
    self.layer_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'softmax': lambda: nn.Softmax(dim=1),
        'dropout': lambda: nn.Dropout(p=0.5),
    }
    self.network = self.build()

  def build(self):
      layers = []
      for i in range(1,len(self.arc)):
        current = self.arc[i]
        # Case 1 if string
        if isinstance(current,str):
          if current.lower() in self.layer_map:
            layers.append(self.layer_map[current.lower()]())
          else:
            raise ValueError(f"Add the required function before using it:{current}")
        # Case 2
        elif isinstance(current,int):
          prev = self.arc[i-1]
          if isinstance(prev,str):
            prev = self.arc[i-2]
          layers.append(nn.Linear(prev,current))

      if (return_layer_list):
        return layers
      else:
        return nn.Sequential(*layers)

  def forward(self,x): #-> torch.tensor
      x = x.view(x.size(0), -1)
      return self.network(x)

  def train_one_epoch(self,data_loader,loss_function,optimizer):
      self.train() #Set the model to training mode
      total_loss=0
      for inputs,labels in data_loader:
        optimizer.zero_grad()
        outputs = self(inputs)
        loss = loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      avg_loss = total_loss/len(data_loader)
      print(f"  Training Loss: {avg_loss:.4f}")


  def test_model(self, data_loader, loss_function):
        self.eval() # Set the model to evaluation mode
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()
                _, predicted_labels = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted_labels == labels).sum().item()
        avg_loss = total_loss / len(data_loader)
        accuracy = (correct_predictions / total_samples) * 100
        print(f"  Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")


  def train_model(self,train_loader,test_loader,loss_fn,optimizer,epochs):
    for epoch in range(epochs):
      print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
      self.train_one_epoch(train_loader, loss_fn, optimizer)
      self.test_model(test_loader, loss_fn)

# === Dataset and Training Setup ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# === Model Setup ===
architecture = [784, 128, 'relu', 64, 'relu', 10, 'softmax']
model = Custom_NN(architecture)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Train the Model ===
model.train_model(train_loader, test_loader, loss_fn, optimizer, epochs=5)