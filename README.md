# 🧠 Custom Neural Network Builder (PyTorch)

A modular, lightweight PyTorch framework to dynamically define, build, and train neural networks using a clean, intuitive list-based architecture format.

Supports both Convolutional Neural Networks (CNNs) and Fully Connected Networks (MLPs), making it great for rapid prototyping, educational purposes, and research experimentation.

---

## 🚀 Features

- 🧩 Simple architecture definition with Python lists
- 🔁 Automatically tracks tensor shapes between layers
- 🔌 Supports:
  - Convolutional layers: `Conv2d`
  - Fully connected layers: `Linear`
  - Activations: `ReLU`, `Sigmoid`, `Tanh`
  - Pooling: `MaxPool2d`
- 🔎 Optional visualization for input data
- 🧪 Clean training loop with evaluation
- 📦 Fully compatible with PyTorch `DataLoader`

---

## 📐 Architecture Format

Define your architecture as a list of tuples:

```python
model_arch = [
    ('conv', 1, 16, 3, 1),     # Conv2d(in_channels, out_channels, kernel_size, stride)
    ('relu',),
    ('maxpool', 2),            # MaxPool2d(kernel_size)
    ('fc', 128),               # Linear output to 128 units
    ('relu',),
    ('fc', 10)                 # Final classification output (e.g., 10 classes)
]
