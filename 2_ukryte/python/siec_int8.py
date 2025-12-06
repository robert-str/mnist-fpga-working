# %%
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Quantization imports
from torch.quantization import QuantStub, DeQuantStub

# %%
# Prepare torchvision MNIST datasets and loaders
print("Preparing MNIST datasets (torchvision)...")

# Quantization is typically done on CPU in PyTorch
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),            # [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
])

# Adjust paths since we are in 2_hidden/
train_dataset = datasets.MNIST(root="../data/MNIST", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="../data/MNIST", train=False, download=True, transform=transform)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

input_dim = 28 * 28
num_classes = 10

print(f"Device: {device}")
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# %%
# Preview a few images from the DataLoader
np.set_printoptions(linewidth=200, threshold=784, suppress=True)

examples, labels = next(iter(train_loader))
fig, axes = plt.subplots(1, 6, figsize=(10, 2))
for i in range(6):
    axes[i].imshow(examples[i, 0].numpy(), cmap='gray')
    axes[i].set_title(int(labels[i]))
    axes[i].axis('off')
plt.show()

# %%
# Define a neural network with 2 hidden layers for Quantization
class MNISTTwoHiddenQuant(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, num_classes: int):
        super().__init__()
        self.quant = QuantStub()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes)
        )
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.net(x)
        x = self.dequant(x)
        return x

hidden_dim1 = 16
hidden_dim2 = 16
model = MNISTTwoHiddenQuant(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)

# %%
# Train (Float32)
num_epochs = 5  # Reduced for speed, but enough for convergence
log_interval = 100

train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(1, num_epochs + 1):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {running_loss/log_interval:.4f}")
            running_loss = 0.0

    train_acc = 100.0 * correct / total
    train_accuracies.append(train_acc)
    print(f"Epoch {epoch} training accuracy: {train_acc:.2f}%")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_acc = 100.0 * correct / total
    test_accuracies.append(test_acc)
    print(f"Epoch {epoch} test accuracy: {test_acc:.2f}%")

# %%
# Apply Post-Training Static Quantization
print("Applying quantization...")
model.eval()
# Use 'fbgemm' for x86 or 'qnnpack' for ARM/mobile
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate
print("Calibrating with training data...")
with torch.no_grad():
    for i, (inputs, targets) in enumerate(train_loader):
        if i > 100: break  # Use a subset for calibration
        inputs = inputs.to(device)
        model(inputs)

# Convert to quantized model
torch.quantization.convert(model, inplace=True)
print("Quantization complete.")
print(model)

# %%
# Evaluate Quantized Model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
print(f"Quantized Test Accuracy: {100.0 * correct / total:.2f}%")

# %%
# Save Quantized Model
os.makedirs("../outputs", exist_ok=True)
pth_path = os.path.join("../outputs", "mnist_2hidden_int8.pth")
torch.save(model.state_dict(), pth_path)
print(f"Saved quantized weights to {pth_path}")

# %%
# Predict digit from test2.png with INT8 model
img_path = "../test2.png"

if not os.path.exists(img_path):
    print("test2.png not found")
else:
    # Load, convert to grayscale, resize to 20x20
    img_orig = Image.open(img_path).convert("L")
    img_20 = img_orig.resize((20, 20), resample=Image.BILINEAR)

    # Create 28x28 black canvas and center the 20x20 digit
    canvas = Image.new("L", (28, 28), color=0)
    offset = ((28 - 20) // 2, (28 - 20) // 2)
    canvas.paste(img_20, offset)

    # Show processed image
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')

    # Preprocess for model
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(canvas)
    if tensor.mean().item() > 0.5:
        tensor = 1.0 - tensor
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    tensor = normalize(tensor).unsqueeze(0)

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        pred = int(logits.argmax(dim=1).item())

    plt.title(f"Predicted (INT8): {pred}")
    plt.show()
    print(f"Prediction: {pred}")


