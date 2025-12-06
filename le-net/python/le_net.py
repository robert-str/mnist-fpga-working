import os
import numpy as np

import torch

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
print("Preparing MNIST datasets for LeNet-5...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard MNIST preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),            # Convert to tensor [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

train_dataset = datasets.MNIST(root="data/MNIST", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data/MNIST", train=False, download=True, transform=transform)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_classes = 10

print(f"Device: {device}")
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")


# ============================================================
# 2. LENET-5 CNN ARCHITECTURE
# ============================================================
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 28x28 -> 28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)               # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)          # 14x14 -> 10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)               # 10x10 -> 5x5
        
        # Classification layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # Feature extraction
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)  # No activation for final layer (logits)
        
        return x


# Initialize model
model = LeNet5(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("LeNet-5 Model Architecture:")
print(model)


# ============================================================
# 3. TRAINING
# ============================================================
def train_model(model, train_loader, test_loader, num_epochs=10):
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
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
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {running_loss/100:.4f}")
                running_loss = 0.0
        
        train_acc = 100.0 * correct / total
        train_accuracies.append(train_acc)
        print(f"Epoch {epoch} training accuracy: {train_acc:.2f}%")
        
        # Testing phase
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
        print("-" * 50)
    
    return train_accuracies, test_accuracies


# Train the model
print("Starting LeNet-5 training...")
train_accs, test_accs = train_model(model, train_loader, test_loader, num_epochs=10)

print(f"Final Training Accuracy: {train_accs[-1]:.2f}%")
print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")


# ============================================================
# 4. SAVE MODEL
# ============================================================
outputs_dir = os.path.join(script_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Save PyTorch model
pth_path = os.path.join(outputs_dir, "mnist_cnn.pth")
torch.save(model.state_dict(), pth_path)
print(f"Saved LeNet-5 weights to {pth_path}")

# Export to ONNX
onnx_path = os.path.join(outputs_dir, "mnist_cnn.onnx")
model_cpu = model.to("cpu").eval()
dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model_cpu,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=13
)
print(f"Exported LeNet-5 ONNX to {onnx_path}")


# ============================================================
# 5. QUANTIZATION AND EXPORT TO .MEM AND .BIN FILES
# ============================================================
print("\nQuantizing weights and biases...")

# Input scale (inputs will be quantized to int8 range)
INPUT_SCALE = 127.0

# Create output directories
mem_dir = os.path.join(outputs_dir, "mem")
bin_dir = os.path.join(outputs_dir, "bin")
data_dir = os.path.join(script_dir, "data")
os.makedirs(mem_dir, exist_ok=True)
os.makedirs(bin_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Extract and quantize weights/biases for each layer
state_dict = model.state_dict()

layers_to_export = [
    ("conv1", "conv1.weight", "conv1.bias"),
    ("conv2", "conv2.weight", "conv2.bias"),
    ("fc1", "fc1.weight", "fc1.bias"),
    ("fc2", "fc2.weight", "fc2.bias"),
    ("fc3", "fc3.weight", "fc3.bias"),
]

scale_info = []
scale_info.append(f"Input scale factor: {INPUT_SCALE}\n")
scale_info.append("-" * 50 + "\n")

# Store shift values for each layer (for FPGA per-layer scaling)
layer_shifts = {}

for layer_name, weight_key, bias_key in layers_to_export:
    print(f"\nProcessing {layer_name}...")
    
    # Get weights and biases
    W = state_dict[weight_key].cpu().numpy()
    b = state_dict[bias_key].cpu().numpy()
    
    # Quantize weights with dynamic scale to use full int8 range
    max_abs_W = np.max(np.abs(W))
    W_SCALE = 127.0 / max_abs_W
    W_int8 = np.clip(np.round(W * W_SCALE), -127, 127).astype(np.int8)
    
    # Quantize biases: scale = weight_scale * input_scale
    BIAS_SCALE = W_SCALE * INPUT_SCALE
    b_int32 = np.round(b * BIAS_SCALE).astype(np.int32)
    
    # Calculate shift for this layer: shift = round(log2(W_SCALE * INPUT_SCALE))
    # This is used in FPGA to rescale accumulator back to int8 range
    total_scale = W_SCALE * INPUT_SCALE
    shift = int(round(np.log2(total_scale)))
    layer_shifts[layer_name] = shift
    
    print(f"  Weight shape: {W.shape}")
    print(f"  Weight range: [{W.min():.4f}, {W.max():.4f}]")
    print(f"  Weight scale factor: {W_SCALE:.4f}")
    print(f"  Quantized weight range: [{W_int8.min()}, {W_int8.max()}]")
    print(f"  Bias shape: {b.shape}")
    print(f"  Bias scale factor: {BIAS_SCALE:.4f}")
    print(f"  Quantized bias range: [{b_int32.min()}, {b_int32.max()}]")
    print(f"  FPGA shift value: {shift} (total_scale={total_scale:.2f}, 2^{shift}={2**shift})")
    
    # Save scale info
    scale_info.append(f"\n{layer_name}:\n")
    scale_info.append(f"  Weight shape: {W.shape}\n")
    scale_info.append(f"  Weight scale factor: {W_SCALE}\n")
    scale_info.append(f"  Original weight range: [{W.min()}, {W.max()}]\n")
    scale_info.append(f"  Bias scale factor: {BIAS_SCALE}\n")
    scale_info.append(f"  Original bias range: [{b.min()}, {b.max()}]\n")
    scale_info.append(f"  FPGA shift: {shift}\n")
    
    # Save weights (.mem format - hex)
    weight_mem_path = os.path.join(mem_dir, f"{layer_name}_weights.mem")
    with open(weight_mem_path, "w") as f:
        for val in W_int8.flatten():
            val = int(val)
            if val < 0:
                val = val + 256
            f.write(f"{val:02x}\n")
    
    # Save weights (.bin format - binary)
    weight_bin_path = os.path.join(bin_dir, f"{layer_name}_weights.bin")
    W_int8.flatten().tofile(weight_bin_path)
    
    # Save biases (.mem format - hex)
    bias_mem_path = os.path.join(mem_dir, f"{layer_name}_biases.mem")
    with open(bias_mem_path, "w") as f:
        for val in b_int32.flatten():
            val = int(val)
            if val < 0:
                val = val + (1 << 32)
            f.write(f"{val:08x}\n")
    
    # Save biases (.bin format - binary, little-endian int32)
    bias_bin_path = os.path.join(bin_dir, f"{layer_name}_biases.bin")
    b_int32.astype('<i4').tofile(bias_bin_path)  # Little-endian int32
    
    print(f"  Saved: {weight_mem_path}, {weight_bin_path}")
    print(f"  Saved: {bias_mem_path}, {bias_bin_path}")

# Save shift values for FPGA (conv1, conv2, fc1, fc2 - fc3 doesn't need shift for argmax)
shifts_bin_path = os.path.join(bin_dir, "shifts.bin")
shifts_array = np.array([
    layer_shifts["conv1"],
    layer_shifts["conv2"], 
    layer_shifts["fc1"],
    layer_shifts["fc2"]
], dtype=np.uint8)
shifts_array.tofile(shifts_bin_path)
print(f"\nSaved FPGA shift values: {shifts_bin_path}")
print(f"  conv1={layer_shifts['conv1']}, conv2={layer_shifts['conv2']}, "
      f"fc1={layer_shifts['fc1']}, fc2={layer_shifts['fc2']}")

# Save scale info for reference
scale_info.append(f"\nFinal Training Accuracy: {train_accs[-1]:.2f}%\n")
scale_info.append(f"Final Test Accuracy: {test_accs[-1]:.2f}%\n")

scale_path = os.path.join(data_dir, "scale_info.txt")
with open(scale_path, "w") as f:
    f.writelines(scale_info)

# Save normalization parameters for inference
np.save(os.path.join(data_dir, "norm_mean.npy"), np.array([0.1307]))
np.save(os.path.join(data_dir, "norm_std.npy"), np.array([0.3081]))

print(f"\nGenerated: {mem_dir}/*.mem")
print(f"Generated: {bin_dir}/*.bin")
print(f"Generated: {scale_path}")
print(f"Generated: {data_dir}/norm_mean.npy, {data_dir}/norm_std.npy")
print("\nDone!")
