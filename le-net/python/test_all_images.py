"""Test Python model predictions on all test images."""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LENET-5 MODEL DEFINITION
# ============================================================
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
# LOAD MODEL
# ============================================================
model_path = os.path.join(script_dir, 'outputs', 'mnist_cnn.pth')
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print("Run le_net.py first to train and save the model.")
    exit(1)

model = LeNet5()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Load normalization parameters
NORM_MEAN = 0.1307
NORM_STD = 0.3081
norm_mean_path = os.path.join(script_dir, 'data', 'norm_mean.npy')
if os.path.exists(norm_mean_path):
    NORM_MEAN = np.load(norm_mean_path)[0]
    NORM_STD = np.load(os.path.join(script_dir, 'data', 'norm_std.npy'))[0]


def predict(img_path):
    """Load and predict a single image."""
    img = Image.open(img_path).convert('L')
    
    # If already 28x28 (MNIST format), use directly without resize or invert
    if img.size == (28, 28):
        img_np = np.array(img).flatten()
    else:
        # Resize and center for non-MNIST images
        img = img.resize((20, 20), Image.Resampling.LANCZOS)
        canvas = Image.new('L', (28, 28), color=0)
        canvas.paste(img, (4, 4))
        img_np = 255 - np.array(canvas).flatten()  # Invert for non-MNIST
    
    # Normalize
    x = img_np.astype(np.float32) / 255.0
    x = (x - NORM_MEAN) / NORM_STD
    x = x.reshape(1, 1, 28, 28)
    x_tensor = torch.from_numpy(x).float()
    
    with torch.no_grad():
        logits = model(x_tensor)
        scores = logits.squeeze().numpy()
    
    return np.argmax(scores), scores


# ============================================================
# TEST ALL IMAGES
# ============================================================
images = [
    '00006.png',  # digit 0
    '00257.png',  # digit 2  
    '01063.png',  # digit 3
    '00153.png',  # digit 4
    '00059.png',  # digit 5
    '00007.png',  # digit 7
    '00032.png',  # digit 8
    '00137.png',  # digit 9
    '00527.png',  # digit 5
    '00001.png',  # digit 1
    '00004.png',  # digit 4
]
test_dir = os.path.join(script_dir, '..', 'test_images')

print("LeNet-5 Model Predictions")
print("=" * 100)
print(f"{'Image':<12} | {'Pred':>4} | {'Scores (0-9)':<80}")
print("-" * 100)

for img in images:
    path = os.path.join(test_dir, img)
    if os.path.exists(path):
        pred, scores = predict(path)
        scores_str = ' '.join([f'{s:6.1f}' for s in scores])
        print(f'{img:<12} | {pred:>4} | [{scores_str}]')
    else:
        print(f'{img:<12} | NOT FOUND')

print("-" * 100)

# Also test first few MNIST images
print("\nMNIST Test Set (first 20 images):")
print("-" * 100)
try:
    from torchvision import datasets, transforms
    
    mnist_test = datasets.MNIST(root='./data', train=False, download=True,
                                 transform=transforms.ToTensor())
    
    correct = 0
    for i in range(20):
        image, label = mnist_test[i]
        img_np = (image.squeeze().numpy() * 255).astype(np.uint8).flatten()
        
        # Normalize
        x = img_np.astype(np.float32) / 255.0
        x = (x - NORM_MEAN) / NORM_STD
        x = x.reshape(1, 1, 28, 28)
        x_tensor = torch.from_numpy(x).float()
        
        with torch.no_grad():
            logits = model(x_tensor)
            scores = logits.squeeze().numpy()
        
        pred = np.argmax(scores)
        status = "✓" if pred == label else "✗"
        if pred == label:
            correct += 1
        
        scores_str = ' '.join([f'{s:6.1f}' for s in scores])
        print(f'MNIST[{i:>3}]   | {pred:>4} | [{scores_str}] (label={label}) {status}')
    
    print("-" * 100)
    print(f"Accuracy: {correct}/20 = {100*correct/20:.1f}%")
    
except ImportError:
    print("torchvision not installed, skipping MNIST test")

print("\n" + "=" * 100)
print("Compare these predictions with what the FPGA displays.")
print("If Python shows correct digits but FPGA always shows '7',")
print("the problem is in the FPGA inference logic, not the model.")
print("=" * 100)


