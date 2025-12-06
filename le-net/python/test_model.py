"""Quick test to verify Python model predictions match FPGA expectations."""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LENET-5 MODEL DEFINITION (same as training)
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
print("LeNet-5 model loaded successfully!")

# Load normalization parameters
NORM_MEAN = 0.1307
NORM_STD = 0.3081
norm_mean_path = os.path.join(script_dir, 'data', 'norm_mean.npy')
norm_std_path = os.path.join(script_dir, 'data', 'norm_std.npy')
if os.path.exists(norm_mean_path):
    NORM_MEAN = np.load(norm_mean_path)[0]
    NORM_STD = np.load(norm_std_path)[0]
print(f"Normalization: mean={NORM_MEAN}, std={NORM_STD}")


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict(image_data):
    """Run inference on a 784-element uint8 array."""
    # Normalize (same as training)
    x = image_data.astype(np.float32) / 255.0
    x = (x - NORM_MEAN) / NORM_STD
    
    # Reshape to (1, 1, 28, 28) for CNN
    x = x.reshape(1, 1, 28, 28)
    x_tensor = torch.from_numpy(x).float()
    
    with torch.no_grad():
        logits = model(x_tensor)
        scores = logits.squeeze().numpy()
    
    return np.argmax(scores), scores


# ============================================================
# TEST CASES
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: All zeros image (black)")
print("=" * 60)
zeros = np.zeros(784, dtype=np.uint8)
pred, scores = predict(zeros)
print(f"Prediction: {pred}")
print(f"Scores: {[f'{s:.2f}' for s in scores]}")
print(f"Top 3: {sorted(enumerate(scores), key=lambda x: -x[1])[:3]}")

print("\n" + "=" * 60)
print("TEST 2: All white image (255)")
print("=" * 60)
white = np.ones(784, dtype=np.uint8) * 255
pred, scores = predict(white)
print(f"Prediction: {pred}")
print(f"Scores: {[f'{s:.2f}' for s in scores]}")
print(f"Top 3: {sorted(enumerate(scores), key=lambda x: -x[1])[:3]}")

print("\n" + "=" * 60)
print("TEST 3: Center pixel pattern (rough '1' shape)")
print("=" * 60)
one_pattern = np.zeros(784, dtype=np.uint8)
for row in range(8, 24):
    one_pattern[row * 28 + 14] = 255  # vertical line
    one_pattern[row * 28 + 13] = 200
    one_pattern[row * 28 + 15] = 200
pred, scores = predict(one_pattern)
print(f"Prediction: {pred}")
print(f"Scores: {[f'{s:.2f}' for s in scores]}")
print(f"Top 3: {sorted(enumerate(scores), key=lambda x: -x[1])[:3]}")

# Test with actual image files
print("\n" + "=" * 60)
print("TEST 4: Test images from test_images folder")
print("=" * 60)

try:
    from PIL import Image
    
    test_images = ['00006.png', '00257.png', '00001.png', '00007.png', '00004.png', '00527.png']
    test_dir = os.path.join(script_dir, '..', 'test_images')
    
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('L')
            
            # Check if already 28x28 (MNIST format)
            if img.size == (28, 28):
                img_np = np.array(img).flatten()
            else:
                # Resize and center
                img = img.resize((20, 20), Image.Resampling.LANCZOS)
                canvas = Image.new('L', (28, 28), color=0)
                canvas.paste(img, (4, 4))
                img_np = 255 - np.array(canvas).flatten()  # Invert
            
            pred, scores = predict(img_np)
            top3 = sorted(enumerate(scores), key=lambda x: -x[1])[:3]
            print(f"{img_name}: pred={pred}, top3={[(i, f'{s:.1f}') for i, s in top3]}")
        else:
            print(f"{img_name}: NOT FOUND")
            
except ImportError:
    print("Pillow not installed, skipping image file tests")

# Test with MNIST dataset
print("\n" + "=" * 60)
print("TEST 5: First 10 MNIST test images")
print("=" * 60)

try:
    from torchvision import datasets, transforms
    
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, 
                                 transform=transforms.ToTensor())
    
    correct = 0
    for i in range(10):
        image, label = mnist_test[i]
        img_np = (image.squeeze().numpy() * 255).astype(np.uint8).flatten()
        pred, scores = predict(img_np)
        status = "✓" if pred == label else "✗"
        if pred == label:
            correct += 1
        print(f"MNIST[{i}]: label={label}, pred={pred} {status}")
    
    print(f"\nAccuracy on first 10: {correct}/10")
    
except ImportError:
    print("torchvision not installed, skipping MNIST test")

print("\n" + "=" * 60)
print("DONE - If Python predictions are correct but FPGA shows '1',")
print("the issue is likely in the Verilog inference logic or weight loading.")
print("=" * 60)


