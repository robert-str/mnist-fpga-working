"""Quick test to verify Python model predictions match FPGA expectations."""
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model weights and biases from .mem files
W_raw = np.loadtxt(os.path.join(script_dir, '../outputs/mem/W.mem'), dtype=str)
W = np.array([int(x, 16) if int(x, 16) < 128 else int(x, 16) - 256 for x in W_raw]).reshape(10, 784)

B_raw = np.loadtxt(os.path.join(script_dir, '../outputs/mem/B.mem'), dtype=str)
B = np.array([int(x, 16) if int(x, 16) < 2**31 else int(x, 16) - 2**32 for x in B_raw])

# Load scaler
mean = np.load(os.path.join(script_dir, '../data/scaler_mean.npy'))
scale = np.load(os.path.join(script_dir, '../data/scaler_scale.npy'))

print("Model loaded successfully!")
print(f"Weights shape: {W.shape}, range: [{W.min()}, {W.max()}]")
print(f"Biases shape: {B.shape}, values: {B}")

def predict(image_data):
    """Simulate FPGA inference."""
    # Apply same preprocessing as send_image.py
    x = image_data.astype(np.float32) / 255.0
    x_scaled = (x - mean) / scale
    x_int = np.round(x_scaled * 127).astype(np.int32)
    
    # Compute scores (same as FPGA)
    scores = x_int @ W.T + B
    return np.argmax(scores), scores

# Test 1: All zeros image
print("\n--- Test 1: All zeros image ---")
zeros = np.zeros(784, dtype=np.uint8)
pred, scores = predict(zeros)
print(f"Prediction: {pred}")
print(f"Scores: {scores}")

# Test 2: All 255 image (white)
print("\n--- Test 2: All white image ---")
white = np.ones(784, dtype=np.uint8) * 255
pred, scores = predict(white)
print(f"Prediction: {pred}")
print(f"Scores: {scores}")

# Test 3: Try loading a test image if available
try:
    from PIL import Image
    test_path = os.path.join(script_dir, '../../test_images/00006.png')
    if os.path.exists(test_path):
        print(f"\n--- Test 3: {test_path} ---")
        img = Image.open(test_path).convert('L')
        img = img.resize((20, 20), Image.Resampling.LANCZOS)
        canvas = Image.new('L', (28, 28), color=0)
        canvas.paste(img, (4, 4))
        img_np = 255 - np.array(canvas).flatten()  # Invert
        pred, scores = predict(img_np)
        print(f"Prediction: {pred}")
        print(f"Top 3 scores: {sorted(enumerate(scores), key=lambda x: -x[1])[:3]}")
except ImportError:
    print("\nPillow not installed, skipping image test")





