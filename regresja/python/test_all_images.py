"""Test Python model predictions on all test images."""
import numpy as np
from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model
W_raw = np.loadtxt(os.path.join(script_dir, '../outputs/mem/W.mem'), dtype=str)
W = np.array([int(x, 16) if int(x, 16) < 128 else int(x, 16) - 256 for x in W_raw]).reshape(10, 784)
B_raw = np.loadtxt(os.path.join(script_dir, '../outputs/mem/B.mem'), dtype=str)
B = np.array([int(x, 16) if int(x, 16) < 2**31 else int(x, 16) - 2**32 for x in B_raw])
mean = np.load(os.path.join(script_dir, '../outputs/npy/scaler_mean.npy'))
scale = np.load(os.path.join(script_dir, '../outputs/npy/scaler_scale.npy'))

def predict(img_path):
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
    
    x = img_np.astype(np.float32) / 255.0
    x_scaled = (x - mean) / scale
    x_int = np.round(x_scaled * 127).astype(np.int32)
    scores = x_int @ W.T + B
    return np.argmax(scores), scores

# Test all images
images = ['00006.png', '00257.png', '01063.png', '00153.png', '00059.png', '00007.png', '00032.png', '00137.png', '00527.png']
test_dir = os.path.join(script_dir, '..', '..', 'test_images')

print('Image       | Prediction | All Scores')
print('-' * 80)
for img in images:
    path = os.path.join(test_dir, img)
    if os.path.exists(path):
        pred, scores = predict(path)
        print(f'{img:11} | {pred:10} | {[int(s) for s in scores]}')
    else:
        print(f'{img:11} | NOT FOUND')

