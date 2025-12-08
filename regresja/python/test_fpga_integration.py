import serial
import time
import sys
import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuration
COM_PORT = 'COM3'     # Change this to your port
BAUD_RATE = 115200
TEST_SAMPLES = 1000    # Number of images to test
INPUT_SCALE = 127.0   # Must match training

# Protocol Constants
IMG_START_MARKER = bytes([0xBB, 0x66])
IMG_END_MARKER = bytes([0x66, 0xBB])
DIGIT_READ_REQUEST = bytes([0xCC])


def load_data_and_scaler():
    """Load scaler params and MNIST test data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npy_dir = os.path.join(script_dir, "..", "outputs", "npy")
    
    # Load Scaler
    try:
        mean = np.load(os.path.join(npy_dir, "scaler_mean.npy"))
        scale = np.load(os.path.join(npy_dir, "scaler_scale.npy"))
    except FileNotFoundError:
        print("Error: Scaler files not found. Run training script first.")
        sys.exit(1)

    # Load MNIST (Cached by sklearn)
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Use the last 10000 as test set (standard split)
    X_test = X[60000:]
    y_test = y[60000:].astype(int)
    
    return X_test, y_test, mean, scale


def preprocess_image(image_f32, mean, scale):
    """Normalize, Scale, and Quantize exactly like the training script."""
    # 1. Normalize 0-255 to 0-1
    img_norm = image_f32 / 255.0
    
    # 2. Standard Scaler
    img_scaled = (img_norm - mean) / scale
    
    # 3. Quantize to int8 range
    img_quant = np.round(img_scaled * INPUT_SCALE)
    img_int8 = np.clip(img_quant, -128, 127).astype(np.int8)
    
    # 4. View as uint8 for UART transmission (Preserves 2's complement bits)
    return img_int8.view(np.uint8)


def run_integration_test():
    X_test, y_test, mean, scale = load_data_and_scaler()
    
    # Pick random samples or first N samples
    indices = range(TEST_SAMPLES)
    
    print(f"\nStarting Integration Test on {TEST_SAMPLES} images...")
    print(f"Port: {COM_PORT}, Baud: {BAUD_RATE}")
    print("-" * 60)
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(1) # Allow reset
        ser.reset_input_buffer()
        
        y_pred = []
        y_true = []
        
        for i, idx in enumerate(indices):
            target_label = y_test[idx]
            y_true.append(target_label)
            
            # Prepare data
            img_data = preprocess_image(X_test[idx], mean, scale)
            
            # 1. Send Image
            ser.write(IMG_START_MARKER)
            ser.write(img_data.tobytes())
            ser.write(IMG_END_MARKER)
            ser.flush()
            
            # 2. Wait for Inference (80us is fast, but UART is slow. 0.05s is safe)
            time.sleep(0.05)
            
            # 3. Request Result
            ser.reset_input_buffer()
            ser.write(DIGIT_READ_REQUEST)
            
            # 4. Read Result
            resp = ser.read(1)
            
            if len(resp) != 1:
                print(f"Error: Timeout on image {i}")
                y_pred.append(-1)
            else:
                pred = int.from_bytes(resp, byteorder='little') & 0x0F
                y_pred.append(pred)
                
            # Live Status
            status = "PASS" if y_pred[-1] == target_label else "FAIL"
            print(f"Img {i:3d} | Label: {target_label} | Pred: {y_pred[-1]:2d} | {status}")

        ser.close()
        
        # Final Stats
        acc = accuracy_score(y_true, y_pred)
        print("-" * 60)
        print(f"Final Accuracy: {acc * 100:.2f}%")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
    except serial.SerialException as e:
        print(f"Serial Error: {e}")


if __name__ == "__main__":
    run_integration_test()
