import serial
import time
import os
import struct
import numpy as np
import argparse
import sys
from torchvision import datasets, transforms

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_PORT = "COM7"  # Update this to your specific COM port
DEFAULT_BAUD = 115200
BIN_DIR = "../outputs/bin"
NPY_DIR = "../outputs/npy"
OUTPUT_FILE = "../outputs/txt/comparison.txt"

# Protocol Markers
IMG_START = bytes([0xBB, 0x66])
IMG_END   = bytes([0x66, 0xBB])
CMD_READ_SCORES = bytes([0xCD])

# Shift parameter (Must match Training & FPGA)
SHIFT_CONV = 8

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def max_pool_2x2(input_vol):
    """
    Simulates 2x2 Max Pooling with Stride 2.
    Input: (Channels, Height, Width)
    Output: (Channels, Height/2, Width/2)
    """
    c, h, w = input_vol.shape
    new_h, new_w = h // 2, w // 2
    output_vol = np.zeros((c, new_h, new_w), dtype=np.int32)
    
    for ch in range(c):
        for r in range(new_h):
            for col in range(new_w):
                # Extract 2x2 window
                window = input_vol[ch, r*2 : r*2+2, col*2 : col*2+2]
                output_vol[ch, r, col] = np.max(window)
                
    return output_vol

def convolve_layer(input_vol, weights, biases, shift):
    """
    Standard Multi-Channel Convolution.
    Input: (In_Channels, H, W)
    Weights: (Out_Channels, In_Channels, 3, 3)
    Output: (Out_Channels, H-2, W-2)
    """
    in_ch, h, w = input_vol.shape
    out_ch, _, k_h, k_w = weights.shape
    out_h, out_w = h - 2, w - 2
    
    output_vol = np.zeros((out_ch, out_h, out_w), dtype=np.int32)

    for f in range(out_ch):
        bias_val = biases[f]
        for r in range(out_h):
            for c in range(out_w):
                acc = 0
                # Sum over all input channels
                for ch in range(in_ch):
                    window = input_vol[ch, r:r+3, c:c+3]
                    w_kernel = weights[f, ch]
                    acc += np.sum(window * w_kernel)
                
                # Add Bias
                acc += bias_val
                
                # FPGA Pipeline Steps:
                acc = acc >> shift             # 1. Arithmetic Right Shift
                if acc < 0: acc = 0            # 2. ReLU
                if acc > 127: acc = 127        # 3. Saturation
                
                output_vol[f, r, c] = acc
                
    return output_vol

# ==========================================
# 2. BIT-EXACT SIMULATION (2-LAYER CNN)
# ==========================================
def simulate_cnn_inference(image, weights_dict):
    """
    Simulates the 2-Layer FPGA CNN pipeline exactly.
    """
    # Unpack weights
    c1_w, c1_b = weights_dict['c1']
    c2_w, c2_b = weights_dict['c2']
    dw, db     = weights_dict['dense']

    # 1. Input Image: (1, 28, 28)
    img_3d = image.reshape(1, 28, 28).astype(np.int32)
    
    # 2. Layer 1: Conv -> ReLU -> Pool
    # Weights: (16 filters, 1 ch, 3, 3)
    c1_w_reshaped = c1_w.reshape(16, 1, 3, 3).astype(np.int32)
    
    # Conv Output: (16, 26, 26)
    x = convolve_layer(img_3d, c1_w_reshaped, c1_b, SHIFT_CONV)
    # Pool Output: (16, 13, 13)
    x = max_pool_2x2(x)

    # 3. Layer 2: Conv -> ReLU -> Pool
    # Weights: (32 filters, 16 ch, 3, 3)
    c2_w_reshaped = c2_w.reshape(32, 16, 3, 3).astype(np.int32)
    
    # Conv Output: (32, 11, 11)
    x = convolve_layer(x, c2_w_reshaped, c2_b, SHIFT_CONV)
    # Pool Output: (32, 5, 5)
    x = max_pool_2x2(x)
    
    # 4. Dense Layer
    # Flatten: (32, 5, 5) -> 800
    flattened = x.flatten().astype(np.int32)
    
    # Reshape Dense: (10 classes, 800 inputs)
    dw_reshaped = dw.reshape(10, 800).astype(np.int32)
    
    scores = np.zeros(10, dtype=np.int32)
    for c in range(10):
        dot_prod = np.dot(flattened, dw_reshaped[c])
        scores[c] = dot_prod + db[c]
        
    return scores

# ==========================================
# 3. UTILITIES
# ==========================================
def load_files():
    """Load weights and normalization parameters."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(script_dir, BIN_DIR)
    npy_path = os.path.join(script_dir, NPY_DIR)

    # Load Weights (New Naming Scheme)
    try:
        c1_w = np.fromfile(os.path.join(bin_path, "conv1_weights.bin"), dtype=np.int8)
        c1_b = np.fromfile(os.path.join(bin_path, "conv1_biases.bin"), dtype=np.int32)
        c2_w = np.fromfile(os.path.join(bin_path, "conv2_weights.bin"), dtype=np.int8)
        c2_b = np.fromfile(os.path.join(bin_path, "conv2_biases.bin"), dtype=np.int32)
        dw = np.fromfile(os.path.join(bin_path, "dense_weights.bin"), dtype=np.int8)
        db = np.fromfile(os.path.join(bin_path, "dense_biases.bin"), dtype=np.int32)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Weight file not found! {e}")
        print("Did you run `train_cnn.py` to generate the new weights?")
        exit(1)

    # Load Norm Params
    mean = np.load(os.path.join(npy_path, "norm_mean.npy"))
    std = np.load(os.path.join(npy_path, "norm_std.npy"))

    weights = {
        'c1': (c1_w, c1_b),
        'c2': (c2_w, c2_b),
        'dense': (dw, db)
    }
    return weights, (mean, std)

def get_mnist_data(num_images=100):
    """Load first N images from MNIST test set."""
    transform = transforms.Compose([transforms.ToTensor()])
    # Adjust root path if needed
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    images = []
    labels = []
    for i in range(num_images):
        img, label = dataset[i]
        img_np = (img.squeeze().numpy() * 255).astype(np.uint8).flatten()
        images.append(img_np)
        labels.append(label)
    return images, labels

def preprocess(image, mean, std):
    """Normalize and Quantize Image (matches FPGA input format)."""
    x = image.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = np.clip(np.round(x * 127.0), -128, 127).astype(np.int8)
    return x

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate")
    parser.add_argument("--index", type=int, default=0, help="Start index in MNIST test set")
    parser.add_argument("--count", type=int, default=100, help="Number of images to test")
    args = parser.parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, OUTPUT_FILE)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    output_file = open(log_path, "w", encoding='utf-8')

    print("Loading weights and data...")
    weights, (mean, std) = load_files()
    
    # Load enough images to cover the requested range
    required_images = args.index + args.count
    images, labels = get_mnist_data(required_images)

    # Open UART
    try:
        print(f"Opening {args.port} at {args.baud}...")
        ser = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(2) # Wait for DTR/RTS
    except Exception as e:
        print(f"Error opening {args.port}: {e}")
        output_file.close()
        return

    print(f"Running comparison on {args.port}...")
    print(f"Testing {args.count} images starting from index {args.index}")
    
    output_file.write(f"Python (2-Layer CNN) vs FPGA Comparison\n")
    output_file.write(f"Start Index: {args.index}, Count: {args.count}\n")
    output_file.write("=" * 80 + "\n\n")

    correct_matches = 0
    accuracy_correct = 0

    for idx in range(args.count):
        real_idx = args.index + idx
        if real_idx >= len(images): break

        img_raw = images[real_idx]
        label = labels[real_idx]
        img_input = preprocess(img_raw, mean, std)
        
        # --- A. PYTHON SIMULATION ---
        expected_scores = simulate_cnn_inference(img_input, weights)
        py_pred = np.argmax(expected_scores)

        # --- B. FPGA INFERENCE ---
        ser.reset_input_buffer()
        ser.write(IMG_START)
        
        # Chunking
        img_bytes = img_input.tobytes()
        for i in range(0, len(img_bytes), 64):
            ser.write(img_bytes[i:i+64])
            time.sleep(0.005) 
            
        ser.write(IMG_END)
        time.sleep(0.15) # Wait for inference
        
        ser.write(CMD_READ_SCORES)
        response = ser.read(40)
        
        if len(response) != 40:
            print(f"Index {real_idx}: Timeout/Error receiving scores.")
            output_file.write(f"Image {real_idx} | TIMEOUT\n")
            continue
            
        fpga_scores = np.array(struct.unpack('<10i', response), dtype=np.int32)
        fpga_pred = np.argmax(fpga_scores)

        # --- C. COMPARE ---
        is_match = np.array_equal(expected_scores, fpga_scores)
        if is_match: correct_matches += 1
        
        if fpga_pred == label: accuracy_correct += 1
        
        status = "MATCH" if is_match else "MISMATCH"
        
        # Console output - compact
        print(f"Img {real_idx:3d}: Label={label} Py={py_pred} FPGA={fpga_pred} [{status}]")
        
        # Log output - detailed
        output_file.write(f"Image {real_idx} | Label: {label} | Status: {status}\n")
        output_file.write(f"Python: {expected_scores.tolist()}\n")
        output_file.write(f"FPGA:   {fpga_scores.tolist()}\n")
        if not is_match:
             output_file.write(f"Diff:   {(fpga_scores - expected_scores).tolist()}\n")
        output_file.write("-" * 50 + "\n")

    # Summary
    print("\n" + "="*30)
    print(" SUMMARY")
    print("="*30)
    print(f"Total Tests: {args.count}")
    print(f"FPGA Accuracy: {accuracy_correct}/{args.count} ({(accuracy_correct/args.count)*100:.1f}%)")
    print(f"Bit-Exact Matches: {correct_matches}/{args.count} ({(correct_matches/args.count)*100:.1f}%)")
    
    output_file.close()
    ser.close()

if __name__ == "__main__":
    main()