import serial
import time
import sys
import os
import argparse
import numpy as np

# Configuration
DEFAULT_PORT = "COM7"
DEFAULT_BAUD = 115200
BIN_DIR = "../outputs/bin"

# Protocol Markers (Must match uart_router.v)
START_MARKER = np.array([0xAA, 0x55], dtype=np.uint8)
END_MARKER   = np.array([0x55, 0xAA], dtype=np.uint8)

def load_files(base_path):
    """Load weights from binary files into numpy arrays."""
    files = [
        ("conv1_weights.bin", "L1 Weights"),
        ("conv1_biases.bin",  "L1 Biases"),
        ("conv2_weights.bin", "L2 Weights"),
        ("conv2_biases.bin",  "L2 Biases"),
        ("dense_weights.bin", "Dense Weights"),
        ("dense_biases.bin",  "Dense Biases")
    ]
    
    data_chunks = []
    
    print(f"Loading files from {base_path}...")
    
    for filename, label in files:
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}\nDid you run train_cnn.py?")
            
        chunk = np.fromfile(path, dtype=np.uint8)
        data_chunks.append(chunk)
        print(f"  {label:<15}: {chunk.size:>6} bytes")
    
    # Calculate total payload size (excluding markers)
    payload_size = sum(c.size for c in data_chunks)
    print(f"  {'Total Payload':<15}: {payload_size:>6} bytes")
    
    # Validation
    EXPECTED_SIZE = 12984 
    if payload_size != EXPECTED_SIZE:
        print(f"\nWARNING: Payload size ({payload_size}) does not match FPGA expected size ({EXPECTED_SIZE})!")
        print("Check uart_router.v 'WEIGHT_SIZE' parameter.")
    
    return data_chunks, payload_size

def send_weights(port, baud):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(script_dir, BIN_DIR)
    
    try:
        chunks, payload_size = load_files(bin_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Concatenate everything
    all_data = np.concatenate([START_MARKER] + chunks + [END_MARKER])
    total_packet_size = all_data.size
    
    print(f"\nTotal Transmission packet: {total_packet_size} bytes")

    # Connect to Serial
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2) 
    except serial.SerialException as e:
        print(f"Error opening port {port}: {e}")
        return 1

    print(f"Sending to {port} (SLOW & SAFE MODE)...")
    start_time = time.time()
    
    # --- SLOW MODE SETTINGS ---
    # Reduced chunk size and increased delay to prevent buffer overflow
    CHUNK_SIZE = 16      
    DELAY = 0.020        # 20ms delay
    # --------------------------

    bytes_sent = 0
    raw_bytes = all_data.tobytes()
    
    for i in range(0, len(raw_bytes), CHUNK_SIZE):
        chunk = raw_bytes[i : i + CHUNK_SIZE]
        ser.write(chunk)
        ser.flush()
        bytes_sent += len(chunk)
        
        # Progress Bar
        progress = (bytes_sent / total_packet_size) * 100
        sys.stdout.write(f"\rProgress: {progress:.1f}% ({bytes_sent}/{total_packet_size})")
        sys.stdout.flush()
        
        time.sleep(DELAY) 

    print("\nDone.")
    elapsed = time.time() - start_time
    print(f"Time elapsed: {elapsed:.2f}s")
    
    ser.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    args = parser.parse_args()
    
    sys.exit(send_weights(args.port, args.baud))