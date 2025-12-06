"""
Send LeNet-5 model parameters (weights, biases, and shifts) to FPGA via UART.

Protocol:
  - Start marker: 0xAA 0x55 (2 bytes)
  - Data: all layer weights and biases in order
  - Shift values: 4 bytes (conv1_shift, conv2_shift, fc1_shift, fc2_shift)
  - End marker: 0x55 0xAA (2 bytes)

Layer sizes:
  - conv1: weights 150 bytes (6*1*5*5), biases 24 bytes (6*4)
  - conv2: weights 2400 bytes (16*6*5*5), biases 64 bytes (16*4)
  - fc1:   weights 48000 bytes (120*400), biases 480 bytes (120*4)
  - fc2:   weights 10080 bytes (84*120), biases 336 bytes (84*4)
  - fc3:   weights 840 bytes (10*84), biases 40 bytes (10*4)
  - shifts: 4 bytes (conv1, conv2, fc1, fc2)

Total: 61470 bytes (weights) + 944 bytes (biases) + 4 bytes (shifts) = 62418 bytes

Usage:
  python send_weights.py [COM_PORT] [BAUD_RATE]
  
  Defaults: COM3, 115200
  
Examples:
  python send_weights.py              # Use defaults
  python send_weights.py COM5         # Use COM5, 115200 baud
  python send_weights.py COM5 9600    # Use COM5, 9600 baud
"""

import serial
import time
import sys
import os


# Protocol markers
START_MARKER = bytes([0xAA, 0x55])
END_MARKER = bytes([0x55, 0xAA])

# Layer configuration
LAYERS = [
    ("conv1", 150, 24),      # 6*1*5*5, 6*4
    ("conv2", 2400, 64),     # 16*6*5*5, 16*4
    ("fc1", 48000, 480),     # 120*400, 120*4
    ("fc2", 10080, 336),     # 84*120, 84*4
    ("fc3", 840, 40),        # 10*84, 10*4
]


def send_with_progress(ser, data, chunk_size=256, description="Sending"):
    """Send data with progress indicator."""
    total = len(data)
    sent = 0
    
    print(f"{description}: ", end="", flush=True)
    
    while sent < total:
        end = min(sent + chunk_size, total)
        chunk = data[sent:end]
        ser.write(chunk)
        sent = end
        
        # Progress bar
        progress = int(50 * sent / total)
        print(f"\r{description}: [{'=' * progress}{' ' * (50 - progress)}] {sent}/{total} bytes", end="", flush=True)
        
        # Small delay to prevent buffer overflow
        time.sleep(0.01)
    
    print()  # New line after progress


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "..", "outputs", "bin")
    
    # Parse command line arguments
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    
    print(f"LeNet-5 Model Parameter Upload")
    print(f"=" * 60)
    print(f"Port:      {port}")
    print(f"Baud rate: {baud}")
    print(f"=" * 60)
    
    # Load all layer data
    all_data = bytearray()
    total_weights = 0
    total_biases = 0
    
    print("\nLoading layer parameters:")
    print("-" * 60)
    
    for layer_name, weight_size, bias_size in LAYERS:
        weights_path = os.path.join(bin_dir, f"{layer_name}_weights.bin")
        biases_path = os.path.join(bin_dir, f"{layer_name}_biases.bin")
        
        # Check if files exist
        if not os.path.exists(weights_path):
            print(f"ERROR: {weights_path} not found!")
            print("Please run convert_to_binary.py first to create the binary files.")
            return 1
        
        if not os.path.exists(biases_path):
            print(f"ERROR: {biases_path} not found!")
            print("Please run convert_to_binary.py first to create the binary files.")
            return 1
        
        # Read binary data
        with open(weights_path, 'rb') as f:
            weights_data = f.read()
        with open(biases_path, 'rb') as f:
            biases_data = f.read()
        
        # Verify sizes
        if len(weights_data) != weight_size:
            print(f"WARNING: {layer_name} weights: expected {weight_size}, got {len(weights_data)}")
        if len(biases_data) != bias_size:
            print(f"WARNING: {layer_name} biases: expected {bias_size}, got {len(biases_data)}")
        
        print(f"  {layer_name:6s}: weights {len(weights_data):6d} bytes, biases {len(biases_data):4d} bytes")
        
        all_data.extend(weights_data)
        all_data.extend(biases_data)
        total_weights += len(weights_data)
        total_biases += len(biases_data)
    
    print("-" * 60)
    print(f"  Total:  weights {total_weights:6d} bytes, biases {total_biases:4d} bytes")
    
    # Load shift values for per-layer dynamic scaling
    shifts_path = os.path.join(bin_dir, "shifts.bin")
    if os.path.exists(shifts_path):
        with open(shifts_path, 'rb') as f:
            shifts_data = f.read()
        if len(shifts_data) != 4:
            print(f"WARNING: shifts.bin has {len(shifts_data)} bytes, expected 4")
        all_data.extend(shifts_data)
        print(f"  Shifts: {len(shifts_data)} bytes (conv1={shifts_data[0]}, conv2={shifts_data[1]}, fc1={shifts_data[2]}, fc2={shifts_data[3]})")
    else:
        print(f"WARNING: {shifts_path} not found! Using default shifts (7)")
        all_data.extend(bytes([7, 7, 7, 7]))  # Default shift of 7 for backwards compatibility
    
    print(f"  Grand total: {len(all_data)} bytes")
    
    # Calculate estimated time
    bits_per_byte = 10  # 1 start + 8 data + 1 stop
    total_bytes = 2 + len(all_data) + 2  # start + data + end
    estimated_time = total_bytes * bits_per_byte / baud
    
    print(f"\nEstimated transmission time: {estimated_time:.2f} seconds")
    
    try:
        # Open serial port
        print(f"\nOpening {port}...")
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(0.5)  # Wait for connection to stabilize
        
        # Clear any pending data
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        start_time = time.time()
        
        # Send start marker
        print("Sending start marker (0xAA 0x55)...")
        ser.write(START_MARKER)
        time.sleep(0.1)
        
        # Send data with progress
        send_with_progress(ser, all_data, chunk_size=256, description="Sending parameters")
        
        # Send end marker
        print("Sending end marker (0x55 0xAA)...")
        ser.write(END_MARKER)
        
        # Wait for transmission to complete
        ser.flush()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'=' * 60}")
        print(f"Transmission complete!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Effective rate: {total_bytes / elapsed_time:.0f} bytes/sec")
        print(f"\nCheck FPGA LEDs for status:")
        print(f"  - LED[1] ON: Waiting for start")
        print(f"  - LED[2] ON: Receiving data")
        print(f"  - LED[3] ON: Transfer complete (success)")
        print(f"  - LED[4] ON: Error (overflow)")
        
        ser.close()
        return 0
        
    except serial.SerialException as e:
        print(f"\nERROR: Could not open serial port: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if the correct COM port is specified")
        print("  2. Make sure no other program is using the port")
        print("  3. Verify the FPGA is connected and powered on")
        return 1
    except KeyboardInterrupt:
        print("\n\nTransfer cancelled by user.")
        if 'ser' in locals() and ser.is_open:
            ser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
