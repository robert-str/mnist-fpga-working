"""
Send model parameters (weights and biases) to FPGA via UART.

Protocol:
  - Start marker: 0xAA 0x55 (2 bytes)
  - Data: weights (7840 bytes) + biases (40 bytes) = 7880 bytes
  - End marker: 0x55 0xAA (2 bytes)

Total transmission: 7884 bytes

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
    
    # Paths to separate binary files
    weights_path = os.path.join(bin_dir, "W.bin")
    biases_path = os.path.join(bin_dir, "B.bin")
    
    # Check if files exist
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found!")
        print("Please run convert_to_binary.py first to create the binary files.")
        return 1
    
    if not os.path.exists(biases_path):
        print(f"ERROR: {biases_path} not found!")
        print("Please run convert_to_binary.py first to create the binary files.")
        return 1
    
    # Read binary data (weights first, then biases)
    with open(weights_path, 'rb') as f:
        weights_data = f.read()
    with open(biases_path, 'rb') as f:
        biases_data = f.read()
    
    data = weights_data + biases_data
    
    print(f"Softmax Regression Model Parameter Upload")
    print(f"=" * 50)
    print(f"Weights file: {weights_path} ({len(weights_data)} bytes)")
    print(f"Biases file:  {biases_path} ({len(biases_data)} bytes)")
    print(f"Total size:   {len(data)} bytes")
    print(f"  - Weights: 7840 bytes (784 x 10 x 8-bit)")
    print(f"  - Biases:  40 bytes (10 x 32-bit)")
    print(f"Port:        {port}")
    print(f"Baud rate:   {baud}")
    print(f"=" * 50)
    
    # Verify expected size
    expected_size = 7840 + 40  # weights + biases
    if len(data) != expected_size:
        print(f"WARNING: Expected {expected_size} bytes, got {len(data)} bytes!")
        print("Continuing anyway...")
    
    # Calculate estimated time
    bits_per_byte = 10  # 1 start + 8 data + 1 stop
    total_bytes = 2 + len(data) + 2  # start + data + end
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
        send_with_progress(ser, data, chunk_size=256, description="Sending parameters")
        
        # Send end marker
        print("Sending end marker (0x55 0xAA)...")
        ser.write(END_MARKER)
        
        # Wait for transmission to complete
        ser.flush()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'=' * 50}")
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


