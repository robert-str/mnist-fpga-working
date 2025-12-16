"""
Send neural network weights to FPGA via UART.

Protocol:
  - Start marker: 0xAA 0x55
  - Data bytes: all weight/bias files concatenated
  - End marker: 0x55 0xAA

Usage:
  python send_weights.py COM3
  python send_weights.py COM3 --baud 115200
"""

import serial
import os
import sys
import time
import argparse

# Protocol markers
START_MARKER = bytes([0xAA, 0x55])
END_MARKER = bytes([0x55, 0xAA])

# Order of files to send (must match FPGA memory layout expectations)
FILES_TO_SEND = [
    "L1_weights.bin",
    "L1_biases.bin",
    "L2_weights.bin",
    "L2_biases.bin",
    "L3_weights.bin",
    "L3_biases.bin",
]


def main():
    parser = argparse.ArgumentParser(description="Send weights to FPGA via UART")
    parser.add_argument("port", help="Serial port (e.g., COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--dir", default="BIN_files", help="Directory with .bin files")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, args.dir)
    
    # Collect all data
    all_data = bytearray()
    
    print("Loading binary files...")
    print("-" * 50)
    
    for filename in FILES_TO_SEND:
        filepath = os.path.join(bin_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"ERROR: {filename} not found!")
            sys.exit(1)
        
        with open(filepath, "rb") as f:
            file_data = f.read()
        
        all_data.extend(file_data)
        print(f"  {filename:20} : {len(file_data):6} bytes")
    
    print("-" * 50)
    print(f"  Total data          : {len(all_data):6} bytes")
    print()
    
    # Check for accidental marker bytes in data
    # (This would cause early termination - very unlikely but let's check)
    marker_check = bytes(all_data)
    if END_MARKER in marker_check:
        print("WARNING: End marker sequence (0x55 0xAA) found in data!")
        print("         This may cause early termination.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Open serial port
    print(f"Opening {args.port} at {args.baud} baud...")
    
    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
    except serial.SerialException as e:
        print(f"ERROR: Could not open serial port: {e}")
        sys.exit(1)
    
    # Wait a moment for FPGA to be ready
    print("Waiting for FPGA to be ready...")
    time.sleep(0.5)
    
    # Calculate expected transmission time
    total_bytes = len(START_MARKER) + len(all_data) + len(END_MARKER)
    expected_time = (total_bytes * 10) / args.baud  # 10 bits per byte (8 data + start + stop)
    
    print()
    print(f"Sending {total_bytes} bytes...")
    print(f"Expected time: {expected_time:.2f} seconds")
    print()
    
    start_time = time.time()
    
    # Send start marker
    print("Sending start marker (0xAA 0x55)...")
    ser.write(START_MARKER)
    ser.flush()
    
    # Send data with progress indicator
    print("Sending data", end="", flush=True)
    chunk_size = 256
    bytes_sent = 0
    
    for i in range(0, len(all_data), chunk_size):
        chunk = all_data[i:i + chunk_size]
        ser.write(chunk)
        bytes_sent += len(chunk)
        
        # Progress indicator
        progress = bytes_sent / len(all_data) * 100
        print(f"\rSending data: {bytes_sent:6}/{len(all_data)} bytes ({progress:5.1f}%)", end="", flush=True)
        
        # Small delay to prevent buffer overflow (optional, usually not needed)
        # time.sleep(0.001)
    
    ser.flush()
    print()
    
    # Send end marker
    print("Sending end marker (0x55 0xAA)...")
    ser.write(END_MARKER)
    ser.flush()
    
    elapsed_time = time.time() - start_time
    
    print()
    print("=" * 50)
    print("TRANSFER COMPLETE!")
    print(f"  Bytes sent: {total_bytes}")
    print(f"  Time: {elapsed_time:.2f} seconds")
    print(f"  Actual rate: {total_bytes * 8 / elapsed_time:.0f} bits/sec")
    print("=" * 50)
    print()
    print("Check FPGA LEDs:")
    print("  - led[3] ON = Transfer successful")
    print("  - led[4] ON = Error (overflow)")
    print()
    
    ser.close()


if __name__ == "__main__":
    main()


