import os
import numpy as np

# Get the absolute path of the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script location
BIN_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "bin")
MEM_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "mem")

def check_files():
    print(f"=== Checking Consistency ===")
    print(f"Looking for binaries in: {BIN_DIR}")
    print(f"Looking for mem files in: {MEM_DIR}\n")
    
    # 1. Load the Binary Bias
    try:
        bin_path = os.path.join(BIN_DIR, "conv1_biases.bin")
        c1_b = np.fromfile(bin_path, dtype=np.int32)
        bias_0 = c1_b[0]
        print(f"[BINARY] Bias 0 (Int32): {bias_0}")
        print(f"[BINARY] Bias 0 (Hex):   {int(bias_0) & 0xFFFFFFFF:08x}")
    except Exception as e:
        print(f"[ERROR] Could not load binary: {e}")
        return

    # 2. Check the MEM file
    try:
        mem_path = os.path.join(MEM_DIR, "sim_conv_biases.mem")
        with open(mem_path, "r") as f:
            line = f.readline().strip()
            print(f"[MEM]    Bias 0 (File):  {line}")
            
            if line.lower() == f"{int(bias_0) & 0xFFFFFFFF:08x}".lower():
                print(">> MATCH: Bias file is consistent.")
            else:
                print(">> MISMATCH: Mem file does not match binary!")
    except Exception as e:
        print(f"[ERROR] Could not read mem file: {e}")

    print("-" * 30)

    # 3. Load the Binary Weight
    try:
        bin_path = os.path.join(BIN_DIR, "conv1_weights.bin")
        c1_w = np.fromfile(bin_path, dtype=np.int8)
        weight_0 = c1_w[0]
        print(f"[BINARY] Weight 0 (Int8): {weight_0}")
        print(f"[BINARY] Weight 0 (Hex):   {int(weight_0) & 0xFF:02x}")
    except Exception as e:
        print(f"[ERROR] Could not load binary weights: {e}")
        return

    # 4. Check the MEM file
    try:
        mem_path = os.path.join(MEM_DIR, "sim_conv_weights.mem")
        with open(mem_path, "r") as f:
            line = f.readline().strip()
            print(f"[MEM]    Weight 0 (File): {line}")
            
            if line.lower() == f"{int(weight_0) & 0xFF:02x}".lower():
                print(">> MATCH: Weight file is consistent.")
            else:
                print(">> MISMATCH: Mem file does not match binary!")
    except Exception as e:
        print(f"[ERROR] Could not read mem file: {e}")

if __name__ == "__main__":
    check_files()