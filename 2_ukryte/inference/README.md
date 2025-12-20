# Two-Hidden-Layer Neural Network - FPGA Inference System

FPGA implementation of weight loading system for a two-hidden-layer neural network (2_ukryte model) on Basys3 board.

## Architecture

```
Input (784) → Layer1 (16) → ReLU → Layer2 (16) → ReLU → Layer3 (10)
```

### Model Parameters

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| L1 (784→16) | 12,544 bytes | 64 bytes | 12,608 bytes |
| L2 (16→16) | 256 bytes | 64 bytes | 320 bytes |
| L3 (16→10) | 160 bytes | 40 bytes | 200 bytes |
| **Total** | **12,960 bytes** | **168 bytes** | **13,128 bytes** |

## Directory Structure

```
2_ukryte/inference/
├── rtl/
│   ├── load_weights.v   - Weight loader with 6 separate BRAMs
│   ├── uart_router.v    - UART protocol router (13,128 bytes)
│   └── uart_rx.v        - UART receiver module
└── constraints/
    └── pins.xdc         - Basys3 pin constraints
```

## Memory Organization

The weight loader uses **6 separate Block RAMs**:

1. **L1_weight_bram** [12,544 bytes] - Layer 1 weights (8-bit signed)
2. **L1_bias_bram** [16 entries] - Layer 1 biases (32-bit signed)
3. **L2_weight_bram** [256 bytes] - Layer 2 weights (8-bit signed)
4. **L2_bias_bram** [16 entries] - Layer 2 biases (32-bit signed)
5. **L3_weight_bram** [160 bytes] - Layer 3 weights (8-bit signed)
6. **L3_bias_bram** [10 entries] - Layer 3 biases (32-bit signed)

### Data Transfer Order

Bytes are received via UART in this order:

```
Bytes 0-12543:      L1_weights (12,544 bytes)
Bytes 12544-12607:  L1_biases (64 bytes)
Bytes 12608-12863:  L2_weights (256 bytes)
Bytes 12864-12927:  L2_biases (64 bytes)
Bytes 12928-13087:  L3_weights (160 bytes)
Bytes 13088-13127:  L3_biases (40 bytes)
```

## UART Protocol

### Weight Loading
- **Start marker:** `0xAA 0x55`
- **Data:** 13,128 bytes (weights + biases)
- **End marker:** `0x55 0xAA`
- **Total transmission:** 13,132 bytes

### Baud Rate
- 115,200 baud (configurable in `uart_rx.v`)
- Estimated transfer time: ~1.1 seconds

## LED Status Indicators

During weight loading:
- `led[0]`: Blinks when receiving bytes
- `led[1]`: Waiting for start marker
- `led[2]`: Receiving data
- `led[3]`: Transfer complete (SUCCESS)
- `led[4]`: Error (overflow detected)
- `led[15:8]`: Progress indicator (lower 8 bits of address)

## Usage

### 1. Synthesize the Design

In Vivado:
1. Create a new project for Basys3 (XC7A35T-1CPG236C)
2. Add RTL files:
   - `rtl/load_weights.v`
   - `rtl/uart_router.v`
   - `rtl/uart_rx.v`
3. Add constraint file:
   - `constraints/pins.xdc`
4. Set `load_weights_top` as the top module
5. Run synthesis, implementation, and generate bitstream

### 2. Load Weights to FPGA

```bash
# From project root
cd 2_ukryte/utils

# Send weights (Windows)
python send_weights.py COM3 --baud 115200 --dir ../outputs/bin

# Send weights (Linux)
python send_weights.py /dev/ttyUSB0 --baud 115200 --dir ../outputs/bin
```

### 3. Verify Transfer

Check the LEDs on the Basys3 board:
- ✅ **led[3] ON** = Transfer successful
- ❌ **led[4] ON** = Transfer failed (error)

## Module Interface

### weight_loader

```verilog
module weight_loader (
    input wire clk,
    input wire rst,
    
    // UART interface (from uart_router)
    input wire [7:0] rx_data,
    input wire rx_ready,
    
    // L1 read ports (for inference module)
    input wire [13:0] L1_weight_rd_addr,  // 0-12543
    output reg [7:0]  L1_weight_rd_data,
    input wire [3:0]  L1_bias_rd_addr,    // 0-15
    output reg [31:0] L1_bias_rd_data,
    
    // L2 read ports
    input wire [7:0]  L2_weight_rd_addr,  // 0-255
    output reg [7:0]  L2_weight_rd_data,
    input wire [3:0]  L2_bias_rd_addr,    // 0-15
    output reg [31:0] L2_bias_rd_data,
    
    // L3 read ports
    input wire [7:0]  L3_weight_rd_addr,  // 0-159
    output reg [7:0]  L3_weight_rd_data,
    input wire [3:0]  L3_bias_rd_addr,    // 0-9
    output reg [31:0] L3_bias_rd_data,
    
    // Status
    output reg transfer_done,
    output reg [15:0] led
);
```

## Key Differences from Regresja Model

| Feature | Regresja | 2_ukryte |
|---------|----------|----------|
| Layers | 1 (softmax regression) | 3 (2 hidden + output) |
| Total bytes | 7,880 | 13,128 |
| Weight BRAMs | 1 | 3 |
| Bias BRAMs | 1 | 3 |
| Read ports | 2 | 6 |
| Max address width | 13 bits | 14 bits |
| Byte counter width | 14 bits | 15 bits |

## Technical Details

### Quantization
- **Weights:** 8-bit signed integers (INT8)
- **Biases:** 32-bit signed integers (INT32)
- **Storage format:** Little-endian for biases

### Bias Assembly
Biases are sent as 4 bytes (little-endian) and assembled into 32-bit values:
```
Byte 0: bits [7:0]
Byte 1: bits [15:8]
Byte 2: bits [23:16]
Byte 3: bits [31:24]
```

### BRAM Synthesis Attributes
All memory arrays use:
```verilog
(* ram_style = "block" *)
```
This directive ensures synthesis tool infers Block RAM (not distributed RAM).

## Next Steps

To complete the full inference system:
1. Create `inference.v` module that reads from the 6 BRAMs
2. Implement layer computations with ReLU activations
3. Add image loading via UART (similar to weight loading)
4. Connect to 7-segment display for showing predictions

## References

- Based on: [`regresja/inference/rtl/load_weights.v`](../../regresja/inference/rtl/load_weights.v)
- Training script: [`2_ukryte/training/siec_2_ukryte.py`](../training/siec_2_ukryte.py)
- Weight sender: [`2_ukryte/utils/send_weights.py`](../utils/send_weights.py)
- Model architecture: [`2_ukryte/README.md`](../README.md)





