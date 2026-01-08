# MNIST CNN on FPGA - Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Directory Structure](#directory-structure)
4. [CNN Model Architecture](#cnn-model-architecture)
5. [Python Training Pipeline](#python-training-pipeline)
6. [Quantization Strategy](#quantization-strategy)
7. [FPGA Implementation](#fpga-implementation)
8. [Communication Protocol](#communication-protocol)
9. [Testing Infrastructure](#testing-infrastructure)
10. [Memory Organization](#memory-organization)
11. [Migration Guide: Converting to LeNet-5](#migration-guide-converting-to-lenet-5)

---

## Project Overview

This project implements a complete end-to-end MNIST digit classification system running on a Basys3 FPGA (Artix-7 XC7A35T). The system achieves **99% accuracy** on MNIST test images with bit-exact results between Python simulation and FPGA hardware.

### Key Features

- **Two-layer CNN architecture** with 3x3 convolutions and max pooling
- **8-bit quantized inference** for efficient FPGA implementation
- **UART communication** at 115200 baud for weight loading and image inference
- **Real-time display** of predictions on 7-segment display
- **Comprehensive testing** infrastructure comparing Python and FPGA outputs

### Target Hardware

| Component | Specification |
|-----------|---------------|
| FPGA Board | Digilent Basys3 |
| FPGA Chip | Artix-7 XC7A35T-1CPG236C |
| System Clock | 100 MHz |
| UART Baud Rate | 115200 |
| Display | 4-digit 7-segment |
| Debug LEDs | 16 status LEDs |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              HOST PC                                     │
│  ┌─────────────┐    ┌────────────────┐    ┌──────────────────────────┐  │
│  │   Training  │───►│   Quantized    │───►│   UART Transmission      │  │
│  │   (PyTorch) │    │   Weights      │    │   (send_weights.py)      │  │
│  └─────────────┘    └────────────────┘    └────────────┬─────────────┘  │
│                                                        │                 │
│  ┌─────────────┐    ┌────────────────┐                │                 │
│  │   MNIST     │───►│   Preprocessed │────────────────┤                 │
│  │   Image     │    │   Pixels       │                │                 │
│  └─────────────┘    └────────────────┘                │                 │
└───────────────────────────────────────────────────────│─────────────────┘
                                                        │
                                              UART (115200 baud)
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            BASYS3 FPGA                                   │
│                                                                          │
│  ┌──────────────┐     ┌────────────────┐     ┌─────────────────────┐    │
│  │  UART Router │────►│  Weight Loader │────►│  Weight RAMs        │    │
│  │              │     └────────────────┘     │  - Conv Weights     │    │
│  │              │                            │  - Conv Biases      │    │
│  │              │     ┌────────────────┐     │  - Dense Weights    │    │
│  │              │────►│  Image Loader  │     │  - Dense Biases     │    │
│  │              │     └───────┬────────┘     └──────────┬──────────┘    │
│  │              │             │                         │               │
│  │              │             ▼                         │               │
│  │              │     ┌────────────────┐                │               │
│  │              │     │   Image RAM    │                │               │
│  │              │     │   (784 bytes)  │                │               │
│  │              │     └───────┬────────┘                │               │
│  │              │             │                         │               │
│  │              │             ▼                         ▼               │
│  │              │     ┌─────────────────────────────────────────┐       │
│  │              │     │           INFERENCE ENGINE (FSM)         │       │
│  │              │     │                                          │       │
│  │              │     │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │       │
│  │              │     │  │ Conv L1 │─►│ Pool L1 │─►│ Conv L2 │  │       │
│  │              │     │  └─────────┘  └─────────┘  └─────────┘  │       │
│  │              │     │       │                          │       │       │
│  │              │     │       ▼                          ▼       │       │
│  │              │     │  ┌─────────┐              ┌─────────┐   │       │
│  │              │     │  │Buffer A │              │ Pool L2 │   │       │
│  │              │     │  └─────────┘              └─────────┘   │       │
│  │              │     │                                │        │       │
│  │              │     │  ┌─────────┐                   ▼        │       │
│  │              │     │  │Buffer B │◄─────────────────────      │       │
│  │              │     │  └─────────┘                            │       │
│  │              │     │       │                                 │       │
│  │              │     │       ▼                                 │       │
│  │              │     │  ┌─────────┐     ┌─────────────────┐    │       │
│  │              │     │  │  Dense  │────►│ Predicted Digit │    │       │
│  │              │     │  └─────────┘     │    + Scores     │    │       │
│  │              │     └─────────────────────────────────────────┘       │
│  │              │                               │                       │
│  │              │◄──────────────────────────────┘                       │
│  │              │                                                       │
│  │              │     ┌────────────────┐     ┌────────────────┐         │
│  │              │────►│ Digit Reader   │────►│    UART TX     │────►TX  │
│  │              │     └────────────────┘     └────────────────┘         │
│  │              │     ┌────────────────┐            │                   │
│  │              │────►│ Scores Reader  │────────────┘                   │
│  └──────────────┘     └────────────────┘                                │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │  7-Segment Display: Shows predicted digit                   │         │
│  │  Status LEDs: Debug and status information                  │         │
│  └────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
mnist-fpga/
├── cnn/                              # Current active implementation
│   ├── training/
│   │   └── train_cnn.py              # Model training and quantization
│   │
│   ├── testing/
│   │   ├── testing_python_model.py   # Bit-exact Python simulation
│   │   ├── compare_fpga_vs_python.py # Full comparison harness
│   │   └── generate_cnn_vectors.py   # Test vector generator
│   │
│   ├── utils/
│   │   ├── send_weights.py           # Weight upload utility
│   │   └── send_image.py             # Image upload and inference
│   │
│   ├── inference/
│   │   ├── rtl/                      # Verilog source files
│   │   │   ├── top.v                 # Top-level module
│   │   │   ├── inference.v           # CNN inference FSM
│   │   │   ├── uart_router.v         # Protocol dispatcher
│   │   │   ├── uart_rx.v             # UART receiver
│   │   │   ├── uart_tx.v             # UART transmitter
│   │   │   ├── weight_loader.v       # Weight parsing
│   │   │   ├── image_loader.v        # Image parsing
│   │   │   ├── ram_cnn.v             # Working buffers
│   │   │   ├── conv_weights_ram.v    # Conv weight storage
│   │   │   ├── conv_biases_ram.v     # Conv bias storage
│   │   │   ├── dense_biases_ram.v    # Dense bias storage
│   │   │   ├── image_ram.v           # Input image storage
│   │   │   ├── scores_ram.v          # Output scores storage
│   │   │   ├── predicted_digit_ram.v # Result storage
│   │   │   ├── digit_reader.v        # Result readback
│   │   │   ├── scores_reader.v       # Scores readback
│   │   │   └── seven_segment_display.v
│   │   │
│   │   ├── tb/                       # Testbenches
│   │   │   └── tb_inference.v
│   │   │
│   │   └── constraints/
│   │       └── pins.xdc              # Basys3 pin mapping
│   │
│   └── outputs/                      # Generated files
│       ├── bin/                      # Binary weight files
│       ├── mem/                      # Hex memory files
│       └── npy/                      # Normalization params
│
├── le-net/                           # LeNet-5 implementation (reference)
├── 2_ukryte/                         # Two-hidden-layer MLP
├── regresja/                         # Softmax regression
├── data/                             # MNIST dataset
└── test_images/                      # Test PNGs
```

---

## CNN Model Architecture

### ImprovedCNN (Current Implementation)

The current model is a 2-layer CNN optimized for FPGA implementation:

```
INPUT: 28x28x1 grayscale image (784 bytes)
       Quantized to int8 range [-128, 127]

       ┌───────────────────────────────────────────┐
       │             INPUT (1x28x28)                │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  CONV LAYER 1                              │
       │  - 16 filters, 3x3 kernel                  │
       │  - No padding, stride 1                    │
       │  - Output: 16x26x26                        │
       │  - Activation: ReLU                        │
       │  - Bit shift: >>8 (div 256)                │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  MAX POOL 1                                │
       │  - 2x2 kernel, stride 2                    │
       │  - Output: 16x13x13                        │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  CONV LAYER 2                              │
       │  - 32 filters, 3x3 kernel                  │
       │  - 16 input channels                       │
       │  - No padding, stride 1                    │
       │  - Output: 32x11x11                        │
       │  - Activation: ReLU                        │
       │  - Bit shift: >>8 (div 256)                │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  MAX POOL 2                                │
       │  - 2x2 kernel, stride 2                    │
       │  - Output: 32x5x5                          │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  FLATTEN                                   │
       │  - 32x5x5 = 800 features                   │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  DENSE (Fully Connected)                   │
       │  - Input: 800                              │
       │  - Output: 10 class scores                 │
       │  - No activation (raw logits)              │
       └───────────────────┬───────────────────────┘
                          │
       ┌───────────────────▼───────────────────────┐
       │  OUTPUT: argmax(scores) = predicted digit  │
       └───────────────────────────────────────────┘
```

### Layer Dimensions Summary

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| Conv1 | 1x28x28 | 16x26x26 | 16x1x3x3 = 144 weights + 16 biases |
| Pool1 | 16x26x26 | 16x13x13 | 0 |
| Conv2 | 16x13x13 | 32x11x11 | 32x16x3x3 = 4,608 weights + 32 biases |
| Pool2 | 32x11x11 | 32x5x5 | 0 |
| Dense | 800 | 10 | 800x10 = 8,000 weights + 10 biases |

**Total Parameters:** 12,810 (weights) + 58 (biases) = 12,868 parameters

---

## Python Training Pipeline

### Training Script (`train_cnn.py`)

```python
# Key hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
INPUT_SCALE = 127.0   # Quantization scale for input
SHIFT_CONV = 8        # Bit shift after convolution (div 256)
```

### Training Process

1. **Data Loading**: MNIST dataset with standard normalization (mean=0.1307, std=0.3081)
2. **Model Definition**: PyTorch `ImprovedCNN` class
3. **Training Loop**: Adam optimizer with CrossEntropyLoss
4. **Weight Export**: Quantized weights exported to binary and hex formats

### Data Preprocessing Pipeline

```python
# Raw image [0, 255] from MNIST
image = dataset[i]

# 1. Convert to tensor [0, 1]
x = transforms.ToTensor()(image)

# 2. Normalize with MNIST statistics
x = (x - 0.1307) / 0.3081

# 3. Quantize to int8
x = round(x * 127.0)
x = clip(x, -128, 127)

# Result: int8 pixel values ready for FPGA
```

---

## Quantization Strategy

The project uses **post-training static quantization** to convert floating-point weights to 8-bit integers while maintaining accuracy.

### Quantization Equations

#### Weight Quantization (INT8)

```
W_scale = 127.0 / max(|W_float|)
W_int8 = clip(round(W_float * W_scale), -128, 127)
```

#### Bias Quantization (INT32)

The bias scale propagates through the network:

```
Layer 1:
  Input_scale = 127.0  (from normalized image quantization)
  W_scale_L1 = 127.0 / max(|conv1_weights|)
  Bias_scale_L1 = Input_scale * W_scale_L1
  Bias_L1_int32 = round(Bias_L1_float * Bias_scale_L1)

Layer 1 Output Scale:
  Output_scale_L1 = (Input_scale * W_scale_L1) / 2^SHIFT_CONV
                  = (Input_scale * W_scale_L1) / 256

Layer 2:
  W_scale_L2 = 127.0 / max(|conv2_weights|)
  Bias_scale_L2 = Output_scale_L1 * W_scale_L2
  Bias_L2_int32 = round(Bias_L2_float * Bias_scale_L2)

Dense Layer:
  Output_scale_L2 = (Output_scale_L1 * W_scale_L2) / 256
  W_scale_FC = 127.0 / max(|dense_weights|)
  Bias_scale_FC = Output_scale_L2 * W_scale_FC
  Bias_FC_int32 = round(Bias_FC_float * Bias_scale_FC)
```

### FPGA Arithmetic Pipeline

For each convolution output:

```verilog
// 1. Accumulate (32-bit signed)
acc = bias + sum(pixel * weight)    // All sign-extended to 32-bit

// 2. Scale down (arithmetic right shift)
temp = acc >>> 8                   // Divide by 256, preserve sign

// 3. ReLU
if (temp < 0) temp = 0

// 4. Saturation
if (temp > 127) temp = 127

// 5. Store as 8-bit
output = temp[7:0]
```

### Export File Formats

| File Type | Format | Description |
|-----------|--------|-------------|
| `.bin` | Binary | Raw bytes, direct memory load |
| `.mem` | Hex ASCII | One value per line, for Verilog `$readmemh` |
| `.npy` | NumPy | Normalization parameters |

#### Weight File Sizes

| File | Size (bytes) | Format |
|------|--------------|--------|
| conv1_weights.bin | 144 | 16x1x3x3 int8 |
| conv1_biases.bin | 64 | 16 int32 |
| conv2_weights.bin | 4,608 | 32x16x3x3 int8 |
| conv2_biases.bin | 128 | 32 int32 |
| dense_weights.bin | 8,000 | 10x800 int8 |
| dense_biases.bin | 40 | 10 int32 |
| **Total** | **12,984** | |

---

## FPGA Implementation

### Module Hierarchy

```
top.v
├── uart_router.v           # Parses UART stream, routes to loaders
│   └── uart_rx.v           # 8N1 UART receiver
│
├── weight_loader.v         # Parses weight packet into RAMs
│
├── image_loader.v          # Parses image packet into RAM
│
├── conv_weights_ram.v      # 4,752 bytes (L1+L2 weights)
├── conv_biases_ram.v       # 48 x 32-bit (L1+L2 biases)
├── dense_biases_ram.v      # 10 x 32-bit
├── image_ram.v             # 784 bytes
│
├── ram_cnn.v               # Working memory
│   ├── buffer_a            # 10,816 bytes (L1 conv output)
│   ├── buffer_b            # 2,704 bytes (pool outputs)
│   └── dense_weights       # 8,000 bytes
│
├── inference.v             # Main FSM engine
│
├── predicted_digit_ram.v   # 1 byte result
├── scores_ram.v            # 40 bytes (10 x int32)
│
├── digit_reader.v          # Response to 0xCC command
├── scores_reader.v         # Response to 0xCD command
├── uart_tx.v               # 8N1 UART transmitter
│
└── seven_segment_display.v # Display driver
```

### Inference FSM States

The inference engine is implemented as a 19-state finite state machine:

```
State                    | Description
-------------------------|-------------------------------------------
IDLE                     | Wait for start signal
L1_LOAD_BIAS             | Request bias from BRAM
L1_LOAD_BIAS_WAIT        | Wait 1 cycle for BRAM read
L1_PREFETCH              | Prefetch first pixel/weight
L1_CONV                  | 3x3 MAC loop (9 cycles per position)
L1_SAVE                  | Apply shift/ReLU/saturation, write result
L1_POOL                  | 2x2 max pooling (5 steps per position)
L2_LOAD_BIAS             | Request L2 bias
L2_LOAD_BIAS_WAIT        | Wait for BRAM
L2_PREFETCH              | Prefetch first activation/weight
L2_CONV                  | 3x3x16 MAC loop (144 cycles per position)
L2_SAVE                  | Apply shift/ReLU/saturation, write result
L2_POOL                  | 2x2 max pooling
DENSE_LOAD_BIAS          | Request FC bias
DENSE_LOAD_BIAS_WAIT     | Wait for BRAM
DENSE_PREFETCH           | Prefetch first feature/weight
DENSE_MULT               | 800 MAC operations per class
DENSE_NEXT               | Store score, track argmax
DONE_STATE               | Signal completion, return to IDLE
```

### Computation Counts

| Layer | Operations per output | Total outputs | Total MACs |
|-------|----------------------|---------------|------------|
| Conv1 | 1x3x3 = 9 | 16x26x26 = 10,816 | 97,344 |
| Conv2 | 16x3x3 = 144 | 32x11x11 = 3,872 | 557,568 |
| Dense | 800 | 10 | 8,000 |
| **Total** | | | **662,912** |

### RAM Architecture

All working RAMs use **distributed memory** (LUT-based) for single-cycle combinational reads:

```verilog
(* ram_style = "distributed" *)
reg [7:0] ram [0:SIZE-1];

// Combinational read (0-cycle latency)
assign rd_data = ram[rd_addr];

// Synchronous write
always @(posedge clk) begin
    if (wr_en) ram[wr_addr] <= wr_data;
end
```

This eliminates pipeline bubbles that would occur with BRAM's registered outputs.

---

## Communication Protocol

### UART Configuration

- **Baud Rate:** 115,200
- **Frame Format:** 8N1 (8 data bits, no parity, 1 stop bit)
- **Clock Frequency:** 100 MHz

### Protocol Markers

| Marker | Bytes | Direction | Purpose |
|--------|-------|-----------|---------|
| `0xAA 0x55` | Weight start | PC to FPGA | Begin weight transfer |
| `0x55 0xAA` | Weight end | PC to FPGA | End weight transfer |
| `0xBB 0x66` | Image start | PC to FPGA | Begin image transfer |
| `0x66 0xBB` | Image end | PC to FPGA | End image transfer |
| `0xCC` | Command | PC to FPGA | Request predicted digit |
| `0xCD` | Command | PC to FPGA | Request all scores |

### Weight Packet Structure (12,984 bytes)

```
Offset    Size      Data
----------------------------------------------
0         144       Conv1 weights (16x1x3x3)
144       64        Conv1 biases (16 x int32)
208       4,608     Conv2 weights (32x16x3x3)
4,816     128       Conv2 biases (32 x int32)
4,944     8,000     Dense weights (10x800)
12,944    40        Dense biases (10 x int32)
----------------------------------------------
Total:    12,984 bytes
```

### Image Packet Structure (784 bytes)

```
Offset    Size      Data
----------------------------------------------
0         784       Pixel values (28x28, row-major)
                    Preprocessed, quantized int8
----------------------------------------------
```

### Binary Safety

The protocol uses **payload-length-aware parsing** to prevent false end-marker detection:

```verilog
// Only check end markers AFTER receiving expected payload size
if (byte_count >= WEIGHT_SIZE &&
    prev_byte == WEIGHT_END1 &&
    rx_data == WEIGHT_END2) begin
    state <= DONE_W;
end
```

### Flow Control

To prevent UART buffer overflow, Python scripts use chunked transmission:

```python
def send_chunked(ser, data, chunk_size=32, delay=0.010):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        ser.write(chunk)
        ser.flush()
        time.sleep(delay)  # 10ms delay per chunk
```

---

## Testing Infrastructure

### Python Simulation (`testing_python_model.py`)

Provides **bit-exact simulation** of the FPGA inference pipeline:

```python
def convolve_layer(input_vol, weights, biases, shift):
    for each output position:
        acc = bias + sum(input * weight)

        # FPGA-exact pipeline:
        acc = acc >> shift        # Arithmetic right shift
        if acc < 0: acc = 0       # ReLU
        if acc > 127: acc = 127   # Saturation

        output = acc
```

### Comparison Framework (`compare_fpga_vs_python.py`)

Tests up to 3,000 MNIST images comparing Python and FPGA results:

```bash
python compare_fpga_vs_python.py --port COM7 --count 3000 --index 0
```

**Test Flow:**

1. Load and preprocess image
2. Run Python simulation - get scores and prediction
3. Send image to FPGA via UART
4. Wait for inference completion (~150ms)
5. Request scores via `0xCD` command
6. Compare bit-exact match and accuracy

### Test Vector Generation (`generate_cnn_vectors.py`)

Generates `.mem` files for Verilog testbenches:

- `test_pixels.mem` - Preprocessed pixel values
- `test_scores.mem` - Expected output scores
- `test_preds.mem` - Expected predictions
- `test_labels.mem` - Ground truth labels

---

## Memory Organization

### Total Memory Footprint

```
Weight Storage (Read-only after loading):
  ├── Conv weights:      4,752 bytes
  ├── Conv biases:         192 bytes (48 x 32-bit)
  ├── Dense weights:     8,000 bytes
  └── Dense biases:         40 bytes (10 x 32-bit)
  Subtotal:             12,984 bytes

Working Memory (Distributed RAM):
  ├── Buffer A:         10,816 bytes (L1 output: 16x26x26)
  └── Buffer B:          2,704 bytes (Pool outputs: max of 16x13x13)
  Subtotal:             13,520 bytes

Input/Output:
  ├── Image RAM:           784 bytes
  ├── Scores RAM:           40 bytes
  └── Digit RAM:             1 byte
  Subtotal:                825 bytes

-----------------------------------------
TOTAL:                 ~27,329 bytes
```

### Address Mapping

**Conv Weights RAM (4,752 bytes):**
```
Address Range          | Content
-----------------------|----------------------
0-143                  | Conv1: 16 filters x 9
144-4,751              | Conv2: 32 filters x 144
```

**Conv Biases RAM (48 entries):**
```
Address    | Content
-----------|----------------------
0-15       | Conv1 biases (16)
16-47      | Conv2 biases (32)
```

**Buffer A Address Calculation:**
```
L1 Conv output: addr = filter_idx * 676 + row * 26 + col
                where 676 = 26 * 26

L2 Conv output: addr = filter_idx * 121 + row * 11 + col
                where 121 = 11 * 11
```

**Buffer B Address Calculation:**
```
L1 Pool output: addr = filter_idx * 169 + row * 13 + col
                where 169 = 13 * 13

L2 Pool output: addr = filter_idx * 25 + row * 5 + col
                where 25 = 5 * 5
```

---

## Migration Guide: Converting to LeNet-5

This section provides a detailed guide for converting the current 2-layer CNN to the classic LeNet-5 architecture.

### Architecture Comparison

| Aspect | Current CNN | LeNet-5 |
|--------|-------------|---------|
| Conv1 | 16 filters, 3x3, no padding | 6 filters, 5x5, padding=2 |
| Pool1 | 2x2 Max Pool | 2x2 Average Pool |
| Conv2 | 32 filters, 3x3 | 16 filters, 5x5 |
| Pool2 | 2x2 Max Pool | 2x2 Average Pool |
| FC Layers | 1 (800 to 10) | 3 (400 to 120 to 84 to 10) |
| Activation | ReLU | Tanh |
| Output Sizes | 26 to 13 to 11 to 5 | 28 to 14 to 10 to 5 |

### LeNet-5 Architecture Detail

```
INPUT: 28x28x1 (with padding for 5x5 kernel compatibility)

CONV1:  6 filters, 5x5, padding=2  -> Output: 6x28x28
POOL1:  2x2 Average Pool           -> Output: 6x14x14
CONV2:  16 filters, 5x5            -> Output: 16x10x10
POOL2:  2x2 Average Pool           -> Output: 16x5x5
FC1:    400 to 120 (Tanh)
FC2:    120 to 84 (Tanh)
FC3:    84 to 10 (Softmax/raw logits)
```

### Step 1: Modify Python Model (`train_cnn.py`)

Replace the `ImprovedCNN` class with LeNet-5:

```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 28 to 28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       # 28 to 14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 14 to 10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)       # 10 to 5

        # Classification
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 400 to 120
        self.fc2 = nn.Linear(120, 84)          # 120 to 84
        self.fc3 = nn.Linear(84, num_classes)  # 84 to 10

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten: 16x5x5 = 400

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x
```

### Step 2: Update Weight Export

Modify the export function to handle 5 layers instead of 3:

```python
# New weight sizes
WEIGHT_SIZES = {
    'conv1': (6, 1, 5, 5),      # 150 weights + 6 biases
    'conv2': (16, 6, 5, 5),     # 2,400 weights + 16 biases
    'fc1':   (120, 400),        # 48,000 weights + 120 biases
    'fc2':   (84, 120),         # 10,080 weights + 84 biases
    'fc3':   (10, 84),          # 840 weights + 10 biases
}

# Total: 61,470 weights + 236 biases ~ 62KB
```

**Note:** LeNet-5 requires significantly more memory (~62KB vs ~13KB). Verify this fits in Artix-7 distributed RAM or consider using BRAM.

### Step 3: Update Quantization

LeNet-5 uses **Tanh activation** instead of ReLU. This requires changes:

```python
# Tanh output range is [-1, 1]
# For int8 quantization, map to [-127, 127]

def quantize_tanh_output(x):
    # Tanh has natural [-1, 1] range
    # Scale to int8 range
    return clip(round(x * 127), -127, 127)
```

**Critical:** Tanh activation is more complex to implement in hardware than ReLU.

### Step 4: Implement Average Pooling

Replace max pooling with average pooling in both Python and Verilog:

**Python:**
```python
def avg_pool_2x2(input_vol):
    c, h, w = input_vol.shape
    new_h, new_w = h // 2, w // 2
    output = np.zeros((c, new_h, new_w), dtype=np.int32)

    for ch in range(c):
        for r in range(new_h):
            for col in range(new_w):
                window = input_vol[ch, r*2:r*2+2, col*2:col*2+2]
                # Integer average: sum >> 2 (divide by 4)
                output[ch, r, col] = np.sum(window) >> 2

    return output
```

**Verilog (in inference.v):**
```verilog
// Replace max pooling logic with average:
L1_POOL: begin
    case(pool_step)
        0: begin
            buf_a_addr <= f_idx*784 + (r*2)*28 + (c*2);
            pool_step <= 1;
        end
        1: begin
            pool_sum <= {24'b0, buf_a_rd_data};  // Start sum
            buf_a_addr <= f_idx*784 + (r*2)*28 + (c*2+1);
            pool_step <= 2;
        end
        2: begin
            pool_sum <= pool_sum + {24'b0, buf_a_rd_data};
            buf_a_addr <= f_idx*784 + (r*2+1)*28 + (c*2);
            pool_step <= 3;
        end
        3: begin
            pool_sum <= pool_sum + {24'b0, buf_a_rd_data};
            buf_a_addr <= f_idx*784 + (r*2+1)*28 + (c*2+1);
            pool_step <= 4;
        end
        4: begin
            // Average = sum >> 2 (divide by 4)
            buf_b_wr_data <= (pool_sum + {24'b0, buf_a_rd_data}) >> 2;
            buf_b_wr_en <= 1;
            // ... continue to next position
        end
    endcase
end
```

### Step 5: Implement Tanh Activation

Tanh is computationally expensive. Options:

#### Option A: Lookup Table (Recommended for FPGA)

```verilog
// 256-entry LUT for tanh approximation
reg signed [7:0] tanh_lut [0:255];

initial begin
    // Pre-compute tanh values: input [-128,127] to output [-127,127]
    $readmemh("tanh_lut.mem", tanh_lut);
end

// Usage: output = tanh_lut[input + 128];
```

Generate the LUT in Python:

```python
import numpy as np

# Generate 256-entry tanh LUT
lut = []
for i in range(-128, 128):
    # Scale input to reasonable tanh range
    x = i / 32.0  # Adjust divisor based on typical activations
    y = np.tanh(x)
    y_int = int(np.clip(np.round(y * 127), -127, 127))
    lut.append(y_int)

# Save as hex file
with open("tanh_lut.mem", "w") as f:
    for val in lut:
        if val < 0:
            val += 256
        f.write(f"{val:02x}\n")
```

#### Option B: Piecewise Linear Approximation

```verilog
function signed [7:0] tanh_approx;
    input signed [31:0] x;
    begin
        if (x < -256)
            tanh_approx = -127;
        else if (x > 256)
            tanh_approx = 127;
        else
            // Linear region: y ~ x/2 (simplified)
            tanh_approx = x >>> 1;
    end
endfunction
```

### Step 6: Update Memory Allocation

**New Buffer Sizes:**

| Buffer | Current CNN | LeNet-5 |
|--------|-------------|---------|
| Conv1 Output | 16x26x26 = 10,816 | 6x28x28 = 4,704 |
| Pool1 Output | 16x13x13 = 2,704 | 6x14x14 = 1,176 |
| Conv2 Output | 32x11x11 = 3,872 | 16x10x10 = 1,600 |
| Pool2 Output | 32x5x5 = 800 | 16x5x5 = 400 |
| FC1 Output | N/A | 120 |
| FC2 Output | N/A | 84 |

**RAM Module Updates:**

```verilog
// Update ram_cnn.v for LeNet-5 buffer sizes
parameter BUF_A_SIZE = 4704;   // 6x28x28 (Conv1 output)
parameter BUF_B_SIZE = 1600;   // 16x10x10 (Conv2 output)
parameter FC1_SIZE = 120;
parameter FC2_SIZE = 84;
```

### Step 7: Update Inference FSM

Add new states for additional FC layers:

```verilog
// Additional states needed:
localparam FC1_LOAD_BIAS      = 5'd19;
localparam FC1_LOAD_BIAS_WAIT = 5'd20;
localparam FC1_PREFETCH       = 5'd21;
localparam FC1_MULT           = 5'd22;
localparam FC1_TANH           = 5'd23;  // Apply tanh LUT
localparam FC2_LOAD_BIAS      = 5'd24;
localparam FC2_LOAD_BIAS_WAIT = 5'd25;
localparam FC2_PREFETCH       = 5'd26;
localparam FC2_MULT           = 5'd27;
localparam FC2_TANH           = 5'd28;
localparam FC3_LOAD_BIAS      = 5'd29;
// ... etc.
```

### Step 8: Update UART Protocol

Modify weight packet structure:

```
LeNet-5 Weight Packet (~62KB):
----------------------------------------------
Offset      Size        Data
----------------------------------------------
0           150         Conv1 weights (6x1x5x5)
150         24          Conv1 biases (6 x int32)
174         2,400       Conv2 weights (16x6x5x5)
2,574       64          Conv2 biases (16 x int32)
2,638       48,000      FC1 weights (120x400)
50,638      480         FC1 biases (120 x int32)
51,118      10,080      FC2 weights (84x120)
61,198      336         FC2 biases (84 x int32)
61,534      840         FC3 weights (10x84)
62,374      40          FC3 biases (10 x int32)
----------------------------------------------
Total:      62,414 bytes
```

Update `weight_loader.v`:

```verilog
// New size constants
localparam CONV1_W_SIZE = 150;
localparam CONV1_B_SIZE = 24;    // 6 x 4
localparam CONV2_W_SIZE = 2400;
localparam CONV2_B_SIZE = 64;    // 16 x 4
localparam FC1_W_SIZE = 48000;
localparam FC1_B_SIZE = 480;     // 120 x 4
localparam FC2_W_SIZE = 10080;
localparam FC2_B_SIZE = 336;     // 84 x 4
localparam FC3_W_SIZE = 840;
localparam FC3_B_SIZE = 40;      // 10 x 4

localparam TOTAL_WEIGHT_SIZE = 62414;
```

### Step 9: Update Python Testing

Modify `testing_python_model.py`:

```python
def simulate_lenet5_inference(image_bytes, weights_dict):
    c1_w, c1_b = weights_dict['conv1']
    c2_w, c2_b = weights_dict['conv2']
    fc1_w, fc1_b = weights_dict['fc1']
    fc2_w, fc2_b = weights_dict['fc2']
    fc3_w, fc3_b = weights_dict['fc3']

    # Reshape image
    img = np.frombuffer(image_bytes, dtype=np.int8)
    x = img.reshape(1, 28, 28).astype(np.int32)

    # Conv1 (with padding=2 for 5x5 kernel)
    x = np.pad(x, ((0,0), (2,2), (2,2)), mode='constant')
    x = convolve_5x5(x, c1_w.reshape(6,1,5,5), c1_b, shift=SHIFT)
    x = tanh_quantized(x)  # Use LUT or approximation
    x = avg_pool_2x2(x)

    # Conv2
    x = convolve_5x5(x, c2_w.reshape(16,6,5,5), c2_b, shift=SHIFT)
    x = tanh_quantized(x)
    x = avg_pool_2x2(x)

    # FC layers
    x = x.flatten()
    x = dense_layer(x, fc1_w.reshape(120,400), fc1_b, shift=SHIFT)
    x = tanh_quantized(x)
    x = dense_layer(x, fc2_w.reshape(84,120), fc2_b, shift=SHIFT)
    x = tanh_quantized(x)
    scores = dense_layer(x, fc3_w.reshape(10,84), fc3_b, shift=0)

    return np.argmax(scores)
```

### Step 10: Resource Estimation

| Resource | Current CNN | LeNet-5 (Est.) |
|----------|-------------|----------------|
| Weight RAM | 13 KB | 62 KB |
| Working RAM | 14 KB | 8 KB |
| LUTs (Logic) | ~2,000 | ~3,000 |
| LUTs (RAM) | ~30,000 | ~70,000 |
| Tanh LUT | N/A | 256 bytes |

**Warning:** The Artix-7 XC7A35T has ~33,280 LUTs. LeNet-5's 62KB weight storage may require using BRAM instead of distributed RAM, or a smaller variant of the network.

### Migration Checklist

- [ ] **Python Training**
  - [ ] Update model architecture to LeNet-5
  - [ ] Change activation from ReLU to Tanh
  - [ ] Update weight export for 5 layers
  - [ ] Generate tanh LUT file

- [ ] **Quantization**
  - [ ] Adjust quantization for tanh output range
  - [ ] Update bias scale propagation for 5 layers
  - [ ] Verify per-layer shift values

- [ ] **FPGA RTL**
  - [ ] Update `inference.v` FSM for 5 layers
  - [ ] Implement average pooling
  - [ ] Add tanh LUT module
  - [ ] Update buffer size parameters
  - [ ] Add FC1, FC2 intermediate storage

- [ ] **Memory**
  - [ ] Create new weight RAM modules
  - [ ] Create FC1 bias RAM (120 x 32-bit)
  - [ ] Create FC2 bias RAM (84 x 32-bit)
  - [ ] Consider BRAM vs distributed RAM trade-off

- [ ] **Communication**
  - [ ] Update `weight_loader.v` for new packet structure
  - [ ] Update `uart_router.v` payload size constants
  - [ ] Update Python `send_weights.py`

- [ ] **Testing**
  - [ ] Update Python simulation for LeNet-5
  - [ ] Generate new test vectors
  - [ ] Verify bit-exact match

---

## References

1. LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition." Proceedings of the IEEE.
2. Digilent Basys3 Reference Manual
3. Xilinx UG901: Vivado Design Suite User Guide - Synthesis

---

*Document generated for the MNIST-FPGA CNN project. Last updated: January 2025.*