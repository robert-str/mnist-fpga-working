# FPGA-Based Neural Network Inference for MNIST Digit Classification

This project implements neural network inference on FPGA hardware for classifying handwritten digits (MNIST dataset).

## Project Goal

- Train neural network models in Python
- Export quantized weights and biases (INT8/INT32)
- Implement inference logic in Verilog (Vivado)
- Load model parameters to FPGA via UART
- Send images to FPGA for real-time classification

## Folder Structure

| Folder | Description |
|--------|-------------|
| `regresja/` | Softmax (logistic) regression model - simple single-layer classifier |
| `2_ukryte/` | Two-hidden-layer neural network (784 → 16 → 16 → 10) |
| `shared/` | Shared Python utilities used by both models |
| `test_images/` | PNG test images for FPGA inference testing |

## Workflow

1. **Train model** - Run Jupyter notebook or Python script
2. **Export weights** - Save as `.mem` files (hex format for Verilog)
3. **Convert to binary** - Use `shared/convert_to_binary.py` for UART transmission
4. **Upload to FPGA** - Use `send_weights.py` via serial port
5. **Run inference** - Send test images with `send_image.py`
6. **Check result** - Read predicted digit from 7-segment display or LEDs

## Hardware Requirements

- FPGA board with UART interface (tested on Basys 3 / Nexys A7)
- USB-to-Serial connection (115200 baud default)

