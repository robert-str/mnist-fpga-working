# Two-Hidden-Layer Neural Network

Multi-layer perceptron with two hidden layers for MNIST classification.

## Model Architecture

```
Input (784) → Hidden1 (16) → ReLU → Hidden2 (16) → ReLU → Output (10)
```

- **Input**: 784 pixels (28×28 image)
- **Hidden Layer 1**: 16 neurons + ReLU activation
- **Hidden Layer 2**: 16 neurons + ReLU activation
- **Output**: 10 classes (digits 0-9)

## Parameters

| Layer | Weights | Biases |
|-------|---------|--------|
| L1 (input → hidden1) | 784 × 16 = 12,544 bytes | 16 × 4 = 64 bytes |
| L2 (hidden1 → hidden2) | 16 × 16 = 256 bytes | 16 × 4 = 64 bytes |
| L3 (hidden2 → output) | 16 × 10 = 160 bytes | 10 × 4 = 40 bytes |
| **Total** | 12,960 bytes | 168 bytes |

## Folder Structure

| Folder | Description |
|--------|-------------|
| `notebooks/` | Jupyter notebooks for training and INT8 quantization |
| `python/` | Python scripts for weight export and UART communication |
| `vivado/` | Verilog HDL source files for FPGA |
| `outputs/` | Exported weights and biases |

## Quantization

Uses PyTorch quantization-aware training:
- Weights: INT8 (per-channel quantization)
- Biases: INT32 (accumulated scale)
- Activations: INT8 with ReLU

## Expected Accuracy

Higher than softmax regression due to non-linear feature extraction.

