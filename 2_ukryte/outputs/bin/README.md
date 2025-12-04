# BIN Files - Two-Hidden-Layer Network

Raw binary files for UART transmission to FPGA.

## Files

| File | Size (bytes) | Description |
|------|--------------|-------------|
| `L1_weights.bin` | 12,544 | Layer 1 weights |
| `L1_biases.bin` | 64 | Layer 1 biases (16 × 4 bytes) |
| `L2_weights.bin` | 256 | Layer 2 weights |
| `L2_biases.bin` | 64 | Layer 2 biases (16 × 4 bytes) |
| `L3_weights.bin` | 160 | Layer 3 weights |
| `L3_biases.bin` | 40 | Layer 3 biases (10 × 4 bytes) |

## Total Size

- Weights: 12,544 + 256 + 160 = 12,960 bytes
- Biases: 64 + 64 + 40 = 168 bytes
- **Total: 13,128 bytes**

## Generation

```bash
uv run ../../shared/convert_to_binary.py ../mem/ ./
```

