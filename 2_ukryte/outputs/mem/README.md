# MEM Files - Two-Hidden-Layer Network

Memory initialization files for Verilog simulation and synthesis.

## Files

| File | Lines | Description |
|------|-------|-------------|
| `L1_weights.mem` | 12,544 | Layer 1 weights (2 hex chars each) |
| `L1_biases.mem` | 16 | Layer 1 biases (8 hex chars each) |
| `L2_weights.mem` | 256 | Layer 2 weights (2 hex chars each) |
| `L2_biases.mem` | 16 | Layer 2 biases (8 hex chars each) |
| `L3_weights.mem` | 160 | Layer 3 weights (2 hex chars each) |
| `L3_biases.mem` | 10 | Layer 3 biases (8 hex chars each) |

## Generation

Run `python/export_weights.py` after training the model in notebooks.

## Format

Same as softmax regression:
- Weights: 2 hex digits (INT8, two's complement for negative)
- Biases: 8 hex digits (INT32, two's complement)

