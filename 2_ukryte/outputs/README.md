# Outputs - Two-Hidden-Layer Model Parameters

Contains exported weights and biases for all three layers.

## Subfolders

| Folder | Description |
|--------|-------------|
| `mem/` | Memory files (`.mem`) - hex text format |
| `bin/` | Binary files (`.bin`) - for UART transmission |

## Files

| File | Description | Size |
|------|-------------|------|
| `L1_weights` | Input → Hidden1 weights | 12,544 values (INT8) |
| `L1_biases` | Hidden1 biases | 16 values (INT32) |
| `L2_weights` | Hidden1 → Hidden2 weights | 256 values (INT8) |
| `L2_biases` | Hidden2 biases | 16 values (INT32) |
| `L3_weights` | Hidden2 → Output weights | 160 values (INT8) |
| `L3_biases` | Output biases | 10 values (INT32) |

