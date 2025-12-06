# Shared Utilities

Python utilities shared between different model implementations.

## Files

| Script | Description |
|--------|-------------|
| `convert_to_binary.py` | Convert `.mem` files (hex text) to `.bin` files (raw binary) |

## convert_to_binary.py

Universal converter for preparing weight files for UART transmission.

### Auto-detection

The script automatically detects file type based on hex character count:
- **2 hex chars** → 8-bit values (weights) → 1 byte each
- **8 hex chars** → 32-bit values (biases) → 4 bytes each (little-endian)

### Usage

```bash
# Default: converts regresja/outputs/mem/ to regresja/outputs/bin/
uv run convert_to_binary.py

# Custom paths
uv run convert_to_binary.py <input_folder> <output_folder>
```

### Example Output

```
Input folder:  regresja/outputs/mem
Output folder: regresja/outputs/bin

  W.mem                     -> W.bin                     (  7840 x 8-bit,   7840 bytes)
  B.mem                     -> B.bin                     (    10 x 32-bit,    40 bytes)

============================================================
Converted: 2 files
Total binary data: 7880 bytes (7.70 KB)
Transmission time at 115200 baud: ~0.7 seconds
```




