# Python Scripts - Softmax Regression

Python scripts for training, testing, and FPGA communication.

## Files

| Script | Description |
|--------|-------------|
| `soft_reg_lepsza_kwant.py` | Train softmax regression with improved quantization |
| `regresja_softamx.py` | Alternative training script |
| `send_weights.py` | Upload model weights to FPGA via UART |
| `send_image.py` | Send test image to FPGA for inference |
| `test_model.py` | Test quantized model locally in Python |
| `test_all_images.py` | Batch test all images from test_images folder |

## UART Protocol

### Weights Upload (`send_weights.py`)

- Start marker: `0xAA 0x55`
- Data: 7840 bytes (weights) + 40 bytes (biases)
- End marker: `0x55 0xAA`

### Image Upload (`send_image.py`)

- Start marker: `0xBB 0x66`
- Data: 784 bytes (28×28 pixels, preprocessed INT8)
- End marker: `0x66 0xBB`

## Image Preprocessing

Before sending to FPGA, images are:

1. Resized to 20×20 pixels
2. Centered in 28×28 canvas (4px padding)
3. Inverted if needed (white digit on black background)
4. Normalized with StandardScaler parameters
5. Quantized to INT8 (-128 to 127)

## Data Folder

Contains MNIST dataset downloaded by PyTorch/torchvision for testing.




