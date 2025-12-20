# Overflow Fix - Step-by-Step Guide

## Problem Summary

The original quantization scheme caused **32-bit integer overflow** in the FPGA hardware, resulting in:
- Logits in the trillions (before fix) or tens of millions (with old weights)
- Always predicting digit "5" regardless of input
- Layer 2 outputs reaching ~40 million, causing overflow in Layer 3

## Root Cause

The training script's quantization used **dynamic per-layer scaling** without considering hardware constraints:
- L1 outputs: ~500k
- L2 outputs: ~40M 
- L3 accumulation: L2_output (40M) × weight (±127) × 16 MACs = **±6 billion** → Overflows 32-bit signed range (±2.1B)

## Solution

Added **hardware right-shifts** after Layer 1 and Layer 2:
- After L1: Right-shift by 7 bits (÷128)
- After L2: Right-shift by 7 bits (÷128)
- Adjusted bias quantization to account for these shifts

This keeps values within safe 32-bit range throughout the network.

## Changes Made

### 1. Training Script (`2_ukryte/training/siec_2_ukryte.py`)
- Added `SHIFT1 = 7` and `SHIFT2 = 7` parameters
- Modified bias scaling to account for right-shifts:
  - L1_OUTPUT_SCALE now divided by 2^SHIFT1
  - L2_OUTPUT_SCALE now divided by 2^SHIFT2
  - L3_BIAS_SCALE uses the shifted L2_OUTPUT_SCALE
- Updated scale_info.txt output to document shifts

### 2. FPGA Verilog (`2_ukryte/inference/rtl/inference.v`)
- Added right-shift (>>> 7) in STATE_L1_NEXT_NEURON (after Layer 1)
- Added right-shift (>>> 7) in STATE_L2_NEXT_NEURON (after Layer 2)
- Updated documentation comments

### 3. Debug Script (`2_ukryte/testing/debug_quantized_inference.py`)
- Updated to simulate hardware shifts
- Can verify quantization works correctly before FPGA synthesis

## Steps to Fix Your System

### Step 1: Re-train and Re-quantize Weights

```bash
cd 2_ukryte/training
uv run python siec_2_ukryte.py
```

This will:
- Train the model (should still achieve ~95% accuracy)
- Generate new quantized weights and biases with proper scaling
- Save to `../outputs/bin/` and `../outputs/mem/`
- Create `../outputs/npy/scale_info.txt` with shift information

**Expected output in scale_info.txt:**
```
Hardware shifts: Layer 1 >> 7, Layer 2 >> 7
Final output scale: ~400-600 (instead of 650 million)
```

### Step 2: Verify Quantization Works (Optional but Recommended)

```bash
cd 2_ukryte/testing
uv run python debug_quantized_inference.py
```

**Expected output:**
- Layer 1 outputs: ~1k-4k (instead of ~500k)
- Layer 2 outputs: ~100k-300k (instead of ~40M)
- Layer 3 outputs: ~1k-200k (NO OVERFLOW warnings!)
- Correct prediction on test image

### Step 3: Re-synthesize FPGA Design

1. Open your Vivado project for `2_ukryte`
2. The source file `inference.v` has been updated with shifts
3. Run synthesis
4. Run implementation
5. Generate bitstream

**Note:** The shifts add minimal overhead - just arithmetic right-shift operations that are very cheap in hardware.

### Step 4: Program FPGA

- Program the FPGA with the new bitstream
- The hardware now includes the right-shifts

### Step 5: Upload New Weights

```bash
cd 2_ukryte/utils
uv run python send_weights.py COM3
```

This uploads the NEW quantized weights (with adjusted bias scaling).

### Step 6: Test!

```bash
# Test single image
uv run python send_image.py ../test_images/00001.png COM3

# Read scores to verify no overflow
uv run python read_scores.py COM3

# Run accuracy test
cd ../testing
uv run python test_fpga_accuracy.py --samples 100
```

**Expected results:**
- Logits in range of ~1k to ~200k (NOT millions or billions)
- Correct predictions (should match Python model ~95% accuracy)
- No more "always predicting 5"

## Verification Checklist

- [ ] Step 1: Re-trained model with new quantization
- [ ] Step 2: Verified with debug_quantized_inference.py (no overflow warnings)
- [ ] Step 3: Re-synthesized FPGA design
- [ ] Step 4: Programmed FPGA with new bitstream
- [ ] Step 5: Uploaded new weights to FPGA
- [ ] Step 6: Tested and confirmed correct predictions

## Technical Details

### Why Right-Shift by 7?

- 8-bit quantization uses scale factor of 127
- Multiplying two 8-bit values: max product = 127 × 127 ≈ 16k
- To normalize back, divide by 127 ≈ shift by 7 (2^7 = 128)
- This is standard practice in fixed-point neural networks

### Impact on Accuracy

- Minimal to none! The shifts are accounted for in bias scaling
- The network still operates on the same mathematical values
- Expected accuracy: ~95% (same as floating-point model)

### Why Not Use Wider Accumulators?

- 40-bit or 64-bit accumulators would work but:
  - Use more FPGA resources (DSP blocks, registers)
  - Slower clock speed
  - More power consumption
- Right-shifts are essentially free in hardware (just wire routing)

## Troubleshooting

### If you still see overflow:
- Check that scale_info.txt shows "Hardware shifts: Layer 1 >> 7, Layer 2 >> 7"
- Verify new weights were uploaded (check file timestamps)
- Ensure FPGA was programmed with new bitstream

### If accuracy is low:
- Run debug_quantized_inference.py to verify software simulation works
- Compare Python model accuracy with FPGA accuracy
- Check that normalization parameters are correct

### If predictions are random:
- Verify weights loaded correctly (use send_weights.py with verbose output)
- Check serial communication settings (baud rate, COM port)
- Ensure image preprocessing matches training (normalization parameters)

## Questions?

If you encounter issues, check:
1. Training script output (scale values should be much smaller now)
2. Debug script output (should show NO overflow warnings)
3. Verilog synthesis report (should show no errors/warnings about inference.v)





