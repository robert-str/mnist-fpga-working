import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize pixel values to [0, 1]
y = y.astype(int)

# Split into train and test
X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 2. TRAIN SOFTMAX REGRESSION
# ============================================================
print("Training softmax regression...")
# Note: multi_class='multinomial' is default in newer sklearn versions
model = LogisticRegression(max_iter=200, solver='lbfgs')
model.fit(X_train_scaled, y_train)

# Extract weights and biases (floats)
W = model.coef_       # Shape: (10, 784)
b = model.intercept_  # Shape: (10,)

# ============================================================
# 3. TEST ACCURACY WITH FLOAT MODEL
# ============================================================
y_pred_float = model.predict(X_test_scaled)
acc_float = accuracy_score(y_test, y_pred_float)
print(f"Float model accuracy: {acc_float * 100:.2f}%")

# ============================================================
# 4. IMPROVED QUANTIZATION FUNCTIONS
# ============================================================

def quantize_to_int8_dynamic(x):
    """
    Quantize to int8 using dynamic scale based on actual data range.
    Returns: quantized values and scale factor
    """
    # Find the maximum absolute value
    max_abs = np.max(np.abs(x))
    
    if max_abs == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0
    
    # Calculate scale to use full int8 range (-127 to 127)
    # We use 127 instead of 128 to keep symmetric range
    scale = 127.0 / max_abs
    
    # Quantize
    x_scaled = np.round(x * scale)
    x_int8 = np.clip(x_scaled, -127, 127).astype(np.int8)
    
    return x_int8, scale

def quantize_to_int32_dynamic(x, scale):
    """
    Quantize to int32 using provided scale.
    For biases: scale = weight_scale * input_scale
    """
    x_scaled = np.round(x * scale)
    return x_scaled.astype(np.int32)

# Input scale (we'll quantize inputs to int8 too)
INPUT_SCALE = 127.0

# ============================================================
# 5. APPLY QUANTIZATION
# ============================================================
print("\nQuantizing weights and biases...")

# Quantize weights with dynamic scale
W_int8, W_scale = quantize_to_int8_dynamic(W)

# Quantize biases: need scale = weight_scale * input_scale
# Because: output = (X * input_scale) @ (W * weight_scale).T + (b * bias_scale)
# For this to work: bias_scale = input_scale * weight_scale
BIAS_SCALE = W_scale * INPUT_SCALE
b_int32 = quantize_to_int32_dynamic(b, BIAS_SCALE)

print(f"Weight range: [{W.min():.4f}, {W.max():.4f}]")
print(f"Dynamic scale factor: {W_scale:.4f}")
print(f"Quantized weight range: [{W_int8.min()}, {W_int8.max()}]")

# ============================================================
# 6. TEST ACCURACY WITH QUANTIZED MODEL
# ============================================================

def predict_quantized(X, W_q, b_q):
    """
    Make predictions using quantized weights.
    This simulates what FPGA will compute.
    """
    # Input quantization (simulate 8-bit input)
    X_int = np.round(X * INPUT_SCALE).astype(np.int32)
    
    # Matrix multiplication (use int32 to avoid overflow)
    # Result scale = INPUT_SCALE * W_SCALE = BIAS_SCALE
    logits = X_int @ W_q.T.astype(np.int32) + b_q
    
    # Just take argmax (no need for softmax in inference)
    predictions = np.argmax(logits, axis=1)
    
    return predictions

y_pred_quant = predict_quantized(X_test_scaled, W_int8, b_int32)
acc_quant = accuracy_score(y_test, y_pred_quant)
print(f"Quantized model accuracy: {acc_quant * 100:.2f}%")
print(f"Accuracy loss: {(acc_float - acc_quant) * 100:.2f}%")

# ============================================================
# 7. SAVE TO .MEM FILES FOR FPGA
# ============================================================
print("\nSaving weights and biases to .mem files...")

# Save weights (int8 as 2-digit hex)
with open("W_improved.mem", "w") as f:
    for i in range(10):      # 10 classes
        for j in range(784):  # 784 pixels
            # Handle negative numbers (two's complement)
            val = int(W_int8[i, j])  # Convert to Python int first
            if val < 0:
                val = val + 256  # Convert to unsigned for hex
            f.write(f"{val:02x}\n")

# Save biases (int32 as 8-digit hex)
with open("B_improved.mem", "w") as f:
    for i in range(10):
        val = int(b_int32[i])  # Convert to Python int first
        if val < 0:
            val = val + (1 << 32)  # Two's complement for 32-bit
        f.write(f"{val:08x}\n")

# Save scale factor (important for FPGA to know!)
with open("scale_info.txt", "w") as f:
    f.write(f"Weight scale factor: {W_scale}\n")
    f.write(f"Input scale factor: {INPUT_SCALE}\n")
    f.write(f"Bias scale factor: {BIAS_SCALE}\n")
    f.write(f"\nOriginal weight range: [{W.min()}, {W.max()}]\n")
    f.write(f"Original bias range: [{b.min()}, {b.max()}]\n")
    f.write(f"\nQuantized model accuracy: {acc_quant * 100:.2f}%\n")

# Save StandardScaler parameters for inference
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

print("Generated: W_improved.mem, B_improved.mem, scale_info.txt")
print("Generated: scaler_mean.npy, scaler_scale.npy")
print("\nDone!")

