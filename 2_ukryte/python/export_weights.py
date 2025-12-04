import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub
import os

# Define the model class (must match the saved model)
class MNISTTwoHiddenQuant(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, num_classes: int):
        super().__init__()
        self.quant = QuantStub()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes)
        )
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.net(x)
        x = self.dequant(x)
        return x

def main():
    device = torch.device("cpu")
    input_dim = 28 * 28
    hidden_dim1 = 16
    hidden_dim2 = 16
    num_classes = 10
    
    # Initialize the floating point model first
    model = MNISTTwoHiddenQuant(input_dim, hidden_dim1, hidden_dim2, num_classes)
    
    # Prepare for quantization (required to load quantized state_dict correctly)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    
    # Load state dict
    # Get the directory of the current script to resolve paths correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pth_path = os.path.join(script_dir, "..", "outputs", "mnist_2hidden_int8.pth")
    pth_path = os.path.abspath(pth_path)

    if not os.path.exists(pth_path):
        print(f"Error: {pth_path} not found.")
        return

    print(f"Loading model from {pth_path}...")
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Helper to write .mem file
    def write_mem(filename, values, is_hex=True, width=2):
        print(f"Writing {len(values)} values to {filename}...")
        with open(filename, 'w') as f:
            for val in values:
                val = int(val) # Convert numpy type to python int to avoid overflow
                if is_hex:
                    # Handle negative values for hex (two's complement)
                    if val < 0:
                        val = (1 << (width * 4)) + val
                    # Format as hex with zero padding
                    f.write(f"{val:0{width}X}\n")
                else:
                    f.write(f"{val}\n")

    # Extract layers
    # Structure: quant -> net[0](Flatten) -> net[1](Linear) -> net[2](ReLU) -> net[3](Linear) -> net[4](ReLU) -> net[5](Linear) -> dequant
    
    layers = [
        ("L1", model.net[1]),
        ("L2", model.net[3]),
        ("L3", model.net[5])
    ]
    
    # Input scale comes from model.quant
    input_scale = model.quant.scale.item()
    print(f"Input scale: {input_scale}")
    
    for name, layer in layers:
        print(f"\nProcessing {name}...")
        
        # Weights
        # weight() returns a packed quantized tensor. 
        # int_repr() gets the integer values (int8/uint8).
        w_int = layer.weight().int_repr()
        
        # Flatten weights for mem file (usually row-major)
        w_flat = w_int.numpy().flatten()
        
        # Check dimensions
        print(f"  Weights shape: {w_int.shape}")
        
        weight_path = os.path.join(script_dir, f"{name}_weights.mem")
        write_mem(weight_path, w_flat, width=2) # 2 hex chars for 8-bit
        
        # Biases
        # Bias is stored as float. For fixed point inference:
        # bias_int = round(bias_float / (input_scale * weight_scale))
        # weight_scale might be per-channel or per-tensor.
        
        b_float = layer.bias()
        if b_float is None:
            print("  No bias for this layer.")
            continue
            
        w_scales = layer.weight().q_per_channel_scales()
        # If per-tensor, this would be a single value, but q_per_channel_scales works for per_channel qscheme
        # Let's check qscheme
        qscheme = layer.weight().qscheme()
        print(f"  Weight qscheme: {qscheme}")
        
        if qscheme == torch.per_channel_affine or qscheme == torch.per_channel_symmetric:
             # Input scale is scalar. Weight scale is vector (one per output channel).
             # We broadcast input_scale.
             scales = input_scale * w_scales
             b_int = (b_float / scales).round().int()
        else:
             w_scale = layer.weight().q_scale()
             scale = input_scale * w_scale
             b_int = (b_float / scale).round().int()
             
        print(f"  Biases shape: {b_int.shape}")
        bias_path = os.path.join(script_dir, f"{name}_biases.mem")
        write_mem(bias_path, b_int.numpy(), width=8) # 8 hex chars for 32-bit bias
        
        # Update input_scale for next layer
        # The output of this layer is the input to the next.
        # layer.scale is the output scale of this layer.
        input_scale = layer.scale
        print(f"  Output scale (next input scale): {input_scale}")

if __name__ == "__main__":
    main()
