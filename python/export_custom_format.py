import torch
import torchvision.models as models
import struct

print("Loading pre-trained VGG16_BN model...")
model = models.vgg16_bn(pretrained=True)
weights_dict = model.state_dict()

output_filename = "../vgg16_weights.bin"
print(f"Saving weights to ROBUST binary format: {output_filename}")

# Define codes for our data types
# 0 = float32, 1 = int64 (long)
dtype_map = {
    torch.float32: 0,
    torch.int64: 1
}

with open(output_filename, "wb") as f:
    for name, tensor in weights_dict.items():
        tensor_cpu = tensor.cpu().contiguous()

        # --- NEW: Get the data type code ---
        dtype_val = tensor_cpu.dtype
        if dtype_val not in dtype_map:
            print(f"!!! SKIPPING TENSOR {name} WITH UNSUPPORTED DTYPE: {dtype_val}")
            continue

        type_code = dtype_map[dtype_val]

        # Get data as raw bytes
        tensor_bytes = tensor_cpu.numpy().tobytes()
        name_bytes = name.encode('utf-8')
        name_len = len(name_bytes)
        num_elements = tensor_cpu.numel()

        # --- NEW ROBUST FORMAT ---
        # 1. Name Length (4 bytes)
        # 2. Name String (N bytes)
        # 3. Dtype Code (1 byte)
        # 4. Num Elements (8 bytes)
        # 5. Tensor Data (M bytes)
        try:
            f.write(struct.pack('I', name_len))      # Unsigned Int
            f.write(name_bytes)
            f.write(struct.pack('B', type_code))     # Unsigned Char (Byte)
            f.write(struct.pack('Q', num_elements)) # Unsigned Long Long
            f.write(tensor_bytes)
        except struct.error as e:
            print(f"Error packing tensor {name}: {e}")
            break

print("Done. Weights saved.")