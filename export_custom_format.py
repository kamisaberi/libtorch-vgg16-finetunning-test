import torch
import torchvision.models as models
import struct # For packing binary data

print("Loading pre-trained VGG16_BN model...")
model = models.vgg16_bn(pretrained=True)
weights_dict = model.state_dict()

output_filename = "vgg16_weights.bin"

print(f"Saving weights to custom binary format: {output_filename}")

with open(output_filename, "wb") as f:
    # Iterate through every parameter and buffer in the state_dict
    for name, tensor in weights_dict.items():
        # Ensure tensor is on the CPU and is contiguous in memory
        tensor_cpu = tensor.cpu().contiguous()

        # 1. Get the tensor data as raw bytes
        tensor_bytes = tensor_cpu.numpy().tobytes()

        # 2. Get the layer name as raw bytes
        name_bytes = name.encode('utf-8')

        # 3. Get the length of the name
        name_len = len(name_bytes)

        # 4. Get the number of elements in the tensor
        num_elements = tensor_cpu.numel()

        # Pack the metadata and data into the file
        # 'I' is an unsigned int (4 bytes) for name_len
        # 'Q' is an unsigned long long (8 bytes) for num_elements
        f.write(struct.pack('I', name_len))
        f.write(name_bytes)
        f.write(struct.pack('Q', num_elements))
        f.write(tensor_bytes)

print("Done. Weights saved.")