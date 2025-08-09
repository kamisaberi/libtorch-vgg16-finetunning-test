import torch
import torchvision.models as models

# Load the official pre-trained VGG16 model with batch normalization
model = models.vgg16_bn(pretrained=True)

# The state_dict contains all the weights and buffers (e.g., batch norm running means)
# It's a simple dictionary of names to tensors.
weights = model.state_dict()

# We only care about the convolutional base, which torchvision calls 'features'.
# Let's also grab the first classifier layer to show how to load it.
# You can inspect the keys to see the names:
# for key in weights.keys():
#     print(key)

output_path = "../vgg16_bn_weights.pt"
torch.save(weights, output_path)

print(f"VGG16_BN state_dict saved to {output_path}")
print(f"Number of tensors saved: {len(weights)}")
