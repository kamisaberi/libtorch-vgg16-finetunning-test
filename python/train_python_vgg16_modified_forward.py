import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os

def fine_tune_vgg16():
    """
    Main function to fine-tune VGG16_BN on the Food-101 dataset.
    """
    # --- 1. Configuration ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure reproducibility for fair comparison, if desired
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)

    DATA_DIR = '/home/kami/Documents/datasets/'  # Directory where the 'food-101' folder will be/is located
    NUM_CLASSES = 101
    NUM_EPOCHS = 3
    BATCH_SIZE = 64  # Match the C++ version
    LEARNING_RATE = 0.001

    # Use more workers if your CPU has more cores to saturate the GPU pipeline
    # Match the number of workers in your C++ DataLoader for a fair test
    NUM_WORKERS = 8

    # --- 2. Data Loading and Transformations ---
    # The transformations must exactly match the logic in the C++ Dataset
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Use Resize, not RandomResizedCrop, to match the C++ version
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading Food-101 dataset...")
    # torchvision.datasets.Food101 handles downloading and parsing automatically
    train_dataset = datasets.Food101(
        root=DATA_DIR,
        split='train',
        transform=data_transforms,
        download=False
    )

    # pin_memory=True is the Python equivalent of creating tensors in pinned memory
    # It speeds up CPU-to-GPU data transfers
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    dataset_size = len(train_dataset)
    print(f"Training data loaded: {dataset_size} images.")

    # --- 3. Model Loading and Modification ---
    print("Loading pre-trained VGG16_BN model...")
    vgg16_bn = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

    # To prevent the pretrained weights from being updated during training
    for param in vgg16_bn.features.parameters():
        param.requires_grad = False

    # 2. Create a new model class to modify the forward method
    class ModifiedVGG16(nn.Module):
        def __init__(self, original_model):
            super(ModifiedVGG16, self).__init__()
            # Separate the features and the classifier from the original model
            self.features = original_model.features
            self.avgpool = original_model.avgpool

            # Example of adding a new custom classifier
            self.classifier = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1024, 10)  # Assuming 10 output classes
            )

        def forward(self, x):
            # Pass the input through the feature extractor
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            print(x.shape)
            # Pass the features through the new classifier
            x = self.classifier(x)
            return x

    # 3. Instantiate your modified model
    model = ModifiedVGG16(vgg16_bn)

    # Print the new model structure
    print(model)

    # --- 4. Freeze Layers and Prepare for Fine-Tuning ---
    # Freeze the convolutional base ('features')
    # for param in model.features.parameters():
    #     param.requires_grad = False

    # Replace the classifier with a new one for our task.
    # The new layers will have requires_grad=True by default.
    num_ftrs_in_classifier = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs_in_classifier, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, NUM_CLASSES)
    )

    model.to(device)

    # Collect only the parameters to be trained
    # This is equivalent to what we did in C++
    params_to_update = model.classifier.parameters()

    print(f"Model modified. Only the classifier parameters will be trained.")

    # --- 5. Training Loop ---
    criterion = nn.CrossEntropyLoss()
    # Use the same optimizer and learning rate as the C++ version
    optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)

    print("\nStarting Python fine-tuning...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Using enumerate to get batch index for logging
        for i, (inputs, labels) in enumerate(train_loader):
            # The DataLoader with pin_memory=True and a non_blocking=True transfer
            # is the most efficient way to move data to the GPU.
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if (i + 1) % 100 == 0:
                print(f'  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print(f'Epoch {epoch+1} Summary -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nPython fine-tuning complete in {total_time:.2f} seconds.")

if __name__ == '__main__':
    fine_tune_vgg16()