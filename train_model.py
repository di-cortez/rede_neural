import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import zipfile
import io

# Paths for the images and labels
#IMG_ZIP_PATH = "images_28x28.zip"
#LABELS_ZIP_PATH = "labels_28x28.zip"
#batch_size = 64
#epochs = 15
#learning_rate = 0.001

# Custom dataset to load image and label
class ShapeDataset(Dataset):
    def __init__(self, img_zip_path, label_zip_path, transform=None):
        self.img_path = zipfile.ZipFile(img_zip_path, 'r')
        self.labels_path = zipfile.ZipFile(label_zip_path, 'r')
        self.transform = transform
        self.file_names = sorted(
            [name for name in self.img_path.namelist() if name.endswith(".png")]
        )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Load image
        img_name = self.file_names[idx]
        img_bytes = self.img_path.read(img_name)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Apply transforms (to tensor, normalization)
        if self.transform:
            image = self.transform(image)

        label_name = img_name.replace(".png", ".txt")
        # Load label (first number in the label file)
        label_content = self.labels_path.read(label_name).decode("utf-8")
        label = int(label_content.split()[0])

        return image, torch.tensor(label, dtype=torch.float32)

# Neural network: MLP with 3 Linear layers and ReLU activations
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), # Saída: 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Saída: 16x14x14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # Saída: 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # Saída: 32x7x7
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5), #aumenta para conter overfitting e diminui para conter underfitting
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1) # Achata o tensor antes de passar para a parte linear
        x = self.classifier(x)
        return x

def train_model(dataset_path, batch_size, epochs, learning_rate, num_images):

    images_zip_path = os.path.join(dataset_path, "images_28x28.zip")
    labels_zip_path = os.path.join(dataset_path, "labels_28x28.zip")
    
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),                      # Converts to [0,1] and shape (C,H,W)
    ])

    dataset = ShapeDataset(images_zip_path, labels_zip_path, transform=transform)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate model, loss function and optimizer
    model = SimpleCNN()
    criterion = nn.BCELoss()                     # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    # Track metrics
    train_acc_list = []
    val_acc_list = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            running_loss += loss.item() * inputs.size(0)
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if epoch == 0:
                print(
                    f"  [Batch {batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} - "
                    f"Batch Acc: {(predictions == labels).float().mean().item() * 100:.2f}%"
                )

        avg_loss = running_loss / total
        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze(1)
                predictions = (outputs >= 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100 * val_correct / val_total
        val_acc_list.append(val_acc)

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Avg Loss: {avg_loss:.4f} - "
            f"Train Acc: {train_acc:.2f}% - "
            f"Val Acc: {val_acc:.2f}% - "
            f"Time: {epoch_time:.2f}s - "
            f"LR: {current_lr:.6f}"
        )

    # Save model
    filename_base = os.path.join(dataset_path, f"grafico_{n_total}_imgs_{epochs}_épocas")
    model_path = f"{filename_base}.pth"

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot
    plot_path = f"{filename_base}_accuracy.png"
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_list, label="Train Accurancy", marker="o")
    plt.plot(val_acc_list, label="Val Accurancy", marker="s")
    plt.title(f"Accuracy per Epoch {num_images} imgs e {epochs}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    return val_acc_list[-1]
    #plt.show()