"""  This code is used to segment flood areas in aerial images using a U-Net model with a ResNet backbone. The dataset used is the "Flood Area Dataset" from Kaggle (https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation). 
The dataset contains images of flood areas and their corresponding masks. The U-Net model is implemented using PyTorch and the ResNet backbone is used to extract features from the input images. The model is trained using a combination 
of Dice Loss and Cross-Entropy Loss. The model is evaluated on a test set and the performance metrics are computed. The predicted masks are saved as images for visualization.
"""
from colorama import Fore, Style
import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import Subset, random_split

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

# !pip install -q -U segmentation-models-pytorch albumentations > /dev/null
# import segmentation_models_pytorch as smp

IMAGE_DIR = "C:/Users/Hadi/Downloads/archive/training/Image/"
MASK_DIR = "C:/Users/Hadi/Downloads/archive/training/Mask/"


BATCH_SIZE = 4           
PATCH_SIZE = (256, 256)   
torch.manual_seed(42)     
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, GaussianBlur, RandomBrightnessContrast
)
from albumentations.pytorch import ToTensorV2

class FloodAreaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None, augmentations=None):
        self.img_labels = sorted([file for file in os.listdir(image_dir)])
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augmentations = augmentations

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load image and mask paths
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        mask_path = os.path.join(self.mask_dir, self.img_labels[idx]).replace(".jpg", ".png")

        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Apply Albumentations augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Apply image-specific transformations
        if self.image_transform:
            image = self.image_transform(Image.fromarray(image))

        # Apply mask-specific transformations
        if self.mask_transform:
            mask = self.mask_transform(Image.fromarray(mask))

        return image, mask.long()



image_transform = transforms.Compose([
    transforms.Resize(size=PATCH_SIZE, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=PATCH_SIZE, antialias=False)
])

augmentations = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Rotate(limit=30, p=0.5),
    GaussianBlur(blur_limit=3, sigma=(0.1, 2.0), p=0.5),
    RandomBrightnessContrast(p=0.5),
])



# Define the dataset
dataset = FloodAreaDataset(IMAGE_DIR, MASK_DIR, image_transform=image_transform, mask_transform=mask_transform, augmentations=augmentations)


# Define the sizes for each split
dataset_size = len(dataset)
print(f"Total number of samples: {dataset_size}")
print(Fore.CYAN + f"Total number of samples:----------------- {dataset_size}" + Style.RESET_ALL)

val_size  = int(0.15 * dataset_size)
train_size = dataset_size  - val_size
print(f"Train size: {train_size}, Validation size: {val_size}")
# Use random_split to create train, test, and val datasets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))


# Create DataLoader instances for each set
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


IMAGE_test_DIR = "C:/Users/Hadi/Downloads/archive/testing/Image/"
MASK_test_DIR = "C:/Users/Hadi/Downloads/archive/testing/Mask/"

torch.manual_seed(42)     # Set a global random seed for reproducibility

class FloodAreaDatasettest(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.img_labels = sorted([file for file in os.listdir(image_dir)])
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        mask_path = os.path.join(self.mask_dir, self.img_labels[idx]).replace(".jpg", ".png")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # if self.augmentations:
        #     image, mask = data_augmentation(image, mask)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask.long()

image_transform = transforms.Compose([
    transforms.Resize(size=PATCH_SIZE, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=PATCH_SIZE, antialias=False)
])

# Define the dataset
datasettest = FloodAreaDataset(IMAGE_test_DIR, MASK_test_DIR, image_transform=image_transform, mask_transform=mask_transform)



# Define the sizes for each split
dataset_test_size = len(datasettest)
print(f"Total number of test samples: {dataset_test_size}")
test_size = int(1 * dataset_test_size)

print(f"Test size: {test_size}")

test_dataset = datasettest


# Create DataLoader instances for each set
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Function to denormalize images
def denormalize_tensor(image, mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image

# Function to visualize images and masks
def visualize_samples(dataset, num_samples=5):
    # Visualize the images and masks
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2 * num_samples))
    for i in range(num_samples):
        image, mask = dataset[np.random.randint(len(dataset))]
        
        # Display images
        axes[i, 0].imshow(denormalize_tensor(image).cpu().permute(1,2,0))
        axes[i, 0].set_title(f'Sample {i + 1} - Image')
        axes[i, 0].axis('off')

        # Display masks
        axes[i, 1].imshow(mask.cpu().squeeze(), cmap='gray')
        axes[i, 1].set_title(f'Sample {i + 1} - Mask')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Visualize samples from the train dataset
visualize_samples(train_dataset)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UpResNet(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, bilinear=True, dropout_rate=0.1):
        super(UpResNet, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_1, in_channels_1, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels_1 + in_channels_2, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

resnet_type = "resnet101"

class ResNetUNet(nn.Module):
    def __init__(self, in_channels, out_channels, resnet_type=resnet_type, bilinear=False, dropout_rate = 0.1):
        super(ResNetUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resnet_type = resnet_type
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate
        
        # Define the backbone network
        if self.resnet_type == "resnet18":
            self.backbone_model = torchvision.models.resnet18(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]
        elif self.resnet_type == "resnet34":
            self.channel_distribution = [3, 64, 64, 128, 256]
            self.backbone_model = torchvision.models.resnet34(weights="DEFAULT")
        elif self.resnet_type == "resnet50":
            self.channel_distribution = [3, 64, 256, 512, 1024]
            self.backbone_model = torchvision.models.resnet50(weights="DEFAULT")
        elif self.resnet_type == "resnet101":
            self.backbone_model = torchvision.models.resnet101(weights="DEFAULT")
            self.channel_distribution = [3, 64, 256, 512, 1024]
        elif self.resnet_type == "resnet152":
            self.backbone_model = torchvision.models.resnet152(weights="DEFAULT")
            self.channel_distribution = [3, 64, 256, 512, 1024]
        else:
            print("Resnet type is not recognized. Loading ResNet 18 as backbone model")
            self.channel_distribution = [3, 64, 64, 128, 256]
            self.backbone_model = torchvision.models.resnet34(weights="DEFAULT")
        
        self.backbone_layers = list(self.backbone_model.children())
        
        # Define the ResNetUNet
        self.inc = DoubleConv(in_channels, 64)
        
        self.block1 = nn.Sequential(*self.backbone_layers[0:3])
        self.block2 = nn.Sequential(*self.backbone_layers[3:5])
        self.block3 = nn.Sequential(*self.backbone_layers[5])
        self.block4 = nn.Sequential(*self.backbone_layers[6])
        
        self.up1 = Up(self.channel_distribution[-1], self.channel_distribution[-2], bilinear=bilinear, dropout_rate = dropout_rate)
        self.up2 = Up(self.channel_distribution[-2], self.channel_distribution[-3], bilinear=bilinear, dropout_rate = dropout_rate)
        self.up3 = UpResNet(self.channel_distribution[-3], 64, self.channel_distribution[-4], bilinear=bilinear, dropout_rate = dropout_rate)
        self.up4 = UpResNet(self.channel_distribution[-4], 64, self.channel_distribution[-4], bilinear=bilinear, dropout_rate = dropout_rate)
        
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        y1 = self.up1(x4, x3)
        y2 = self.up2(x3, x2)
        y3 = self.up3(x2, x1)
        y4 = self.up4(x1, x0)

        logits = self.outc(y4)
        
        return logits
    

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_unet = ResNetUNet(in_channels=3, out_channels=2, resnet_type=resnet_type).to(device)

# Freeze parameters in blocks 1, 2, 3, and 4
for block in [resnet_unet.block1, resnet_unet.block2, resnet_unet.block3, resnet_unet.block4]:
    for param in block.parameters():
        param.requires_grad_(False)
        

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define constants
LEARNING_RATE = 0.001
LR_FACTOR = 0.5
LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 20
NUM_EPOCHS = 50

# Define the loss function, optimizer, and learning rate scheduler

# criterion = smp.losses.DiceLoss('multiclass')    # If you want to use DiceLoss
from monai.losses import DiceCELoss

# Define the loss function
criterion = DiceCELoss(
    include_background=True,  # If you want to use DiceCELoss
    to_onehot_y=True,         
    softmax=True,             
    lambda_dice=0.5,          
    lambda_ce=0.5             
)
optimizer = optim.Adam(resnet_unet.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, verbose=False)

# Initialize early stopping parameters
early_stop_counter = 0
best_val_loss = float('inf')

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

def compute_epoch_metrics(true_masks, pred_masks):
    metrics = {}
    # Flatten arrays to compute metrics for all pixels
    true_flat = true_masks.ravel()
    pred_flat = pred_masks.ravel()
    
    metrics['Accuracy'] = accuracy_score(true_flat, pred_flat)
    metrics['Precision'] = precision_score(true_flat, pred_flat, average='binary', zero_division=0)
    metrics['Recall'] = recall_score(true_flat, pred_flat, average='binary', zero_division=0)
    metrics['F1'] = f1_score(true_flat, pred_flat, average='binary', zero_division=0)
    metrics['IoU'] = jaccard_score(true_flat, pred_flat, average='binary')
    
    return metrics

def train_model_with_validation_metrics(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, early_stop_patience=EARLY_STOP_PATIENCE):
    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses, val_losses = [], []
    val_metrics = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        # Training loop
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        true_val_masks, pred_val_masks = [], []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Collect predictions for metrics
                pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
                true_val_masks.append(masks.cpu().numpy())
                pred_val_masks.append(pred_masks)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Compute validation metrics
        true_val_masks = np.concatenate(true_val_masks)
        pred_val_masks = np.concatenate(pred_val_masks)
        epoch_val_metrics = compute_epoch_metrics(true_val_masks, pred_val_masks)
        val_metrics.append(epoch_val_metrics)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Print metrics
        print(f"Epoch [{epoch}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation Metrics: {epoch_val_metrics}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), '20.pth')
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping after {early_stop_patience} epochs without improvement.")
            break

    # Save training and validation metrics to Excel
    metrics_data = {
        'Epoch': list(range(1, len(val_metrics) + 1)),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Accuracy': [m['Accuracy'] for m in val_metrics],
        'IoU': [m['IoU'] for m in val_metrics],
        'F1 Score': [m['F1'] for m in val_metrics],
        'Precision': [m['Precision'] for m in val_metrics],
        'Recall': [m['Recall'] for m in val_metrics],
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_excel('best_model_256_augmentation_resnet101.xlsx', index=False)


    # Plot validation metrics
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_data['Epoch'], metrics_data['Accuracy'], label='Accuracy', marker='o')
    plt.plot(metrics_data['Epoch'], metrics_data['IoU'], label='IoU', marker='o')
    plt.plot(metrics_data['Epoch'], metrics_data['F1 Score'], label='F1 Score', marker='o')
    plt.plot(metrics_data['Epoch'], metrics_data['Precision'], label='Precision', marker='o')
    plt.plot(metrics_data['Epoch'], metrics_data['Recall'], label='Recall', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_metrics_plot.png')
    plt.show()

    return val_losses, val_metrics

# Train the model with validation metrics tracking
val_losses, val_metrics = train_model_with_validation_metrics(
    resnet_unet, train_loader, val_loader, criterion, optimizer, scheduler, 
    num_epochs=NUM_EPOCHS, early_stop_patience=EARLY_STOP_PATIENCE
)

import matplotlib.pyplot as plt
import os
import torch

# Function to save the predicted mask and image separately
output_dir = "C:/Users/Hadi/Downloads/twitter-flood-dataset-master/twitter-flood-dataset-master/outputs/"

def save_example(image, mask, pred_mask, idx):
    # Save the image
    image_path = os.path.join(output_dir, f"image_{idx + 1}.png")
    plt.imsave(image_path, denormalize_tensor(image).permute(1, 2, 0).numpy())  # Save image
    print(f"Saved image to {image_path}")

    # Save the true mask
    true_mask_path = os.path.join(output_dir, f"true_mask_{idx + 1}.png")
    # Convert the mask to 2D and save as grayscale
    plt.imsave(true_mask_path, mask.squeeze().numpy(), cmap='gray')  # Save mask
    print(f"Saved true mask to {true_mask_path}")

    # Save the predicted mask
    pred_mask_path = os.path.join(output_dir, f"pred_mask_{idx + 1}.png")
    # Convert the predicted mask to 2D and save as grayscale
    plt.imsave(pred_mask_path, pred_mask.squeeze().numpy(), cmap='gray')  # Save predicted mask
    print(f"Saved predicted mask to {pred_mask_path}")

def plot_examples(model, dataset, num_examples=len(test_dataset)):
    model.eval()

    for i in range(num_examples):
        image, mask = dataset[i]

        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device)).cpu()

        pred_mask = torch.argmax(output, dim=1)

        # Save the image, true mask, and predicted mask separately
        save_example(image, mask, pred_mask, i)

# Plot examples from the test dataloader
plot_examples(resnet_unet, test_dataset, num_examples=42)


# Function to compute metrics
def compute_test_metrics(model, test_loader):
    model.eval()
    true_test_masks = []
    pred_test_masks = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Get predictions
            pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
            true_test_masks.append(masks.cpu().numpy())
            pred_test_masks.append(pred_masks)
    
    # Flatten arrays for metric computation
    true_test_masks = np.concatenate(true_test_masks).ravel()
    pred_test_masks = np.concatenate(pred_test_masks).ravel()
    
    # Compute metrics
    metrics = {
        "Accuracy": accuracy_score(true_test_masks, pred_test_masks),
        "Precision": precision_score(true_test_masks, pred_test_masks, average="binary", zero_division=0),
        "Recall": recall_score(true_test_masks, pred_test_masks, average="binary", zero_division=0),
        "F1 Score": f1_score(true_test_masks, pred_test_masks, average="binary", zero_division=0),
        "IoU (Jaccard Index)": jaccard_score(true_test_masks, pred_test_masks, average="binary")
    }
    
    return metrics

# Compute and print test metrics
test_metrics = compute_test_metrics(resnet_unet, test_loader)
print("Test Metrics:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Compute and print validation metrics
validation2_metrics = compute_test_metrics(resnet_unet, val_loader)
print("Validation2 Metrics:")
for metric, value in validation2_metrics.items():
    print(f"  {metric}: {value:.4f}")
    
    
    










