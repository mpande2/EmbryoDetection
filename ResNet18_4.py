import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define a custom dataset
class EggDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.transform = transform

    def load_annotations(self, annotation_file):
        annotations = []
        with open(annotation_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append((class_id, x_center, y_center, width, height))
        return annotations

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        annotation_file = os.path.join(self.annotations_dir, os.path.splitext(img_file)[0] + '.txt')
        
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        
        annotations = self.load_annotations(annotation_file)
        
        crops = []
        labels = []
        for class_id, x_center, y_center, width, height in annotations:
            x = int((x_center - width / 2) * img_width)
            y = int((y_center - height / 2) * img_height)
            w = int(width * img_width)
            h = int(height * img_height)
            cropped_image = image.crop((x, y, x + w, y + h))
            if self.transform:
                cropped_image = self.transform(cropped_image)
            crops.append(cropped_image)
            labels.append(class_id)

        return torch.stack(crops), torch.tensor(labels)

# Define transformations for training data with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transformations for validation data without augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset with transformations
dataset = EggDataset(img_dir='synthetic_frog_eggs/images', annotations_dir='synthetic_frog_eggs/annotations', transform=transforms)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Apply transformations to datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Custom collate function to handle batches with varying numbers of crops
def custom_collate_fn(batch):
    crops = []
    labels = []
    for b in batch:
        crops.append(b[0])
        labels.append(b[1])
    crops = torch.cat(crops)
    labels = torch.cat(labels)
    return crops, labels

# Reduce batch size to minimize memory usage
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Create a new ResNet18 model
model = models.resnet18(pretrained=False)  # Initialize without pretrained weights

# Modify the final layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Use mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with gradient accumulation and mixed precision
def train_model(model, criterion, optimizer, num_epochs=10, accumulation_steps=4):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Normalize loss by accumulation steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * inputs.size(0) * accumulation_steps  # Multiply back by accumulation steps
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += labels.size(0)
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                val_total_samples += labels.size(0)
        
        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects / val_total_samples
        
        print(f'Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
    
    return model

model = train_model(model, criterion, optimizer, num_epochs=10)

# Save the trained model
torch.save(model.state_dict(), 'resnet18_egg_classifier.pth')

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    running_corrects = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += labels.size(0)
    
    accuracy = running_corrects / total_samples
    print(f'Accuracy: {accuracy:.4f}')

evaluate_model(model, val_loader)

# Visualize and save predictions
def visualize_predictions(model, dataloader, output_dir='predictions', num_images=6):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                img = transforms.ToPILImage()(inputs.cpu().data[j])
                pred_label = preds[j].item()
                img.save(os.path.join(output_dir, f'predicted_{images_so_far}_label_{pred_label}.png'))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

visualize_predictions(model, val_loader)

