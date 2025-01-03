import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from model import get_model, save_model
from tqdm import tqdm

def get_transforms():
    """
    Define the image transformations
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def get_data(subset_size=None):
    """
    Load and prepare the dataset
    Args:
        subset_size (int): If provided, return only a subset of data
    """
    transform = get_transforms()
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    if subset_size:
        indices = torch.randperm(len(trainset))[:subset_size]
        trainset = Subset(trainset, indices)
    
    trainloader = DataLoader(
        trainset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    return trainloader

def train_model(model, trainloader, epochs=100, device='cuda'):
    """
    Train the model
    Args:
        model: The ResNet50 model
        trainloader: DataLoader for training data
        epochs (int): Number of epochs to train
        device (str): Device to train on ('cuda' or 'cpu')
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'max', 
        patience=5
    )
    
    best_acc = 0.0
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc='Training')
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create batch progress bar
        batch_pbar = tqdm(trainloader, leave=False, desc=f'Epoch {epoch+1}')
        
        for inputs, labels in batch_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update batch progress bar
            batch_pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        epoch_acc = 100. * correct / total
        avg_loss = running_loss/len(trainloader)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'loss': f'{avg_loss:.3f}',
            'accuracy': f'{epoch_acc:.2f}%'
        })
        
        scheduler.step(epoch_acc)
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_model(model, 'best_model.pth')
            
        if epoch_acc > 70:
            print(f"\nReached target accuracy of 70%!")
            break

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data
    trainloader = get_data(subset_size=5000)  # Using subset for initial testing
    
    # Initialize model
    model = get_model(num_classes=10)
    
    # Train model
    train_model(model, trainloader, epochs=10, device=device) 