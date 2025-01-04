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

def get_data(subset_size=None, train=True):
    """
    Load and prepare the dataset
    Args:
        subset_size (int): If provided, return only a subset of data
        train (bool): If True, return training data, else test data
    """
    transform = get_transforms()
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=train,
        download=True, 
        transform=transform
    )
    
    if subset_size:
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True if train else False,
        num_workers=2
    )
    
    return dataloader

def evaluate_model(model, testloader, device):
    """
    Evaluate the model on test data
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def train_model(model, trainloader, testloader, epochs=100, device='cuda'):
    """
    Train the model
    Args:
        model: The ResNet50 model
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
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
    
    # Create epoch progress bar without a description (we'll use it for stats only)
    epoch_pbar = tqdm(range(epochs), desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create batch progress bar with position below epoch bar
        batch_pbar = tqdm(trainloader, 
                         desc=f'Epoch {epoch+1}', 
                         position=1, 
                         leave=True)
        
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
        
        # Evaluate on test data
        test_acc = evaluate_model(model, testloader, device)
        epoch_pbar.write(f'Epoch {epoch+1}: Train Loss: {avg_loss:.3f} | Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}%')
        
        scheduler.step(test_acc)  # Using test accuracy for scheduler
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, 'best_model.pth')
            epoch_pbar.write(f'New best test accuracy: {test_acc:.2f}%')
        
        if test_acc > 70:
            epoch_pbar.write(f"\nReached target accuracy of 70% on test data!")
            break

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get train and test data
    trainloader = get_data(subset_size=5000, train=True)
    testloader = get_data(subset_size=1000, train=False)
    
    # Initialize model
    model = get_model(num_classes=10)
    
    # Train model
    train_model(model, trainloader, testloader, epochs=100, device=device) 