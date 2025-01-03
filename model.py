import torch
import torch.nn as nn
from torchvision.models import resnet50

def get_model(num_classes):
    """
    Initialize a ResNet50 model from scratch
    Args:
        num_classes (int): Number of output classes
    Returns:
        model: ResNet50 model with custom final layer
    """
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def save_model(model, path):
    """
    Save model state dict
    """
    torch.save(model.state_dict(), path)

def load_model(num_classes, path):
    """
    Load a saved model
    """
    model = get_model(num_classes)
    model.load_state_dict(torch.load(path))
    return model 