import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

# Load model
model = resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Define classes (for CIFAR-10)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        
    return classes[predicted.item()]

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    examples=[["example1.jpg"], ["example2.jpg"]]
)

iface.launch() 