import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

# Function to load the model checkpoint
def load_checkpoint(arch='vgg16'):
    checkpoint = torch.load('checkpoint.pth')
    if arch == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
    elif arch == 'alexnet':
        model = models.alexnet(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Architecture {arch} is not supported.")
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Function to process an image
def process_image(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = preprocess(img)
    return img

# Function to predict the class of an image
def predict(image_path, model, topk=5, device='cpu'):
    model.to(device)
    model.eval()
    image = process_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    probabilities = torch.exp(output).topk(topk)
    probs = probabilities[0].tolist()[0]
    classes = probabilities[1].tolist()[0]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[c] for c in classes]
    return probs, classes

# Function to display an image and its top 5 predicted classes
def imshow(image, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

# Function to visualize the prediction
def visualize_prediction(image_path, model, cat_to_name, topk=5, device='cpu'):
    probs, classes = predict(image_path, model, topk=topk, device=device)
    class_names = [cat_to_name[str(cls)] for cls in classes]
    img = process_image(image_path)
    
    _, (ax1, ax2) = plt.subplots(figsize=(6,9), nrows=2)
    imshow(img, ax=ax1)
    ax1.set_title(class_names[0])
    
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.invert_yaxis()
    plt.show()

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Predict image class using a trained network.')
    parser.add_argument('image_path', type=str, help='Path to the image.')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or alexnet).')
    parser.add_argument('--top_k', type=int, default=5, help='Top K most likely classes.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to JSON file mapping categories to names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for prediction.')
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    args = parse_args()

    model = load_checkpoint(args.arch)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    visualize_prediction(args.image_path, model, cat_to_name, topk=args.top_k, device=device)
