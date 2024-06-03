import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

# Function to get data loaders
def get_data_loaders():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Loading datasets
    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(root=test_dir, transform=data_transforms['test']),
    }
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
    }
    dataset_sizes = {
        'train': len(image_datasets['train']),
        'valid': len(image_datasets['valid']),
        'test': len(image_datasets['test']),
    }
    
    return dataloaders, dataset_sizes, image_datasets

# Function to build the model
def build_model(arch='vgg16', hidden_units=4096, learning_rate=0.001):
    if arch == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
        input_size = 25088
    elif arch == 'alexnet':
        model = models.alexnet(weights='IMAGENET1K_V1')
        input_size = 9216
    else:
        print(f"Architecture {arch} is not supported.")
        return None
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer

# Function to train the model
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, epochs):
    model.to(device)
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                accuracy += torch.sum(preds == labels.data)
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {running_loss/len(dataloaders['train'])}, "
              f"Validation Loss: {val_loss/len(dataloaders['valid'])}, "
              f"Accuracy: {accuracy.double()/dataset_sizes['valid']}")
    
    return model

# Function to save the model checkpoint
def save_checkpoint(model, optimizer, image_datasets, save_path='checkpoint.pth'):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset of images.')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or alexnet).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units in the classifier.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    args = parse_args()
    
    dataloaders, dataset_sizes, image_datasets = get_data_loaders()
    model, criterion, optimizer = build_model(arch=args.arch, hidden_units=args.hidden_units, learning_rate=args.learning_rate)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    trained_model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, args.epochs)
    save_checkpoint(trained_model, optimizer, image_datasets)
