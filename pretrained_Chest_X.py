import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# You need to specify the correct path to your dataset on your local machine
dataset_path = r"C:\Users\Students\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray"

# Create directories for checkpoints and results
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_results")

for directory in [checkpoint_dir, results_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory at: {directory}")

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Datasets
try:
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")
    
    print(f"Checking paths: train={os.path.exists(train_path)}, val={os.path.exists(val_path)}, test={os.path.exists(test_path)}")
    
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Successfully loaded datasets with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test images")
    
    # Print class names
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

except Exception as e:
    print(f"Error loading datasets: {e}")
    print("\nPossible solution: Please update the 'dataset_path' variable to point to where chest_xray folder is located")
    import sys
    sys.exit(1)

# Load pre-trained DenseNet121 model
def load_pretrained_model(num_classes=2):
    # Load pre-trained DenseNet121
    model = models.densenet121(pretrained=True)
    
    # Freeze all the parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    
    return model

# Function to save checkpoint
def save_checkpoint(state, is_best=False, checkpoint_dir=checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)
        print(f"Best model saved to {best_filepath}")

# Function to load checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        train_losses = checkpoint.get('train_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"Successfully loaded checkpoint from epoch {start_epoch}")
        return start_epoch, best_accuracy, train_losses, val_accuracies
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0.0, [], []

# Function to save training results as graphs
def save_training_graphs(train_losses, val_accuracies, save_path=results_dir):
    if not train_losses and not val_accuracies:
        print("No data to plot")
        return
    
    # Plot training loss
    if train_losses:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "training_loss.png"))
        plt.close()
    
    # Plot validation accuracy
    if val_accuracies:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='x', linestyle='--', color='g')
        plt.title('Validation Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "validation_accuracy.png"))
        plt.close()
    
    print(f"Training graphs saved to {save_path}")

# Function to evaluate the model
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

# Function for inference on a single image
def predict_image(model, image_path, transform):
    from PIL import Image
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = F.softmax(output, dim=1)
    
    # Get probability of predicted class
    prob = probabilities[0][predicted.item()].item() * 100
    
    return predicted.item(), prob

# Initialize model, loss function, and optimizer
model = load_pretrained_model(num_classes=2)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)  # Only train the classifier

# Check if checkpoint exists
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth.tar")
start_epoch = 0
best_accuracy = 0.0
train_losses = []
val_accuracies = []

if os.path.exists(checkpoint_path):
    print("=> Loading checkpoint...")
    start_epoch, best_accuracy, train_losses, val_accuracies = load_checkpoint(checkpoint_path, model, optimizer)
    print(f"=> Loaded checkpoint from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting training from scratch.")

# Training Loop
epochs = 52
print(f"Starting training from epoch {start_epoch + 1}")

try:
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Print batch progress
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate on validation set
        val_accuracy, _, _ = evaluate(model, val_loader)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save checkpoint if validation accuracy improved
        is_best = val_accuracy > best_accuracy
        if is_best:
            best_accuracy = val_accuracy
        
        # Save checkpoint every 5 epochs or if it's the best model
        if (epoch + 1) % 5 == 0 or is_best or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            }
            save_checkpoint(checkpoint, is_best=is_best)
            save_training_graphs(train_losses, val_accuracies)

    # Final evaluation on test set
    test_accuracy, test_preds, test_labels = evaluate(model, test_loader)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    # Save final training graphs
    save_training_graphs(train_losses, val_accuracies)

    print("Training complete!")
    
except KeyboardInterrupt:
    print("\nTraining interrupted! Saving checkpoint...")
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }
    save_checkpoint(checkpoint)
    save_training_graphs(train_losses, val_accuracies)
    print("Checkpoint saved. You can resume training later.")

except Exception as e:
    print(f"An error occurred during training: {e}")
    # Save emergency checkpoint
    checkpoint = {
        'epoch': epoch if 'epoch' in locals() else start_epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }
    emergency_path = os.path.join(checkpoint_dir, "emergency_checkpoint.pth.tar")
    torch.save(checkpoint, emergency_path)
    print(f"Emergency checkpoint saved to {emergency_path}")

# Example of inference on a single image
def demo_inference():
    # Load the best model
    best_model_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded best model for inference")
    
    # Example: test on a few images from the test set
    test_path = os.path.join(dataset_path, "test")
    normal_dir = os.path.join(test_path, "NORMAL")
    pneumonia_dir = os.path.join(test_path, "PNEUMONIA")
    
    # Get a few sample images
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)[:3]]
    pneumonia_images = [os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir)[:3]]
    
    print("\nInference Demo:")
    print("--------------")
    
    for img_path in normal_images + pneumonia_images:
        pred_class, confidence = predict_image(model, img_path, transform)
        true_class = "NORMAL" if "NORMAL" in img_path else "PNEUMONIA"
        pred_class_name = class_names[pred_class]
        
        print(f"Image: {os.path.basename(img_path)}")
        print(f"True Class: {true_class}")
        print(f"Predicted Class: {pred_class_name} with {confidence:.2f}% confidence")
        print("--------------")

def save_enhanced_training_graphs(train_losses, val_accuracies, save_path=results_dir):
    if not train_losses and not val_accuracies:
        print("No data to plot")
        return

    # Create figure with two y-axes for loss and accuracy
    fig, ax1 = plt.figure(figsize=(14, 8)), plt.gca()
    
    # Plot training loss on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Training Loss', color=color, fontsize=12)
    ax1.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', 
             color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create a second y-axis for validation accuracy
    if val_accuracies:
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=12)
        ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='x', 
                 linestyle='--', color=color, label='Validation Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and legend
    plt.title('Training Progress: Loss vs Epochs', fontsize=14)
    
    # Create custom legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if val_accuracies:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        plt.legend(lines, labels, loc='best')
    else:
        plt.legend(lines1, labels1, loc='best')
    
    plt.tight_layout()
    
    # Save the combined plot
    combined_path = os.path.join(save_path, "training_progress.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced training graph saved to {combined_path}")

    # Also save the original individual plots
    save_individual_training_graphs(train_losses, val_accuracies, save_path)

def save_individual_training_graphs(train_losses, val_accuracies, save_path=results_dir):
    # Plot training loss
    if train_losses:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "training_loss_pretrained.png"))
        plt.close()

    # Plot validation accuracy
    if val_accuracies:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='x', linestyle='--', color='g')
        plt.title('Validation Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "validation_accuracy_pretrained.png"))
        plt.close()

# Run inference demo after training
if __name__ == "__main__":
    demo_inference()