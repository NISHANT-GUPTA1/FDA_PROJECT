import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
from collections import OrderedDict
import re

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# You need to specify the correct path to your dataset on your local machine
# Change this to where your dataset is actually stored
dataset_path = r"C:\Users\Students\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray"

# Create directories for checkpoints and results
base_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(base_dir, "checkpoints")
results_dir = os.path.join(base_dir, "training_results")
gradcam_dir = os.path.join(base_dir, "gradcam_results")

for directory in [checkpoint_dir, results_dir, gradcam_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory at: {directory}")

# Create a temp directory for frequent automatic checkpoints
temp_checkpoint_dir = os.path.join(checkpoint_dir, "temp")
if not os.path.exists(temp_checkpoint_dir):
    os.makedirs(temp_checkpoint_dir)
    print(f"Created temporary checkpoint directory at: {temp_checkpoint_dir}")

# Image Transformations - Standard preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data augmentation transformations (for domain shift robustness)
augment_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Datasets
try:
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")
    
    print(f"Checking paths: train={os.path.exists(train_path)}, val={os.path.exists(val_path)}, test={os.path.exists(test_path)}")
    
    # Default datasets (original domain)
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    
    # Augmented dataset (simulated domain shift)
    train_dataset_augmented = datasets.ImageFolder(root=train_path, transform=augment_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_loader_augmented = DataLoader(train_dataset_augmented, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Successfully loaded datasets with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test images")
    
    # Get class names
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

except Exception as e:
    print(f"Error loading datasets: {e}")
    print("\nPossible solution: Please update the 'dataset_path' variable to point to where chest_xray folder is located")
    import sys
    sys.exit(1)

# Custom Dataset for Domain Shift - Used for visualization and analysis
class DomainShiftDataset(Dataset):
    def __init__(self, base_dataset, transform=None, domain_shift_type='default'):
        self.base_dataset = base_dataset
        self.transform = transform
        self.domain_shift_type = domain_shift_type
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # If we're using an instance of torchvision's ImageFolder, 
        # we need to get the actual image path
        if hasattr(self.base_dataset, 'samples'):
            image_path = self.base_dataset.samples[idx][0]
            # Load the image from path
            image = Image.open(image_path).convert('RGB')
        
        # Apply domain shifts based on the type
        if self.domain_shift_type == 'contrast':
            # Reduce contrast
            enhancer = transforms.ColorJitter(contrast=0.5)
            image = enhancer(image)
        elif self.domain_shift_type == 'noise':
            # Convert to numpy array
            img_array = np.array(image)
            # Add Gaussian noise
            noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy_img)
        elif self.domain_shift_type == 'blur':
            # Convert to numpy array and apply blur
            img_array = np.array(image)
            blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
            image = Image.fromarray(blurred)
        elif self.domain_shift_type == 'intensity':
            # Adjust brightness/intensity
            enhancer = transforms.ColorJitter(brightness=0.3)
            image = enhancer(image)
        
        # Apply the transform on the modified image
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_model(pretrained=True, num_classes=2):
    """
    Load DenseNet121 model with appropriate modifications for chest X-ray classification
    """
    if pretrained:
        model = models.densenet121(pretrained=True)
        # Freeze early layers if using pretrained
        for param in list(model.parameters())[:-20]:  # Freeze all except the last few layers
            param.requires_grad = False
    else:
        model = models.densenet121(pretrained=False)
    
    # Replace the classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    
    return model

# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        logits = self.model(input_image)
        
        # If target class is not specified, use the predicted class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Zero all gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.zeros_like(logits)
        one_hot_output[0, target_class] = 1
        
        # Backward pass
        logits.backward(gradient=one_hot_output, retain_graph=True)
        
        # Get mean gradients and activations
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activation maps with gradients
        for i in range(pooled_gradients.shape[0]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # Generate heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze().detach().cpu().numpy()
        
        # ReLU on the heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize the heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap, target_class

# Function to apply GradCAM and visualize results
def visualize_gradcam(model, image_tensor, original_image, class_names, target_class=None, save_path=None):
    # Get the final feature layer (for DenseNet121)
    target_layer = model.features[-1]  # final dense block's output
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate the CAM
    heatmap, predicted_class = grad_cam.generate_cam(image_tensor.unsqueeze(0).to(device), target_class)
    
    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image to BGR (for OpenCV)
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    # Superimpose the heatmap on the original image
    superimposed = cv2.addWeighted(original_image_bgr, 0.5, heatmap, 0.5, 0)
    
    # Convert back to RGB for matplotlib
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    
    # Create the visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original image
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # GradCAM visualization
    ax[1].imshow(superimposed)
    ax[1].set_title(f'GradCAM: {class_names[predicted_class]}')
    ax[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()
        return None

# Function to analyze domain shift effect with GradCAM
def analyze_domain_shift(model, dataset, domain_shift_types, class_names, num_samples=3, save_dir=gradcam_dir):
    """
    Analyze how GradCAM heatmaps change under different domain shifts
    """
    model.eval()
    
    # Create a directory for each domain shift type
    for domain_type in domain_shift_types:
        domain_dir = os.path.join(save_dir, domain_type)
        if not os.path.exists(domain_dir):
            os.makedirs(domain_dir)
    
    # Get samples for analysis (one from each class if possible)
    class_samples = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in class_samples:
            class_samples[label] = []
        if len(class_samples[label]) < num_samples:
            class_samples[label].append(i)
        
        # Break if we have enough samples from each class
        if all(len(samples) >= num_samples for samples in class_samples.values()):
            break
    
    # For each sample and domain shift type, generate and save GradCAM visualizations
    for label, sample_indices in class_samples.items():
        for idx in sample_indices:
            # Get original image path and load the image
            image_path = dataset.samples[idx][0]
            original_image = Image.open(image_path).convert('RGB')
            original_image_np = np.array(original_image)
            
            # Apply standard transform
            image_tensor = transform(original_image).to(device)
            
            # Predict class
            with torch.no_grad():
                outputs = model(image_tensor.unsqueeze(0))
                _, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()
                
            # Generate and save GradCAM for original domain
            original_save_path = os.path.join(save_dir, 'default', f'sample_{idx}_class_{label}_pred_{predicted_class}.jpg')
            visualize_gradcam(model, image_tensor, original_image_np, class_names, target_class=None, save_path=original_save_path)
            
            # For each domain shift type, generate and save GradCAM
            for domain_type in domain_shift_types:
                if domain_type == 'default':
                    continue
                
                # Create a temporary dataset with the domain shift
                shifted_dataset = DomainShiftDataset(dataset, transform, domain_type)
                
                # Get the shifted image
                shifted_image_tensor, _ = shifted_dataset[idx]
                
                # Convert the tensor to a PIL image for visualization
                shifted_image = transforms.ToPILImage()(shifted_image_tensor)
                shifted_image_np = np.array(shifted_image)
                
                # Apply model and GradCAM
                shifted_save_path = os.path.join(save_dir, domain_type, f'sample_{idx}_class_{label}_pred_{predicted_class}.jpg')
                visualize_gradcam(model, shifted_image_tensor.to(device), shifted_image_np, class_names, target_class=None, save_path=shifted_save_path)
                
                print(f"Processed sample {idx} with domain shift {domain_type}")
    
    print(f"Domain shift analysis complete. Results saved to {save_dir}")

# Function to save confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Generate and plot a confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()

# Function to save checkpoint
def save_checkpoint(state, is_best=False, checkpoint_dir=checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    
    # First save to a temporary file to prevent corruption if power fails during saving
    temp_filepath = filepath + '.temp'
    torch.save(state, temp_filepath)
    
    # Then rename the file to the final name (this is an atomic operation)
    if os.path.exists(filepath):
        os.remove(filepath)  # Remove existing file if it exists
    os.rename(temp_filepath, filepath)
    
    print(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)
        print(f"Best model saved to {best_filepath}")

# Function to save temporary checkpoint (called more frequently)
def save_temp_checkpoint(epoch, batch_idx, model, optimizer, scheduler, train_losses, val_accuracies, domain_metrics=None):
    state = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'domain_metrics': domain_metrics,
        'timestamp': time.time()
    }
    
    filepath = os.path.join(temp_checkpoint_dir, 'temp_checkpoint.pth.tar')
    
    # Use the same safe saving approach
    temp_filepath = filepath + '.temp'
    torch.save(state, temp_filepath)
    
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(temp_filepath, filepath)

# Function to load checkpoint
def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    try:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        batch_idx = checkpoint.get('batch_idx', 0)  # Default to 0 if not present
        
        train_losses = checkpoint.get('train_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        domain_metrics = checkpoint.get('domain_metrics', {})
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler']:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Successfully loaded checkpoint from epoch {start_epoch}, batch {batch_idx}")
        return start_epoch, batch_idx, train_losses, val_accuracies, domain_metrics
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0, [], [], {}

# Function to check if training was interrupted
def check_for_interruption():
    temp_checkpoint_path = os.path.join(temp_checkpoint_dir, 'temp_checkpoint.pth.tar')
    if os.path.exists(temp_checkpoint_path):
        try:
            checkpoint = torch.load(temp_checkpoint_path)
            # Check if the temporary checkpoint is more recent than the main checkpoint
            main_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
            
            if not os.path.exists(main_checkpoint_path):
                return temp_checkpoint_path
            
            main_checkpoint = torch.load(main_checkpoint_path)
            
            # If temp checkpoint is from a later epoch or later batch in the same epoch
            if (checkpoint['epoch'] > main_checkpoint['epoch'] or 
                (checkpoint['epoch'] == main_checkpoint['epoch'] and 
                 checkpoint.get('batch_idx', 0) > main_checkpoint.get('batch_idx', 0))):
                return temp_checkpoint_path
        except Exception as e:
            print(f"Failed to check temporary checkpoint: {e}")
    
    return None

# Function to evaluate the model
def evaluate(model, data_loader, device, return_predictions=False):
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
            
            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    if return_predictions:
        return accuracy, all_preds, all_labels
    return accuracy

# Function to evaluate domain shift performance
def evaluate_domain_shift(model, base_dataset, transform, domain_types, device):
    """
    Evaluate model performance across different domain shifts
    """
    results = {}
    
    for domain_type in domain_types:
        # Create shifted dataset
        shifted_dataset = DomainShiftDataset(base_dataset, transform, domain_type)
        shifted_loader = DataLoader(shifted_dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        accuracy = evaluate(model, shifted_loader, device)
        results[domain_type] = accuracy
        print(f"Domain: {domain_type}, Accuracy: {accuracy:.2f}%")
    
    return results

# Function to save domain shift performance graph
def save_domain_shift_graph(domain_metrics, save_path=results_dir):
    """
    Generate a graph showing model performance across domain shifts
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.figure(figsize=(14, 8))
    
    # If we have multiple epochs of domain shift data, plot the trend
    if isinstance(domain_metrics, dict) and domain_metrics and isinstance(list(domain_metrics.values())[0], list):
        # This means we have data across epochs
        epochs = range(1, len(list(domain_metrics.values())[0]) + 1)
        
        for domain, accuracies in domain_metrics.items():
            plt.plot(epochs, accuracies, marker='o', linestyle='-', label=f'{domain}')
    else:
        # Just plot the latest results
        domains = list(domain_metrics.keys())
        accuracies = list(domain_metrics.values())
        
        plt.bar(domains, accuracies, color='skyblue')
        plt.ylim(0, 100)  # Accuracy scale
        
        # Add value labels on top of bars
        for i, v in enumerate(accuracies):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    plt.title('Model Performance Across Domain Shifts')
    plt.xlabel('Domain')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    # Save the figure
    domain_graph_path = os.path.join(save_path, "domain_shift_performance.png")
    plt.savefig(domain_graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Domain shift graph saved to {domain_graph_path}")

# Function to save training and validation graphs
def save_training_graphs(train_losses, val_accuracies, domain_metrics=None, save_path=results_dir):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Create figure with two y-axes for loss and accuracy
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot training loss on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Training Loss', color=color, fontsize=12)
    ax1.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', 
             color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create a second y-axis for validation accuracy
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=12)
    ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='x', 
             linestyle='--', color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and legend
    plt.title('Training Progress: Loss vs Accuracy', fontsize=14)
    
    # Create custom legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)
    
    plt.tight_layout()
    
    # Save the combined plot
    combined_path = os.path.join(save_path, "training_progress.png")
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training graph saved to {combined_path}")
    
    # Also save individual plots
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "training_loss.png"), dpi=150)
    plt.close()
    
    # Plot validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='x', linestyle='--', color='g')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "validation_accuracy.png"), dpi=150)
    plt.close()
    
    # If we have domain metrics, plot them too
    if domain_metrics:
        save_domain_shift_graph(domain_metrics, save_path)

# Training function with domain shift analysis
def train_model_with_domain_shift(model, train_loader, val_loader, test_dataset, domain_types, 
                                criterion, optimizer, scheduler, device, 
                                start_epoch=0, start_batch=0, train_losses=None, val_accuracies=None, 
                                domain_metrics=None, epochs=100, checkpoint_frequency=100,
                                gradcam_frequency=10):
    
    if train_losses is None:
        train_losses = []
    if val_accuracies is None:
        val_accuracies = []
    if domain_metrics is None:
        domain_metrics = {domain: [] for domain in domain_types}
    
    # Create a flag file to indicate training is in progress
    with open(os.path.join(temp_checkpoint_dir, 'training_in_progress'), 'w') as f:
        f.write('1')
    
    best_val_accuracy = max(val_accuracies) if val_accuracies else 0
    
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            epoch_start_time = time.time()
            
            # If we're resuming in the middle of an epoch
            start_idx = start_batch if epoch == start_epoch else 0
            
            for i, (images, labels) in enumerate(train_loader):
                # Skip batches we've already processed if resuming
                if i < start_idx:
                    continue
                
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
                    elapsed_time = time.time() - epoch_start_time
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, Time: {elapsed_time:.2f}s")
                
                # Frequent temporary checkpointing
                if (i + 1) % checkpoint_frequency == 0:
                    save_temp_checkpoint(epoch, i + 1, model, optimizer, scheduler, train_losses, val_accuracies, domain_metrics)
            
            # Update learning rate scheduler if provided
            if scheduler:
                scheduler.step()
            
            # Calculate epoch loss
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            
            # Evaluate on validation set
            val_accuracy = evaluate(model, val_loader, device)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            
            # Periodically evaluate domain shift performance
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print("\nEvaluating domain shift performance...")
                domain_results = evaluate_domain_shift(model, test_dataset, transform, domain_types, device)
                
                # Store domain shift results for this epoch
                for domain, accuracy in domain_results.items():
                    domain_metrics[domain].append(accuracy)
                
                print("Domain shift evaluation complete\n")
            
            # Generate GradCAM visualizations periodically
            if (epoch + 1) % gradcam_frequency == 0 or epoch == epochs - 1:
                print("\nGenerating GradCAM visualizations for domain shift analysis...")
                epoch_gradcam_dir = os.path.join(gradcam_dir, f"epoch_{epoch+1}") 
                os.makedirs(epoch_gradcam_dir, exist_ok=True)
                
                analyze_domain_shift(
                    model=model,
                    dataset=test_dataset,
                    domain_shift_types=domain_types,
                    class_names=class_names,
                    save_dir=epoch_gradcam_dir
                )
                print("GradCAM analysis complete.\n")
            
            # Save checkpoint at the end of the epoch
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'domain_metrics': domain_metrics,
            }, is_best=(val_accuracy > best_val_accuracy))
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
            
            # Reset batch index after completing epoch
            start_batch = 0

        # Final visualization after training
        save_training_graphs(train_losses, val_accuracies, domain_metrics)

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'domain_metrics': domain_metrics,
        })

    finally:
        # Remove the training-in-progress flag file
        in_progress_flag = os.path.join(temp_checkpoint_dir, 'training_in_progress')
        if os.path.exists(in_progress_flag):
            os.remove(in_progress_flag)
        print("Training completed.")


if __name__ == "__main__":
    # Load model
    model = load_model(pretrained=True, num_classes=len(class_names)).to(device)
    
    # Loss function, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Domain shift types to analyze
    domain_types = ['default', 'contrast', 'noise', 'blur', 'intensity']
    
    # Train the model and analyze domain shift
    train_model_with_domain_shift(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_dataset=test_dataset,
        domain_types=domain_types,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=100,  
        checkpoint_frequency=100,
        gradcam_frequency=1  # Generate GradCAM every epoch
    )
