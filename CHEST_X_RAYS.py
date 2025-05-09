import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.optim as optim
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# You need to specify the correct path to your dataset on your local machine
# Change this to where your dataset is actually stored
# For example:
dataset_path = r"C:\Users\Students\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray"

# If this doesn't match the actual location, update it to the correct path
print(f"Using dataset path: {dataset_path}")
print(f"Checking if path exists: {os.path.exists(dataset_path)}")

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize
])

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
}

# Load Datasets - with error handling to help debug path issues
try:
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")
    
    print(f"Checking train path: {train_path}, exists: {os.path.exists(train_path)}")
    print(f"Checking val path: {val_path}, exists: {os.path.exists(val_path)}")
    print(f"Checking test path: {test_path}, exists: {os.path.exists(test_path)}")
    
    # Load Datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    
    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Successfully loaded datasets with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test images")

except Exception as e:
    print(f"Error loading datasets: {e}")
    print("\nPossible solution: Please update the 'dataset_path' variable in the code to point to where chest_xray folder is located on your computer")
    import sys
    sys.exit(1)

class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)



# Create checkpoint directory (in your code's directory)
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print(f"Created checkpoint directory at: {checkpoint_dir}")

# Create a temp directory for frequent automatic checkpoints
temp_checkpoint_dir = os.path.join(checkpoint_dir, "temp")
if not os.path.exists(temp_checkpoint_dir):
    os.makedirs(temp_checkpoint_dir)
    print(f"Created temporary checkpoint directory at: {temp_checkpoint_dir}")

# Save path for training results
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory at: {results_dir}")

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
def save_temp_checkpoint(epoch, batch_idx, model, optimizer, train_losses, val_accuracies):
    state = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
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
def load_checkpoint(checkpoint_path, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        batch_idx = checkpoint.get('batch_idx', 0)  # Default to 0 if not present
        
        try:
            train_losses = checkpoint['train_losses']
        except KeyError:
            train_losses = []
        
        try:
            val_accuracies = checkpoint['val_accuracies']
        except KeyError:
            val_accuracies = []
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"Successfully loaded checkpoint from epoch {start_epoch}, batch {batch_idx}")
        return start_epoch, batch_idx, train_losses, val_accuracies
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0, [], []

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
# Function to save training graph
def save_training_graph(train_losses, save_path=results_dir):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Save the figure
    graph_path = os.path.join(save_path, "training_loss.png")
    plt.savefig(graph_path)
    plt.close()
    
    print(f"Training graph saved to {graph_path}")

# Function to save validation graph
def save_validation_graph(val_accuracies, save_path=results_dir):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='x', linestyle='--', color='g')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    graph_path = os.path.join(save_path, "validation_accuracy.png")
    plt.savefig(graph_path)
    plt.close()

    print(f"Validation graph saved to {graph_path}")

# Make sure this function is defined before the train_model function where it's called

# Training Loop with automatic checkpoint recovery
def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                start_epoch=0, start_batch=0, train_losses=None, val_accuracies=None, 
                epochs=100, checkpoint_frequency=100):  # Save every 100 batches
    
    if train_losses is None:
        train_losses = []
    if val_accuracies is None:
        val_accuracies = []
    
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
                    save_temp_checkpoint(epoch, i + 1, model, optimizer, train_losses, val_accuracies)
            
            # Calculate epoch loss
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            
            # Evaluate on validation set
            val_accuracy = evaluate(model, val_loader, device)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            
            # Check if this is the best model
            is_best = val_accuracy > best_val_accuracy
            if is_best:
                best_val_accuracy = val_accuracy
            
            # Save checkpoint every epoch
            checkpoint = {
                'epoch': epoch + 1,
                'batch_idx': 0,  # Reset batch index for new epoch
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'timestamp': time.time()
            }
            
            # Save regular checkpoint
            save_checkpoint(checkpoint, is_best=is_best)
            
            # Save training graphs using the existing functions in your code
            save_training_graph(train_losses)
            save_validation_graph(val_accuracies)
            
            # Also save epoch-specific checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth.tar")
                save_checkpoint(checkpoint, checkpoint_dir=checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
        
        # Training completed successfully
        if os.path.exists(os.path.join(temp_checkpoint_dir, 'training_in_progress')):
            os.remove(os.path.join(temp_checkpoint_dir, 'training_in_progress'))
        
        return train_losses, val_accuracies
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        checkpoint = {
            'epoch': epoch,
            'batch_idx': i if 'i' in locals() else 0,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'timestamp': time.time()
        }
        save_checkpoint(checkpoint)
        save_training_graph(train_losses)
        save_validation_graph(val_accuracies)
        print("Checkpoint saved. You can resume training later.")
        
        # Remove the training_in_progress flag since we're exiting gracefully
        if os.path.exists(os.path.join(temp_checkpoint_dir, 'training_in_progress')):
            os.remove(os.path.join(temp_checkpoint_dir, 'training_in_progress'))
        
        return train_losses, val_accuracies
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Save emergency checkpoint
        checkpoint = {
            'epoch': epoch if 'epoch' in locals() else start_epoch,
            'batch_idx': i if 'i' in locals() else 0,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'timestamp': time.time()
        }
        emergency_path = os.path.join(checkpoint_dir, "emergency_checkpoint.pth.tar")
        torch.save(checkpoint, emergency_path)
        print(f"Emergency checkpoint saved to {emergency_path}")
        
        # Don't remove the training_in_progress flag since this was an error
        return train_losses, val_accuracies

# Function to evaluate the model
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_classes = 2  # Normal and Pneumonia
learning_rate = 0.001
epochs = 200

# Initialize the model, loss function, and optimizer
model = densenet121(pretrained=False, num_classes=num_classes)

# Replace the final classifier layer with a new one for our number of classes
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, num_classes)

# Move model to device
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check for interrupted training
interrupted_checkpoint = check_for_interruption()
start_epoch = 0
start_batch = 0
train_losses = []
val_accuracies = []

if interrupted_checkpoint:
    print(f"Detected interrupted training. Resuming from temporary checkpoint: {interrupted_checkpoint}")
    start_epoch, start_batch, train_losses, val_accuracies = load_checkpoint(interrupted_checkpoint, model, optimizer)
    print(f"Resuming from epoch {start_epoch}, batch {start_batch}")
else:
    # Check for regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth.tar")
    if os.path.exists(checkpoint_path):
        print("=> Loading checkpoint...")
        start_epoch, _, train_losses, val_accuracies = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"=> Loaded checkpoint from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

# Check if training was previously interrupted abnormally (power failure, crash)
if os.path.exists(os.path.join(temp_checkpoint_dir, 'training_in_progress')):
    print("WARNING: Previous training session appears to have been interrupted abnormally.")
    # Clean up the flag file
    os.remove(os.path.join(temp_checkpoint_dir, 'training_in_progress'))

print(f"Starting training from epoch {start_epoch + 1}, batch {start_batch + 1}")

# Start training
train_losses, val_accuracies = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    start_epoch=start_epoch,
    start_batch=start_batch,
    train_losses=train_losses,
    val_accuracies=val_accuracies,
    epochs=epochs,
    checkpoint_frequency=50  # Save temporary checkpoint every 50 batches
)

# Final evaluation on test set
test_accuracy = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save final training graphs
save_training_graph(train_losses)
save_validation_graph(val_accuracies)

print("Training complete!")