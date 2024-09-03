import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierModel, self).__init__()
        # Apply adaptive average pooling to convert (512, 14, 14) to (512)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Define multiple fully connected layers
        self.fc1 = nn.Linear(512, 256)  # First FC layer, reducing to 256 features
        self.fc2 = nn.Linear(256, 128)  # Second FC layer, reducing to 128 features
        self.fc3 = nn.Linear(128, num_classes)  # Final FC layer, outputting num_classes for classification
        
        #dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten the output from the adaptive pooling
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Pass through the fully connected layers with ReLU activations and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation, raw scores
        x = F.softmax(x, dim=1)
        
        return x