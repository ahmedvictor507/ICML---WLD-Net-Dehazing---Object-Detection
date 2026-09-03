import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP with one hidden layer
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply global average pooling
        avg_out = self.avg_pool(x)
        
        # Forward pass through the MLP
        out = self.fc1(avg_out)
        out = self.relu(out)
        out = self.fc2(out)
        
        # Apply sigmoid activation
        out = self.sigmoid(out)
        
        # Scale the input tensor by the attention map
        return x * out


class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.in_channels = in_channels
        
        # 1x1 convolution to generate the attention map
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Sigmoid activation to generate the attention coefficients
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate the attention map using a 1x1 convolution
        attention_map = self.conv1(x)
        attention_map = torch.relu(attention_map)
        attention_map = self.conv2(attention_map)
        
        # Apply the sigmoid activation to the attention map
        attention_map = self.sigmoid(attention_map)
        
        # Scale the input by the attention map
        out = x * attention_map
        
        return out

