import torch
import torch.nn as nn



#Create an Efficent Channel Attention Class
class Efficient_Channel_Attention(nn.Module):
    
    def __init__(self, channel, k_size=3): # Initialize the class
        super(Efficient_Channel_Attention, self).__init__() # inherit from nn.Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Create an adaptive average pooling layer
        self.conv=nn.Conv1d(1,1,kernel_size=k_size, padding=((k_size-1)//2), bias=False) # Create a convolutional layer
        self.sigmoid=nn.Sigmoid() # Create a sigmoid activation layer
    
    def forward(self, x): # Forward pass of the function
        y=self.avg_pool(x) # Apply average pooling layer
        y=y.squeeze(-1).permute(0,2,1) # Change the dimensions of the tensor
        y=self.conv(y)
        y=self.sigmoid(y) # Apply sigmoid activation
        y=y.permute(0,2,1).unsqueeze(-1) # Change the dimensions of the tensor
        
        return x*y.expand_as(x) # Multiply elementwise




