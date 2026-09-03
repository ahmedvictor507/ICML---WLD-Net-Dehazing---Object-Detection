import torch
import torch.nn as nn



class Efficient_Spatial_Attention(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(Efficient_Spatial_Attention, self).__init__()
        
        self.avg_pool=nn.AdaptiveAvgPool2d(1) # Create an adaptive average pooling layer across both dimensions
        
        self.conv1d=nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=int(kernel_size/2), bias=False) # Create a 1D convolutional layer
    
    def forward(self, x):
        avg_pool=self.avg_pool(x)
        b,c,_,_=avg_pool.size()
        avg_pool=avg_pool.view(b,c,-1)
        conv1d=self.conv1d(avg_pool)
        sigmoid_conv1d=torch.sigmoid(conv1d)
        sigmoid_conv1d=sigmoid_conv1d.view(b,c,1,1)
        
        return x*sigmoid_conv1d.expand_as(x)
