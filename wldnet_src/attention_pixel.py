import torch
import torch.nn as nn



class Efficient_Pixel_Attention(nn.Module):
    
    def __init__(self, channel, k_size=3):
        super(Efficient_Pixel_Attention, self).__init__()
        self.conv1=nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.conv2=nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.relu=nn.ReLU(inplace=True)
    
    def forward(self, x):
        
        first=self.conv1(x)
        first_relu=self.relu(first)
        second=self.conv2(first_relu)
        sigmoid_second=torch.sigmoid(second)
        
        return x*sigmoid_second.expand_as(x)

