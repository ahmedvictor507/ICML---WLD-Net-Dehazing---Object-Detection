import torch
import torch.nn as nn

class FCB(nn.Module):
    # Initialize the class
    def __init__(self, n_inputs, n_outputs, bias=False):
        super(FCB, self).__init__() # Inherit from nn.Module
        self.pad1 = nn.ReflectionPad2d((0,1,0,1)) # Add reflection padding for conv_layer_1
        self.pad2 = nn.ReflectionPad2d((1,0,1,0)) # Add reflection padding for conv_layer_2
        self.conv_layer_1=nn.Conv2d(n_inputs, n_outputs, kernel_size=(1,3), stride=1, padding=0, bias=bias) # 1x3 convolution
        self.conv_layer_2=nn.Conv2d(n_outputs, n_outputs, kernel_size=(3,1), stride=1, padding=0, bias=bias) # 3x1 convolution
        self.relu=nn.ReLU() # ReLU activation
    
    #Perform the forward computation
    def forward(self, x):
        x = self.pad1(x) # Apply reflection padding
        x1=self.conv_layer_1(x)
        x = self.pad2(x1) # Apply reflection padding
        x2=self.conv_layer_2(x)
        x3=self.relu(x2)
        return x3

class FCB_No_Act(nn.Module):
    # Initialize the class
    def __init__(self, n_inputs, n_outputs, bias=False):
        super(FCB_No_Act, self).__init__() # Inherit from nn.Module
        self.pad1 = nn.ReflectionPad2d((0,1,0,1)) # Add reflection padding for conv_layer_1
        self.pad2 = nn.ReflectionPad2d((1,0,1,0)) # Add reflection padding for conv_layer_2
        self.conv_layer_1=nn.Conv2d(n_inputs, n_outputs, kernel_size=(1,3), stride=1, padding=0, bias=bias) # 1x3 convolution
        self.conv_layer_2=nn.Conv2d(n_outputs, n_outputs, kernel_size=(3,1), stride=1, padding=0, bias=bias) # 3x1 convolution
    
    #Perform the forward computation
    def forward(self, x):
        x = self.pad1(x) # Apply reflection padding
        x1=self.conv_layer_1(x)
        x = self.pad2(x1) # Apply reflection padding
        x2=self.conv_layer_2(x)
        return x2

