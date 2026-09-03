import torch
import torch.nn as nn
import ill_module

class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()
        
        # Depthwise Convolution
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # Layer Normalization
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Pointwise Convolution (1x1)
        self.pw_conv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        
        # GELU Activation
        self.act = nn.GELU()
        
        # Pointwise Convolution (1x1) - Back to original dimensions
        self.pw_conv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        
        self.simam1=simam_module()
        self.simam2=simam_module()
        self.simam3=simam_module()
        self.simam4=simam_module()
        
        
    def forward(self, x):
        shortcut = x
        shortcut = self.simam4(shortcut)
        x = self.dw_conv(x)  # Depthwise convolution
        x = self.simam1(x)
        x = x.permute(0, 2, 3, 1)  # Permute to (batch_size, height, width, channels) for LayerNorm
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # Permute back to (batch_size, channels, height, width)
        
        x = self.pw_conv1(x)  # First pointwise convolution
        x = self.simam2(x)
        x = self.act(x)        # GELU activation
        x = self.pw_conv2(x)  # Second pointwise convolution
        x = self.simam2(x)
        
        return x + shortcut  # Residual connection

class dehazing_module(nn.Module):
    def __init__(self, in_channels=4):
        super(dehazing_module, self).__init__()
        
        self.down1=nn.PixelUnshuffle(2)
        self.mid_conv1=nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=3, stride=1, padding=1)
        self.down2=nn.PixelUnshuffle(2)
        
        self.convv=nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.acm1=ConvNeXtBlock(64)
        self.acm2=ConvNeXtBlock(64)
        self.acm3=ConvNeXtBlock(64)
        self.acm4=ConvNeXtBlock(64)
        
        self.fa1=ill_module.FA_Block(64)
        self.fa2=ill_module.FA_Block(64)
        self.fa3=ill_module.FA_Block(64)
        self.fa4=ill_module.FA_Block(64)
        
        self.con=nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=1)
        
        self.up1=nn.PixelShuffle(2)
        self.mid_conv2=nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=3, stride=1, padding=1)
        self.up2=nn.PixelShuffle(2)
        
        self.final_conv=nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh=nn.Tanh()
    
    def forward(self, input):
        x=input
        
        x=self.down1(x)
        x=self.mid_conv1(x)
        x=self.down2(x)
        x=self.convv(x)
        x=self.acm1(x)
        x=self.acm2(x)
        x=self.acm3(x)
        
        x=self.fa1(x)
        x=self.fa2(x)
        x=self.fa3(x)
        
        x=self.con(x)
        x=self.up1(x)
        x=self.mid_conv2(x)
        x=self.up2(x)
        
        x=self.final_conv(x)
        x=torch.relu(x)
        return x
    

