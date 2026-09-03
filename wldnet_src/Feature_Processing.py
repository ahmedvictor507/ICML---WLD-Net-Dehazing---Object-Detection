import torch
import torch.nn as nn
import FCB_class
import Attention_Block
import attention_channel
import attention_pixel
import attention_spatial
import torch.nn.functional as F

class FEAB(nn.Module):
    def __init__(self, n_inputs=32, n_outputs=32, k_size=3, bias=False):
        super(FEAB, self).__init__()
        self.fcb1=FCB_class.FCB(n_inputs, n_outputs, bias)
        self.conv1x1=nn.Conv2d(n_outputs*2, n_outputs, kernel_size=1, stride=1, padding=0, bias=bias)
        self.instance_norm=nn.InstanceNorm2d(n_outputs)
        self.Attention_Block=Attention_Block.Attn_Block(n_outputs, k_size)
        
    
    def forward(self, x):
        x1=self.fcb1(x)
        attention_output=self.Attention_Block(x1)
        x5=self.conv1x1(attention_output)
        x_norm=self.instance_norm(x)
        x_res=x5+x_norm
        return x_res

class HFP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(HFP, self).__init__()
        self.pad=nn.ReflectionPad2d(1)
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.insnorm=nn.InstanceNorm2d(in_channels)
        self.relu=nn.ReLU()
        

    def forward(self, x):
        x1=self.pad(x)
        x2=self.conv(x1)
        x3=self.insnorm(x2)
        x4=self.relu(x3)
        x4=x4+x
        return x4

class HF_Down(nn.Module):
    def __init__(self, channels, sampling=2):
        super(HF_Down, self).__init__()
        self.pad=nn.ReflectionPad2d(1)
        self.conv=nn.Conv2d(channels, channels, kernel_size=3, stride=sampling, padding=0)
        self.fcb=FCB_class.FCB(channels,channels)
        

    def forward(self, x):
        x=self.pad(x)
        x=self.conv(x)
        x=self.fcb(x)
        return x


class HF_Up(nn.Module):
    def __init__(self, channels):
        super(HF_Up, self).__init__()
        self.fcb_no_act=FCB_class.FCB_No_Act(channels,channels)
        self.fcb2=FCB_class.FCB(channels,channels)
        self.pixel_attention=attention_pixel.Efficient_Pixel_Attention(channels)
        self.up=nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
    def forward(self, x2_h,x2_l):
        x=x2_h+x2_l
        x=self.fcb_no_act(x)
        x=self.pixel_attention(x)
        x=self.up(x)
        x=self.fcb2(x)
        return x

class Fusion_Up(nn.Module):
    def __init__(self, channels):
        super(Fusion_Up, self).__init__()
        self.fcb_no_act=FCB_class.FCB_No_Act(channels,channels)
        self.fcb2=FCB_class.FCB(channels,channels)
        self.pixel_attention=attention_pixel.Efficient_Pixel_Attention(channels)
        self.up=nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
    def forward(self, x_prev, x_hf, x_lf):
        x=x_prev+x_hf
        x=x+x_lf
        x=self.fcb_no_act(x)
        x=self.pixel_attention(x)
        x=self.up(x)
        x=self.fcb2(x)
        return x


class FEAG(nn.Module):
    def __init__(self,n_FEAB_Blocks=3, channels=32):
        super(FEAG, self).__init__()
        self.FEAB_Blocks = nn.ModuleList([FEAB(n_inputs=channels, n_outputs=channels) for i in range(n_FEAB_Blocks)])
        
    def forward(self, x):
        for block in self.FEAB_Blocks:
            x=block(x)
        return x

def normalize(tensor):
    return tensor * 2 - 1

def denormalize(tensor):
    return (tensor + 1) / 2