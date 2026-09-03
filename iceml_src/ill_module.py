import torch
import torch.nn as nn
import attentions

class FA_Block(nn.Module):
    def __init__(self, channels):
        super(FA_Block, self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.channel_attention=attentions.ChannelAttention(in_channels=channels)
        self.pixel_attention=attentions.PixelAttention(in_channels=channels)
    
    def forward(self,x):
        x1=self.conv1(x)
        x1=torch.relu(x1)
        x1=x1+x
        x1=self.conv2(x1)
        x1=self.channel_attention(x1)
        x1=self.pixel_attention(x1)
        x1=x1+x
        
        return x1

class single_level(nn.Module):
    def __init__(self, downsampling_ratio, channels, no_FA_Block):
        super(single_level, self).__init__()
        
        
        self.down=nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=downsampling_ratio, padding=1)
        self.init_conv=nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        
        self.fa_blocks = nn.ModuleList([FA_Block(channels) for _ in range(no_FA_Block)])

        
        if downsampling_ratio==2:
            self.up=nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=downsampling_ratio, padding=1, output_padding=1)
        if downsampling_ratio==4:
            self.up=nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=downsampling_ratio, padding=0, output_padding=1)
        if downsampling_ratio==8:
            self.up=nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=downsampling_ratio, padding=0, output_padding=5)
        
        self.final_conv=nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        
    
    
    def forward(self, x):
        x1 = self.down(x)
        x1=torch.relu(x1)
        x1 = self.init_conv(x1)
        
        # Pass through all FA_Block instances
        for block in self.fa_blocks:
            x1 = block(x1)
        
        x1 = self.up(x1)
        x1=torch.relu(x1)
        x1 = self.final_conv(x1)
        x1=torch.relu(x1)
        return x1

class ill_gen(nn.Module):
    def __init__(self, channels, no_FA_Block):
        super(ill_gen, self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.level_1=single_level(downsampling_ratio=2, channels=channels, no_FA_Block=no_FA_Block)
        self.level_2=single_level(downsampling_ratio=4, channels=channels, no_FA_Block=no_FA_Block)
        self.level_3=single_level(downsampling_ratio=8, channels=channels, no_FA_Block=no_FA_Block)
        self.conv2=nn.Conv2d(in_channels=channels*3, out_channels=channels*3, kernel_size=3, stride=1, padding=1)
        self.conv3=nn.Conv2d(in_channels=channels*3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.final_conv=nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
       #self.tanh=nn.Tanh()
    
    def forward(self, x):
        x1=self.conv1(x)
        x_l1=self.level_1(x1)
        x_l2=self.level_2(x1)
        x_l3=self.level_3(x1)
        x_conc=torch.cat((x_l1, x_l2, x_l3), dim=1)
        x_conc=self.conv2(x_conc)
        x_conc=torch.relu(x_conc)
        x_conc=self.conv3(x_conc)
        x_conc=torch.relu(x_conc)
        x_conc=torch.cat((x_conc,x),dim=1)
        x_final=self.final_conv(x_conc)
        x_final=torch.relu(x_final)
        return x_final

