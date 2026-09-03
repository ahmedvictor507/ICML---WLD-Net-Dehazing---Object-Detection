import attention_channel
import attention_pixel
import attention_spatial
import torch
import torch.nn as nn

class Attn_Block(nn.Module):
    def __init__(self, channel, k_size=3):
        super(Attn_Block, self).__init__()
        self.channel=channel
        self.k_size=k_size
        self.channel_attention=attention_channel.Efficient_Channel_Attention(channel, k_size)
        self.pixel_attention=attention_pixel.Efficient_Pixel_Attention(channel, k_size)
        self.spatial_attention=attention_spatial.Efficient_Spatial_Attention(channel, k_size)
    
    def forward(self, x):
        channel_attention_output=self.channel_attention(x)
        pixel_attention_output=self.pixel_attention(channel_attention_output) + x
        spatial_attention_output=self.spatial_attention(channel_attention_output) + x
        output=torch.cat((pixel_attention_output, spatial_attention_output), dim=1)
        
        return output

