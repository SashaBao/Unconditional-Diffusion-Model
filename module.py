import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)
        
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.gsc1 = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),    
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.gsc2 = nn.Sequential(
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),    
        )

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t):
        gsc1_x = self.gsc1(x)
        emb = self.emb_layer(t)[:, :, None, None]
        emb = emb.repeat(1, 1, x.shape[-2], x.shape[-1])
        emb = emb + gsc1_x
        out = self.gsc2(emb)
        return out + self.skip_connection(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.ln = nn.LayerNorm([channels])
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        out = attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        return out
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, size, emb_dim=256):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.rnb = ResNetBlock(in_channels, out_channels, emb_dim)
        self.sa = SelfAttention(out_channels, size // 2)

    def forward(self, x, t):
        x = self.maxpool(x)
        x = self.rnb(x, t)
        x = self.sa(x)
        return x

class Mid(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, size, emb_dim=256):
        super().__init__()
        self.rnb1 = ResNetBlock(in_channels, mid_channels, emb_dim)
        self.sa = SelfAttention(mid_channels, size // 2)
        self.rnb2 = ResNetBlock(mid_channels, out_channels, emb_dim)

    def forward(self, x, t):
        x = self.rnb1(x, t)
        x = self.sa(x)
        return self.rnb2(x, t)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, size, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rnb = ResNetBlock(in_channels, out_channels, emb_dim)
        self.sa = SelfAttention(out_channels, size * 2)

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.rnb(x, t)
        out = self.sa(x)
        return out
        