import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
                
        #left
        self.conv1 = spectral_norm(nn.Conv2d(in_channels , in_channels , kernel_size=3 , padding=1 , padding_mode='relf'))
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv2 = spectral_norm(nn.Conv2d(in_channels , in_channels , kernel_size=3 , padding=1 , padding_mode='relf'))
        self.in2 = nn.InstanceNorm2d(in_channels, affine=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.in1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        
        out = out + residual
        
        return out

class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlockDown, self).__init__()

        # Right Side
        self.conv_r1 = spectral_norm(nn.Conv2d(in_channels , out_channels , kernel_size , padding=1 , padding_mode='relf'))
        self.conv_r2 = spectral_norm(nn.Conv2d(out_channels , out_channels , kernel_size , padding=1 , padding_mode='relf'))

        # Left Side
        self.conv_l = spectral_norm(nn.Conv2d(in_channels , out_channels , kernel_size=1))

    def forward(self, x):
        residual = x

        # Right Side
        out = F.relu(x)
        out = self.conv_r1(out)
        out = F.relu(out)
        out = self.conv_r2(out)
        out = F.avg_pool2d(out, 2)

        # Left Side
        residual = self.conv_l(residual)
        residual = F.avg_pool2d(residual, 2)

        # Merge
        out = residual + out
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()
    
        self.q = spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        self.k = spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        self.v = spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        bs, c, h, w = x.shape
        Q = self.q(x) #BxC'xHxW, C'=C//8
        K = self.k(x) #BxC'xHxW
        V = self.v(x) #BxCxHxW
        
        Q_ = Q.view(bs, -1, h*w) #BxC'xN
        K_ = K.view(bs, -1, h*w) #BxC'xN
        V_ = V.view(bs, -1, h*w) #BxCxN
        
        attention = torch.bmm(Q_.permute(0,2,1), K_).softmax(-1)    
        out = torch.bmm(V_, attention) #BxCxN
        out = out.view(bs,c,h,w)
        
        out = self.gamma*out + x
        return out

class AdaIn(nn.Module):
    def __init__(self):
        super(AdaIn, self).__init__()
        self.eps = 1e-5

    def forward(self, x, mean_style, std_style):
        bs, c, h, w = x.shape

        feature = x.view(bs, c, -1)

        std_feat = feature.std(dim=2 , keepdim=True) + self.eps
        mean_feat = feature.mean(dim=2 , keepdim=True)
        
        adain = std_style.unsqueeze(2) * (feature - mean_feat) / std_feat + mean_style.unsqueeze(2)
        adain = adain.view(bs, c, h, w)
        
        return adain

class AdaptiveResBlock(nn.Module):
    def __init__(self, channels , e_dim=512):
        super(AdaptiveResBlock, self).__init__()
        
        self.conv1 = spectral_norm(nn.Conv2d(channels , channels , kernel_size=3 , padding=1 , padding_mode='relf'))
        self.in1 = AdaIn()
        self.conv2 = spectral_norm(nn.Conv2d(channels , channels , kernel_size=3 , padding=1 , padding_mode='relf'))
        self.in2 = AdaIn()
        self.linear = nn.Linear(e_dim , channels*4 ,bias=False)

    def forward(self, x, e):
        mean1, std1 , mean2, std2 = self.linear(e).chunk(4,dim=1)
        
        residual = x

        out = self.conv1(x)
        out = self.in1(out, mean1, std1)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out, mean2, std2)

        out = out + residual
        return out

class AdaptiveResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsample=2 , e_dim=512):
        super(AdaptiveResBlockUp, self).__init__()

        self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')

        # Right Side
        self.conv_r1 = spectral_norm(nn.Conv2d(in_channels , out_channels , kernel_size=3 , padding=1 , padding_mode='relf'))
        self.in1 = AdaIn()
        self.conv_r2 = spectral_norm(nn.Conv2d(out_channels , out_channels , kernel_size=3 , padding=1 , padding_mode='relf'))
        self.in2 = AdaIn()

        # Left Side
        self.conv_l = spectral_norm(nn.Conv2d(in_channels , out_channels , kernel_size=1))
        
        self.linear1 = nn.Linear(e_dim , in_channels*2 ,bias=False)
        self.linear2 = nn.Linear(e_dim , out_channels*2 ,bias=False)

    def forward(self, x, e):
        mean1, std1 = self.linear1(e).chunk(2,dim=1)
        mean2, std2 = self.linear2(e).chunk(2,dim=1)
        
        residual = x

        # Right Side
        out = self.in1(x, mean1, std1)
        out = F.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.in2(out, mean2, std2)
        out = F.relu(out)
        out = self.conv_r2(out)

        # Left Side
        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        # Merge
        out = residual + out
        return out