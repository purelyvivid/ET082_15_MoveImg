import torch
import torch.nn as nn
from module import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv1 = ResBlockDown(3, 64)
        self.in1_e = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = ResBlockDown(64, 128)
        self.in2_e = nn.InstanceNorm2d(128, affine=True)

        self.conv3 = ResBlockDown(128, 256)
        self.in3_e = nn.InstanceNorm2d(256, affine=True)

        self.att1 = SelfAttention(256)

        self.conv4 = ResBlockDown(256, 512)
        self.in4_e = nn.InstanceNorm2d(512, affine=True)

        self.conv5 = ResBlockDown(512, 512)
        self.in5_e = nn.InstanceNorm2d(512, affine=True)
        
        self.conv6 = ResBlockDown(512, 512) # add
        self.in6_e = nn.InstanceNorm2d(512, affine=True) # add
        
        self.res1 = AdaptiveResBlock(512)
        self.res2 = AdaptiveResBlock(512)
        self.res3 = AdaptiveResBlock(512)
        self.res4 = AdaptiveResBlock(512)
        self.res5 = AdaptiveResBlock(512)
        
        self.deconv6 = AdaptiveResBlockUp(512, 512, upsample=2) #add
        self.in6_d = nn.InstanceNorm2d(512, affine=True) #add
        
        self.deconv5 = AdaptiveResBlockUp(512, 512, upsample=2)
        self.in5_d = nn.InstanceNorm2d(512, affine=True)

        self.deconv4 = AdaptiveResBlockUp(512, 256, upsample=2)
        self.in4_d = nn.InstanceNorm2d(256, affine=True)

        self.deconv3 = AdaptiveResBlockUp(256, 128, upsample=2)
        self.in3_d = nn.InstanceNorm2d(128, affine=True)

        self.att2 = SelfAttention(128)

        self.deconv2 = AdaptiveResBlockUp(128, 64, upsample=2)
        self.in2_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv1 = AdaptiveResBlockUp(64, 3, upsample=2)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)
        
    
    def forward(self , source_landmark , emb):
        
        out = source_landmark # [B, 64, 256, 256]
        
        out = self.in1_e(self.conv1(out))  # [B, 64, 128, 128]
        out = self.in2_e(self.conv2(out))  # [B, 128, 64, 64]
        out = self.in3_e(self.conv3(out))  # [B, 256, 32, 32]
        out = self.att1(out)
        out = self.in4_e(self.conv4(out))  # [B, 512, 16, 16]
        out = self.in5_e(self.conv5(out))  # [B, 512, 8, 8]
        out = self.in6_e(self.conv6(out))  # [B, 512, 4, 4] 
        
        out = self.res1(out, emb) # emb: [B, 512] 
        out = self.res2(out, emb)
        out = self.res3(out, emb)
        out = self.res4(out, emb)
        out = self.res5(out, emb)
        
        out = self.in6_d(self.deconv6(out, emb))  # [B, 512, 8, 8]
        out = self.in5_d(self.deconv5(out, emb))  # [B, 512, 16, 16]
        out = self.in4_d(self.deconv4(out, emb))  # [B, 256, 32, 32]
        out = self.in3_d(self.deconv3(out, emb))  # [B, 128, 64, 64]
        out = self.att2(out)
        out = self.in2_d(self.deconv2(out, emb))  # [B, 64, 128, 128]
        out = self.in1_d(self.deconv1(out, emb))  # [B, 3, 256, 256]

        out = torch.sigmoid(out)
        
        return out    

class Embedder(nn.Module):
    def __init__(self, emb_size=512):
        super(Embedder, self).__init__()
        self.emb_size = emb_size
        self.conv1 = ResBlockDown(6, 64)
        self.conv2 = ResBlockDown(64, 128)
        self.conv3 = ResBlockDown(128, 256)
        self.att = SelfAttention(256)
        self.conv4 = ResBlockDown(256, 512)
        self.conv5 = ResBlockDown(512, 512)
        self.conv6 = ResBlockDown(512, 512) # add

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        bs,K,c,h,w = x.shape # [B, K, 6, 256, 256]
        out = x.view(-1,c,h,w)  # [BxK, 6, 256, 256]

        # Encode
        out = self.conv1(out)  # [BxK, 64, 128, 128]
        out = self.conv2(out)  # [BxK, 128, 64, 64]
        out = self.conv3(out)  # [BxK, 256, 32, 32]
        out = self.att(out)
        out = self.conv4(out)  # [BxK, 512, 16, 16]
        out = self.conv5(out)  # [BxK, 512, 8, 8]
        out = self.conv6(out)  # [BxK, 512, 4, 4]

        # Vectorize
        out = self.pooling(out).relu().view(bs,K,self.emb_size)# [B, K, 512]
        #out = F.relu(self.pooling(out).view(-1, config.E_VECTOR_LENGTH))

        return out.mean(1)# [B, 512]

class Discriminator(nn.Module):
    def __init__(self, num_videos):
        super(Discriminator, self).__init__()
        self.conv1 = ResBlockDown(6, 64)
        self.conv2 = ResBlockDown(64, 128)
        self.conv3 = ResBlockDown(128, 256)
        self.att = SelfAttention(256)
        self.conv4 = ResBlockDown(256, 512)
        self.conv5 = ResBlockDown(512, 512)
        self.conv6 = ResBlockDown(512, 512) # add
        self.res = ResBlock(512)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        
        self.W = nn.Parameter(torch.rand(512, num_videos).normal_(0.0, 0.02))
        self.w_0 = nn.Parameter(torch.rand(512, 1).normal_(0.0, 0.02))
        self.b = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))
        
        #self.linear = nn.Linear(512 , num_videos)
        
    def set_W(self, i, emb): # emb: [B,512]
        for j, i_ in enumerate(i):
            self.W.data[:,i_] = emb[j]
            
                
    def forward(self, x, i): 
        #print(x.shape, i.shape) # torch.Size([BxK, 6, 256, 256]) torch.Size([BxK])
        out = x                # [BxK,  6, 256, 256]
        out = self.conv1(out)  # [BxK, 64, 128, 128]
        out = self.conv2(out)  # [BxK, 128, 64, 64]
        out = self.conv3(out)  # [BxK, 256, 32, 32]
        out = self.att(out)
        out = self.conv4(out)  # [BxK, 512, 16, 16]
        out = self.conv5(out)  # [BxK, 512, 8, 8]
        out = self.conv6(out)  # [BxK, 512, 4, 4]
        out = self.res(out)    # [BxK, 512, 4, 4]
        out = self.pooling(out).view(-1,512,1)# [BxK, 512, 1]
        
        # Calculate Realism Score
        _out = out.transpose(1, 2) # [BxK, 1, 512]
        _W_i = (self.W[:, i].unsqueeze(-1)).transpose(0, 1) # [BxK, 512, 1]
        out = torch.bmm(_out, _W_i + self.w_0) + self.b # [BxK, 1]
        
        """
        out = self.pooling(out).view(-1,512)# [BxK, 512]
        out = self.linear(out) # [BxK, 512]
        score = out.gather(1 , torch.LongTensor(i).to(out.device).view(out.size(0),1))
        
        return score
        """
        return out
    
                
    def forward_few_shot(self, x, new_emb):  
        """
        x: [BxK,  6, 256, 256]
        new_emb: [B,512]
        """
        BxK = x.shape[0]
        out = x                # [BxK,  6, 256, 256]
        out = self.conv1(out)  # [BxK, 64, 128, 128]
        out = self.conv2(out)  # [BxK, 128, 64, 64]
        out = self.conv3(out)  # [BxK, 256, 32, 32]
        out = self.att(out)
        out = self.conv4(out)  # [BxK, 512, 16, 16]
        out = self.conv5(out)  # [BxK, 512, 8, 8]
        out = self.conv6(out)  # [BxK, 512, 4, 4]
        out = self.res(out)    # [BxK, 512, 4, 4]
        out = self.pooling(out).view(-1,512,1)# [BxK, 512, 1]
        
        # Calculate Realism Score
        _out = out.transpose(1, 2) # [BxK, 1, 512]
        _new_emb = new_emb.view(-1,512,1).repeat(BxK, 1, 1) # [BxK, 512, 1]
        #print(_new_emb.shape)
        out = torch.bmm(_out, _new_emb + self.w_0) + self.b  # [BxK, 1]
        
        return out