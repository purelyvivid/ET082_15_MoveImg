from vgg import vggface, VGG_Activations
from torchvision.models import vgg19

import torch
from torch import nn
from torch.nn import functional as F

class LossD(nn.Module):
    def __init__(self):
        super(LossD, self).__init__()

    def forward(self, r_x, r_x_hat):
        return (F.relu(1 + r_x_hat) + F.relu(1 - r_x)).mean()


class LossEG(nn.Module):
    def __init__(self, vggface, vgg19):
        super(LossEG, self).__init__()
        self.VGG_FACE_AC = VGG_Activations(vggface, [1, 6, 11, 18, 25])
        self.VGG19_AC = VGG_Activations(vgg19, [1, 6, 11, 20, 29])
        
    def loss_cnt(self, x_hat, x):
        
        vgg19_x_hat = self.VGG19_AC(x_hat)
        vgg19_x = self.VGG19_AC(x)

        vgg19_loss = 0
        for i in range(0, len(vgg19_x)):
            vgg19_loss += F.l1_loss(vgg19_x_hat[i], vgg19_x[i])

        # VGG Face Loss
        vgg_face_x_hat = self.VGG_FACE_AC(x_hat)
        vgg_face_x = self.VGG_FACE_AC(x)

        vgg_face_loss = 0
        for i in range(0, len(vgg_face_x)):
            vgg_face_loss += F.l1_loss(vgg_face_x_hat[i], vgg_face_x[i])

        return 1e-2*vgg19_loss + 2e-3*vgg_face_loss

    def loss_cnt_(self,x_hat,x):
        return F.l1_loss(x_hat , x)
    
    def loss_adv(self, r_x_hat):
        return -r_x_hat.mean()
        
    def forward(self, x_hat, x, r_x_hat):
        cnt = self.loss_cnt(x_hat, x)
        adv = self.loss_adv(r_x_hat)
        
        return cnt+adv