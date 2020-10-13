from collections import OrderedDict
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import vgg

class VGG_Activations(nn.Module):
    def __init__(self, vgg_network, feature_idx):
        super(VGG_Activations, self).__init__()
        self.model = vgg_network.features
        self.output = []
        for idx in feature_idx:
            self.model[idx].register_forward_hook(self.hook)
    
    def hook(self , module, input, output):
        self.output.append(output)
        
    def forward(self , x):
        self.output = []
        mean = torch.tensor([0.485, 0.456, 0.406] , device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225] , device=x.device).view(1,3,1,1)
        x = x.sub(mean).div(std)
        _ = self.model(x)
        return self.output

def vggface(weight_path):
    network = vgg.VGG(vgg.make_layers(vgg.cfgs['D'], batch_norm=False), num_classes=2622)
    default = torch.load(weight_path)
    state_dict = OrderedDict({
        'features.0.weight': default['conv1_1.weight'],
        'features.0.bias': default['conv1_1.bias'],
        'features.2.weight': default['conv1_2.weight'],
        'features.2.bias': default['conv1_2.bias'],
        'features.5.weight': default['conv2_1.weight'],
        'features.5.bias': default['conv2_1.bias'],
        'features.7.weight': default['conv2_2.weight'],
        'features.7.bias': default['conv2_2.bias'],
        'features.10.weight': default['conv3_1.weight'],
        'features.10.bias': default['conv3_1.bias'],
        'features.12.weight': default['conv3_2.weight'],
        'features.12.bias': default['conv3_2.bias'],
        'features.14.weight': default['conv3_3.weight'],
        'features.14.bias': default['conv3_3.bias'],
        'features.17.weight': default['conv4_1.weight'],
        'features.17.bias': default['conv4_1.bias'],
        'features.19.weight': default['conv4_2.weight'],
        'features.19.bias': default['conv4_2.bias'],
        'features.21.weight': default['conv4_3.weight'],
        'features.21.bias': default['conv4_3.bias'],
        'features.24.weight': default['conv5_1.weight'],
        'features.24.bias': default['conv5_1.bias'],
        'features.26.weight': default['conv5_2.weight'],
        'features.26.bias': default['conv5_2.bias'],
        'features.28.weight': default['conv5_3.weight'],
        'features.28.bias': default['conv5_3.bias'],
        'classifier.0.weight': default['fc6.weight'],
        'classifier.0.bias': default['fc6.bias'],
        'classifier.3.weight': default['fc7.weight'],
        'classifier.3.bias': default['fc7.bias'],
        'classifier.6.weight': default['fc8.weight'],
        'classifier.6.bias': default['fc8.bias']})
    
    network.load_state_dict(state_dict)
    return network