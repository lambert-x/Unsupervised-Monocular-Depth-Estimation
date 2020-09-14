import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_, kaiming_uniform_
import mono2net



class DispNet_resnet(nn.Module):

    def __init__(self,encoder,decoder):
        super(DispNet_resnet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_uniform_(m.weight.data)
                # xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        features = self.encoder(x)
        disps = self.decoder(features)
        return disps
        # if self.training:
        #     return disp1, disp2, disp3, disp4
        # else:
        #     return disp1
