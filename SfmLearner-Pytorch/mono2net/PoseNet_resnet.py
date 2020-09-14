import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_, kaiming_uniform_
import mono2net



class PoseNet_resnet(nn.Module):

    def __init__(self,encoder,decoder,mode):
        super(PoseNet_resnet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mode = mode

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #             kaiming_uniform_(m.weight.data)
    #             # xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 zeros_(m.bias)

    def forward(self, target_image, ref_imgs):
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)
        if self.mode == 'seperate':
            features = self.encoder(input)
        elif self.mode == 'share':
            features = []
            for i in range(input.size(1)):
                features.append(self.encoder(input[:,i]))

        pose = self.decoder([features])  
        return None,pose
        # if self.training:
        #     return disp1, disp2, disp3, disp4
        # else:
        #     return disp1
