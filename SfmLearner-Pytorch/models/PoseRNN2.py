import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_, kaiming_uniform_
import torchvision as tv


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class PoseRNN2(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False, encoder='conv'):
        super(PoseRNN2, self).__init__()
        assert(output_exp == False)
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        if encoder == 'conv':
            self.encoder = nn.Sequential(
                conv(3, conv_planes[0], kernel_size=7),
                conv(conv_planes[0], conv_planes[1], kernel_size=5),
                conv(conv_planes[1], conv_planes[2]),
                conv(conv_planes[2], conv_planes[3]),
                conv(conv_planes[3], conv_planes[4])
                )

            self.lstm = nn.LSTM(int(4*13*conv_planes[4]), 1000, 2, batch_first=True)
        elif encoder == 'resnet':
            resnet = tv.models.resnet18(pretrained=False)
            self.encoder = nn.Sequential(*(list(resnet.children())[:-2]))
            self.lstm = nn.LSTM(4*13*512, 1000, 2, batch_first=True)
        self.pose_pred = nn.Linear(1000, 6*self.nb_ref_imgs)
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # xavier_uniform_(m.weight.data)
                kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        output_encoder = []
        for i in range(int(self.nb_ref_imgs/2)):
            out = self.encoder(ref_imgs[i])
            out = out.view(out.size(0), 1, -1)
            output_encoder.append(out)
        out = self.encoder(target_image)
        out = out.view(out.size(0), 1, -1)
        output_encoder.append(out)
        for i in range(int(self.nb_ref_imgs/2), self.nb_ref_imgs):
            out = self.encoder(ref_imgs[i])
            out = out.view(out.size(0), 1, -1)
            output_encoder.append(out)

        in_lstm = torch.cat(output_encoder, 1)
        out_lstm = self.lstm(in_lstm)
        in_pose = out_lstm[0][:, -1, :]
        pose = self.pose_pred(in_pose)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        exp_mask4 = None
        exp_mask3 = None
        exp_mask2 = None
        exp_mask1 = None

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose
