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


class PoseSep(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False, encoder='conv'):
        super(PoseSep, self).__init__()
        assert(output_exp == False)
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        if encoder == 'conv':
            self.encoder = nn.Sequential(
                conv(6, conv_planes[0], kernel_size=7),
                conv(conv_planes[0], conv_planes[1], kernel_size=5),
                conv(conv_planes[1], conv_planes[2]),
                conv(conv_planes[2], conv_planes[3]),
                conv(conv_planes[3], conv_planes[4]),
                conv(conv_planes[4], conv_planes[5]),
                conv(conv_planes[5], conv_planes[6])
                )

            self.pose_pred = nn.Conv2d(conv_planes[6], 6, kernel_size=1, padding=0)
        elif encoder == 'resnet':
            resnet = tv.models.resnet18(pretrained=False)
            self.encoder = nn.Sequential(*(list(resnet.children())[:-2]))
            self.encoder[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.pose_pred = nn.Conv2d(512, 6, kernel_size=1, padding=0)
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # xavier_uniform_(m.weight.data)
                kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        poses = []
        for i in range(self.nb_ref_imgs):
            in_encoder = torch.cat([target_image, ref_imgs[i]], 1)
            out_encoder = self.encoder(in_encoder)
            out_pose = self.pose_pred(out_encoder)
            out_pose = out_pose.mean(3).mean(2)
            out_pose = 0.01 * out_pose.view(out_pose.size(0), 1, 6)
            poses.append(out_pose)

        pose = torch.cat(poses, 1)

        exp_mask4 = None
        exp_mask3 = None
        exp_mask2 = None
        exp_mask1 = None

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose
