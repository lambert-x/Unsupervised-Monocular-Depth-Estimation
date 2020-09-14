from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def photometric_reconstruction_loss_ssim(tgt_img, ref_imgs, intrinsics,
                                    depth, explainability_mask, pose,
                                    rotation_mode='euler', padding_mode='zeros', mini_reproj=False, auto_mask=False):
    def one_scale(depth, explainability_mask):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []
        pes = []
        ssim = SSIM()
        ssim = ssim.cuda()
        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                        intrinsics_scaled,
                                                        rotation_mode, padding_mode)
            diff = (0.15 * (tgt_img_scaled - ref_img_warped)+ 0.85 * ssim(tgt_img_scaled,ref_img_warped)) 

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)

            reproj_loss = diff.abs().mean(1, True)
            pes.append(reproj_loss)

            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])
            
        if mini_reproj:
            pes = torch.cat(pes, 1)
            pe, _ = torch.min(pes, dim=1, keepdim=True)
        else:
            pe = torch.cat(pes, 1)

        if auto_mask:
            pes_o = []
            for i, ref_img in enumerate(ref_imgs_scaled):
                diff = 0.15 * (tgt_img_scaled - ref_img)+ 0.85 * ssim(tgt_img_scaled,ref_img)
                reproj_loss = diff.abs().mean(1, True)
                pes_o.append(reproj_loss)
            if mini_reproj:
                pes_o = torch.cat(pes_o, 1)
                pe_o, _ = torch.min(pes_o, dim=1, keepdim=True)
            else:
                pe_o = torch.cat(pes_o, 1)
            mask = (pe < pe_o).float()
            pe = pe * mask

        reconstruction_loss = pe.mean()

        return reconstruction_loss, warped_imgs, diff_maps

    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    total_loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss, warped, diff = one_scale(d, mask)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_loss, warped_results, diff_results


def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics,
                                    depth, explainability_mask, pose,
                                    rotation_mode='euler', padding_mode='zeros', mini_reproj=False, auto_mask=False):
    def one_scale(depth, explainability_mask):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        #reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []
        pes = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                        intrinsics_scaled,
                                                        rotation_mode, padding_mode)
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)
            l1_loss = diff.abs().mean(1, True)
            pes.append(l1_loss)

            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])

        if mini_reproj:
            pes = torch.cat(pes, 1)
            pe, _ = torch.min(pes, dim=1, keepdim=True)
        else:
            pe = torch.cat(pes, 1)

        if auto_mask:
            pes_o = []
            for i, ref_img in enumerate(ref_imgs_scaled):
                diff = tgt_img_scaled - ref_img
                l1_loss = diff.abs().mean(1, True)
                pes_o.append(l1_loss)
            if mini_reproj:
                pes_o = torch.cat(pes_o, 1)
                pe_o, _ = torch.min(pes_o, dim=1, keepdim=True)
            else:
                pe_o = torch.cat(pes_o, 1)
            mask = (pe < pe_o).float()
            pe = pe * mask
        reconstruction_loss = pe.mean() * len(ref_imgs)
        return reconstruction_loss, warped_imgs, diff_maps

    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    total_loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss, warped, diff = one_scale(d, mask)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_loss, warped_results, diff_results


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss


def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss


def smooth_loss_2(pred_map, tgt_img):
    def gradient(img):
        grad_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        return grad_x, grad_y

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        b, _, h, w = scaled_map.size()
        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        mean_map = scaled_map.mean(2, True).mean(3, True)
        norm_map = scaled_map / (mean_map + 1e-7)
        grad_disp_x, grad_disp_y = gradient(norm_map)
        grad_img_x, grad_img_y = gradient(tgt_img_scaled)
        
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        loss += (grad_disp_x.mean() + grad_disp_y.mean())*weight
        weight /= 2
    return loss        
    

@torch.no_grad()
def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]
