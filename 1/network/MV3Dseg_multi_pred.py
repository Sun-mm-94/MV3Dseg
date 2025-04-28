import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from network.basic_block_c import Lovasz_loss
from network.baseline import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block_c import PEBlock

class xModalKD(nn.Module):
    def __init__(self,config):
        super(xModalKD, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)
        self.scale_list = [1] + self.scale_list

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        self.fcs3 = nn.ModuleList()
        for i in range(self.num_scales):
            self.fcs1.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs3.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))

        self.point_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.point_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, self.hiden_size),
                    nn.ReLU(True),
                    nn.Linear(self.hiden_size, self.num_classes),
                )
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size, self.hiden_size),
            nn.ReLU(True),
            nn.Linear(self.hiden_size, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights, ignore_index=config['dataset_params']['ignore_label'])
        self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

    @staticmethod
    def calculate_entropy(prob_dist):
        prob_dist_softmax = F.softmax(prob_dist, dim=1)
        return -(prob_dist_softmax * torch.log(prob_dist_softmax + 1e-10)).sum(1)

    @staticmethod
    def normalize_weights(entropy_point, entropy_voxel):
        exp_neg_entropy_point = torch.exp(-entropy_point)
        exp_neg_entropy_voxel = torch.exp(-entropy_voxel)
        sum_exp = exp_neg_entropy_point + exp_neg_entropy_voxel
        weight_point = exp_neg_entropy_point / sum_exp
        weight_voxel = exp_neg_entropy_voxel / sum_exp
        return weight_point, weight_voxel

    @staticmethod
    def weighted_fusion(prob_point, prob_voxel, weight_point, weight_voxel):
        return weight_point[..., np.newaxis] * prob_point + weight_voxel[..., np.newaxis] * prob_voxel

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    def fusion_to_single_KD(self, data_dict, idx):


        return loss

    def forward(self, data_dict):
        loss = 0

        for idx in range(self.num_scales):
            singlescale_loss = self.fusion_to_single_KD(data_dict, idx)
            loss += singlescale_loss
        full_pts_feat = data_dict['pfeat_scale1'][data_dict['scale_1']['coors_inv']]
        point_seg_logits = self.classifier(full_pts_feat)
        loss += self.seg_loss(point_seg_logits, data_dict['labels'])
        data_dict['loss'] += loss

        return data_dict


class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)

        self.model_3d = SPVCNN(config)
        if not self.baseline_only:
            self.point_enc = PEBlock(
                in_channels=self.hiden_size,
                out_channels=self.hiden_size,
                scale=self.num_scales,
                num_classes=self.num_classes
            )
            self.fusion = xModalKD(config)
        else:
            print('Start vanilla training!')

    def forward(self, data_dict):
        # 3D network
        data_dict = self.model_3d(data_dict)

        # training with 2D network
        if self.training and not self.baseline_only:
            data_dict = self.point_enc(data_dict)
            data_dict = self.fusion(data_dict)

        return data_dict
