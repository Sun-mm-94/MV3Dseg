#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: basic_block.py
@time: 2021/12/16 20:34
'''
import torch
import torch_scatter
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet34
from utils.lovasz_loss import lovasz_softmax


class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, indice_key):
        super(SparseBasicBlock, self).__init__()
        self.indice_key = indice_key
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        identity = self.layers_in(x)
        output = self.layers(x)
        return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1))


class PEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale, num_classes):
        super(PEBlock, self).__init__()
        self.num_scales = scale
        self.num_classes = num_classes
        self.in_cha = in_channels
        self.out_cha = [64, 64, 64, 64, 64]
        self.cha = self.out_cha[0]
        # encoder layers
        self.point_enc = nn.ModuleList()
        for i in range(self.num_scales+1):
            self.point_enc.append(
                nn.Sequential(
                    nn.Linear(self.in_cha, self.out_cha[i] // 2),
                    nn.LeakyReLU(0.1, True),
                    nn.BatchNorm1d(self.out_cha[i] // 2),
                    nn.Linear(self.out_cha[i] // 2, self.out_cha[i]),
                    nn.LeakyReLU(0.1, True),
                )
            )
            self.in_cha = self.out_cha[i]

        # decoder layers
        self.MLP = nn.ModuleList()
        for i in range(self.num_scales+1):
            self.cha = self.out_cha[-1-i]
            self.MLP.append(
                nn.Sequential(
                    nn.Linear(self.cha,  self.cha),
                    nn.LeakyReLU(0.1, True),
                )
            )

        self.point_dec = nn.ModuleList()
        self.in_cha = self.out_cha[-1]
        for i in range(self.num_scales+1):
            self.cha = self.out_cha[-1-i]
            self.point_dec.append(
                nn.Sequential(
                    nn.Linear(self.in_cha + self.cha, self.cha),
                    nn.BatchNorm1d(self.cha),
                    nn.LeakyReLU(0.1, True),
                )
            )
            self.in_cha = self.cha

        self.classifier = nn.Sequential(
            nn.Linear(self.out_cha[0] * 2, self.out_cha[0]),
            nn.ReLU(True),
            nn.Linear(self.out_cha[0], self.num_classes)
        )

    # def DR_block_forward(self, pt_fea, down_coors, down_num):
    #     x = self.point_enc[down_num](pt_fea)
    #     out, _ = torch_scatter.scatter_max(x, down_coors, dim=0)
    #     return out

    def UR_block_forward(self, cur_feat, last_layer_feat, up_num):
        cur_feat_skip = self.MLP[up_num](cur_feat)
        fusion_feat = torch.concat([cur_feat_skip, last_layer_feat], dim=1)
        fusion_feat = self.point_dec[up_num](fusion_feat)
        return fusion_feat

    def forward(self, data_dict):

        pt_fea = data_dict['pt_fea']

        p1_en = self.point_enc[0](pt_fea)  # 0.1
        out, _ = torch_scatter.scatter_max(p1_en, data_dict['scale_2']['coors_inv'], dim=0)

        p2_en = self.point_enc[1](out)  # 0.2
        out, _ = torch_scatter.scatter_max(p2_en, data_dict['scale_4']['coors_inv'], dim=0)

        p3_en = self.point_enc[2](out)  # 0.4
        out, _ = torch_scatter.scatter_max(p3_en, data_dict['scale_8']['coors_inv'], dim=0)

        p4_en = self.point_enc[3](out)  # 0.8
        out, _ = torch_scatter.scatter_max(p4_en, data_dict['scale_16']['coors_inv'], dim=0)
        p5_en = self.point_enc[4](out)  # 1.6

        p5_feat = self.UR_block_forward(p5_en, p5_en, 0)  # 1.6
        out = p5_feat[data_dict['scale_16']['coors_inv']]

        p4_feat = self.UR_block_forward(p4_en, out, 1)  # 0.8
        out = p4_feat[data_dict['scale_8']['coors_inv']]

        p3_feat = self.UR_block_forward(p3_en, out, 2)  # 0.4
        out = p3_feat[data_dict['scale_4']['coors_inv']]

        p2_feat = self.UR_block_forward(p2_en, out, 3)  # 0.2
        out = p2_feat[data_dict['scale_2']['coors_inv']]

        p1_feat = self.UR_block_forward(p1_en, out, 4)  # 0.1

        data_dict['pfeat_scale1'] = p1_feat
        data_dict['pfeat_scale2'] = p2_feat
        data_dict['pfeat_scale4'] = p3_feat
        data_dict['pfeat_scale8'] = p4_feat
        data_dict['pfeat_scale16'] = p5_feat

        return data_dict


# class ResNetFCN(nn.Module):
#     def __init__(self, backbone="resnet34", pretrained=True, config=None):
#         super(ResNetFCN, self).__init__()
#
#         if backbone == "resnet34":
#             net = resnet34(pretrained)
#         else:
#             raise NotImplementedError("invalid backbone: {}".format(backbone))
#         self.hiden_size = config['model_params']['hiden_size']
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
#         self.conv1.weight.data = net.conv1.weight.data
#         self.bn1 = net.bn1
#         self.relu = net.relu
#         self.maxpool = net.maxpool
#         self.layer1 = net.layer1
#         self.layer2 = net.layer2
#         self.layer3 = net.layer3
#         self.layer4 = net.layer4
#
#         # Decoder
#         self.deconv_layer1 = nn.Sequential(
#             nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=2),
#         )
#         self.deconv_layer2 = nn.Sequential(
#             nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=4),
#         )
#         self.deconv_layer3 = nn.Sequential(
#             nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=4),
#         )
#         self.deconv_layer4 = nn.Sequential(
#             nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=4),
#         )
#
#     def forward(self, data_dict):
#         x = data_dict['img']
#         h, w = x.shape[2], x.shape[3]
#         if h % 16 != 0 or w % 16 != 0:
#             assert False, "invalid input size: {}".format(x.shape)
#
#         # Encoder
#         conv1_out = self.relu(self.bn1(self.conv1(x)))
#         layer1_out = self.layer1(self.maxpool(conv1_out))
#         layer2_out = self.layer2(layer1_out)
#         layer3_out = self.layer3(layer2_out)
#         layer4_out = self.layer4(layer3_out)
#
#         # Deconv
#         layer1_out = self.deconv_layer1(layer1_out)
#         layer2_out = self.deconv_layer2(layer2_out)
#         layer3_out = self.deconv_layer3(layer3_out)
#         layer4_out = self.deconv_layer4(layer4_out)
#
#         data_dict['img_scale2'] = layer1_out
#         data_dict['img_scale4'] = layer2_out
#         data_dict['img_scale8'] = layer3_out
#         data_dict['img_scale16'] = layer4_out
#
#         process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
#         img_indices = data_dict['img_indices']
#
#         temp = {k: [] for k in process_keys}
#
#         for i in range(x.shape[0]):
#             for k in process_keys:
#                 temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])  # find img_indices中对应的图像特征
#
#         for k in process_keys:
#             data_dict[k] = torch.cat(temp[k], 0)
#
#         return data_dict

class Lovasz_loss(nn.Module):
    def __init__(self, ignore=None):
        super(Lovasz_loss, self).__init__()
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, ignore=self.ignore)