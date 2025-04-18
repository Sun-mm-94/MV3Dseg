import torch
import torch_scatter
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv


class voxelization(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list):
        super(voxelization, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = [1] + scale_list
        self.coors_range_xyz = coors_range_xyz

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def return_xyz(self, grid_ind, scale):
        coors_range_xyz = torch.Tensor(self.coors_range_xyz)
        cur_grid_size = torch.Tensor(self.spatial_shape / scale)
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        intervals = (crop_range / cur_grid_size).to(grid_ind.device)
        pc_center = grid_ind * intervals + coors_range_xyz[:, 0].to(grid_ind.device)
        return pc_center

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]
        batch_idx = data_dict['batch_idx']
        labels = data_dict['labels']
        for idx, scale in enumerate(self.scale_list):
            xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], np.ceil(self.spatial_shape[0] / scale))
            yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], np.ceil(self.spatial_shape[1] / scale))
            zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], np.ceil(self.spatial_shape[2] / scale))

            bxyz_indx = torch.stack([batch_idx, xidx, yidx, zidx], dim=-1).long()
            labels = self.voxelize_labels(labels, bxyz_indx)

            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            pc = self.return_xyz(unq[:, 1:4], scale)
            # pc_indx = torch.cat([unq[:, 0:1], pc], dim=1)
            batch_idx = unq[:, 0:1].squeeze(1)
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)

            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32),
                'voxel_label': labels,
                # 'points': pc_indx,
            }
        return data_dict


class voxel_3d_generator(nn.Module):
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape):
        super(voxel_3d_generator, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels + 6, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )

    def prepare_input(self, point, grid_ind, inv_idx):
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]
        nor_pc = point[:, :3] - pc_mean

        coors_range_xyz = torch.Tensor(self.coors_range_xyz)
        cur_grid_size = torch.Tensor(self.spatial_shape)
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        intervals = (crop_range / cur_grid_size).to(point.device)
        voxel_centers = grid_ind * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point = point[:, :3] - voxel_centers

        pc_feature = torch.cat((point, nor_pc, center_to_point), dim=1) # 原始点，原始点-相同点的平均值，原始点-每个原始点的中心位置
        return pc_feature

    def forward(self, data_dict):
        pt_fea = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv']
        )
        pt_fea = self.PPmodel(pt_fea)

        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0)
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=features,
            indices=data_dict['scale_1']['coors'].int(),
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),
            batch_size=data_dict['batch_size']
        )

        data_dict['pt_fea'] = features
        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']

        return data_dict