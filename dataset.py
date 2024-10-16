import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from utils import relative_pose

class OdometryDataset(Dataset):
    def __init__(self, data_path, flag='vio', mode='train', sample_size=10):
        
        self.imu_data = []
        self.gt_data = []
        self.batched_imu_data = []
        self.batched_ground_truth_data = []
        self.idx = []
        self.data_path = data_path
        self.flag = flag

        with open(data_path+'camera_poses.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split()
                self.gt_data.append([float(line[4]), float(line[6]), float(line[8]), float(line[13]), float(line[15]), float(line[17])])

        if self.flag != 'vo':
            with open(data_path+'real_imu_data.txt', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip().split()
                    self.imu_data.append([float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7])])

        self._samplify_data(mode, sample_size)

    def _samplify_data(self, mode, sample_size):

        num_batches = len(self.gt_data) // sample_size
        
        if mode == 'test':
            _ids = 0
            _ide = num_batches
            _batch_idx = np.arange(_ids, _ide)

            for i in _batch_idx:
                start_idx = i * sample_size
                end_idx = start_idx + sample_size
                if end_idx >= 4998: continue
                self.idx.append([start_idx, end_idx])

                if self.flag != 'vo': self.batched_imu_data.append(self.imu_data[start_idx:end_idx+1])

                _pose = relative_pose(self.gt_data[start_idx], self.gt_data[end_idx])
                self.batched_ground_truth_data.append(_pose.tolist())
        
        elif mode == 'train':
            _ids = 0
            _ide = int(num_batches*0.8)
            _batch_idx = np.arange(_ids, _ide)

            for i in _batch_idx:
                # for j in range(sample_size):
                start_idx = i * sample_size
                end_idx = start_idx + sample_size
                if end_idx >= 4998: continue
                self.idx.append([start_idx, end_idx])

                if self.flag != 'vo': self.batched_imu_data.append(self.imu_data[start_idx:end_idx+1])

                _pose = relative_pose(self.gt_data[start_idx], self.gt_data[end_idx])
                self.batched_ground_truth_data.append(_pose.tolist())

        elif mode == 'val':
            _ids = int(num_batches*0.8)
            _ide = num_batches
            _batch_idx = np.arange(_ids, _ide)

            for i in _batch_idx:
                # for j in range(sample_size):
                start_idx = i * sample_size
                end_idx = start_idx + sample_size
                if end_idx >= 4998: continue
                self.idx.append([start_idx, end_idx])

                if self.flag != 'vo': self.batched_imu_data.append(self.imu_data[start_idx:end_idx+1])

                _pose = relative_pose(self.gt_data[start_idx], self.gt_data[end_idx])
                self.batched_ground_truth_data.append(_pose.tolist())

    def __len__(self):
        return len(self.batched_ground_truth_data)

    def __getitem__(self, idx):

        gt_data = torch.Tensor(self.batched_ground_truth_data[idx])
        imu_data = torch.zeros((2, 2))
        img_data = torch.zeros((2, 2))


        if self.flag == 'vio':
            imu_data = torch.Tensor(self.batched_imu_data[idx])
            
            img_idx = self.idx[idx]
            img1 = cv2.imread(self.data_path+'images/frame_'+str(img_idx[0]+3).zfill(4)+'.png')
            img2 = cv2.imread(self.data_path+'images/frame_'+str(img_idx[1]+3).zfill(4)+'.png')

            img_data = torch.Tensor(np.concatenate((img1, img2), axis=2)).permute(2, 0, 1)

            return imu_data, img_data, gt_data
        
        elif self.flag == 'vo':
            img_idx = self.idx[idx]
            img1 = cv2.imread(self.data_path+'images/frame_'+str(img_idx[0]+3).zfill(4)+'.png')
            img2 = cv2.imread(self.data_path+'images/frame_'+str(img_idx[1]+3).zfill(4)+'.png')

            img_data = torch.Tensor(np.concatenate((img1, img2), axis=2)).permute(2, 0, 1)

            return imu_data, img_data, gt_data
        
        elif self.flag == 'io':
            imu_data = torch.Tensor(self.batched_imu_data[idx])

            return imu_data, img_data, gt_data
