import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from models import VIONet, IMGNet, IMUNet
from dataset import OdometryDataset
from utils import transform_to_world_frame, plot_poses, calculate_ate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
        

if __name__ == '__main__':

    data_path = '/home/turane/temp_cv/P4/data/'
    log_dir = 'IO_mt_3/'
    log_path = '/home/turane/temp_cv/P4/VIO/logs/'+log_dir
    flag = 'io'

    test_data_dir = 'fig_8'
    _path = data_path + test_data_dir + '/'

    test_dataset = OdometryDataset(_path, flag=flag, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print('read data complete')

    if flag == 'vio':
        model = VIONet().to(DEVICE)
    elif flag == 'vo':
        model = IMGNet().to(DEVICE)
    elif flag == 'io':
        model = IMUNet().to(DEVICE)

    model.load_state_dict(torch.load(log_path+'best_model.pt'))

    prev_pred_pose = np.array([0., 0., 0., 0., 0., 0.])
    prev_gt_pose = np.array([0., 0., 0., 0., 0., 0.])
    pred = []
    gt = []
    pred_lines = []
    gt_lines = []

    print('Testing...')
    # Predict
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            input_imu = batch[0].to(DEVICE)
            input_img = batch[1].to(DEVICE)
            label = batch[2].to(DEVICE)
            
            if flag == 'vio':
                output = model(input_imu, input_img)
            elif flag == 'vo':
                output = model(input_img)
            elif flag == 'io':
                output = model(input_imu)

            pred_pose = transform_to_world_frame(prev_pred_pose, np.squeeze(output.cpu().numpy(), axis=0))
            gt_pose = transform_to_world_frame(prev_gt_pose, np.squeeze(label.cpu().numpy(), axis=0))

            pred_lines.append(str(pred_pose[0])+' '+str(pred_pose[1])+' '+str(pred_pose[2])+' '+str(pred_pose[3])+' '+str(pred_pose[4])+' '+str(pred_pose[5])+'\n')
            gt_lines.append(str(gt_pose[0])+' '+str(gt_pose[1])+' '+str(gt_pose[2])+' '+str(gt_pose[3])+' '+str(gt_pose[4])+' '+str(gt_pose[5])+'\n')

            pred.append(pred_pose)
            gt.append(gt_pose)

            prev_pred_pose = pred_pose
            prev_gt_pose = gt_pose

    print('Testing complete')  

    mean_ate, median_ate = calculate_ate(gt, pred)
    print('mean ate: ', mean_ate)
    print('median ate: ', median_ate)

    with open(log_path+test_data_dir+'_metric.txt', 'a') as file:
        file.writelines(['mean ate: ' + str(mean_ate) + ' median ate: ' + str(median_ate) + '\n'])

    with open(log_path+test_data_dir+'_pred_pose.txt', 'a') as file:
        file.writelines(pred_lines)

    with open(log_path+test_data_dir+'_gt_pose.txt', 'a') as file:
        file.writelines(gt_lines)

    plot_poses(pred, gt, log_path+test_data_dir+'_poses.png')