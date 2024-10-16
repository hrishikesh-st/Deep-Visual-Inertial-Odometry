import os, time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from models import VIONet, IMGNet, IMUNet
from loss import TransformationLoss
from dataset import OdometryDataset
from utils import transform_to_world_frame, plot_poses, plot_loss, calculate_ate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="bent_spiral/",help="Data directory")
    parser.add_argument("--log_dir", default="IO_bs/", help="Log Directory")
    parser.add_argument("--flag", default="io", help="Type of Odometry (io, vo, vio)")
    args = parser.parse_args()

    # Example file paths
    data_path = '/home/turane/temp_cv/P4/data/'
    data_dir = [args.data_dir]
    log_dir = args.log_dir
    log_path = '/home/turane/temp_cv/P4/VIO/logs/bs/'+log_dir

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # Hyperparameters
    num_samples = 10
    input_size = 6  # (3 accelerometer + 3 gyroscope readings)
    hidden_size = 64
    output_size = 12  # 3 for position (x, y, z) + 3 for rotation (euler angles)
    batch_size = 32  # Number of IMU readings per batch
    num_epochs = 50
    flag = args.flag

    print('training', flag)

    # Read input and ground truth data from files
    # train_dataset = OdometryDataset(data_path+data_dir[0], flag=flag, mode='train')
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = OdometryDataset(data_path+data_dir[0], flag=flag, mode='val')
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    dataset = OdometryDataset(data_path+data_dir[0], flag=flag, mode='test')
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = OdometryDataset(data_path+data_dir[0], flag=flag, mode='val')
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    print('read data complete')

    # Initialize model, loss function, and optimizer
    
    if flag == 'vio':
        model = VIONet().to(DEVICE)
        lr = 1e-3
    elif flag == 'vo':
        model = IMGNet().to(DEVICE)
        lr = 1e-3
    elif flag == 'io':
        model = IMUNet().to(DEVICE)
        lr = 1e-3

    criterion = TransformationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    train_losses = []
    val_losses = []
    best_loss = 1000

    print('Training...')
    for epoch in range(num_epochs):
        _start = time.time()
        model.train()
        train_loss = 0
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        for batch in train_dataloader:
            input_imu = batch[0].to(DEVICE)
            input_img = batch[1].to(DEVICE)
            label = batch[2].to(DEVICE)

            optimizer.zero_grad()
            
            if flag == 'vio':
                output = model(input_imu, input_img)
            elif flag == 'vo':
                output = model(input_img)
            elif flag == 'io':
                output = model(input_imu)

            loss = criterion(output, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        _avg_train_loss = train_loss/len(train_dataset)
        train_losses.append(_avg_train_loss)

        model.eval()
        val_loss = 0
        for batch in val_dataloader:
            input_imu = batch[0].to(DEVICE)
            input_img = batch[1].to(DEVICE)
            label = batch[2].to(DEVICE)

            if flag == 'vio':
                output = model(input_imu, input_img)
            elif flag == 'vo':
                output = model(input_img)
            elif flag == 'io':
                output = model(input_imu)

            loss = criterion(output, label)
            val_loss += loss.item()

        _avg_val_loss = val_loss/len(val_dataset)
        val_losses.append(_avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {_avg_train_loss}, Val Loss: {_avg_val_loss}, Avg Time: {time.time()-_start} secs')

        plot_loss(train_losses, val_losses, log_path+'loss.png')

        if _avg_val_loss < best_loss:
            torch.save(model.state_dict(), log_path+'best_model.pt')
            best_loss = _avg_val_loss
        
    print('Training complete') 

    test_data_dir = args.data_dir
    _path = data_path + test_data_dir

    test_dataset = OdometryDataset(_path, flag=flag, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prev_pred_pose = np.array([0., 0., 0., 0., 0., 0.])
    prev_gt_pose = np.array([0., 0., 0., 0., 0., 0.])

    pred = []
    gt = []
    pred_lines = []
    gt_lines = []

    print('Testing...')
    model.load_state_dict(torch.load(log_path+'best_model.pt'))
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

            pred.append(pred_pose)
            gt.append(gt_pose)

            pred_lines.append(str(pred_pose[0])+' '+str(pred_pose[1])+' '+str(pred_pose[2])+' '+str(pred_pose[3])+' '+str(pred_pose[4])+' '+str(pred_pose[5])+'\n')
            gt_lines.append(str(gt_pose[0])+' '+str(gt_pose[1])+' '+str(gt_pose[2])+' '+str(gt_pose[3])+' '+str(gt_pose[4])+' '+str(gt_pose[5])+'\n')

            prev_pred_pose = pred_pose
            prev_gt_pose = gt_pose

    print('Testing complete')  

    mean_ate, median_ate = calculate_ate(gt, pred)
    print('mean ate: ', mean_ate)
    print('median ate: ', median_ate)

    _f = open(log_path+'metric.txt', 'w')
    _f = open(log_path+'pred_pose.txt', 'w')
    _f = open(log_path+'gt_pose.txt', 'w')

    with open(log_path+'metric.txt', 'a') as file:
        file.writelines(['mean ate: ' + str(mean_ate) + ' median ate: ' + str(median_ate) + '\n'])

    with open(log_path+'pred_pose.txt', 'a') as file:
        file.writelines(pred_lines)

    with open(log_path+'gt_pose.txt', 'a') as file:
        file.writelines(gt_lines)

    plot_poses(pred, gt, log_path+'poses.png')
