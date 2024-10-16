import torch
import torch.nn as nn


class IMGNet(nn.Module):
    def __init__(self, in_channels=6, hidden_size=128, output_size=6, concat=False):
        super(IMGNet, self).__init__()

        self.batchNorm = True
        self.concat = concat

        self.conv0   = self.conv(self.batchNorm, in_channels, 64)
        self.conv1   = self.conv(self.batchNorm, 64, 64, stride=2)
        self.conv1_1 = self.conv(self.batchNorm, 64, 128)
        self.conv2   = self.conv(self.batchNorm, 128, 128, stride=2)
        self.conv2_1 = self.conv(self.batchNorm, 128, 128)
        self.conv3   = self.conv(self.batchNorm, 128, 256, stride=2)
        self.conv3_1 = self.conv(self.batchNorm, 256, 256)
        self.conv4   = self.conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = self.conv(self.batchNorm, 512, 512)
        self.conv5   = self.conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = self.conv(self.batchNorm, 512, 512)
        self.conv6   = self.conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = self.conv(self.batchNorm, 1024, 1024)

        self.lstm = nn.LSTM(81920, hidden_size, num_layers=2, dropout=0, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def conv(self, batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1,inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.LeakyReLU(0.1,inplace=True)
            )

    def encode_image(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        return out_conv6

    def forward(self, x):

        x = self.encode_image(x)
        x = x.reshape(-1, 81920)
        out, _ = self.lstm(x)
        out = self.lstm_dropout(out)
        if self.concat == False:
            out = self.fc(out)
        return out
    

class IMUNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=6, concat=False):
        super(IMUNet, self).__init__()
        self.hidden_size = hidden_size
        self.concat = concat

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc_concat = nn.Linear(hidden_size, 128)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.prelu = nn.PReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        if self.concat == False:
            out, _ = self.lstm1(x)
            out, _ = self.lstm2(out)
            out = self.fc1(out[:, -1, :])
            out = self.prelu(out)
            out = self.fc2(out)
        else:
            out, _ = self.lstm1(x)
            out = self.fc_concat(out[:, -1, :])

        return out
    

class VIONet(nn.Module):
    def __init__(self, hidden_size=128, output_size=6):
        super(VIONet, self).__init__()

        self.IMU_model = IMUNet(concat=True)
        self.IMG_model = IMGNet(concat=True)
        self._load_weights()

        self.lstm = nn.LSTM(hidden_size*2, hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.prelu = nn.PReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def _load_weights(self):

        weights = torch.load('/home/turane/temp_cv/P4/VO/FlowNet2-SD_checkpoint.pth.tar')
        for key in weights["state_dict"].keys():
            if key in self.IMG_model.state_dict().keys():
                self.IMG_model.state_dict()[key] = weights["state_dict"][key]

        params_to_train = ['conv0.0.weight', 'conv0.1.weight', 'conv0.1.bias', 'conv1.0.weight', 'conv1.1.weight', 'conv1.1.bias', 'conv1_1.0.weight', 'conv1_1.1.weight', 'conv1_1.1.bias', 'conv2.0.weight', 'conv2.1.weight', 'conv2.1.bias', 'conv2_1.0.weight', 'conv2_1.1.weight', 'conv2_1.1.bias', 'conv3.0.weight', 'conv3.1.weight', 'conv3.1.bias', 'conv3_1.0.weight', 'conv3_1.1.weight', 'conv3_1.1.bias', 'conv4.0.weight', 'conv4.1.weight', 'conv4.1.bias', 'conv4_1.0.weight', 'conv4_1.1.weight', 'conv4_1.1.bias', 'conv5.0.weight', 'conv5.1.weight', 'conv5.1.bias', 'conv5_1.0.weight', 'conv5_1.1.weight', 'conv5_1.1.bias', 'conv6.0.weight', 'conv6.1.weight', 'conv6.1.bias', 'conv6_1.0.weight', 'conv6_1.1.weight', 'conv6_1.1.bias']
        for name, param in self.IMG_model.named_parameters():
            param.requires_grad = False if name in params_to_train else True

    def forward(self, x_imu, x_img):

        out_imu = self.IMU_model(x_imu)
        out_img = self.IMG_model(x_img)

        x_concat = torch.cat((out_imu.view(out_imu.size(0), -1), out_img.view(out_img.size(0), -1)), dim=1)
        out, _ = self.lstm(x_concat)
        out = self.fc1(out)
        out = self.prelu(out)
        out = self.fc2(out)

        return out
