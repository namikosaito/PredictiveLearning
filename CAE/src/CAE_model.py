 # -*- coding: utf-8 -*-

from numpy.core.defchararray import decode, encode
import torch
import torch.nn as nn
import torch.nn.functional as F

class CAE_2ims(nn.Module):
    def __init__(self, height, width, color_ch, cnn_ch0, cnn_ch1, cnn_ch2, cnn_ch3, h_dim0, h_dim1, h_dim2):
        super(CAE_2ims, self).__init__()
        convoluted_size = [cnn_ch3, 1, height/2/2/2/3, width/2/2/2/3]

        # エンコーダー用の関数
        self.en_conv1 = nn.Conv3d(in_channels=color_ch, out_channels=cnn_ch0, kernel_size=[2,7,7], stride=[1,3,3], padding=[0,2,2])
        self.en_norm1 = nn.BatchNorm3d(cnn_ch0)
        self.en_conv2 = nn.Conv3d(in_channels=cnn_ch0,  out_channels=cnn_ch1, kernel_size=[1,6,6], stride=[1,2,2], padding=[0,2,2])
        self.en_norm2 = nn.BatchNorm3d(cnn_ch1)
        self.en_conv3 = nn.Conv3d(in_channels=cnn_ch1,  out_channels=cnn_ch2, kernel_size=[1,6,6], stride=[1,2,2], padding=[0,2,2])
        self.en_norm3 = nn.BatchNorm3d(cnn_ch2)
        self.en_conv4 = nn.Conv3d(in_channels=cnn_ch2,  out_channels=cnn_ch3, kernel_size=[1,6,6], stride=[1,2,2], padding=[0,2,2])
        self.en_norm4 = nn.BatchNorm3d(cnn_ch3)
        self.en_flat = nn.Flatten()
        self.en_linear1 = nn.Linear(convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3], h_dim0)
        self.en_linear2 = nn.Linear(h_dim0, h_dim1)
        self.en_linear3 = nn.Linear(h_dim1, h_dim2)

        # デコーダー用の関数
        self.de_linear3 = nn.Linear(h_dim2, h_dim1)
        self.de_linear2 = nn.Linear(h_dim1, h_dim0)
        self.de_linear1 = nn.Linear(h_dim0, convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3])
        self.de_conv4 = nn.ConvTranspose3d(in_channels=cnn_ch3, out_channels=cnn_ch2,  kernel_size=[1,6,6], stride=[1,2,2], padding=[0,2,2])
        self.de_norm4 = nn.BatchNorm3d(cnn_ch2)
        self.de_conv3 = nn.ConvTranspose3d(in_channels=cnn_ch2, out_channels=cnn_ch1,  kernel_size=[1,6,6], stride=[1,2,2], padding=[0,2,2])
        self.de_norm3 = nn.BatchNorm3d(cnn_ch1)
        self.de_conv2 = nn.ConvTranspose3d(in_channels=cnn_ch1, out_channels=cnn_ch0,  kernel_size=[1,6,6], stride=[1,2,2], padding=[0,2,2])
        self.de_norm2 = nn.BatchNorm3d(cnn_ch0)
        self.de_conv1 = nn.ConvTranspose3d(in_channels=cnn_ch0, out_channels=color_ch, kernel_size=[2,7,7], stride=[1,3,3], padding=[0,2,2])
        self.de_norm1 = nn.BatchNorm3d(color_ch)

        # 活性化関数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # エンコーダー
    def encoder(self, x):
        x = self.en_conv1(x)
        x = self.en_norm1(x)
        x = self.relu(x)

        x = self.en_conv2(x)
        x = self.en_norm2(x)
        x = self.relu(x)

        x = self.en_conv3(x)
        x = self.en_norm3(x)
        x = self.relu(x)

        x = self.en_conv4(x)
        x = self.en_norm4(x)
        x = self.relu(x)

        self.size = x.size()
        x = self.en_flat(x)
        
        x = self.en_linear1(x)
        x = self.relu(x)

        x = self.en_linear2(x)
        x = self.relu(x)

        x = self.en_linear3(x)
        h = self.sigmoid(x)

        return h

    # デコーダー
    def decoder(self, h):
        y = self.de_linear3(h)
        y = self.relu(y)

        y = self.de_linear2(y)
        y = self.relu(y)

        y = self.de_linear1(y)
        y = self.relu(y)

        y = y.reshape(self.size)

        y = self.de_conv4(y)
        y = self.de_norm4(y)
        y = self.relu(y)

        y = self.de_conv3(y)
        y = self.de_norm3(y)
        y = self.relu(y)

        y = self.de_conv2(y)
        y = self.de_norm2(y)
        y = self.relu(y)

        y = self.de_conv1(y)
        y = self.de_norm1(y)
        # y = self.sigmoid(y)
        y = self.relu(y)

        return y

    def forward(self, x, device):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return h, x_hat

class CAE_2ims_all2d(nn.Module):
    def __init__(self, height, width, color_ch, cnn_ch0, cnn_ch1, cnn_ch2, cnn_ch3, h_dim0, h_dim1, h_dim2):
        super(CAE_2ims_all2d, self).__init__()
        convoluted_size = [cnn_ch3, 1, height/2/2/2/3, width/2/2/2/3]

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=color_ch*2, out_channels=cnn_ch0, kernel_size=[7,7], stride=[3,3], padding=[2,2]),
            nn.BatchNorm2d(cnn_ch0),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_ch0,  out_channels=cnn_ch1, kernel_size=[6,6], stride=[2,2], padding=[2,2]),
            nn.BatchNorm2d(cnn_ch1),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_ch1,  out_channels=cnn_ch2, kernel_size=[6,6], stride=[2,2], padding=[2,2]),
            nn.BatchNorm2d(cnn_ch2),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_ch2,  out_channels=cnn_ch3, kernel_size=[6,6], stride=[2,2], padding=[2,2]),
            nn.BatchNorm2d(cnn_ch3),
            nn.ReLU()
        )
        self.encoer_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3], h_dim0),
            nn.BatchNorm1d(h_dim0),
            nn.ReLU(),
            nn.Linear(h_dim0, h_dim1),
            nn.BatchNorm1d(h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.BatchNorm1d(h_dim2),
            nn.Sigmoid(),
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(h_dim2, h_dim1),
            nn.BatchNorm1d(h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim0),
            nn.BatchNorm1d(h_dim0),
            nn.ReLU(),
            nn.Linear(h_dim0, convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3]),
            nn.BatchNorm1d(convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3]),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=cnn_ch3, out_channels=cnn_ch2,  kernel_size=[6,6], stride=[2,2], padding=[2,2]),
            nn.BatchNorm2d(cnn_ch2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=cnn_ch2, out_channels=cnn_ch1,  kernel_size=[6,6], stride=[2,2], padding=[2,2]),
            nn.BatchNorm2d(cnn_ch1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=cnn_ch1, out_channels=cnn_ch0,  kernel_size=[6,6], stride=[2,2], padding=[2,2]),
            nn.BatchNorm2d(cnn_ch0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=cnn_ch0, out_channels=color_ch*2, kernel_size=[7,7], stride=[3,3], padding=[2,2]),
            nn.BatchNorm2d(color_ch*2),
            nn.ReLU(),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        self.size = h.size()
        h = self.encoer_linear(h)
        return h

    def decode(self, h):
        h = self.decoder_linear(h)
        h = h.reshape(self.size)
        x = self.decoder_conv(h)
        return x
    
    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return h, x_hat

class CAE_1ims(nn.Module):
    def __init__(self, height, width, color_ch, cnn_ch0, cnn_ch1, cnn_ch2, cnn_ch3, h_dim0, h_dim1, h_dim2):
        super(CAE_1ims, self).__init__()
        convoluted_size = [cnn_ch3, 1, int(height/2/2/2/2), int(width/2/2/2//2)]

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=color_ch, out_channels=cnn_ch0, kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch0),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_ch0,  out_channels=cnn_ch1, kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch1),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_ch1,  out_channels=cnn_ch2, kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch2),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_ch2,  out_channels=cnn_ch3, kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch3),
            nn.ReLU()
        )
        self.encoder_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3], h_dim0),
            nn.BatchNorm1d(h_dim0),
            nn.ReLU(),
            nn.Linear(h_dim0, h_dim1),
            nn.BatchNorm1d(h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.BatchNorm1d(h_dim2),
            nn.Sigmoid(),
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(h_dim2, h_dim1),
            nn.BatchNorm1d(h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim0),
            nn.BatchNorm1d(h_dim0),
            nn.ReLU(),
            nn.Linear(h_dim0, convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3]),
            nn.BatchNorm1d(convoluted_size[0]*convoluted_size[1]*convoluted_size[2]*convoluted_size[3]),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=cnn_ch3, out_channels=cnn_ch2,  kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=cnn_ch2, out_channels=cnn_ch1,  kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=cnn_ch1, out_channels=cnn_ch0,  kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.BatchNorm2d(cnn_ch0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=cnn_ch0, out_channels=color_ch, kernel_size=[4,4], stride=[2,2], padding=[1,1]),
            nn.ReLU(),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        self.size = h.size()
        h = self.encoder_linear(h)
        return h

    def decode(self, h, size=None):
        if size != None:
            self.size = size
        h = self.decoder_linear(h)
        h = h.reshape(self.size)
        x = self.decoder_conv(h)
        return x
    
    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return h, x_hat