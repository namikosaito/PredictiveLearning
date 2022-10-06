# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt

import time
import os
import sys
import random

import torch
import torch.optim as optim

import CAE_model as caemodel
from util import datautil

def main():
    ########## Setting parameter ##########
    cnn_ch0 = 16
    cnn_ch1 = 32
    cnn_ch2 = 64
    cnn_ch3 = 128
    h_dim0 = 1024
    h_dim1 = 128
    h_dim2 = 15


    train_sequence = [
        0
    ]
    test_sequence = [
        1
    ]

    n_epoch = 5000
    save_frequency = 100
    batch_size = 300
    torch_device = "cuda" # "cuda" or "cpu"
    
    ########## Directory setting ##########
    data_path = "../sample_data"
    save_dir = "../model_CAE/CAE_" + time.strftime("%Y%m%d_%H%M")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ########## Read dataset ##########
    train_dataset = datautil.Dataset_1ims_train(data_path, train_sequence)
    test_dataset = datautil.Dataset_1ims_train(data_path, test_sequence)
    height, width, color_ch = train_dataset[0].shape

    ########## Setting the model ##########
    torch.manual_seed(1234)
    min_loss = 1000000000000000000000.0

    model = caemodel.CAE_1ims(height, width, color_ch, cnn_ch0, cnn_ch1, cnn_ch2, cnn_ch3, h_dim0, h_dim1, h_dim2).to(torch_device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.MSELoss()

    ########## make files ##########
    with open(save_dir + "/property.txt", "w") as f:
        f.write("########## model information ##########\n")
        f.write("cnn_ch0: {}\n".format(cnn_ch0))
        f.write("cnn_ch1: {}\n".format(cnn_ch1))
        f.write("cnn_ch2: {}\n".format(cnn_ch2))
        f.write("cnn_ch3: {}\n".format(cnn_ch3))
        f.write("h_dim0: {}\n".format(h_dim0))
        f.write("h_dim1: {}\n".format(h_dim1))
        f.write("h_dim2: {}\n".format(h_dim2))
        f.write("\n")
        f.write("{}\n".format(model))
        f.write("{}\n".format(optimizer))
        f.write("\n")
        f.write("########## train data information ##########\n")
        f.write("train_data_path: {}\n".format(data_path))
        f.write("data_shape: [{}, {}, {}, {}]\n".format(len(train_dataset), height, width, color_ch))
        f.write("batch_size: {}\n".format(batch_size))
        f.write("test_sequence: {}\n".format(test_sequence))

    ########## training ##########
    train_loss_record = []
    test_loss_record = []

    # sys.exit()

    for epoch in range(n_epoch):
        start_time = time.time()
        train_loss = 0.0
        num = 0
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for data in train_dataloader:
            model.train()
            optimizer.zero_grad()
            
            concat = data
            noise = np.random.normal(loc=0, scale=1, size=[concat.shape[0], height, width, color_ch])
            x = np.clip(concat + noise, 0, 1)
            y = concat
            x = x.permute(0, 3, 1, 2).to(torch.float32).to(torch_device)
            y = y.permute(0, 3, 1, 2).to(torch.float32).to(torch_device)
            h, x_hat = model.forward(x)
            loss = criterion(x_hat, y)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.to("cpu").detach().numpy().copy()
            # print("    x: {} loss: {}".format(x.shape, loss))
            del x, y, h, x_hat, loss
            num = num+1
        train_loss = train_loss / num

        test_loss = 0.0
        num = 0
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        for data in test_dataloader:
            model.eval()
            
            concat = data
            x = np.clip(concat, 0, 1)
            y = concat
            x = x.permute(0, 3, 1, 2).to(torch.float32).to(torch_device)
            y = y.permute(0, 3, 1, 2).to(torch.float32).to(torch_device)
            h, x_hat = model.forward(x)
            loss = criterion(x_hat, y)
            test_loss = test_loss + loss.to("cpu").detach().numpy().copy()
            # print("    x: {} loss: {}".format(x.shape, loss))
            del x, y, h, x_hat, loss
            num = num+1
        test_loss = test_loss / num

        end_time = time.time()
        epoch_time = end_time - start_time
        print("EPOCH: {} total_loss: {:.7f} time: {:.2f}".format(epoch, train_loss, epoch_time))
        train_loss_record.append(train_loss)
        test_loss_record.append(test_loss)

        if test_loss < min_loss:
            if not os.path.exists(save_dir + "/epoch_best"):
                os.makedirs(save_dir + "/epoch_best")
            torch.save(model.state_dict(), save_dir + "/epoch_best/epoch_best.pth")
            min_loss = test_loss
            print("    best model saved!!")
            
            plt.figure()
            plt.plot(train_loss_record, label = "train loss")
            plt.plot(test_loss_record, label = "test loss", alpha=0.5)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.ylim(ymin = -0.01)
            plt.legend()
            plt.savefig(save_dir + "/loss_curve.png")
            plt.close()

            ########## make file ##########
            with open(save_dir + "/property.txt", "w") as f:
                f.write("########## model information ##########\n")
                f.write("cnn_ch0: {}\n".format(cnn_ch0))
                f.write("cnn_ch1: {}\n".format(cnn_ch1))
                f.write("cnn_ch2: {}\n".format(cnn_ch2))
                f.write("cnn_ch3: {}\n".format(cnn_ch3))
                f.write("h_dim0: {}\n".format(h_dim0))
                f.write("h_dim1: {}\n".format(h_dim1))
                f.write("h_dim2: {}\n".format(h_dim2))
                f.write("\n")
                f.write("{}\n".format(model))
                f.write("{}\n".format(optimizer))
                f.write("\n")
                f.write("########## train data information ##########\n")
                f.write("train_data_path: {}\n".format(data_path))
                f.write("data_shape: [{}, {}, {}, {}]\n".format(len(train_dataset), height, width, color_ch))
                f.write("batch_size: {}\n".format(batch_size))
                f.write("test_sequence: {}\n".format(test_sequence))
                f.write("########## best model ##########\n")
                f.write("epoch: {}\n".format(epoch+1))
                f.write("test_loss: {}\n".format(test_loss))

        if (epoch+1)%save_frequency == 0 and epoch > 45:
            if not os.path.exists(save_dir + "/epoch_{}".format(epoch+1)):
                os.makedirs(save_dir + "/epoch_{}".format(epoch+1))
            torch.save(model.state_dict(), save_dir + "/epoch_{0}/epoch_{0}.pth".format(epoch+1))

            plt.figure()
            plt.plot(train_loss_record, label = "train loss")
            plt.plot(test_loss_record, label = "test loss", alpha=0.5)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.ylim(ymin = -0.01)
            plt.legend()
            plt.savefig(save_dir + "/loss_curve.png")
            plt.close()

if __name__ == "__main__":
    main()