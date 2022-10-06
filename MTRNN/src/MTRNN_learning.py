# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

#from IPython.display import HTML

from tqdm import tqdm
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

import MTRNN_model as mtmodel
import MTRNN_util as mtutil

def main():
    ########## ディレクトリ管理 ##########
    model_path = "../model_MTRNN/MTRNN_" + time.strftime("%Y%m%d_%H%M")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    ########## モデルの設定 ##########
    torch_device = "cpu" # "cuda" or "cpu"
    data_num = 500
    batch_size = 500
    n_epoch = 100

    loss_record = []
    min_loss = 10.0

    layer_size = {"in": 2, "out": 2, "io": 10, "cf": 10, "cs": 10}
    tau = {"tau_io": 2.0, "tau_cf": 5.0, "tau_cs": 10.0}
    open_rate = 1

    model = mtmodel.MTRNN(layer_size, tau, open_rate, torch_device).to(torch_device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    print(model)

    ########## 教師データ作成 ##########
    t, xy, x, y = mtutil.createLissajous(a=1, b=2, r=1, data_num = data_num)
    xy_train = mtutil.createBatches(xy, batch_size, torch_device)

    ########## 学習 ##########
    start_time = time.time()
    model.train()

    for epoch in tqdm(range(n_epoch)):
        total_loss = 0.0
        optimizer.zero_grad()
        model.init_state(xy_train.shape[1])
        # print(xy_train.shape[1])
        for i in range(xy_train.shape[0]-1):
            input = xy_train[i]
            output = model.forward(input)
            # print(output)
            # print(xy_train[i+1])
            loss = criterion(output, xy_train[i+1])
            total_loss = total_loss + loss
            del input, output, loss
        total_loss.backward()
        optimizer.step()

        loss_record.append(total_loss.to("cpu").detach().numpy().copy())
        if total_loss.to("cpu").detach().numpy().copy() < min_loss and epoch > 50:
            torch.save(model.to("cpu").state_dict(), model_path + "/MTRNN_best.pth")
            min_loss = total_loss.to("cpu").detach().numpy().copy()
            
        del total_loss

    end_time = time.time()
    learning_time = end_time - start_time
    print("learning time : {:.2f}[sec]".format(learning_time))

    fig = plt.figure()
    plt.plot(loss_record, "ro--", label = "")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.ylim(ymin = -0.1, ymax = 1.0)
    plt.show()


    ########## テスト ##########
    x_test = []
    y_test = []

    model.load_state_dict(torch.load(model_path + "/MTRNN_best.pth", map_location=torch.device(torch_device)))
    model.init_state(1)
    model.eval()
    for i in range(xy.shape[0]):
        input = torch.Tensor(xy[i].reshape(1, 2)).to(torch_device)
        output = model.forward(input).to("cpu")
        x_test.append(output[0][0].data)
        y_test.append(output[0][1].data)

    plt.plot(x, y)
    plt.plot(x_test, y_test)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(xmin = -1.1, xmax = 1.1)
    plt.ylim(ymin = -1.1, ymax = 1.1)
    plt.show()

if __name__ == "__main__":
    main()