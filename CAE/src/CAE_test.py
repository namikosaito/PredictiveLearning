# -*- coding: utf-8 -*-

# from PIL.Image import Image
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import os
import sys
from tqdm import tqdm

import torch

import CAE_model as caemodel
from util import datautil

def main():
    ########## ディレクトリ管理 ##########
    file_name = "CAE_20220428_1812"
    data_path = "../sample_data"
    save_dir = "../model_CAE/" + file_name

    # learned_models = os.listdir(save_dir)
    # learned_models = [s for s in learned_models if "epoch" in s]
    learned_model = "epoch_best"

    ########## プロパティ・データ読み込み ##########
    with open(save_dir + "/property.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("cnn_ch0"):
            param = line.strip().split(": ")
            cnn_ch0 = int(param[1])
            print("cnn_ch0: {}".format(cnn_ch0))
        if line.startswith("cnn_ch1"):
            param = line.strip().split(": ")
            cnn_ch1 = int(param[1])
            print("cnn_ch1: {}".format(cnn_ch1))
        if line.startswith("cnn_ch2"):
            param = line.strip().split(": ")
            cnn_ch2 = int(param[1])
            print("cnn_ch2: {}".format(cnn_ch2))
        if line.startswith("cnn_ch3"):
            param = line.strip().split(": ")
            cnn_ch3 = int(param[1])
            print("cnn_ch3: {}".format(cnn_ch3))
        if line.startswith("h_dim0"):
            param = line.strip().split(": ")
            h_dim0 = int(param[1])
            print("h_dim0: {}".format(h_dim0))
        if line.startswith("h_dim1"):
            param = line.strip().split(": ")
            h_dim1 = int(param[1])
            print("h_dim1: {}".format(h_dim1))
        if line.startswith("h_dim2"):
            param = line.strip().split(": ")
            h_dim2 = int(param[1])
            print("h_dim2: {}".format(h_dim2))

    dataset = datautil.Dataset_1ims_test(data_path)
    height, width, color_ch = dataset[0][0].shape

    ########## モデルの設定 ##########
    torch_device = "cuda" # "cuda" or "cpu"

    model = caemodel.CAE_1ims(height, width, color_ch, cnn_ch0, cnn_ch1, cnn_ch2, cnn_ch3, h_dim0, h_dim1, h_dim2).to(torch_device)
        
    ########## テスト ##########
    model_path = save_dir + "/" + learned_model
    print(model_path)
    model.load_state_dict(torch.load(model_path + "/" + learned_model + ".pth", map_location=torch.device(torch_device)))
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    num = 0

    for data in tqdm(dataloader):
        if not os.path.exists(model_path + "/reconstruct/{:0>3}/lr_concat".format(num)):
            os.makedirs(model_path + "/reconstruct/{:0>3}/lr_concat".format(num))
        concat = data[0]
        x = concat
        x = x.permute(0, 3, 1, 2).to(torch.float32).to(torch_device)
        h, x_hat = model.forward(x)
        h_sample = h.to("cpu").detach().numpy().copy()
        reconst = x_hat.to("cpu").detach().numpy().copy().transpose(0, 2, 3, 1)*255
        reconst = reconst.astype(np.uint8)
        for ts in range(reconst.shape[0]):
            concat_image = Image.fromarray(reconst[ts])
            concat_image.save(model_path + "/reconstruct/{:0>3}/lr_concat/{:0>4}.png".format(num, ts*2))
        if not os.path.exists(model_path + "/h_sample"):
            os.makedirs(model_path + "/h_sample")
        np.save(model_path + "/h_sample/{:0>3}.npy".format(num), h_sample)

        del x, h, x_hat
        num = num+1

if __name__ == "__main__":
    main()