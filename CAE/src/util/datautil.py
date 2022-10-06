# -*- coding: utf-8 -*-

import numpy as np
import os

import math

import torch

def normalize(data, indataRange, outdataRange):
    """
    return normalized data
    it need two list (indataRange[x1,x2] and outdataRange[y1,y2])
    """
    if indataRange[0]!=indataRange[1]:
        data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
        data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    else:
        data = (outdataRange[0] + outdataRange[1]) / 2.
    return data

def denormalize(data, indataRange, outdataRange):
    """
    上記のinとoutのrangeを入れ替えればよい
    return denormalized data
    it need two list (indataRange[x1,x2] and outdataRange[y1,y2])
    """
    if indataRange[0]!=indataRange[1]:
        data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
        data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    else:
        data = (outdataRange[0] + outdataRange[1]) / 2.
    return data

########## CAE/VAE 2枚 学習用 ##########
class Dataset_2ims_train(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform

        dirs = os.listdir(data_path)
        dirs.sort()
        for i in range(len(dirs)):
            if i == 0:
                self.left = np.load(data_path + "/" + dirs[i] + "/left_image_wide.npy")
                self.right = np.load(data_path + "/" + dirs[i] + "/right_image_wide.npy")
                self.left_gap = np.load(data_path + "/" + dirs[i] + "/left_image_wide_gap.npy")
                self.right_gap = np.load(data_path + "/" + dirs[i] + "/right_image_wide_gap.npy")
            else:
                self.left = np.concatenate([self.left, np.load(data_path + "/" + dirs[i] + "/left_image_wide.npy")])
                self.right = np.concatenate([self.right, np.load(data_path + "/" + dirs[i] + "/right_image_wide.npy")])
                self.left_gap = np.concatenate([self.left_gap, np.load(data_path + "/" + dirs[i] + "/left_image_wide_gap.npy")])
                self.right_gap = np.concatenate([self.right_gap, np.load(data_path + "/" + dirs[i] + "/right_image_wide_gap.npy")])

    def __len__(self):
        return self.left.shape[0]
                
    def __getitem__(self, idx):
        out_left = self.left[idx]
        out_right = self.right[idx]
        out_left_gap = self.left_gap[idx]
        out_right_gap = self.right_gap[idx]

        return out_left, out_right, out_left_gap, out_right_gap

########## CAE/VAE 2枚 テスト用 ##########
class Dataset_2ims_test(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform

        dirs = os.listdir(data_path)
        dirs.sort()
        self.left = []
        self.right = []
        for dir in dirs:
            self.left.append(np.load(data_path + "/" + dir + "/left_image_wide.npy"))
            self.right.append(np.load(data_path + "/" + dir + "/right_image_wide.npy"))

    def __len__(self):
        return len(self.left)

    def __getitem__(self, idx):
        out_left = self.left[idx]
        out_right = self.right[idx]

        return out_left, out_right

########## CAE/VAE 1枚 学習用 ##########
class Dataset_1ims_train(torch.utils.data.Dataset):
    def __init__(self, data_path, use_seq_idx, transform=None):
        self.transform = transform

        first_flag = True

        dirs = os.listdir(data_path)
        dirs.sort()
        for i in range(len(dirs)):
            if i in use_seq_idx:
                if first_flag:
                    self.concat = np.load(data_path + "/" + dirs[i] + "/concat_image_wide.npy")
                    #self.concat_gap = np.load(data_path + "/" + dirs[i] + "/concat_gap_image_wide.npy")
                    first_flag = False
                else:
                    self.concat = np.concatenate([self.concat, np.load(data_path + "/" + dirs[i] + "/concat_image_wide.npy")])
                    #self.concat_gap = np.concatenate([self.concat_gap, np.load(data_path + "/" + dirs[i] + "/concat_gap_image_wide.npy")])

    def __len__(self):
        return self.concat.shape[0]
                
    def __getitem__(self, idx):
        out_concat = self.concat[idx]
        #out_concat_gap = self.concat_gap[idx]

        return out_concat#, out_concat_gap

########## CAE/VAE 1枚 テスト用 ##########
class Dataset_1ims_test(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform

        dirs = os.listdir(data_path)
        dirs.sort()
        self.concat = []
        for dir in dirs:
            self.concat.append(np.load(data_path + "/" + dir + "/concat_image_wide.npy"))

    def __len__(self):
        return len(self.concat)

    def __getitem__(self, idx):
        out_concat = self.concat[idx]

        return out_concat

########## SubgoalCAE 学習用 ##########
class Dataset_SubCAE(torch.utils.data.Dataset):
    def __init__(self, data_path, use_seq_idx, subgoal_timestep, transform=None):
        self.transform = transform

        first_flag = True

        dirs = os.listdir(data_path)
        dirs.sort()
        for i in range(len(dirs)):
            if i in use_seq_idx:
                data = np.load(data_path + "/" + dirs[i] + "/concat_image_wide.npy")
                data_gap = np.load(data_path + "/" + dirs[i] + "/concat_gap_image_wide.npy")

                for ts in range(data.shape[0]):
                    for subs in range(len(subgoal_timestep[i])):
                        if math.fabs(ts - subgoal_timestep[i][subs]) < 4:
                            if first_flag:
                                self.concat = data[ts].reshape(1, data.shape[1], data.shape[2], data.shape[3])
                                self.concat_gap = data_gap[ts].reshape(1, data_gap.shape[1], data_gap.shape[2], data_gap.shape[3])
                                first_flag = False
                            else:
                                self.concat = np.concatenate([self.concat, data[ts].reshape(1, data.shape[1], data.shape[2], data.shape[3])])
                                self.concat_gap = np.concatenate([self.concat_gap, data_gap[ts].reshape(1, data_gap.shape[1], data_gap.shape[2], data_gap.shape[3])])

    def __len__(self):
        return self.concat.shape[0]
                
    def __getitem__(self, idx):
        out_concat = self.concat[idx]
        out_concat_gap = self.concat_gap[idx]

        return out_concat, out_concat_gap

########## サブゴール提案とMTRNN 学習用 ##########
class Dataset_subgoal_mtrnn(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform

        dirs = os.listdir(data_path)
        dirs.sort()
        self.image = []
        self.motion = []
        for dir in dirs:
            self.image.append(np.load(data_path + "/" + dir + "/concat_image_wide.npy"))
            self.motion.append(np.load(data_path + "/" + dir + "/motion.npy"))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        out_image = self.image[idx]
        out_motion = self.motion[idx]

        return out_image, out_motion


########## 以下今は使ってない ##########
def load_mnist(data_path):
    train_data = []
    test_data = []
    for label in range(10):
        train = torch.tensor(np.load(data_path + "/train_{}.npy".format(label)).transpose(0, 3, 1, 2))
        test = torch.tensor(np.load(data_path + "/test_{}.npy".format(label)).transpose(0, 3, 1, 2))
        train_data.append(train)
        test_data.append(test)
    return train_data, test_data

def load_wide_image(data_path):
    dirs = os.listdir(data_path)
    dirs.sort()
    print(dirs)
    for i in range(len(dirs)):
        if i == 0:
            left_image_wide = np.load(data_path + "/" + dirs[i] + "/left_image_wide.npy")
            right_image_wide = np.load(data_path + "/" + dirs[i] + "/right_image_wide.npy")
        else:
            left_image_wide = np.concatenate([left_image_wide, np.load(data_path + "/" + dirs[i] + "/left_image_wide.npy")])
            right_image_wide = np.concatenate([right_image_wide, np.load(data_path + "/" + dirs[i] + "/right_image_wide.npy")])
    return left_image_wide, right_image_wide

def load_wide_gap_image(data_path):
    dirs = os.listdir(data_path)
    dirs.sort()
    for i in range(len(dirs)):
        if i == 0:
            left_image_wide_gap = np.load(data_path + "/" + dirs[i] + "/left_image_wide_gap.npy")
            right_image_wide_gap = np.load(data_path + "/" + dirs[i] + "/right_image_wide_gap.npy")
        else:
            left_image_wide_gap = np.concatenate([left_image_wide_gap, np.load(data_path + "/" + dirs[i] + "/left_image_wide_gap.npy")])
            right_image_wide_gap = np.concatenate([right_image_wide_gap, np.load(data_path + "/" + dirs[i] + "/right_image_wide_gap.npy")])
    return left_image_wide_gap, right_image_wide_gap

def load_wide_image_test(data_path):
    dirs = os.listdir(data_path)
    dirs.sort()
    left_image_wide = []
    right_image_wide = []
    for dir in dirs:
        left_image_wide.append(np.load(data_path + "/" + dir + "/left_image_wide.npy"))
        right_image_wide.append(np.load(data_path + "/" + dir + "/right_image_wide.npy"))
    return left_image_wide, right_image_wide

def load_concat_image(data_path):
    dirs = os.listdir(data_path)
    dirs.sort()
    for i in range(len(dirs)):
        if i == 0:
            concat = np.load(data_path + "/" + dirs[i] + "/concat_image_wide.npy")
        else:
            concat = np.concatenate([concat, np.load(data_path + "/" + dirs[i] + "/concat_image_wide.npy")])
    return concat

def load_narrow_image(data_path):
    left_image_object = []
    left_image_rarm = []
    right_image_larm = []
    right_image_object = []
    dirs = os.listdir(data_path)
    dirs.sort()
    for dir in dirs:
        left_image_object.append(np.load(data_path + "/" + dir + "/left_image_object.npy"))
        left_image_rarm.append(np.load(data_path + "/" + dir + "/left_image_rarm.npy"))
        right_image_larm.append(np.load(data_path + "/" + dir + "/right_image_larm.npy"))
        right_image_object.append(np.load(data_path + "/" + dir + "/right_image_object.npy"))
    return left_image_object, left_image_rarm, right_image_larm, right_image_object