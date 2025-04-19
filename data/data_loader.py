import os
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings

warnings.filterwarnings('ignore')


# 自定数据预处理类
class Dataset_Custom(Dataset):
    def __init__(self,
                 root_path,size=None,features='S',data_path='DST_0C.csv',
                 target='SOC',scale=True,inverse=False,timeenc=0,freq='s',
                 cols=None,SOC=True,label='range',begin=1,valiRatio=None,flag = 'test'):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.SOC = SOC
        self.label = label
        self.begin = begin
        self.valiIndex = [] #   验证集索引，测试集未使用
        self.valiRatio = valiRatio
        self.flag = flag    # vali train test
        self.vinMasks = []  #  测试集车辆vin码
        self.vinNumCounts = []  # 测试集每辆车的充放电过程数
        self.dataSet = np.empty((0, 660 * 5 + self.pred_len))
        self.scaler = StandardScaler()
        self.__read_data__()


    def __read_data__(self):
        folderPath = self.root_path +  self.data_path
        for vinMask in os.listdir(folderPath):
            df = pd.read_csv(filepath_or_buffer=folderPath + vinMask, header=None)
            rows = df.shape[0]
            valiIndexs = np.linspace(0, rows - 1, math.ceil(rows * self.valiRatio)).astype(int).tolist()  # 验证集使用的下标linspace(startIndex,endIndex,number)
            if self.flag == 'train':
                df.drop(valiIndexs, axis=0, inplace=True)   #训练集和测试集分开
            elif self.flag == 'vali':
                df = df.loc[valiIndexs]
            else:pass   # test集合保持原状
            df = df.to_numpy()
            self.dataSet = np.concatenate((self.dataSet, df[0:]), axis=0)
            self.vinMasks.append(vinMask[0:-4])
            self.vinNumCounts.append(df.shape[0])
            # break
        print(self.dataSet.shape)
        if self.scale:
            self.scaler.fit(self.dataSet)             # 预处理，归一化
            self.dataSetScale = self.scaler.transform(self.dataSet)



    def __getitem__(self, index):   # 用来遍历dataloader，当dataloader设置的shuffle_flag=true, index按乱序返回，否则按照增序有序返回；batchsize作为每一组index里包含的元素个数；index变化范围在__len__()中定义
        return self.dataSetScale[index], self.dataSet[index]
        # return index

    def __len__(self):
        # return len(self.dataSet) - self.seq_len - self.pred_len + 1
        return len(self.dataSet)


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


