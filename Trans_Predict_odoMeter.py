import os
import numpy as np
from math import sqrt
import pandas as pd
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
# import cv2
import torch
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import init
import copy
import random

import datetime
import time
from baselines.ViT.ViT_explanation_generator import LRP

loss_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''不同参数设置'''
outputLen = 30  #30
# readPathPre = r'./data/discharge/4th'
# savePathPre = r'./vitTransResult/4th'
# readPathPre = r'./data/discharge/4th_leastSquareReduce'
# savePathPre = r'./vitTransResult/4th_leastSquareReduce'
# readPathPre = r'./data/discharge/4th_maxminReduce'
# savePathPre = r'./vitTransResult/4th_maxminReduce'

# readPathPre = r'./data/discharge/4th_lassoReduce_0.25'
# savePathPre = r'./vitTransResult/4th_lassoReduce_0.25'
# readPathPre = r'./data/discharge/4th_leastSquareReduce_0.25'
# savePathPre = r'./vitTransResult/4th_leastSquareReduce_0.25'
# readPathPre = r'./data/discharge/4th_maxminReduce_0.25'
# savePathPre = r'./vitTransResult/4th_maxminReduce_0.25'
# readPathPre = r'./data/discharge/4th_lassoReduce_0.35'
# savePathPre = r'./vitTransResult/4th_lassoReduce_0.35'
# readPathPre = r'./data/discharge/4th_leastSquareReduce_0.35'
# savePathPre = r'./vitTransResult/4th_leastSquareReduce_0.35'
# readPathPre = r'./data/discharge/4th_maxminReduce_0.35'
# savePathPre = r'./vitTransResult/4th_maxminReduce_0.35'

'''使用SG-filter后数据'''
readPathPre = r'./data/discharge/4th_SGReduce'
savePathPre = r'./vitTransResult/4th_SGReduce'
# readPathPre = r'./data/discharge/4th_SGReduce_0.25'
# savePathPre = r'./vitTransResult/4th_SGReduce_0.25'
# readPathPre = r'./data/discharge/4th_SGReduce_0.35'
# savePathPre = r'./vitTransResult/4th_SGReduce_0.35'


# outputLen = 20  #30
# readPathPre = r'./data/discharge/5th'
# savePathPre = r'./vitTransResult/5th'
# readPathPre = r'./data/discharge/5th_lassoReduce'
# savePathPre = r'./vitTransResult/5th_lassoReduce'
# readPathPre = r'./data/discharge/5th_leastSquareReduce'
# savePathPre = r'./vitTransResult/5th_leastSquareReduce'
# readPathPre = r'./data/discharge/5th_maxminReduce'
# savePathPre = r'./vitTransResult/5th_maxminReduce'
# readPathPre = r'./data/discharge/5th_lassoReduce_0.25'
# savePathPre = r'./vitTransResult/5th_lassoReduce_0.25'
# readPathPre = r'./data/discharge/5th_leastSquareReduce_0.25'
# savePathPre = r'./vitTransResult/5th_leastSquareReduce_0.25'
# readPathPre = r'./data/discharge/5th_maxminReduce_0.25'
# savePathPre = r'./vitTransResult/5th_maxminReduce_0.25'
# readPathPre = r'./data/discharge/5th_lassoReduce_0.35'
# savePathPre = r'./vitTransResult/5th_lassoReduce_0.35'
# readPathPre = r'./data/discharge/5th_leastSquareReduce_0.35'
# savePathPre = r'./vitTransResult/5th_leastSquareReduce_0.35'
# readPathPre = r'./data/discharge/5th_maxminReduce_0.35'
# savePathPre = r'./vitTransResult/5th_maxminReduce_0.35'





# outputLen = 25
# readPathPre = r'./data/discharge/6th'
# savePathPre = r'./vitTransResult/6th'
# readPathPre = r'./data/discharge/6th_lassoReduce'
# savePathPre = r'./vitTransResult/6th_lassoReduce'
# readPathPre = r'./data/discharge/6th_leastSquareReduce'
# savePathPre = r'./vitTransResult/6th_leastSquareReduce'
# readPathPre = r'./data/discharge/6th_maxminReduce'
# savePathPre = r'./vitTransResult/6th_maxminReduce'

# readPathPre = r'./data/discharge/6th_lassoReduce_0.25'
# savePathPre = r'./vitTransResult/6th_lassoReduce_0.25'
# readPathPre = r'./data/discharge/6th_leastSquareReduce_0.25'
# savePathPre = r'./vitTransResult/6th_leastSquareReduce_0.25'
# readPathPre = r'./data/discharge/6th_maxminReduce_0.25'
# savePathPre = r'./vitTransResult/6th_maxminReduce_0.25'
# readPathPre = r'./data/discharge/6th_lassoReduce_0.35'
# savePathPre = r'./vitTransResult/6th_lassoReduce_0.35'
# readPathPre = r'./data/discharge/6th_leastSquareReduce_0.35'
# savePathPre = r'./vitTransResult/6th_leastSquareReduce_0.35'
# readPathPre = r'./data/discharge/6th_maxminReduce_0.35'
# savePathPre = r'./vitTransResult/6th_maxminReduce_0.35'

# outputLen = 16
# readPathPre = r'./data/discharge/7th'
# savePathPre = r'./vitTransResult/7th'
# readPathPre = r'./data/discharge/7th_lassoReduce'
# savePathPre = r'./vitTransResult/7th_lassoReduce'
# readPathPre = r'./data/discharge/7th_leastSquareReduce'
# savePathPre = r'./vitTransResult/7th_leastSquareReduce'
# readPathPre = r'./data/discharge/7th_maxminReduce'
# savePathPre = r'./vitTransResult/7th_maxminReduce'
# readPathPre = r'./data/discharge/7th_lassoReduce_0.25'
# savePathPre = r'./vitTransResult/7th_lassoReduce_0.25'
# readPathPre = r'./data/discharge/7th_leastSquareReduce_0.25'
# savePathPre = r'./vitTransResult/7th_leastSquareReduce_0.25'
# readPathPre = r'./data/discharge/7th_maxminReduce_0.25'
# savePathPre = r'./vitTransResult/7th_maxminReduce_0.25'
# readPathPre = r'./data/discharge/7th_lassoReduce_0.35'
# savePathPre = r'./vitTransResult/7th_lassoReduce_0.35'
# readPathPre = r'./data/discharge/7th_leastSquareReduce_0.35'
# savePathPre = r'./vitTransResult/7th_leastSquareReduce_0.35'
# readPathPre = r'./data/discharge/7th_maxminReduce_0.35'
# savePathPre = r'./vitTransResult/7th_maxminReduce_0.35'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(27)


# scale train and test data to [-1, 1]
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [v for v in value]  #   数组合并
    # new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))    #传递二维数组实现invert
    inverted = scaler.inverse_transform(array)
    # inverted = torch.Tensor(inverted)
    # return inverted[0, -1]
    return inverted[0]  #返回一维数组


class DataPrepare:   #  outputLen,预测未来数据的段数
    def __init__(self, dataSet, outputLen):
        self.len = dataSet.shape[0]
        x_set = dataSet[:, 0:-1 * outputLen]    #   划分X值和目标值
        x_set = x_set.reshape(x_set.shape[0], 660, 5)
        y_set = dataSet[:, -1 * outputLen:]
        y_set = y_set.reshape(y_set.shape[0], outputLen)
        self.x_data = torch.from_numpy(x_set)
        self.y_data = torch.from_numpy(y_set)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Depth_Pointwise_Conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if (k == 1):
            self.depth_conv = nn.Identity()
        else:
            self.depth_conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k // 2  #   /精确除法、//除法向下取整、%取余除法
            )
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )

    def forward(self, x):
        out = self.pointwise_conv(self.depth_conv(x))
        return out


class MUSEAttention(nn.Module): #类的继承

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):

        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)
        self.dy_paras = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        # Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)  # bs,dim,n
        self.dy_paras = nn.Parameter(self.softmax(self.dy_paras))
        out2 = self.dy_paras[0] * self.conv1(v2) + self.dy_paras[1] * self.conv3(v2) + self.dy_paras[2] * self.conv5(v2)
        out2 = out2.permute(0, 2, 1)  # bs.n.dim

        out = out + out2
        return out


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = 660, 5
        patch_height, patch_width = 5, 5

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) #   132*1
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Conv2d(1, num_patches, kernel_size=5, stride=5),
            # Rearrange('b c h w -> b c (h w)'),
            # nn.Conv1d(num_patches//2, num_patches, kernel_size=10, stride=10),
            Rearrange('b c h w -> b c (h w)')
        )

        self.muse = MUSEAttention(d_model=int(660/5), d_k=int(660/5), d_v=int(660/5), h=8)
        """
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
            # x.shape = (b (hw) dim)
        )
        """
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # print(img.shape)
        x = self.to_patch_embedding(img)
        x = self.muse(x, x, x)
        visual2 = x
        x = rearrange(x, 'b r p -> b p r')
        # print(x.shape)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        a = x
        return self.mlp_head(x), a, visual2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.LSTM = nn.LSTM(4, 1, 2)
        self.MultiheadAttention = nn.MultiheadAttention(640, 64)

        self.layer = nn.Sequential(
            nn.Linear(660, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x, _ = self.LSTM(x)
        x = x.reshape(x.shape[0], 660)
        x = self.layer(x)
        return x


model = ViT(
    dim=int(660/5),
    image_size=660*5,
    patch_size=5,
    #num_classes=1,
    num_classes = outputLen, #viT.mlpHead,nn.linear 输出数据的维度
    channels=1,
    depth=12,
    mlp_dim=3,
    heads=8
)
model.to(device)
# model = torch.load('./result/soh_model.h5')
attribution_generator = LRP(model)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.015, amsgrad=False, betas=(0.9, 0.999), eps=1e-08)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back, 2641):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    for i in range(len(dataY)):
        if dataY[i].astype("float64") == 0:
            dataY[i] = str(dataY[i - 1][0].astype("float64"))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset, dataY


def experiment(train,test,outputLen,testvinMasks, n_epoch, batch_size, testVinMaskRows, savePathFolder):
    index = []
    np.random.shuffle(train)
    scaler, trainScaled, testScaled = scale(train, test)
    # joblib.dump(scaler, r'./vitTranResult/scaler_soh.pickle')

    starttime = time.time()
    # fit the model
    endtime = time.time()
    dtime = endtime - starttime

    trainScaledTensor = DataPrepare(trainScaled,outputLen)
    train_loader = DataLoader(dataset = trainScaledTensor, batch_size=batch_size, shuffle=False)
    bs, ic, image_h, image_w = 14, 1, 660, 5
    patch_size = 5
    model_dim = 8
    patch_depth = patch_size * patch_size * ic
    weight = torch.randn(patch_depth, model_dim)

    dic_loss = dict()
    for epoch in range(n_epoch):
        tmpStr = f"epoch:{epoch}"
        dic_loss[tmpStr] = []
        dic_loss[tmpStr].append(datetime.datetime.now())
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.reshape(14, 1, 660, 5)
            optimizer.zero_grad()
            # print(inputs.shape)
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            # y_pred = ViT(inputs, weight)
            y_pred, _, _ = model(inputs)
            # y_pred = y_pred.reshape(14, 1)
            loss = criterion(labels, y_pred)
            # print(epoch, i, loss.item())
            dic_loss[tmpStr].append(loss.cpu().item())
            loss.backward()
            optimizer.step()

            if i % 10 == 1:
                print(datetime.datetime.now(),epoch, i, loss.item())
                loss_list.append(loss.cpu().item())
                # print('Cycle=%d, Predicted=%f, Expected=%f' % (i + 1, y_pred[j], float(labels[j])))
    # torch.save(model, r'./vitTransResult/soh_model.h5')
    pd.DataFrame(dic_loss).to_csv(savePathFolder+r'/epochLoss.csv',index=False)

    # forecast the test data(#5)
    print('Forecasting Testing Data')
    predictions_test = list()
    UP_Pre = list()
    Down_Pre = list()
    expected = list()
    testScaledTensor = DataPrepare(testScaled,outputLen)
    test_loader = DataLoader(dataset = testScaledTensor, batch_size=batch_size, shuffle=False, drop_last=True)
    for i, data in enumerate(test_loader, 0):
        # make one-step forecast
        inputs, labels = data   #inputs shape: (14,660,5)   labels shape:(14,30)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.reshape(14, 1, 660, 5)
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        yhat, a, b = model(inputs)  #yhat shape:14,30  ashape:(14,132)  bshape:(14,132,132)
        # inverting scale
        print(a.shape)
        print(b.shape)
        print(yhat.shape)
        # fig, ax = plt.subplots(1, 14)
        # fig = plt.figure(figsize=[10, 4])
        # sns.heatmap(data=a[13].reshape(int(660/5), 1).cpu().detach().numpy())

        for j in range(yhat.shape[0]):  #   yhat.shape[0]=batch_size (14)
            # print(f'Cycle={i + 1}, Predicted={yhat[j]}, Expected={labels[j]}')
            predict = invert_scale(scaler, inputs[j].reshape(660*5, ).cpu(), yhat[j].cpu().detach().numpy())
            expect = invert_scale(scaler, inputs[j].reshape(660*5, ).cpu(), labels[j].cpu().detach().numpy())
            # predictions_test.append(predict.cpu().detach().numpy() - 0.01)
            predictions_test.append(predict[-1*outputLen:])
            # expected.append(expect.cpu().detach().numpy())
            expected.append(expect[-1*outputLen:])
            # UP_Pre.append(predict.cpu().detach().numpy() + 0.005*np.random.randn() - 0.01)
            UP_Pre.append(predict[-1*outputLen:] + 0.005 * np.random.randn() - 0.01)
            # Down_Pre.append(expect.cpu().detach().numpy() - 0.005*np.random.randn() - 0.01)
            Down_Pre.append(expect[-1*outputLen:] - 0.005 * np.random.randn() - 0.01)
        # store forecast
        # expected = dataY_5[len(train_5) + i]
        for j in range(yhat.shape[0]):
            # print('Cycle=%d, Predicted=%f, Expected=%f' % (i + 1, yhat[j], float(labels[j])))
            print(  f'Cycle={i+1}, Predicted={yhat[j]}, Expected={labels[j]}' )

    """删除batchsize中的冗余项,保存预测结果"""
    predictions_test = predictions_test[0:sum(testVinMaskRows)]
    expected  = expected[0:sum(testVinMaskRows)]
    UP_Pre = UP_Pre[0:sum(testVinMaskRows)]
    Down_Pre = Down_Pre[0:sum(testVinMaskRows)]

    deviation = 0 # 前面车辆的偏置，确定每辆车放电过程
    for i in range(len(testvinMasks)):  #为每辆车保存结果
        dic = dict()
        for j in range(testVinMaskRows[i]):
            dic[j] = np.concatenate((expected[ j+deviation ], predictions_test[ j+deviation ]), axis=0)
        pd.DataFrame(dic).to_csv(savePathFolder+r'/%s.csv'%testvinMasks[i], index=False)
        print(f'vin：{testvinMasks[i]} is saved')
        deviation += testVinMaskRows[i]


    # region to be explored
    # # report performance using RMSE
    # rmse_test = sqrt(
    #     mean_squared_error(np.array(expected) / 2, np.array(predictions_test) / 2))
    # print('Test RMSE: %.3f' % rmse_test)
    # # AE = np.sum((dataY_5[-len(test18_scaled):-9].astype("float64")-np.array(predictions_test))/len(predictions_test))
    # AE = np.sum((np.array(expected).astype("float64") - np.array(predictions_test)) / len(predictions_test))
    # print('Test AE:', AE.tolist())
    # print("程序训练时间：%.8s s" % dtime)
    #
    # index.append(rmse_test)
    # index.append(dtime)
    # with open(r'./result/soh_prediction_result.txt', 'a', encoding='utf-8') as f:
    #     for j in range(len(index)):
    #         f.write(str(index[j]) + "\n")
    #
    # with open(r'./result/soh_prediction_data_#5.txt', 'a', encoding='utf-8') as f:
    #     for k in range(len(predictions_test)):
    #         f.write(str(predictions_test[k]) + "\n")
    #
    # # line plot of observed vs predicted
    # num2 = len(expected)
    # Cyc_X = np.linspace(0, num2, num2)
    # UP_Pre = np.array(UP_Pre).reshape(len(UP_Pre), )
    # Down_Pre = np.array(Down_Pre).reshape(len(Down_Pre),  )
    # print(UP_Pre.shape)
    # fig = plt.figure(figsize=[8, 6], dpi=400)
    # sub = fig.add_subplot(111)
    # sub.plot(expected, c='r', label='Real Capacity', linewidth=2)
    # sub.plot(predictions_test, c='b', label='Predicted Capacity', linewidth=2)
    # sub.fill_between(Cyc_X, UP_Pre, Down_Pre, color='aqua', alpha=0.3)
    # sub.scatter(Cyc_X, predictions_test, s=25, c='orange', alpha=0.6, label='Predicted Capacity')
    # ax = plt.gca()
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # plt.tick_params(labelsize=13)
    # plt.legend(loc=1, edgecolor='w', fontsize=13)
    # plt.ylabel('Capacity (Ah)', fontsize=13)
    # plt.xlabel('Discharge Cycle', fontsize=13)
    # plt.title('MVIP-Trans SOH Estimation', fontsize=13)
    # plt.savefig(r'./result/soh_result.png')
    # plt.show()
    # endregion


def run(outputLen,readPathPre,savePathPre,type):
    train = np.empty((0,660*5 + outputLen))
    test = np.empty((0,660*5 + outputLen))
    neurons = [64, 64]
    n_epochs = 64
    # n_epochs = 120
    # n_epochs = 1
    updates = 1
    batch_size = 14 #为利用显卡算力，最好使用2的幂次方
    # batch_size = 20
    readPathTrain = ''
    readPathTest = ''
    savePathFolder = ''
    testVinMasks = list()
    testVinMaskRows = list()    #每辆车的挑选放电段数
    if type == 'capacity':
        readPathTrain = readPathPre + r'/capacityTransTrain/'
        readPathTest = readPathPre + r'/capacityTransTest/'
        savePathFolder = savePathPre + r'/capacity'
    else:   # if type == 'odoMeter':
        readPathTrain = readPathPre + r'/odoTransTrain/'
        readPathTest = readPathPre + r'/odoTransTest/'
        savePathFolder = savePathPre + r'/odoMeter'
    for vinMask in os.listdir(readPathTrain):
        df = pd.read_csv(filepath_or_buffer = readPathTrain + vinMask, header=None)
        trainSet = df.to_numpy()
        train = np.concatenate((train, trainSet[0:]), axis=0)
    for vinMask in os.listdir(readPathTest):
        df = pd.read_csv(filepath_or_buffer = readPathTest + vinMask, header=None)
        testSet = df.to_numpy()
        test = np.concatenate((test, testSet[0:]), axis=0)
        testVinMasks.append(vinMask[0:-4])
        testVinMaskRows.append(testSet.shape[0])
    """batchsize需要被train.shape[0]整除, 同理，test验证集也需要整除batchsize, 因其训练和验证是以图片帧(14,660,5)为单位，不足部分丢弃"""
    trainRows = train.shape[0]
    if trainRows % batch_size != 0:
        lacks = batch_size - trainRows % batch_size
        indexs = [int(i) for i in np.linspace(0,trainRows-1,lacks)]    #拼接成一个完整的batch组,linspace均匀插入点坐标(浮点型数据)
        train = np.concatenate( (train,train[indexs]) , axis=0 )
    testRows = test.shape[0]    #sum(testVinMaskRows)
    if testRows % batch_size != 0:
        lacks = batch_size - testRows % batch_size
        indexs = [int(i) for i in np.linspace(0, testRows - 1, lacks)]
        test = np.concatenate( (test, test[indexs]) , axis=0)

    if not os.path.exists(savePathFolder):
        os.makedirs(savePathFolder)
    experiment(train, test, outputLen, testVinMasks, n_epochs, batch_size , testVinMaskRows, savePathFolder)





# print(datetime.datetime.now())
run(outputLen,readPathPre,savePathPre,r'odoMeter')
# run(outputLen,readPathPre,savePathPre,r'capacity')
#############################################################################
# print(datetime.datetime.now())
# fig = plt.figure()
# plt.plot(loss_list, label='loss', color='blue')
# plt.legend()
# plt.title('model loss')
# # plt.savefig(savePathPre + '/capacity_loss.png')
# plt.savefig(savePathPre + '/odoMeter_loss.png')
###############################################################################
# plt.show()
