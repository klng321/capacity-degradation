import datetime

from data.data_loader import Dataset_Custom
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
# import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    # 继承Exp_Basic
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    # 函数重载 self.model = self._build_model().to(self.device)
    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, dataType):    #dataType (train test vali)
        args = self.args
        Data = Dataset_Custom  # 类重命名
        timeenc = 0 if args.embed != 'timeF' else 1
        drop_last = True    # drop_last: 对不足一个batch的数据丢弃; freq: 以h还是min还是s为单位处理数据
        batch_size = args.batch_size
        freq = args.freq
        dataPath = args.train_file_folder
        if dataType == 'test':
            shuffle_flag = False
            dataPath = args.test_file_folder
        else:   #train和vali 需要打乱遍历顺序# 当遍历返回的 dataloader 时，使用的 index 下标乱序排序
            shuffle_flag = True
        data_set = Data(
            root_path=args.root_path, data_path=dataPath,
            size=[args.seq_len, args.label_len, args.pred_len],features=args.features,
            target=args.target, timeenc=timeenc, freq=freq, cols=args.cols,
            SOC=args.SOC,label=args.label, begin=args.begin, valiRatio=args.vali_Ratio, flag=dataType
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (singleProcessScaled, singleProcess) in enumerate(vali_loader):
            targetScaled = singleProcessScaled[0][-1 * self.args.pred_len:]
            target = singleProcess[0][-1 * self.args.pred_len:]
            disChargeDataScaled = np.resize(singleProcessScaled[0][:-1 * self.args.pred_len], (1, 660, 5))
            batch_x = disChargeDataScaled[:, :660 - self.args.pred_len, :]
            batch_y = disChargeDataScaled[:, 660 - self.args.label_len - self.args.pred_len:, :]
            df_stamp = range(660)  # 660组电池数据
            data_stamp = np.expand_dims((df_stamp - np.mean(df_stamp)) / np.std(df_stamp),
                                        axis=1)  # 实现standardScaler，扩充数据维度
            data_stamp = np.expand_dims(data_stamp, axis=0)
            predictScaled = self._process_one_batch(batch_x, batch_y, data_stamp[:, :660 - self.args.pred_len, :],
                                                    data_stamp[:, 660 - self.args.pred_len - self.args.label_len:, :])
            targetScaled = targetScaled.to(self.device).float()
            loss = criterion(predictScaled.detach().cpu(), targetScaled.detach().cpu())
            total_loss.append(loss)
        avg_loss = np.average(total_loss)
        self.model.train()
        return avg_loss

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('vali')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(self.args.data_save_folder):
            os.makedirs(self.args.data_save_folder)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        '''# 分布式，Linux加
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()'''
        train_loss = dict()
        for epoch in range(self.args.epochs):
            tmpStr = 'epoch'+str(epoch+1)
            train_loss[tmpStr] = list()
            train_loss[tmpStr].append(datetime.datetime.now())
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (singleProcessScaled,singleProcess) in enumerate(train_loader):   #每次返回660组放电过程数据调用dataloader.__getitem() {return self.dataSetScale[index], self.dataSet[index]}  函数；每一批有batchSize个矩阵，相邻矩阵采用滑动窗口策略，步长为1； 每一层循环生成的batch起始地址以batchsize等差数列递进
                # print(singleProcessScaled[0][1])
                # print(singleProcessScaled[:-1*self.args.pred_len])
                targetScaled = singleProcessScaled[0][-1*self.args.pred_len:]
                target = singleProcess[0][-1*self.args.pred_len:]
                disChargeDataScaled = np.resize( singleProcessScaled[0][:-1 * self.args.pred_len],(1,660,5) )
                batch_x = disChargeDataScaled[:,:660-self.args.pred_len,:]
                batch_y = disChargeDataScaled[:,660 - self.args.label_len - self.args.pred_len:,:]
                df_stamp = range(660)  # 660组电池数据
                data_stamp = np.expand_dims((df_stamp - np.mean(df_stamp)) / np.std(df_stamp), axis=1)  # 实现standardScaler，扩充数据维度
                data_stamp = np.expand_dims(data_stamp, axis=0)
                predictScaled = self._process_one_batch( batch_x, batch_y, data_stamp[:,:660-self.args.pred_len,:], data_stamp[:,660-self.args.pred_len-self.args.label_len:,:])
                iter_count += 1
                model_optim.zero_grad()
                # print(predictScaled.device) # device cuda0
                # print(targetScaled.device)  # device cpu
                targetScaled = targetScaled.to(self.device).float()
                # print(targetScaled.device)
                # print(targetScaled.type())
                # print(predictScaled.type())
                loss = torch.sqrt(criterion(predictScaled, targetScaled ))
                train_loss[tmpStr].append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss_mean = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            early_stopping(vali_loss, self.model, path)  #  实例化对象early_stopping当作函数添加参数，调用类内__call()__函数， 存储训练模型
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        best_model_path = path + '/' + 'checkpoint.pth'
        # loss_path = path + '/' + 'loss.csv'
        self.model.load_state_dict(torch.load(best_model_path,map_location={'cuda:0':self.args.device_select}))
        # train_loss.to_csv(loss_path)
        return pd.DataFrame(train_loss, index=None)

    def test(self):
        test_data, test_loader = self._get_data('test')
        self.model.eval()
        preds = []
        trues = []
        dic = dict()
        j = 0   #  每辆车内部充放电周期的遍历下标
        k = 0   #  每辆车的遍历下标
        count = test_data.vinNumCounts[k]   #车辆计数

        for i, (singleProcessScaled, singleProcess) in enumerate(test_loader):
            # print(singleProcessScaled[0][1])
            # print(singleProcessScaled[:-1*self.args.pred_len])
            # targetScaled = singleProcessScaled[0][-1 * self.args.pred_len:]
            dataScaled = singleProcessScaled[0][:-1 * self.args.pred_len].to(self.args.device_select)   #   拆分单次放电过程，划分训练数据和预测结果
            target = singleProcess[0][-1 * self.args.pred_len:].to(self.args.device_select)
            disChargeDataScaled = np.resize(singleProcessScaled[0][:-1 * self.args.pred_len], (1, 660, 5))
            batch_x = disChargeDataScaled[:, :660 - self.args.pred_len, :]
            batch_y = disChargeDataScaled[:, 660 - self.args.label_len - self.args.pred_len:, :]
            df_stamp = range(660)  # 660组电池数据
            data_stamp = np.expand_dims((df_stamp - np.mean(df_stamp)) / np.std(df_stamp),
                                        axis=1)  # 实现standardScaler，扩充数据维度
            data_stamp = np.expand_dims(data_stamp, axis=0)
            predictScaled = self._process_one_batch(batch_x, batch_y, data_stamp[:, :660 - self.args.pred_len, :],
                                                    data_stamp[:, 660 - self.args.pred_len - self.args.label_len:, :])
            predict = test_data.inverse_transform( torch.cat((dataScaled,predictScaled.resize(self.args.pred_len)),dim=0) )[-1*self.args.pred_len:]
            dic[j] = torch.cat((target, predict), dim=0).cpu().detach().numpy() #GPU取出到内存，转numpy
            if i == count-1:    #同一辆车充放电过程数据存储到硬盘
                if not os.path.exists(self.args.data_save_folder):
                    os.makedirs(self.args.data_save_folder)
                pd.DataFrame(dic).to_csv(self.args.data_save_folder + '/%s.csv'%test_data.vinMasks[k],index=False)
                k += 1
                j = 0
                dic.clear()
                try:
                    count += test_data.vinNumCounts[k]
                except:pass # 最后一辆车k自增后越界
                finally:
                    print("测试集车辆%s数据已缓存"%test_data.vinMasks[k-1])
            else:
                j += 1






    def _process_one_batch(self,  batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = torch.from_numpy(batch_x.astype(np.float32)).to(self.device)
        # batch_y = batch_y.float()
        batch_y = torch.from_numpy(batch_y.astype(np.float32)).to(self.device)

        batch_x_mark = torch.from_numpy(batch_x_mark.astype(np.float32)).to(self.device)
        batch_y_mark = torch.from_numpy(batch_y_mark.astype(np.float32)).to(self.device)

        # # decoder input
        # if self.args.padding == 0:
        #     dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float() #shape[32,24,4]
        # else:
        #     dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # batch_y[:, :self.args.label_len, :] = [batch, label_len, feature] 第1,3个维度全取,第2个维度取前label_len=48个
        # label_len个SOC 和 pred_len个0拼接，48+24，dim=1即第1个维度拼接
        # dec_inp: [batch, label_len+pred_len, feature_size]
        if self.args.SOC:    # 如果SOC作为特征值输入网络
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], batch_y], dim=1).float().to(self.device)
        else:
            # 去掉SOC特征
            # dec_inp = batch_y[:, :, :-1].float().to(self.device)    #去掉第三个维度，最后一位指SOC，略去不使用
            # batch_x = batch_x[:, :, :-1].float().to(self.device)
            print(batch_x.shape)
        # encoder - decoder

        if self.args.use_amp:  # 分布式Linux
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:  #      windows       model结构再model.py中    执行forward
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

        # f_dim = -1 if self.args.features == 'MS' else 0
        # # batch_y: [batch, label_len+pred_len, 1]
        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # batch_y尚未删除SOC值
        # # batch_y = batch_y[:, -1, f_dim:].to(self.device)

        return outputs

    def Model_load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
