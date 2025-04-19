import datetime
import math
import os
import time
import keras
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
import shutil
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from xgboost import XGBRegressor
import joblib
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, LSTM #ATSLSTM
from keras.utils import plot_model

# region cudaSet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['PYTHONHASHSEED'] = '0'

seed = 7
np.random.seed(seed)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
tf.compat.v1.set_random_seed(1234)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
# endregion

class capacityForecast:
    def __init__(self, dataReadPathPre, dataSavaPathPre, batchSize, nEpochs,outputLen):
        self.batchSize = batchSize
        self.nEpochs = nEpochs
        self.dataReadPathPre = dataReadPathPre
        self.dataSavePathPre = dataSavaPathPre
        self.outputLen = outputLen
        if not os.path.exists(self.dataSavePathPre):
            os.makedirs(self.dataSavePathPre)
        # delStat = False
        # while delStat != True:
        #     try:
        #         if os.path.exists(r'./result/%s' % self.folderName):
        #             shutil.rmtree(r'./result/%s' % self.folderName)
        #         os.mkdir(r'./result/%s' % self.folderName)
        #         delStat = True
        #     except Exception as e:
        #         print(r'./result/%s' % self.folderName + "清理失败", e)
        #         time.sleep(10)
        train = np.empty((0, 660*5 +self.outputLen))  # empty第一维不是0时，随机指定值，非空值；训练集
        # # train = np.delete(train,0,axis=0)	#删除第0行，创建空数组
        self.test = np.empty((0, 660*5 +self.outputLen))  # 测试集，验证训练效果
        # # test = np.delete(test,0,axis=0)
        self.nowStat = np.empty((0, 660*5 +self.outputLen))  # 当前状态输入，预测未来soh
        self.vinMasks = []  #   测试集车辆编码
        self.vinTestCounts = []   #   记录经过输入格式转换后，每台测试集车辆在testScaled中占据的数据行数
        for vinMask in os.listdir(self.dataReadPathPre + r'/capacityTransTrain'):  # 车辆数据集合并
            print(vinMask)
            df = pd.read_csv(filepath_or_buffer=self.dataReadPathPre + r'/capacityTransTrain/%s' % vinMask, header=None)
            dataSet = df.to_numpy()
            train = np.concatenate((train, dataSet[0:]), axis=0)  # dataSet[0:-1]保留最后一行，作为预测值
            # self.test = np.concatenate((self.test, dataSet[-2:-1]), axis=0)  # 验证训练效果
            # self.nowStat = np.concatenate((self.nowStat, dataSet[-1:]), axis=0)  # 二维dataset[-1:]，一维dataset[-1]
        for vinMask in os.listdir(self.dataReadPathPre + r'/capacityTransTest'):
            df = pd.read_csv(filepath_or_buffer = self.dataReadPathPre + r'/capacityTransTest/%s'%vinMask,header=None)
            testSet = df.to_numpy()
            self.test = np.concatenate((self.test , testSet[0:]) , axis=0)
            self.vinTestCounts.append( testSet.shape[0] )
            self.vinMasks.append(vinMask[0:-4]) # 对应车辆编码，和test.scaled保持对应
        np.random.shuffle(train)  # 打乱里程顺序，抑制 过拟合
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # 数据归一化，提高模型收敛速度
        self.scaler = self.scaler.fit(train)  # fit(train)之后的scaler直接transform(test)
        self.trainScaled = self.scaler.transform(train)
        self.testScaled = self.scaler.transform(self.test)
        # self.nowStatScaled = self.scaler.transform(self.nowStat)
        self.model = Sequential()

    def fitLSTM(self):
        X, y = self.trainScaled[:, 0:-1*self.outputLen], self.trainScaled[:,
                                           -1*self.outputLen:]  # 输入数据，每行代表一段充电过程，提取3330个数据，前660组电压、电流、里程、温度，后30个数据是未来30段充电过程的soh
        X = X.reshape(X.shape[0], 660, 5)
        # model = Sequential()
        self.model.add(Conv1D(filters=46, kernel_size=7, strides=4, padding='same', activation='relu',
                              input_shape=(X.shape[1], X.shape[2])))
        # print(model.output_shape)(None, 165, 46)
        self.model.add(MaxPooling1D(pool_size=2, padding='valid'))  # strides默认和pool_size相同，池化后，相邻数据选变成660*2，
        # print(model.output_shape)(None, 82, 46)
        self.model.add(LSTM(24, return_sequences=True))
        # print(model.output_shape)(None, 82, 24)
        self.model.add(LSTM(28, return_sequences=False))
        # print(model.output_shape)(None, 28)
        self.model.add(Dropout(0.0609))

        self.model.add(Dense(self.outputLen))

        adam = optimizers.Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='mean_squared_error', optimizer=adam)
        for i in range(self.nEpochs):  # nb_epoch=153,人为设置的训练次数
            print('Epoch:', i)
            print(datetime.datetime.now())
            history = self.model.fit(X, y, epochs=240, batch_size=self.batchSize, verbose=1, shuffle=True)
            # loss_list.append(history.history['loss'][0])
            print(datetime.datetime.now())
            print(history.history['loss'])
            with open(self.dataSavePathPre + r'/soh_loss.txt', 'a', encoding='utf-8') as f:
                f.write(str(history.history['loss'][0]) + "\n")
            self.model.reset_states()
        # plot_model(self.model, to_file=r'./result/%s/soh_model_structure.png'%self.folderName, show_shapes=True)
        self.model.save(self.dataSavePathPre + r'/soh_model.h5')
        # self.model.save(r'./result/2nd/capacity/soh_model')
    # model.save('path/to/location')
    # model = keras.models.load_model('path/to/location')

    def loadModel(self):
        self.model = keras.models.load_model(self.dataSavePathPre + r'/soh_model.h5')
    def forecast(self):
        print('测试集结果验证中... ')
        count = 0 #计数以区分testScale内不同的车辆
        for j in range(len(self.vinTestCounts)):
            dic = dict()
            for i in range(self.vinTestCounts[j]):
                # 验证测试集效果
                xtestScale = self.testScaled[i+count][0:-1*self.outputLen]  # 最后30个是soh
                ytestScale = self.model.predict(xtestScale.reshape(1, 660, 5),
                                                batch_size=self.batchSize).flatten()  # 模型预测结果
                xtestScale = np.concatenate((xtestScale, ytestScale), axis=0).reshape(1,
                                                                                      -1)  # 预测结果合并实际输入，从归一化数据还原实际soh;list可以相加合并，numpy不适用;或X1.reshape(1,len(X1))
                testPredict = self.scaler.inverse_transform(xtestScale).flatten()[-1*self.outputLen:]  # 验证预测soh结果
                testExpect = self.test[i+count][-1*self.outputLen:]  # 实际soh
                testSmoothModel = Lasso( alpha = 0.01 )
                testSmoothModel.fit(np.array([x for x in range(self.outputLen)]).reshape(-1, 1), np.array(testPredict).reshape(-1, 1))
                testPredictSmooth = testSmoothModel.predict(np.array([x for x in range(self.outputLen)]).reshape(-1, 1))
                # testPredictSmooth.append(   math.sqrt(mse(testExpect, testPredict))   )#最后一位存储误差
                dic[i] = np.concatenate((testExpect, testPredict, testPredictSmooth),   axis=0)

            count += self.vinTestCounts[j]
            print("正在保存车辆%s结果到result文件夹中..."%self.vinMasks[j])
            pd.DataFrame(dic).to_csv(self.dataSavePathPre + '/%s.csv'%self.vinMasks[j], index=False)
            # pd.DataFrame(list(dic_mse.items()), columns=['vin', 'rmse']).to_csv(r'./result/capacity/误差.csv',index=False)
        # region 未来SOH预测
            # 未来soh预测
            # xnowScale = self.nowStatScaled[i][0:-30]
            # ynowScale = self.model.predict(xnowScale.reshape(1, 660, 4), batch_size=self.batchSize).flatten()
            # xnowScale = np.concatenate((xnowScale, ynowScale), axis=0).reshape(1, -1)
            # nowStatPredict = self.scaler.inverse_transform(xnowScale).flatten()[-30:]
            # # 数据平滑处理
            # smoothModel = Lasso(alpha=0.01)
            # smoothModel.fit(np.array([x for x in range(30)]).reshape(-1, 1), np.array(nowStatPredict).reshape(-1, 1))
            # nowStatSooth = smoothModel.predict(np.array([x for x in range(30)]).reshape(-1, 1))
            # bias = nowStatSooth[0] - (sum(testPredictSmooth[-3:]) + sum(testExpect[-3:])) / 6
            # nowStatSooth -= bias

        # endregion
        # return r'./result/%s/result.csv' % self.folderName

class xgbCapacityForecast:
    def __init__(self, dataReadPathPre, dataSavePathPre, outputLen):
        self.dataReadPathPre = dataReadPathPre
        self.dataSavePathPre = dataSavePathPre
        self.outputLen = outputLen
        if not os.path.exists(self.dataSavePathPre):
            os.makedirs(self.dataSavePathPre)

        self.test_size = 0.2  # proportion of dataset to be used as test set
        self.N = 3  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        self.model_seed = 100
        # parameters={'max_depth':range(2,10,1)}

        train = np.empty((0, 660 * 5 + self.outputLen))  # empty第一维不是0时，随机指定值，非空值；训练集
        self.test = np.empty((0, 660 * 5 + self.outputLen))  # 测试集，验证训练效果
        self.nowStat = np.empty((0, 660 * 5 + self.outputLen))  # 当前状态输入，预测未来soh
        self.vinMasks = []  # 测试集车辆编码
        self.vinTestCounts = []  # 记录经过输入格式转换后，每台测试集车辆在testScaled中占据的数据行数
        for vinMask in os.listdir(dataReadPathPre +r'/capacityTransTrain'):  # 车辆数据集合并
            df = pd.read_csv(filepath_or_buffer=dataReadPathPre + r'/capacityTransTrain/%s' % vinMask, header=None)
            dataSet = df.to_numpy()
            train = np.concatenate((train, dataSet[0:]), axis=0)  # dataSet[0:-1]保留最后一行，作为预测值
            # self.test = np.concatenate((self.test, dataSet[-2:-1]), axis=0)  # 验证训练效果
            # self.nowStat = np.concatenate((self.nowStat, dataSet[-1:]), axis=0)  # 二维dataset[-1:]，一维dataset[-1]
        for vinMask in os.listdir(dataReadPathPre + r'/capacityTransTest'):
            df = pd.read_csv(filepath_or_buffer=dataReadPathPre + r'/capacityTransTest/%s' % vinMask, header=None)
            testSet = df.to_numpy()
            self.test = np.concatenate((self.test, testSet[0:]), axis=0)
            self.vinTestCounts.append(testSet.shape[0])
            self.vinMasks.append(vinMask[0:-4])  # 对应车辆编码，和test.scaled保持对应
        np.random.shuffle(train)  # 打乱里程顺序，抑制 过拟合
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # 数据归一化，提高模型收敛速度
        self.scaler = self.scaler.fit(train)  # fit(train)之后的scaler直接transform(test)
        self.trainScaled = self.scaler.transform(train)
        self.testScaled = self.scaler.transform(self.test)
        self.yTrainScaled = None
        self.xTrainScaled = None
        # self.nowStatScaled = self.scaler.transform(self.nowStat)
        self.model = XGBRegressor(seed=self.model_seed,
                                  n_estimators=90,
                                  max_depth=7,
                                  eval_metric='rmse',
                                  learning_rate=0.3,
                                  min_child_weight=12,
                                  subsample=1,
                                  colsample_bytree=1,
                                  colsample_bylevel=1,
                                  gamma=0)
        parameters = {'n_estimators': [90],
                      'max_depth': [7],
                      'learning_rate': [0.3],
                      'min_child_weight': range(5, 21, 1),
                      }
        self.gs = GridSearchCV(estimator=self.model, param_grid=parameters, cv=5, refit=True,
                               scoring='neg_mean_squared_error')
    def fitXGboost(self):
        self.xTrainScaled, self.yTrainScaled = self.trainScaled[:, 0:-1 * self.outputLen], self.trainScaled[:, -1 * self.outputLen:]  # 输入数据，每行代表一段充电过程，提取3330个数据，前660组电压、电流、里程、温度，后30个数据是未来30段充电过程的soh
        self.gs.fit(self.xTrainScaled, self.yTrainScaled)
        print('最优参数: ' + str(self.gs.best_params_))
        # self.model.save(self.dataSavePathPre + r'/xgb_model.xgb')

    def loadModel(self):
        self.model = self.model.load_model(self.dataSavePathPre + r'/xgb_model.xgb')    #待修改,或者使用pickledump保存模型

    def forecast(self):
        # est_scaled = self.gs.predict(self.X_train_scaled)
        # train['est'] = est_scaled * math.sqrt(scaler.var_[0]) + scaler.mean_[0]
        X_test_scaled,y_test_scaled = self.testScaled[:, 0:-1 * self.outputLen], self.testScaled[:,-1 * self.outputLen:]
        predict_y_test_scaled = self.gs.predict(X_test_scaled)
        """按照每台车辆输出预测结果"""
        predictTestScaled = np.concatenate((X_test_scaled,predict_y_test_scaled),axis=1)
        testPredicts = self.scaler.inverse_transform(predictTestScaled)
        count = 0  # 计数以区分testScale内不同的车辆
        for j in range(len(self.vinTestCounts)):    #共j辆车
            dic = dict()
            for i in range(self.vinTestCounts[j]):
                # testPredict = self.scaler.inverse_transform(predictTestScaled[i + count]).flatten()[-1*self.outputLen:]
                testPredict = testPredicts[i + count][-1 * self.outputLen:]
                testExpect = self.test[i + count][-1 * self.outputLen:] #实际值
                dic[i] = np.concatenate((testExpect, testPredict), axis=0)
            count += self.vinTestCounts[j]
            print("正在保存车辆%s结果到result文件夹中..." % self.vinMasks[j])
            pd.DataFrame(dic).to_csv(self.dataSavePathPre + '/%s.csv' % self.vinMasks[j], index=False)

# region CNN-ALSTM模型预测
# caForecast = capacityForecast(r'./data/discharge/2nd',r'./result/2nd/capacity',14,142,30) #输入1次，输出30段，未使用降噪 (batchsize,nEpochs,outputlen)
# caForecast = capacityForecast(r'./data/discharge/4th',r'./result/4th/capacity',14,142,30) #输入1次，输出30段，使用lasso降噪15%
# caForecast = capacityForecast(r'./data/discharge/4th_leastSquareReduce',r'./result/4th_leastSquareReduce/capacity',14,142,30) #输入1次，输出30段，使用最小二乘降噪15%
# caForecast = capacityForecast(r'./data/discharge/4th_maxminReduce',r'./result/4th_maxminReduce/capacity',14,142,30) #输入1次，输出30段，使用去除极值降噪15%

# caForecast = capacityForecast(r'./data/discharge/5th',r'./result/5th/capacity',14,142,20)
# caForecast = capacityForecast(r'./data/discharge/5th_lassoReduce',r'./result/5th_lassoReduce/capacity',14,142,20)
# caForecast = capacityForecast(r'./data/discharge/5th_leastSquareReduce',r'./result/5th_leastSquareReduce/capacity',14,142,20)
# caForecast = capacityForecast(r'./data/discharge/5th_maxminReduce',r'./result/5th_maxminReduce/capacity',14,142,20)

# caForecast = capacityForecast(r'./data/discharge/6th',r'./result/6th/capacity',14,142,25)
# caForecast = capacityForecast(r'./data/discharge/6th_lassoReduce',r'./result/6th_lassoReduce/capacity',14,142,25)
# caForecast = capacityForecast(r'./data/discharge/6th_leastSquareReduce',r'./result/6th_leastSquareReduce/capacity',14,142,25)
# caForecast = capacityForecast(r'./data/discharge/6th_maxminReduce',r'./result/6th_maxminReduce/capacity',14,142,25)

# caForecast = capacityForecast(r'./data/discharge/7th',r'./result/7th/capacity',14,142,16)
# caForecast = capacityForecast(r'./data/discharge/7th_lassoReduce',r'./result/7th_lassoReduce/capacity',14,142,16)
# caForecast = capacityForecast(r'./data/discharge/7th_leastSquareReduce',r'./result/7th_leastSquareReduce/capacity',14,142,16)
# caForecast = capacityForecast(r'./data/discharge/7th_maxminReduce',r'./result/7th_maxminReduce/capacity',14,142,16)

#region 批量运行
# counts = ['4th']
# predLens = [30]
#
# # noiseTypes = ['lassoReduce','leastSquareReduce','maxminReduce']   4th_leastSquareReduce_0.25
# noiseTypes = ['maxminReduce']
# predLengths = dict(zip(counts,predLens))
#
# for count in counts:
#     for noiseType in noiseTypes:
#         for noiseRatio in ['0.35']:
#             path = f'./data/discharge/{count}_{noiseType}_{noiseRatio}'
#             if count == '6th' and ( noiseType != 'maxminReduce' or noiseRatio != '0.35' ) :
#                 print(f'Train is skipped for: {path}')
#                 continue
#             print(f'Train is start for: {path}')
#             caForecast = capacityForecast(path,f'./result/{count}_{noiseType}_{noiseRatio}/capacity',14,142,predLengths[count])
#             caForecast.fitLSTM()
#             caForecast.forecast()
#             del caForecast
#             print(f'Train is over for: {path}')
#endregion

#region  论文补做实验 SG-filter reduction

# caForecast = capacityForecast(r'./data/discharge/4th_SGReduce',r'./result/4th_SGReduce/capacity',16,1,30) #输入1次，输出30段，使用去除极值降噪15%
# caForecast = capacityForecast(r'./data/discharge/4th_SGReduce_0.25',r'./result/4th_SGReduce_0.25/capacity',16,1,30) #输入1次，输出30段，使用去除极值降噪15%
# caForecast = capacityForecast(r'./data/discharge/4th_SGReduce_0.35',r'./result/4th_SGReduce_0.35/capacity',16,1,30) #输入1次，输出30段，使用去除极值降噪15%
# caForecast.fitLSTM()
# caForecast.forecast()

#endregion



# caForecast.loadModel()
# endregion

#region xgbBoost模型预测
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/2nd',r'./xgbResult/2nd/capacity',30)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/3th',r'./xgbResult/3th/capacity',30)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/4th',r'./xgbResult/4th/capacity',30)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/4th_leastSquareReduce',r'./xgbResult/4th_leastSquareReduce/capacity',30)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/4th_maxminReduce',r'./xgbResult/4th_maxminReduce/capacity',30)
#

xgbForecast = xgbCapacityForecast(r'./data/discharge/4th_SGReduce',r'./xgbResult/4th_SGReduce/capacity',30)
# xgbForecast = xgbCapacityForecast(r'./data/discharge/4th_SGReduce_0.25',r'./xgbResult/4th_SGReduce_0.25/capacity',30)
# xgbForecast = xgbCapacityForecast(r'./data/discharge/4th_SGReduce_0.35',r'./xgbResult/4th_SGReduce_0.35/capacity',30)
# xgbForecast.fitXGboost()
# xgbForecast.forecast()

# # xgbForecast = xgbCapacityForecast(r'./data/discharge/5th',r'./xgbResult/5th/capacity',20)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/5th_lassoReduce',r'./xgbResult/5th_lassoReduce/capacity',20)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/5th_leastSquareReduce',r'./xgbResult/5th_leastSquareReduce/capacity',20)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/5th_maxminReduce',r'./xgbResult/5th_maxminReduce/capacity',20)
#
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/6th',r'./xgbResult/6th/capacity',25)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/6th_lassoReduce',r'./xgbResult/6th_lassoReduce/capacity',25)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/6th_leastSquareReduce',r'./xgbResult/6th_leastSquareReduce/capacity',25)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/6th_maxminReduce',r'./xgbResult/6th_maxminReduce/capacity',25)
#
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/7th',r'./xgbResult/7th/capacity',16)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/7th_lassoReduce',r'./xgbResult/7th_lassoReduce/capacity',16)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/7th_leastSquareReduce',r'./xgbResult/7th_leastSquareReduce/capacity',16)
# # xgbForecast = xgbCapacityForecast(r'./data/discharge/7th_maxminReduce',r'./xgbResult/7th_maxminReduce/capacity',16)
#
# # xgbForecast1 = xgbCapacityForecast(r'./data/discharge/7th_lassoReduce_0.25',r'./xgbResult/7th_lassoReduce_0.25/capacity',16)
# # xgbForecast2 = xgbCapacityForecast(r'./data/discharge/7th_leastSquareReduce_0.25',r'./xgbResult/7th_leastSquareReduce_0.25/capacity',16)
# # xgbForecast3 = xgbCapacityForecast(r'./data/discharge/7th_maxminReduce_0.25',r'./xgbResult/7th_maxminReduce_0.25/capacity',16)
# # xgbForecast4 = xgbCapacityForecast(r'./data/discharge/7th_lassoReduce_0.35',r'./xgbResult/7th_lassoReduce_0.35/capacity',16)
# # xgbForecast5 = xgbCapacityForecast(r'./data/discharge/7th_leastSquareReduce_0.35',r'./xgbResult/7th_leastSquareReduce_0.35/capacity',16)
# # xgbForecast6 = xgbCapacityForecast(r'./data/discharge/7th_maxminReduce_0.35',r'./xgbResult/7th_maxminReduce_0.35/capacity',16)
#
# xgbForecast1 = xgbCapacityForecast(r'./data/discharge/6th_lassoReduce_0.25',r'./xgbResult/6th_lassoReduce_0.25/capacity',25)
# xgbForecast2 = xgbCapacityForecast(r'./data/discharge/6th_leastSquareReduce_0.25',r'./xgbResult/6th_leastSquareReduce_0.25/capacity',25)
# xgbForecast3 = xgbCapacityForecast(r'./data/discharge/6th_maxminReduce_0.25',r'./xgbResult/6th_maxminReduce_0.25/capacity',25)
# xgbForecast4 = xgbCapacityForecast(r'./data/discharge/6th_lassoReduce_0.35',r'./xgbResult/6th_lassoReduce_0.35/capacity',25)
# xgbForecast5 = xgbCapacityForecast(r'./data/discharge/6th_leastSquareReduce_0.35',r'./xgbResult/6th_leastSquareReduce_0.35/capacity',25)
# xgbForecast6 = xgbCapacityForecast(r'./data/discharge/6th_maxminReduce_0.35',r'./xgbResult/6th_maxminReduce_0.35/capacity',25)
#
# print("---------开始训练模型--------------")
# print(datetime.datetime.now())
# xgbForecast1.fitXGboost()
# xgbForecast1.forecast()
# print("---------模型训练结束1--------------")
# xgbForecast2.fitXGboost()
# xgbForecast2.forecast()
# print("---------模型训练结束2--------------")
# xgbForecast3.fitXGboost()
# xgbForecast3.forecast()
# print("---------模型训练结束3--------------")
# xgbForecast4.fitXGboost()
# xgbForecast4.forecast()
# print("---------模型训练结束4--------------")
# xgbForecast5.fitXGboost()
# xgbForecast5.forecast()
# print("---------模型训练结束5--------------")
# xgbForecast6.fitXGboost()
# xgbForecast6.forecast()
# print("---------模型训练结束6--------------")
# print(datetime.datetime.now())

Pic1 = ['4th', '6th_lassoReduce', '5th_lassoReduce_0.25', '7th_lassoReduce_0.35',
        '4th_leastSquareReduce', '6th_leastSquareReduce', '5th_leastSquareReduce_0.25',
        '7th_leastSquareReduce_0.35',
        '4th_maxminReduce', '6th_maxminReduce', '5th_maxminReduce_0.25', '7th_maxminReduce_0.35'
        ]
Pic2 = ['4th_lassoReduce_0.35', '6th_lassoReduce_0.25', '5th_lassoReduce', '7th_lassoReduce',
        '4th_leastSquareReduce_0.35', '6th_leastSquareReduce_0.25', '5th_leastSquareReduce',
        '7th_leastSquareReduce',
        '4th_maxminReduce_0.35', '6th_maxminReduce_0.25', '5th_maxminReduce', '7th_maxminReduce'
        ]


# Pic = ['4th_maxminReduce_0.35', '6th_maxminReduce_0.25']
# predlen = [30,25]
# for i in range(len(Pic)):
#     print(datetime.datetime.now())
    # xgbForecast = xgbCapacityForecast(f'./data/discharge/{Pic[i]}',
    #                                    f'./xgbResult/{Pic[i]}/capacity', predlen[i])
    # xgbForecast.fitXGboost()
    # xgbForecast.forecast()
    # del xgbForecast
    # print(Pic[i],predlen[i],r'Training Over')






data = ['2nd', '4th', '4th_lassoReduce_0.25', '4th_lassoReduce_0.35', '4th_leastSquareReduce',
        '4th_leastSquareReduce_0.25', '4th_leastSquareReduce_0.35', '4th_maxminReduce', '4th_maxminReduce_0.25',
        '4th_maxminReduce_0.35',
        '5th', '5th_lassoReduce', '5th_lassoReduce_0.25', '5th_lassoReduce_0.35', '5th_leastSquareReduce',
        '5th_leastSquareReduce_0.25', '5th_leastSquareReduce_0.35', '5th_maxminReduce', '5th_maxminReduce_0.25',
        '5th_maxminReduce_0.35',
        '6th', '6th_lassoReduce', '6th_lassoReduce_0.25', '6th_lassoReduce_0.35', '6th_leastSquareReduce',
        '6th_leastSquareReduce_0.25', '6th_leastSquareReduce_0.35', '6th_maxminReduce', '6th_maxminReduce_0.25',
        '6th_maxminReduce_0.35',
        '7th', '7th_lassoReduce', '7th_lassoReduce_0.25', '7th_lassoReduce_0.35', '7th_leastSquareReduce',
        '7th_leastSquareReduce_0.25', '7th_leastSquareReduce_0.35', '7th_maxminReduce', '7th_maxminReduce_0.25',
        '7th_maxminReduce_0.35']
preLength = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
             20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
             25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
             16, 16, 16, 16, 16, 16, 16, 16, 16, 16]



# endregion