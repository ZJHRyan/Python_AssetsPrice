#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:34:47 2021

@author: z
"""


from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input,Dense,LSTM,GRU,BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

start=time.time()

def Quote_change(today,yesterday):
    change = (today-yesterday)/yesterday
    return change

all_0050 = pd.read_csv("0050_all_stock.csv")
tokens = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyMS0wNS0xNCAxNDoxNToxNiIsInVzZXJfaWQiOiJaIiwiaXAiOiIyMjMuMTM3LjIyMi4xNDkifQ.s_o3egQEhgJmIxzxc2y42ozyKWxy6doSsF75uW-ubL8"
url = "https://api.finmindtrade.com/api/v4/data"

b = []


for i in range(0,all_0050.shape[0]):#all_0050.shape[0]
    a=[]
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanStockPrice",
        "data_id": "2330",
        "start_date": "2020-01-01",
        "end_date": "2021-05-29",
        "token": tokens, # 參考登入，獲取金鑰
    }
    a.append(parameter["data_id"])
    resp = requests.get(url, params=parameter)
    data = resp.json()
    data = pd.DataFrame(data["data"])
    
    parameter1 = {
    "dataset": "TaiwanStockPER",
    "data_id": "2330",
    "start_date": "2020-01-01",
    "end_date": "2021-05-30",
    "token": tokens # 參考登入，獲取金鑰
    }
    resp1 = requests.get(url, params=parameter1)
    data1 = resp1.json()
    data1 = pd.DataFrame(data1["data"])
    
    dataset = []
    for i in range(0, len(data)):
        row = []
        for j in range(0, 17):
            row.append(0)
        dataset.append(row)
    dataset[0][0] = "date"
    dataset[0][1] = "stock_id"
    dataset[0][2] = "Trading_Volume"
    dataset[0][3] = "Trading_money"
    dataset[0][4] = "open"
    dataset[0][5] = "max"
    dataset[0][6] = "min"
    dataset[0][7] = "close"
    dataset[0][8] = "Trading_turnover"
    dataset[0][9] = "dividend_yield"
    dataset[0][10] = "PER"
    dataset[0][11] = "PBR"
    dataset[0][12] = "OBV"
    dataset[0][13] = "VVA"
    dataset[0][14] = "PTV"
    dataset[0][15] = "EM"
    dataset[0][16] = "TAPI"
    
    for i in range(1, len(data)):
        dataset[i][0] = data.iloc[i,0]
        dataset[i][1] = float(data.iloc[i,1])
        dataset[i][2] = float(data.iloc[i,2])
        dataset[i][3] = float(data.iloc[i,3])
        dataset[i][4] = float(data.iloc[i,4])
        dataset[i][5] = float(data.iloc[i,5])
        dataset[i][6] = float(data.iloc[i,6])
        dataset[i][7] = float(data.iloc[i,7])
        dataset[i][8] = float(data.iloc[i,9])
        dataset[i][9] = float(data1.iloc[i,2])
        dataset[i][10] = float(data1.iloc[i,3])
        dataset[i][11] = float(data1.iloc[i,4])
    
    for i in range(2, len(dataset)):
        if dataset[i][7] > dataset[i-1][7]: 
            dataset[i][12]=dataset[i-1][12] + dataset[i][2]
        elif dataset[i][7] < dataset[i-1][7]:
            dataset[i][12]=dataset[i-1][12] - dataset[i][2]
        else:
            dataset[i][12]=dataset[i-1][12]
    
    #VVA VVAt=VVAt-1 + (Close - Open) / (High - Low) * Volume
    for i in range(2, len(data)):
        if dataset[i][5] != dataset[i][6]:
            dataset[i][13]=dataset[i-1][13]+(dataset[i][7]-dataset[i][4])/(dataset[i][5]-dataset[i][6])*dataset[i][2]
        else:
            dataset[i][13]=dataset[i-1][13]
    
    #PVT = [((CurrentClose - PreviousClose) / PreviousClose) x Volume] + PreviousPVT
    for i in range(2, len(dataset)):
        if dataset[i-1][7]!=0:
            dataset[i][14]=((dataset[i][7] - dataset[i-1][7])/dataset[i-1][7])*(dataset[i][2])+dataset[i-1][14]
        else:
            dataset[i][14]=dataset[i-1][14]
    
    
    
    #EM=((high[i]+low[i])/2-(high[i-1]+low[i-1])/2)*(high[i]-low[i])/volum[i]
    for i in range(2, len(data)):
        if dataset[i][2] != 0 :    
            dataset[i][15]=((dataset[i][5]+dataset[i][6])/2-(dataset[i-1][5]+dataset[i-1][6])/2)*(dataset[i][5]-dataset[i][6])/dataset[i][2]
        else:
            dataset[i][15]=dataset[i-1][15]
    
    #TAPIｔ=  成交張數ｔ ／　收盤價ｔ
    for i in range(2, len(data)):
        if dataset[i][7]!=0:
            dataset[i][16]=dataset[i][2]/dataset[i][7]
        else:
            dataset[i][16]=dataset[i-1][16]
    
    #delete 0
    datasettemp=pd.DataFrame(dataset)
    datasettemp.columns = dataset[0]
    datasettemp=datasettemp.drop(index=[0,1])
    datasettemp=datasettemp.reset_index(drop = True)
    pd.DataFrame(datasettemp)
    
    #标准化数据集
    outputCol = ['close']#输出列
    inputCol = ['close', 'Trading_turnover','Trading_Volume','dividend_yield','OBV','VVA','PTV','EM','TAPI']#输入列
    X = datasettemp[inputCol]
    Y = datasettemp[outputCol]
    xScaler = StandardScaler()
    yScaler = StandardScaler()
    X = xScaler.fit_transform(X)
    Y = yScaler.fit_transform(Y)
    X[:5,:]
    
    #按时间步组成输入输出集
    timeStep = 5#输入天数
    outStep = 1#输出天数
    xAll = list()
    yAll = list()
    #按时间步整理数据 输入数据尺寸是(timeStep,5) 输出尺寸是(outSize)
    for row in range(datasettemp.shape[0]-timeStep-outStep+1):
        x = X[row:row+timeStep]
        y = Y[row+timeStep:row+timeStep+outStep]
        xAll.append(x)
        yAll.append(y)
    xAll = np.array(xAll).reshape(-1,timeStep,len(inputCol))
    yAll = np.array(yAll).reshape(-1,outStep)
    print('输入集尺寸',xAll.shape)
    print('输出集尺寸',yAll.shape)
    
    #分成测试集，训练集
    splitIndex = -11
    xTrain = xAll[:splitIndex]
    xTest = xAll[splitIndex:]
    yTrain = yAll[:splitIndex]
    yTest = yAll[splitIndex:]
    
    def buildLSTM(timeStep,inputColNum,outStep,learnRate=1e-4):
        '''
        搭建LSTM网络，激活函数为tanh
        timeStep：输入时间步
        inputColNum：输入列数
        outStep：输出时间步
        learnRate：学习率    
        '''
        #输入层
        inputLayer = Input(shape=(timeStep,inputColNum))
    
        #中间层
        middle = LSTM(100,activation='tanh')(inputLayer)
        middle = Dense(100,activation='tanh')(middle)
    
        #输出层 全连接
        outputLayer = Dense(outStep)(middle)
        
        #建模
        model = Model(inputs=inputLayer,outputs=outputLayer)
        optimizer = Adam(lr=learnRate)
        model.compile(optimizer=optimizer,loss='mse') 
        model.summary()
        return model
    
    #搭建LSTM
    lstm = buildLSTM(timeStep=timeStep,inputColNum=len(inputCol),outStep=outStep,learnRate=1e-4)
    
    #训练网络
    epochs = 1000#迭代次数
    batchSize = 500#批处理量
    lstm.fit(xTrain,yTrain,epochs=epochs,verbose=0,batch_size=batchSize)
    
    yPredict = lstm.predict(xTest)
    yPredict = yScaler.inverse_transform(yPredict)[:,0]
    yTest = yScaler.inverse_transform(yTest)[:,0]
    result = {'观测值':yTest,'预测值':yPredict}
    result = pd.DataFrame(result)
    result.index = data.index[timeStep+xTrain.shape[0]:result.shape[0]+timeStep+xTrain.shape[0]]
    
    predition_data = []
    for k in range(0,10):        #抓後10天的資料
        g = []
        g.append(datasettemp.iloc[-10+k,0])
        g.append(result.iloc[k+1,0])
        g.append(result.iloc[k+1,1])
        g.append(Quote_change(result.iloc[k+1,1],result.iloc[k,1]))
        predition_data.append(g)
    
    
        a.append(predition_data)
    
    b.append(a)
        
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


predition_date  = []
for z in range(0,len(b[0][1])):
    predition_date.append(b[0][1][z][0])
    
by_date=[]
for i in range(0,len(predition_date)):
    for n in range(0,len(b)):
        for m in range(0,len(b[0][1])):
            if predition_date[i] == b[n][1][m][0]:   #紀錄時間
                kk = []
                kk.append(predition_date[i])
                kk.append(b[n][0])
                kk.append(b[n][1][m][3])
                print(predition_date[i],b[n][0],b[n][1][m][3])
                by_date.append(kk)


by_date_data = pd.DataFrame(by_date)
pd.DataFrame(by_date_data).columns=["date","stock_id","updown"]  #命名

rank = list(range(1,50+1))

b = pd.DataFrame()
for k in range(0,len(predition_date)):
    fliter = by_date_data["date"] == predition_date[k]
    a = by_date_data[fliter].sort_values(by=['updown'], ascending=False)
    a["rank"] = rank
    b = pd.concat([b,a], ignore_index=True)
del b["updown"]

b.sort_values(by=["date","stock_id"], inplace=True)
b= b.reset_index(drop=True)    #把index重設
b = pd.concat([pd.Series(b.index),b["stock_id"],b["date"],b["rank"]],axis=1)
b.columns.values[0] = "index"
b.head()

AutoML_index_etc=[]
for j in range(0,len(b)):
    AutoML_index_etc.append(str(b["index"][j])+" ; "+str(b["date"][j])+" ; "+str(b["stock_id"][j]))

AutoML = pd.DataFrame()
AutoML["index ; date ; stock_id"] = AutoML_index_etc
AutoML["rank"] = b["rank"]/50

AutoML.to_csv("LSTM1.csv",index=False)

end=time.time()
print('Running time: %s Seconds'%(end-start))
    
    