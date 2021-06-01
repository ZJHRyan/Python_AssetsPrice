#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:44:21 2021

@author: z
"""

import requests
import pandas as pd
from sklearn import preprocessing
from autogluon.tabular import TabularDataset, TabularPredictor
from datetime import datetime
import time
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
        "data_id": "{}".format(all_0050.iloc[i,1]),
        "start_date": "2020-01-01",
        "end_date": "2021-03-18",
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
    "end_date": "2021-03-18",
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
    
    '''
    OBVt = OBVt-1 + Volume      IF Closet > Closet-1
    OBVt = OBVt-1 - Volume       IF Closet < Closet-1
    '''
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
    
    #二阶差分difference
    difference=datasettemp.drop(columns=['date','stock_id','close']).diff(periods=2)
    difference=difference.drop(index=[0,1])
    difference=difference.reset_index(drop = True)
    
    #Z-Score
    values = difference.values #dataframe转换为array
    values = values.astype('float32') #定义数据类型
    data_zscore = preprocessing.scale(values) 
    df_zscore=pd.DataFrame(data_zscore) #将array还原为dataframe
    df_zscore.columns = difference.columns
    
    #final data
    temp=datasettemp.drop(index=[0,1])
    temp=temp.reset_index(drop = True)
    temp
    data_final = temp.loc[:,['date','stock_id','close']].join(df_zscore)
    #data_final = data_final.drop(columns=[''])
    
    '''
    #adjust data move 1odays
    temp_adjust=data_final.iloc[:-10,3:]
    temp_adjust=temp_adjust.reset_index(drop = True)
    date_adjust=data_final.iloc[10:,:3]
    date_adjust=date_adjust.reset_index(drop = True)
    data_adjust=date_adjust.join(temp_adjust)
    data_adjust=data_adjust.reset_index(drop = True)
    '''
    #train
    train_data=TabularDataset(datasettemp.drop(columns=['Trading_money','open','max','min','PER','PBR'])
                              .iloc[:-11])
    
    #predictor
    predictor = TabularPredictor(label='close').fit(train_data.drop(columns=['date','stock_id']))
                                                   # , num_stack_levels=1,num_bag_folds=2)
    
    #test
    test_data=datasettemp.iloc[-11:len(datasettemp)]
    preds=predictor.predict(test_data.drop(columns=['date','stock_id','close']))
    test_hat=pd.DataFrame({'date':test_data['date'],'stock_id':test_data['stock_id'],'close':preds})
    test_hat
    
    
    predition_data = []
    for k in range(0,10):        #抓後10天的資料
        g = []
        g.append(test_hat.iloc[k,0])
        g.append(test_hat.iloc[k,2])
        g.append(datasettemp.iloc[k-10,2])
        g.append(Quote_change(test_hat.iloc[k+1,2],test_hat.iloc[k,2]))
        predition_data.append(g)
    predition_data
    
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

AutoML.to_csv("AutoML18.csv",index=False)

end=time.time()
print('Running time: %s Seconds'%(end-start))
    
    
    
    