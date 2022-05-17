"""
Authoer:王楠
Date:2022-02-20
Title:模型微调与参数衰减
步骤0 ：读取任务5的数据集，并完成数据划分。
步骤1 ：学习使用LightGBM微调的步骤逐步完成1k数据分批次训练，训练集分批次验证集不划分，记录下验证集AUC精度。
步骤2 ：学习使用LightGBM学习率衰减的方法，使用指数衰减&阶梯衰减，记录下验证集AUC精度。
"""

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
import lgb_learn.config as config
np.random.seed(2022)

# 步骤0 ：运行以下代码得到训练集和验证集
# 读取数据
# data = pd.read_csv("https://cdn.coggle.club/kaggle-flight-delays/flights_10k.csv.zip")
data = pd.read_csv(r"../data/flight.csv")
# 提取有用的列
data = data[["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]]
data.dropna(inplace=True) #去掉空值,默认按行，存在任一空值，即删除改行，axis=1,how='any',thresh=None
# 筛选出部分数据
data["ARRIVAL_DELAY"] = (data["ARRIVAL_DELAY"]>10)*1
# 进行编码
cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
for item in cols:
    data[item] = data[item].astype("category").cat.codes +1  #类别编码
# 划分训练集和测试集
train, test, y_train, y_test = train_test_split(data.drop(["ARRIVAL_DELAY"], axis=1),
                                                data["ARRIVAL_DELAY"], random_state=10, test_size=0.25)

# 步骤1 ：学习使用LightGBM微调的步骤逐步完成1k数据分批次训练，训练集分批次验证集不划分，记录下验证集AUC精度。
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'bagging_fraction': 0.85,
        'feature_fraction': 0.76,
        'learning_rate': 0.034,
        'max_depth': 4,
        'num_leaves': 15,
        'verbose': -1,
        'random_state': 2022
    }
#将之前的train，test的index重置，并将test作为测试集进行五折预测
train = train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
test = test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

score_ls = []
for i in range(int(train.shape[0]/1000)-1):
    print(f'{i}:{i + 1}')
    if i == 0:
        # 定义lgb接口的数据集
        lgb_train = lgb.Dataset(train.iloc[:(i+1)*1000,:], y_train[:(i+1)*1000], free_raw_data=False)
        lgb_valid = lgb.Dataset(test, y_test, free_raw_data=False)
        lgb_clf = lgb.train(params=lgb_params,
                            train_set=lgb_train,
                            valid_sets=lgb_valid,
                            num_boost_round=config.num_boost_round,
                            callbacks=[lgb.log_evaluation(10),
                                       lgb.early_stopping(config.early_stopping_rounds)]
                            )
    else:
        lgb_train = lgb.Dataset(train.iloc[(i)*1000:(i+1)*1000,:], y_train[(i)*1000:(i+1)*1000], free_raw_data=False)
        lgb_valid = lgb.Dataset(test, y_test, free_raw_data=False)
        print('starting loading model_'+str(i-1)+'.txt')
        lgb_clf = lgb.train(params=lgb_params,
                            train_set=lgb_train,
                            valid_sets=lgb_valid,
                            num_boost_round=config.num_boost_round,
                            init_model='./model/model_'+str(i-1)+'.txt',
                            callbacks=[lgb.log_evaluation(10),
                                       lgb.early_stopping(config.early_stopping_rounds)]
                            )

    pred = lgb_clf.predict(test,num_iteration=lgb_clf.best_iteration)
    score = roc_auc_score(y_test,pred)
    score_ls.append(score)
    print(f'score:{score}')
    print(f'saving model_{i}.txt')
    lgb_clf.save_model(f'./model/model_{i}.txt')
ret = pd.DataFrame({'score':score_ls})

if __name__ == '__main__':
    # print(train.shape)
    print('starting task:')
    print('task finished')
    print(f'{ret}')
    #     score
    # 0 0.725023
    # 1 0.741800
    # 2 0.741695
    # 3 0.742602
    # 4 0.747406
    # 5 0.751837