"""
Authoer:王楠
Date:2022-02-20
Title:模型微调与参数衰减
步骤0 ：读取任务5的数据集，并完成数据划分。
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
        'learning_rate': 0.04,
        'max_depth': 4,
        'num_leaves': 15,
        'verbose': -1,
        'random_state': 2022
    }

# 定义lgb接口的数据集
lgb_train = lgb.Dataset(train, y_train, free_raw_data=False)
lgb_valid = lgb.Dataset(test, y_test, free_raw_data=False)

# 步骤2 ：学习使用LightGBM学习率衰减的方法，使用指数衰减&阶梯衰减，记录下验证集AUC精度。
# 2-1 设置学习率衰减-指数衰减
evals_result1 = {}
lgb_clf1 = lgb.train(params=lgb_params,
                    train_set=lgb_train,
                    valid_sets=lgb_valid,
                    num_boost_round=config.num_boost_round,
                    callbacks=[lgb.log_evaluation(250),
                               lgb.early_stopping(config.early_stopping_rounds),
                               lgb.record_evaluation(evals_result1),
                               lgb.reset_parameter(learning_rate=lambda iter:0.04*(0.998**iter)) #指数衰减
                               ]
                    )

pred1 = lgb_clf1.predict(test,num_iteration=lgb_clf1.best_iteration)
score1 = roc_auc_score(y_test,pred1)
ret1 = pd.DataFrame(evals_result1['valid_0'])
# 2-2 设置学习率衰减-阶梯衰减
evals_result2 = {}
lgb_clf2 = lgb.train(params=lgb_params,
                    train_set=lgb_train,
                    valid_sets=lgb_valid,
                    num_boost_round=config.num_boost_round,
                    callbacks=[ lgb.log_evaluation(250),
                                lgb.record_evaluation(evals_result2),
                                lgb.early_stopping(config.early_stopping_rounds),
                                lgb.reset_parameter(
                                        learning_rate=lambda iter:0.04-0.04/config.num_boost_round if iter%100==0 else 0.04
                                )
                               ]
                    )

pred2 = lgb_clf2.predict(test,num_iteration=lgb_clf2.best_iteration)
score2 = roc_auc_score(y_test,pred2)
ret2 = pd.DataFrame(evals_result2['valid_0'])

#不设置学习率衰减
evals_result3 = {}
lgb_clf3 = lgb.train(params=lgb_params,
                    train_set=lgb_train,
                    valid_sets=lgb_valid,
                    num_boost_round=config.num_boost_round,
                    callbacks=[ lgb.log_evaluation(250),
                                lgb.record_evaluation(evals_result3),
                                lgb.early_stopping(config.early_stopping_rounds)
                               ]
                    )

pred3 = lgb_clf2.predict(test,num_iteration=lgb_clf3.best_iteration)
score3 = roc_auc_score(y_test,pred3)
ret3 = pd.DataFrame(evals_result3['valid_0'])

if __name__ == '__main__':
    # print(train.shape)
    print('starting task')
    print('task finished')
    print(f'学习率指数衰减_score1:{score1}')
    print(f'学习率阶梯衰减_score2:{score2}')
    print(f'不设置学习率衰减_score3:{score3}')
    print(f'{ret1}')
    print(f'{ret2}')
    print(f'{ret3}')
    # 学习率指数衰减_score1:0.7673808983475381
    # 学习率阶梯衰减_score2:0.7656494276060317
    # 不设置学习率衰减_score3:0.7650747222082136