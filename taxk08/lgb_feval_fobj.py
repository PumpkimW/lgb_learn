"""
Author:王楠
Date:2022-01-01
Title:自定义损失函数
https://gitee.com/mirrors/lightgbm/blob/master/examples/python-guide/advanced_example.py
步骤0 ：读取任务5的数据集，并完成数据划分。
步骤1 ：自定义损失函数，预测概率小于0.1的正样本（标签为正样本，但模型预测概率小于0.1），梯度增加一倍。
步骤2 ：自定义评价函数，阈值大于0.8视为正样本（标签为正样本，但模型预测概率大于0.8）。
"""
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
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


# 步骤1 ：自定义损失函数，预测概率小于0.1的正样本（标签为正样本，但模型预测概率小于0.1），梯度增加一倍。
# 自定义损失函数（fobj）

def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess
def binary_error(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'error', np.mean(labels != (preds > 0.5)), False

def accuracy(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'accuracy', np.mean(labels == (preds > 0.5)), True


# 定义参数字典
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
# 定义lgb接口的数据集
lgb_train = lgb.Dataset(train, y_train, free_raw_data=False)
lgb_valid = lgb.Dataset(test, y_test, free_raw_data=False)

evals_result = {}
lgb_clf = lgb.train(params=lgb_params,
                    train_set=lgb_train,
                    valid_sets=lgb_valid,
                    num_boost_round=config.num_boost_round,
                    early_stopping_rounds = 50,
                    fobj=loglikelihood,
                    feval=[binary_error,accuracy],
                    callbacks=[lgb.log_evaluation(10),
                               # lgb.early_stopping(config.early_stopping_rounds), #0.7666811656328273
                               lgb.record_evaluation(evals_result),
                               lgb.reset_parameter(learning_rate=lambda iter:0.04*(0.998**iter)) #指数衰减
                               ]
                    )

pred = lgb_clf.predict(test,num_iteration=lgb_clf.best_iteration)
score = roc_auc_score(y_test,pred)

if __name__ == '__main__':
    print(score)
    print(lgb_clf.params)
    print(lgb_clf.best_iteration)