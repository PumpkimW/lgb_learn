"""
Auther:PumpkinW
Date:2022-01-22
Title:GridSearchCV tuning parameters
"""

import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split,GridSearchCV
import lightgbm as lgb
np.random.seed(2022)

# 步骤1 ：运行以下代码得到训练集和验证集
# 读取数据
data = pd.read_csv("https://cdn.coggle.club/kaggle-flight-delays/flights_10k.csv.zip")
# data.to_csv(r"../data/flight.csv",index=False)

# 提取有用的列
data = data[["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]]
data.dropna(inplace=True)
# 筛选出部分数据
data["ARRIVAL_DELAY"] = (data["ARRIVAL_DELAY"]>10)*1
# 进行编码
cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
for item in cols:
    data[item] = data[item].astype("category").cat.codes +1
# 划分训练集和测试集
train, test, y_train, y_test = train_test_split(data.drop(["ARRIVAL_DELAY"], axis=1),
                                                data["ARRIVAL_DELAY"], random_state=10, test_size=0.25)

# 步骤4 ：学习网格搜索原理，使用GridSearchCV完成其他超参数搜索，其他超参数设置可以选择learning_rate、num_leaves等

param_grid = {
    'learning_rate':[0.01,0.02,0.03,0.04,0.05,0.1],
    'num_leaves':[7,15,31]
}
gbm = lgb.LGBMClassifier(objective = 'binary',
                         metric = 'binary_logloss,auc',
                         max_depth = 3,
                         num_leaves = 31,
                         learning_rate = 0.05,
                         colsample_bytree = 0.9,
                         subsample = 0.8,
                         subsample_freq = 5,
                         n_estimators = 100,
                         random_state=2022
                        )
grearch = GridSearchCV(gbm, param_grid=param_grid, scoring='roc_auc',cv=3,verbose=4,return_train_score=True)
grearch.fit(train, y_train)

if __name__ == '__main__':
    print('start training:')
    for i in range(len(grearch.cv_results_['params'])):
        print(grearch.cv_results_['params'])
        print(grearch.cv_results_['params'][i])
        print(grearch.cv_results_['mean_train_score'])
        print(grearch.cv_results_['mean_test_score'])
    print('参数的最佳取值:{0}'.format(grearch.best_params_))
    print('最佳模型得分:{0}'.format(grearch.best_score_))

# 参数的最佳取值:{'learning_rate': 0.1, 'num_leaves': 15}
# 最佳模型得分:0.737613786362719