"""
Auther:PumpkinW
Date:2022-01-22
Title:模型调参（网格、随机、贝叶斯）
步骤1 ：运行以下代码得到训练集和验证集
步骤2 ：构建LightGBM分类器，并设置树模型深度分别为[3,5,6,9]，设置训练集和验证集，分别记录下验证集AUC精度
步骤3 ：构建LightGBM分类器，在fit函数中将category变量设置为categorical_feature，训练并记录下分别记录下验证集AUC精度
步骤4 ：学习网格搜索原理，使用GridSearchCV完成其他超参数搜索，其他超参数设置可以选择learning_rate、num_leaves等
步骤5 ：学习随机搜索原理，使用GridSearchCV完成其他超参数搜索，其他超参数设置可以选择
步骤6 ：学习贝叶斯调参原理，使用BayesianOptimization完成超参数搜索，具体过程可以参考
"""
import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import auc
import lightgbm as lgb
np.random.seed(2022)

# 步骤1 ：运行以下代码得到训练集和验证集
# 读取数据
data = pd.read_csv("https://cdn.coggle.club/kaggle-flight-delays/flights_10k.csv.zip")
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

# 步骤2 ：构建LightGBM分类器，并设置树模型深度分别为[3,5,6,9]，设置训练集和验证集，分别记录下验证集AUC精度。
max_depth = [3,5,6,9]
param = {
    'boosting_type': 'gbdt',
    'objective' : 'binary',
    'metric':'auc',
    'max_depth':3,
    'num_leaves' : 15,
    'learning_rate':0.1,
    'feature_fraction':0.9,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'verbose':-1,
    'random_state':2022
}

lgb_train = lgb.Dataset(train,label=y_train,free_raw_data=False)
lgb_test = lgb.Dataset(test,label=y_test,free_raw_data=False)

auc_list = []
# 步骤3 ：构建LightGBM分类器，在fit函数中将category变量设置为categorical_feature，训练并记录下分别记录下验证集AUC精度
categorical_feature = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]

for mx_dp in max_depth:
    param['max_depth'] = mx_dp

    lgb_clf = lgb.train(params=param,
                        train_set=lgb_train,
                        num_boost_round=100,
                        valid_sets=lgb_test,
                        categorical_feature=categorical_feature,
                        callbacks=[lgb.log_evaluation(10)]
                        )
    pred = lgb_clf.predict(test)
    print('max_depth:',mx_dp)
    print('auc',lgb_clf.best_score.values())
    auc_list.append(dict(lgb_clf.best_score['valid_0']).get('auc'))



if __name__ == '__main__':
    print('start training:')
    pass
    # print(dict(lgb_clf.best_score['valid_0']))
    # print('set max_depth:')
    # print(f'max_depth:{max_depth}')
    # print(f'auc_list:{auc_list}')
    # max_depth:[3, 5, 6, 9]
    # auc_list:[0.7514866693563299, 0.7557496806024846, 0.7582518029149226, 0.7588611797535596]

    # print('add categorical_feature:')
    # print(f'max_depth:{max_depth}')
    # print(f'auc_list:{auc_list}')
    # max_depth:[3, 5, 6, 9]
    # auc_list:[0.7614536579420714, 0.7720268714172844, 0.7711411737018172, 0.7729230756299695]

    # print('after  grearch learing_rate=0.1,num_leaves=15:')
    # print(f'max_depth:{max_depth}')
    # print(f'auc_list:{auc_list}')
    # max_depth:[3, 5, 6, 9]
    # auc_list:[0.7669369988400827, 0.7618129801469229, 0.7657334795837746, 0.7644726999176291]