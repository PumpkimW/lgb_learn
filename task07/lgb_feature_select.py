"""
Author:王楠
Date:2022-01-01
Title:特征筛选方法
步骤0 ：读取任务5的数据集，并完成数据划分。
步骤1 ：使用LightGBM计算特征重要性，并筛选最重要的3个特征。
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

# 步骤1 ：基于lgb自带的特征重要性计算方式筛选特征
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
                    callbacks=[lgb.log_evaluation(10),
                               lgb.early_stopping(config.early_stopping_rounds),
                               lgb.record_evaluation(evals_result),
                               lgb.reset_parameter(learning_rate=lambda iter:0.04*(0.998**iter)) #指数衰减
                               ]
                    )

pred = lgb_clf.predict(test,num_iteration=lgb_clf.best_iteration)
score = roc_auc_score(y_test,pred)
ret = pd.DataFrame(evals_result['valid_0'])

#输出特征重要性
"""
importance_type : str, optional (default="split")
How the importance is calculated.
If "split", result contains numbers of times the feature is used in a model.
If "gain", result contains total gains of splits which use the feature.
"""
fea_imp_dict = {'feature_name':lgb_clf.feature_name(),
           'feature_importance':lgb_clf.feature_importance(importance_type='gain')}
fea_imp = pd.DataFrame(fea_imp_dict).sort_values(by='feature_importance',ascending=False,ignore_index=True)

if __name__ == '__main__':
    print((fea_imp[:3]))

    """  
    split:
    feature_name     feature_importance
    0 DESTINATION_AIRPORT 448
    1 DEPARTURE_TIME    438
    2 ORIGIN_AIRPORT    404
    """

    """   
    gain:      
    feature_name  feature_importance
    0       DEPARTURE_TIME         6684.322787
    1  DESTINATION_AIRPORT         5129.102557
    2       ORIGIN_AIRPORT         3143.066795
    """