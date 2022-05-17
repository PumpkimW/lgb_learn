"""
@Author:PumpkinW
Date:2022-01-22
Title:回归任务
步骤1 ：学习LightGBM中sklearn接口的使用，导入分类、回归和排序接口。
步骤2 ：学习LightGBM中原生train接口的使用。
步骤5 ：回归任务
使用make_regression，创建一个回归数据集。
使用sklearn接口完成训练和预测。
使用原生train接口完成训练和预测。
"""
from lightgbm import LGBMRegressor
import lightgbm as lgb
from  sklearn.datasets import make_regression
from  sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
np.random.seed(2022)

# 使用make_classification，创建一个二分类数据集
X,y = make_regression(n_samples=10000,n_features=20,random_state=2022)
##划分训练集与测试集
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=200)

# 使用sklearn接口完成训练和预测
params1 = {
        'boosting_type': 'gbdt',
         'objective': 'regression',
         'n_estimators': 10000,
         'subsample': 0.9,
         'colsample_bytree': 0.8,
         'subsample_freq' :5,
        'learning_rate': 0.05,
         'num_leaves': 31,
         'random_state': 2022
}
#定义模型
lgb_clf1 = LGBMRegressor(**params1)
lgb_clf1.fit(train_x,train_y,eval_set=[(test_x,test_y)],eval_names=['validation'],
             eval_metric=['l2','l1'],
             callbacks=[lgb.log_evaluation(100),lgb.early_stopping(100)]
             )
pred1 = lgb_clf1.predict(test_x)

# 使用原生train接口完成训练和预测
params2 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    # 'num_class':2,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 2022
}
#定义Dataset
train_matrix = lgb.Dataset(train_x,label=train_y,free_raw_data=False)
test_matrix = lgb.Dataset(test_x,label=test_y,free_raw_data=False,reference=train_matrix)

lgb_clf2 = lgb.train(params=params2,
                     train_set=train_matrix,
                     valid_sets=test_matrix,
                     # learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                     num_boost_round=10000,
                    callbacks=[lgb.log_evaluation(100),lgb.early_stopping(100)],
                     # early_stopping_rounds=100,
                    # callbacks=[lgb.reset_parameter(learning_rate=lambda iter: 0.5 * (0.99 ** iter))]
                     )
pred2 = lgb_clf2.predict(test_x)



if __name__ == '__main__':
    print(pred1[:20])
    print(pred2[:20])
    print(test_y[:20])
    print('mearn square error')
    print(mean_squared_error(test_y,pred1)**0.5)
    print(mean_squared_error(test_y,pred2)**0.5)