"""
@Author:PumpkinW
Date:2022-01-22
Title:多分类分类任务
步骤1 ：学习LightGBM中sklearn接口的使用，导入分类、回归和排序接口。
步骤2 ：学习LightGBM中原生train接口的使用。
步骤4 ：多分类任务
使用make_classification，创建一个多分类数据集。
使用sklearn接口完成训练和预测。
使用原生train接口完成训练和预测。
"""
from lightgbm import LGBMClassifier,LGBMRegressor,LGBMRanker
import lightgbm as lgb
from  sklearn.datasets import make_classification
from  sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(2022)

# 使用make_classification，创建一个二分类数据集
X,y = make_classification(n_samples=1000,n_features=10,n_classes=4,n_clusters_per_class=1,random_state=2022)
##划分训练集与测试集
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=200)
# 使用sklearn接口完成训练和预测
params1 = {
        'boosting_type': 'gbdt',
         'objective': 'multiclass',
         'num_class':4,
         'n_estimators': 100,
         'subsample': 0.9,
         'colsample_bytree': 0.8,
         'learning_rate': 0.05,
         'num_leaves': 31,
         'importance_type': 'split',
         'min_child_samples': 20,
         'min_child_weight': 0.001,
         'min_split_gain': 0.0,
         'reg_alpha': 0.0,
         'reg_lambda': 1,
         'random_state': 2022
}
#定义模型
lgb_clf1 = LGBMClassifier(**params1)
lgb_clf1.fit(train_x,train_y,eval_set=[(train_x,train_y),(test_x,test_y)],
             eval_names=['train','valid'],
             callbacks=[lgb.early_stopping(10),
                        lgb.log_evaluation(10)]
             )
pred1 = lgb_clf1.predict(test_x)
proba1 = lgb_clf1.predict_proba(test_x,num_iteration=lgb_clf1.best_iteration_)

# 使用原生train接口完成训练和预测
params2 = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class':4,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}
#定义Dataset
train_matrix = lgb.Dataset(train_x,label=train_y,free_raw_data=False)
test_matrix = lgb.Dataset(test_x,label=test_y,free_raw_data=False,reference=train_matrix)
#
lgb_clf2 = lgb.train(params=params2,
                     train_set=train_matrix,
                     valid_sets=test_matrix,
                     # learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                     num_boost_round=100,
                     callbacks=[
                                # lgb.reset_parameter(learning_rate=lambda iter: 0.5 * (0.99 ** iter)),
                               lgb.early_stopping(10),
                               lgb.log_evaluation(10)]
                     )
pred2 = lgb_clf2.predict(test_x,num_iteration=lgb_clf2.best_iteration)

if __name__ == '__main__':
    print(pred1[:10])
    # print(proba1[:10])
    # print(pred2[:10])
    print(pred2.argmax(axis=1)[:10])