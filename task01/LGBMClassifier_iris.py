"""
@Author:pumpkinW
Date:2022-01-01
Title:模型训练与预测
步骤1 ：导入LightGBM库
步骤2 ：使用LGBMClassifier对iris进行训练。
步骤3 ：将预测的模型对iris进行预测。
"""
# 步骤1 ：导入LightGBM库
import lightgbm as lgb
# 步骤2 ：使用LGBMClassifier对iris进行训练
from  lightgbm import  LGBMClassifier
#加载iris数据
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import lgb_learn.config
import numpy as np
np.random.seed(2022)

# 步骤3 ：将预测的模型对iris进行预测
iris = load_iris()
data = iris.data
target = iris.target
# target = target.reshape(-1,1)
train_x,test_x,trian_y,test_y = train_test_split(data,target,test_size=0.2,random_state=2022)

#lightgbm的skleran接口
param = {'boosting_type': 'gbdt',
         'objective': 'binary',
         'n_estimators': 100,
         'subsample': 1.0,
         'colsample_bytree': 1.0,
         'max_depth': 4,
         'num_leaves': 31,
         'importance_type': 'split',
         'learning_rate': 0.01,
         'min_child_samples': 20,
         'min_child_weight': 0.001,
         'min_split_gain': 0.0,
         'reg_alpha': 0.0,
         'reg_lambda': 1,
         'random_state': 2022
}

# lgb_clf =  LGBMClassifier(**param)
# lgb_clf.fit(train_x,trian_y)
# # pred = lgb_clf.predict(test_x)
# pred_proba = lgb_clf.predict_proba(test_x)

#lightgbm原生接口
params2 = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class':3,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}
train_data = lgb.Dataset(train_x,trian_y,free_raw_data=False)
# if you want to re-use data, remember to set free_raw_data=False,free_raw_data=True时，test_data.data为None
test_data = lgb.Dataset(test_x,test_y,free_raw_data=False,reference=train_data)

evals_result = {}  #记录训练结果所用
lgb_clf2 = lgb.train(params2,
                     train_set=train_data,
                     valid_sets=test_data,
                     # learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                     num_boost_round=100,
                     callbacks=[lgb.log_evaluation(5),
                                lgb.reset_parameter(learning_rate=lambda iter: 0.5 * (0.99 ** iter)),
                                lgb.record_evaluation(evals_result),
                                lgb.early_stopping(10)]
                     )
pred2= lgb_clf2.predict(test_x,num_iteration=lgb_clf2.best_iteration)

if __name__ == '__main__':
    # print(lgb_clf.get_params())
    # print(accuracy_score(test_y,pred))
    # print(confusion_matrix(test_y,pred))
    # print(pred_proba.shape)
    # print(pred.shape)
    # print(np.hstack((pred.reshape(-1,1),pred_proba)))
    # print(np.hstack((pred.reshape(-1,1),pred_proba.argmax(axis=1).reshape(-1,1))))
    # print(pred2)
    # print(test_data.data)
    print(lgb_clf2.feature_name())
    # feature importances
    print('Feature importances:', list(lgb_clf2.feature_importance()))










