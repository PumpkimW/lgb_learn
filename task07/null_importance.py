"""
Author:王楠
Date:2022-01-01
Title:特征筛选方法
步骤0 ：读取任务5的数据集，并完成数据划分。
步骤3 ：学习null importance重要性，并手动实现其过程，计算出最重要的3个特征。
https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
"""

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.simplefilter('ignore', UserWarning)

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
use_cols = list(data)

train, test, y_train, y_test = train_test_split(data.drop(["ARRIVAL_DELAY"], axis=1),
                                                data["ARRIVAL_DELAY"], random_state=10, test_size=0.25)

# 步骤3 ：基于null_importance方法计算特征重要性
def get_feature_importances(data, shuffle, seed=None):
    # Gather real features
    train_features = list(data.drop(["ARRIVAL_DELAY"],axis=1))
    # Go over fold and keep track of CV score (train and valid) and feature importances

    # Shuffle target if required
    y = data['ARRIVAL_DELAY'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['ARRIVAL_DELAY'].copy().sample(frac=1.0)

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    lgb_train = lgb.Dataset(data.drop(["ARRIVAL_DELAY"], axis=1), y, free_raw_data=False,silent=True)
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
    lgb_clf = lgb.train(params=lgb_params,
                        train_set=lgb_train,
                        num_boost_round=200,
                        callbacks=[lgb.log_evaluation(10)]
                        )
    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = lgb_clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = lgb_clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, lgb_clf.predict(data[train_features]))

    return imp_df

#shullfe多次，获取null_importance分布
null_imp_df = pd.DataFrame()
nb_runs = 30
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=data, shuffle=True)
    imp_df['run'] = i + 1
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)

def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(),
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(),
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())
    plt.show()


if __name__ == '__main__':
    actual_imp_df = get_feature_importances(data=data[use_cols],shuffle=False)
    print(get_feature_importances(data=data[use_cols],shuffle=True))
    print(get_feature_importances(data=data[use_cols],shuffle=False))
    # print(null_imp_df)
    display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='ORIGIN_AIRPORT')