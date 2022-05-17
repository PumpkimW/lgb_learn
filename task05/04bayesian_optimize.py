"""
Auther:PumpkinW
Date:2022-01-22
Title:Bayesian tuning parameters
"""
import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import warnings
import lgb_learn.config as config
np.random.seed(2022)

# 步骤1 ：运行以下代码得到训练集和验证集
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

# 步骤6 ：学习贝叶斯调参原理，使用BayesianOptimization完成超参数搜索，具体过程可以参考
# pip install -i https://mirrors.aliyun.com/pypi/simple bayesian-optimization

# 步骤6-1 基于LightGBM创建黑盒函数以查找参数
def lgb_bayesian(max_depth,num_leaves,learning_rate,feature_fraction,bagging_fraction):

    #lightgbm需要有些参数为整型数据,可以做个声明
    max_depth = int(max_depth)
    num_leaves = int(num_leaves)

    assert type(max_depth) == int
    assert type(num_leaves) == int

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'verbose': -1,
        'random_state': 2022
    }

    lgb_train = lgb.Dataset(train, label=y_train, free_raw_data=False)
    lgb_test = lgb.Dataset(test, label=y_test, free_raw_data=False)

    lgb_clf = lgb.train(params=params,
                        train_set=lgb_train,
                        num_boost_round=config.num_boost_round,
                        valid_sets=lgb_test,
                        callbacks=[lgb.log_evaluation(50),
                                   lgb.early_stopping(config.early_stopping_rounds)]
                        )
    pred = lgb_clf.predict(test,num_iteration=lgb_clf.best_iteration)
    score = roc_auc_score(y_test,pred)
    return score
# 步骤6-2 为参数提供边界，以便贝叶斯优化仅在边界内搜索
lgb_bound = {
        'max_depth':(3,5),
        'num_leaves' : (15,31),
        'learning_rate':(0.01,0.1),
        'feature_fraction':(0.7,0.9),
        'bagging_fraction':(0.7,0.9)
    }

# 步骤6-3 初始化贝叶斯优化器与优化器参数
lgb_bo = BayesianOptimization(lgb_bayesian,lgb_bound,random_state=2022)
# 步骤6-3 运行优化器搜索最优参数组合
lgb_bo.maximize(init_points=5, n_iter=5)
# 步骤6-4 基于获取的参数组合进行模型训练
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': int(lgb_bo.max['params']['max_depth']),
        'num_leaves': int(lgb_bo.max['params']['num_leaves']),
        'learning_rate': lgb_bo.max['params']['learning_rate'],
        'feature_fraction': lgb_bo.max['params']['feature_fraction'],
        'bagging_fraction': lgb_bo.max['params']['bagging_fraction'],
        'verbose': -1,
        'random_state': 2022
    }

n_fold = 5 #定义五折交易验证
skf = StratifiedKFold(n_fold,shuffle=True,random_state=2022)
#将之前的train，test的index重置，并将test作为测试集进行五折预测
train = train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
test = test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

oof = np.zeros(train.shape[0])
test_pred = np.zeros((test.shape[0],n_fold))
for i,(train_idx,valid_idx) in enumerate(skf.split(train,y_train)):
    print(f'{i+1}th split')
    train_data = lgb.Dataset(train.iloc[train_idx,:],y_train[train_idx],free_raw_data=False)
    valid_data  = lgb.Dataset(train.iloc[valid_idx,:],y_train[valid_idx],free_raw_data=False)

    lgb_clf = lgb.train(params=lgb_params,
                        train_set=train_data,
                        num_boost_round=config.num_boost_round,
                        valid_sets=valid_data,
                        callbacks=[lgb.log_evaluation(50),
                                   lgb.early_stopping(config.early_stopping_rounds)]
                        )
    # 预测需要用原始数据，不能用lgb格式的
    oof[valid_idx] = lgb_clf.predict(train.iloc[valid_idx,:],num_iteration=lgb_clf.best_iteration)
    test_pred[:,i] =  lgb_clf.predict(test,num_iteration=lgb_clf.best_iteration)
    print(f'{i+1}th training finished')

#计算五折的平均预测结果
rank_prediction = test_pred.mean(axis=1)
test['rank_prediction'] = rank_prediction

if __name__ == '__main__':
    # print(train.shape)
    # print(test.shape)
    # print(y_train.value_counts())
    # print(y_test.value_counts())
    # print(data[cols])
    # print(data['FLIGHT_NUMBER'].astype("category").cat.codes)
    # print(data.shape)

    # 查看参数空间名称
    # print(lgb_bo.space.keys)
    print(lgb_bo.max['target'])
    print(lgb_bo.max['params'])
    # 0.7693151444853498
    # {'bagging_fraction': 0.8488412424509231, 'feature_fraction': 0.7584998948190763,
    #  'learning_rate': 0.036880781586845705, 'max_depth': 4.504946945379715, 'num_leaves': 15.298619643281205}

    #打印最终预测结果
    print(test)

