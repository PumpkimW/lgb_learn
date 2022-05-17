import lightgbm as lgb
import numpy as np
import pandas as pd
from  sklearn.model_selection import train_test_split
import  warnings
warnings.filterwarnings("ignore")
from bayes_opt import BayesianOptimization

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

def lgb_eval(max_depth, learning_rate, num_leaves, n_estimators):
    params = {
        "metric": 'auc'
    }
    params['max_depth'] = int(max(max_depth, 1))
    params['learning_rate'] = np.clip(0, 1, learning_rate)
    params['num_leaves'] = int(max(num_leaves, 1))
    params['n_estimators'] = int(max(n_estimators, 1))
    lgb_train = lgb.Dataset(train, label=y_train, free_raw_data=False)
    cv_result = lgb.cv(params, lgb_train, nfold=5, seed=0, verbose_eval=200, stratified=False)
    return 1.0 * np.array(cv_result['auc-mean']).max()


lgbBO = BayesianOptimization(lgb_eval, {'max_depth': (4, 8),
                                        'learning_rate': (0.05, 0.2),
                                        'num_leaves': (20, 1500),
                                        'n_estimators': (5, 200)}, random_state=0)

lgbBO.maximize(init_points=5, n_iter=50, acq='ei')


if __name__ == '__main__':
    print(lgbBO.max)