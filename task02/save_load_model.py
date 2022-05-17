"""
@Auther:PumpukinW
Date:2022-01-20
Title:save and load model
步骤1 ：将任务1训练得到的模型，使用pickle进行保存。
步骤2 ：将任务1训练得到的模型，使用txt进行保存。
步骤3 ：加载步骤1和步骤2的模型，并进行预测。
"""
import numpy as np
from lgb_learn.task01.LGBMClassifier_iris import lgb_clf2,test_x
import pickle
import json

# pickle 保存模型txt
print('saving model file as .txt:')
with open(r'./model/lgbclf.txt','wb') as file:
    pickle.dump(lgb_clf2,file)

# pickle 保存模型pkl
print('saving model file as .pickle:')
with open(r'./model/lgbclf.pkl','wb') as file:
    pickle.dump(lgb_clf2,file)

# json 保存模型json
print('saving model file as .json:')
model_json = lgb_clf2.dump_model()
with open(r'./model/lgbclf.json','w+') as file:
    json.dump(model_json,file) #对json进行数据格式化输出使用indent=4 这个参数


# pickle 加载模型txt文件
print('loading model txt file:')
with open(r'./model/lgbclf.txt','rb') as file2:
    lgb_clf1 = pickle.load(file2)

# pickle 加载模型pkl文件
print('loading model pkl file:')
with open(r'./model/lgbclf.pkl','rb') as file2:
    lgb_clf2 = pickle.load(file2)

# json 加载模型.json文件
print('saving model file:')
with open(r'./model/lgbclf.json','rb') as file3:
    lgb_clf3 = json.load(file3) #对json进行数据格式化输出使用indent=4 这个参数


if __name__ == '__main__':
    #加载步骤1和步骤2的模型，并进行预测。
    pred1 = lgb_clf1.predict(test_x)
    pred2 = lgb_clf2.predict(test_x)
    # pred3 = lgb_clf3.predict(test_x)
    print(np.hstack((pred1,pred2)))



