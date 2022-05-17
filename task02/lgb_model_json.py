"""
Auther:PumpkinW
Date:2022-01-23
Title:在任务2中我们保存了json版本的树模型，其手动读取
"""
import pandas as pd
import numpy as np
import json
from lgb_learn.task01.LGBMClassifier_iris import test_x


# json 加载模型.json文件
print('saving model file:')
with open(r'./model/lgbclf.json','rb') as file3:
    lgb_clf = json.load(file3)
# 定义一个函数判断每一个leaf是走left还是right

feature_names = lgb_clf['feature_names']        # 获取模型中所用的特征变量
def decison(data,threshold,default_left):
    '''
    :param data:  特征值
    :param threshold: 分割判断值
    :param default_left: 默认分支 default_left= True or False
    :return: 返回结果left_child or right_child
    '''
    if ((np.isnan(data)) and (default_left is True)):
        return 'left_child'
    elif data <= threshold:
        return 'left_child'
    else:
        return 'right_child'

# 定义预测函数
def predict_gbm(data):
    score = 0
    for i in range(len(lgb_clf['tree_info'])):              # 遍历每一个节点
        num_leaves = lgb_clf['tree_info'][i]['num_leaves']  # 获取每颗树的节点数
        tree = lgb_clf['tree_info'][i]['tree_structure']    # 获取每一颗树结构
        for i in range(num_leaves):  # 遍历节点数
            # 到达节点leaf,进行走向判断
            threshold = tree.get('threshold')
            default_left = tree.get('default_left')
            split_feature = feature_names[tree['split_feature']]  # 获取叶子节点的分割特征变量
            next_decison = decison(data[split_feature],threshold,default_left)
            # 获取下一个分支leaf
            tree = tree[next_decison]
            if tree.get('left_child','not found') == 'not found':   # 如果到达节点节点停止遍历，返回对应值
                score = score + tree['leaf_value']
                break
    return(score)

if __name__ == '__main__':
    df = pd.DataFrame(test_x,columns=feature_names)
    # 进行测试
    predict_df = []
    for i in range(len(df)):
        predict_data = predict_gbm(df.iloc[i,:])  # 分值
        predict_dt = 1 / (np.exp(-predict_data) + 1)  # 将预测分值转为p值
        predict_df.append(predict_dt)
    print(predict_df)