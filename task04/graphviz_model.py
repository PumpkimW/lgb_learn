"""

步骤1 ：安装graphviz
步骤2 ：将树模型预测结果进行可视化，
步骤3（可选，不参与积分） ：在任务2中我们保存了json版本的树模型，其中一家包含了每棵树的
"""
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from lgb_learn.task01 import LGBMClassifier_iris
import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin'

evals_result = LGBMClassifier_iris.evals_result

# pickle 加载模型pkl文件
print('loading model pkl file:')
with open(r'../task02/model/lgbclf.txt','rb') as file2:
    lgb_clf = pickle.load(file2)

if __name__ == '__main__':
    print('plot 1th tree of lgb:')
    graph = lgb.create_tree_digraph(lgb_clf,tree_index=1,name='Tree2')
    # graph.view()
    graph.render(view=True)
    # print('画出训练结果...')
    # ax = lgb.plot_metric(evals_result, metric='multi_logloss')  # metric的值与之前的params里面的值对应
    # print('画特征重要性排序...')
    # ax = lgb.plot_importance(lgb_clf, max_num_features=4)  # max_features表示最多展示出前10个重要性特征，可以自行设置
    # print('Plot 3th tree...')  # 画出决策树，其中的第三颗
    # ax = lgb.plot_tree(lgb_clf, tree_index=3, figsize=(8, 5), show_info=['split_gain'])
    # plt.show()