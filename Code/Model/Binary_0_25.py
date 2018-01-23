import threading

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
import operator
from multiprocessing import Pool
import datetime
from collections import defaultdict
from model_utils import yyq_split, move_to_wifi1, vote, yyq_score, Cal_mall_area, Cal_open_sorce, get_newLable, \
    GetFinalPred, two_list_score, xgb_GetFinalPred, Cal_Manhattan, Cal_Euler, get_xgb_param, Cal_Wifi_distance
from pandocfilters import Math
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsRegressor


import matplotlib.pyplot as plt


def dd():
    return defaultdict(list)
#
#
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

if __name__ == '__main__':

    print('Load data...')
    with open('../../Data/pre_test.pkl', 'rb') as f:
        test = pickle.load(f)
    with open('../../Data/pre_train.pkl', 'rb') as f:
        train = pickle.load(f)
    mall_list = np.unique(train['mall_id'])
    submission = pd.DataFrame({
        "row_id": test['row_id'],
        "shop_id": np.zeros(len(test['row_id']))
    })
    Final_Final_score = []
    all_score = []
    cover_score = []
    mall_count = 0
    for mall in mall_list:
        # with open('wifi_dict/%d_wifi_dict.pkl'%mall,'rb') as f:
        #     wifi_dict = pickle.load(f)
        # 是cv看分还是直接跑结果   False跑结果  True看分
        cv_or_not = False
        # 每个mall的计算平均得分的字典
        mark_dict = defaultdict(lambda: 0)
        start = datetime.datetime.now()
        current_cover_rate = []
        current_mall_dict = {}
        current_mall = pd.DataFrame()
        print('\033[1;32m' + '共有%d个mall,当前处理第%d个' % (len(mall_list), mall_count + 1) + '\033[0m')
        # TODO：一定注意，这里面的特征不能和CallMallFeat里的重合
        x_train = train[['row_id', 'wifi1',  'Eating']][
            train['mall_id'] == mall]
        y_train = train[['shop_id']][train['mall_id'] == mall]
        test_id = test['row_id'][test['mall_id'] == mall]
        x_test = test[['row_id', 'wifi1', 'Eating']][test['mall_id'] == mall]
        # 添加其他特征
        with open('OpenFeat/train_%d.pkl' % mall, 'rb') as f:
            open_train = pickle.load(f)
        with open('OpenFeat/test_%d.pkl' % mall, 'rb') as f:
            open_test = pickle.load(f)
        x_train = pd.merge(x_train, open_train, how='left', on='row_id')
        x_test = pd.merge(x_test, open_test, how='left', on='row_id')
        x_train = x_train.drop('row_id', axis=1)
        x_test = x_test.drop('row_id', axis=1)
        print(x_train.shape)
        print(x_test.shape)
        # 随机下采样

        # 本次mall的结果
        print(type(test_id))
        result = pd.DataFrame({
            "row_id": test_id,
            "shop_id": np.zeros(len(test_id))
        })
        print(result.index)
        print('\033[1;32m' + '共有shop_id:%d个' % len(np.unique(y_train['shop_id'])) + '\033[0m')
        shop_list = np.unique(y_train['shop_id']).tolist()

        # C V
        y = train[['cv']][train['mall_id'] == mall]
        train_index, test_index = yyq_split(y['cv'].tolist())
        X1_train, X1_valid = x_train.iloc[train_index], x_train.iloc[test_index]
        y1_train, y1_valid = y_train.iloc[train_index], y_train.iloc[test_index]
        # 取出在验证集和测试集都有的shop_id
        good_shop_list = []
        for item in np.unique(y1_train['shop_id']).tolist():
            if item in np.unique(y1_valid['shop_id']).tolist():
                good_shop_list.append(item)
        # 是否cv验证
        # -----------------------------------------------------------------------------------------#
        if cv_or_not:
            for shop in good_shop_list:
                current_mall_dict[shop] = 0
                y1_train_label = get_newLable(y1_train['shop_id'].tolist(), shop)
                y1_valid_label = get_newLable(y1_valid['shop_id'].tolist(), shop)
                # 收集最佳参数 收集完注释掉
                d_train = xgb.DMatrix(X1_train.values, y1_train_label)
                d_valid = xgb.DMatrix(X1_valid.values, y1_valid_label)
                watchlist = [(d_train, 'train'), (d_valid, 'valid')]
                d_test = xgb.DMatrix(x_test.values)
                # 调参的参数
                params = {
                    'objective': 'binary:logistic',
                    'max_depth': 5,
                    'eval_metric': 'auc',
                    'eta': 0.1,
                    'silent': 1,
                    'seed': 0,
                    'num_leaves': 10
                }
                # 传统参数
                # params = {
                #     'objective': 'binary:logistic',
                #     'max_depth': 4,
                #     'eval_metric': 'auc',
                #     'eta': 0.14,
                #     'silent': 1,
                #     'seed': 0,
                #     'num_leaves': 10
                # }
                mdl = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=50,
                                verbose_eval=100)
                pred = mdl.predict(d_test)
                cover_score.append(mdl.best_score)
                current_cover_rate.append(mdl.best_score)
                print('\033[1;32m' + str(mdl.best_score) + '\033[0m')
                current_mall['%s' % shop] = pred
            mall_count += 1
            end = datetime.datetime.now()
            Final_pred = xgb_GetFinalPred(current_mall, good_shop_list, current_mall_dict)
            result['shop_id'] = Final_pred

            submission.update(result)
            print(
                '\033[1;32m' + '-------------------------------------------------------------------------------------------------------------------------------------' + '\033[0m')
            print('\033[1;32m' + '当前mall' + str(mall) + '\033[0m')
            print('\033[1;32m' + '当前mall时间:' + str(end - start) + '\033[0m')
            print('\033[1;32m' + '当前mall覆盖率:' + str(np.mean(current_cover_rate)) + '\033[0m')
            print(
                '\033[1;32m' + '-------------------------------------------------------------------------------------------------------------------------------------' + '\033[0m')
            current_cover_rate = []
        # -----------------------------------------------------------------------------------------#
        else:
            for shop in shop_list:
                current_mall_dict[shop] = 0
                new_label = get_newLable(y_train['shop_id'].tolist(), shop)
                # TODO：对于每个shop，选出所有在这个shop出现过的wifi来作指纹，强度也是，然后测试集的特征也一一对应上，这样就把模型的训练集的粒度分为shop而不是mall了
                d_train = xgb.DMatrix(x_train.values, new_label)
                watchlist = [(d_train, 'train'), (d_train, 'test')]
                d_test = xgb.DMatrix(x_test.values)
                print(x_train.columns)
                print(x_test.columns)
                print(x_test['super_magic_feature'])
                print('shop:' + str(shop))
                params = {
                    'objective': 'binary:logistic',
                    'max_depth': 5,
                    'eval_metric': 'auc',
                    'eta': 0.1,
                    'silent': 1,
                    'seed': 0,
                    'num_leaves': 10
                }
                mdl = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=50,
                                verbose_eval=100)
                pred = mdl.predict(d_test)
                # 重要性--------------------------------------------
                features = x_train.columns.tolist()
                ceate_feature_map(features)
                importance = mdl.get_fscore(fmap='xgb.fmap')
                importance = sorted(importance.items(), key=operator.itemgetter(1))
                df = pd.DataFrame(importance, columns=['feature', 'fscore'])
                df['fscore'] = df['fscore'] / df['fscore'].sum()
                plt.figure()
                df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
                plt.title('XGBoost Feature Importance')
                plt.xlabel('relative importance')
                plt.show()
                # ---------------------------------------------------
                cover_score.append(mdl.best_score)
                current_cover_rate.append(mdl.best_score)
                print('\033[1;32m' + str(mdl.best_score) + '\033[0m')
                current_mall['%s' % shop] = pred
            mall_count += 1
            Final_pred = xgb_GetFinalPred(current_mall)
            with open('Probability_mall/%d_probability_xgb.pkl' % mall, 'wb') as f:
                pickle.dump(current_mall, f, -1)
            result['shop_id'] = Final_pred
            submission.update(result)
            end = datetime.datetime.now()
            # 每跑完一个mall都保存
            submission.to_csv('submission/%d_binary_submission.csv' % mall, index=False)
            print(
                '\033[1;32m' + '-------------------------------------------------------------------------------------------------------------------------------------' + '\033[0m')
            print('\033[1;32m' + '当前mall' + str(mall) + '\033[0m')
            print('\033[1;32m' + '当前mall时间:' + str(end - start) + '\033[0m')
            print('\033[1;32m' + '当前mall覆盖率:' + str(np.mean(current_cover_rate)) + '\033[0m')
            print(
                '\033[1;32m' + '-------------------------------------------------------------------------------------------------------------------------------------' + '\033[0m')
            current_cover_rate = []
    # 整合结果
    print('覆盖率为:' + str(np.mean(cover_score)))
    submission['row_id'] = submission['row_id'].astype(int)
    submission['shop_id'] = submission['shop_id'].astype(int)
    sort = lambda line: 's_' + str(line)
    print('All Done')
    submission['shop_id'] = submission['shop_id'].apply(sort)
    submission.to_csv('binary_submission.csv', index=False)
    #
