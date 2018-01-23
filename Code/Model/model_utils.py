import re
import numpy as np
import pandas as pd
from collections import defaultdict
from bayes_opt import BayesianOptimization
import xgboost as xgb
import lightgbm as lgb


def get_xgb_param(X1_train, y1_train, X1_valid, y1_valid, good_shop_list):
    def xgbCV(max_depth, eta):
        acc_list = []
        params = {
            'objective': 'binary:logistic',
            'eta': max(min(eta, 0.5), 0),
            'silent': 1,
            'seed': 1024,
            'eval_metric': 'auc',
            'max_depth': int(round(max_depth)),
        }
        for shop in good_shop_list:
            y1_train_new = get_newLable(y1_train['shop_id'].tolist(), shop)
            y1_valid_new = get_newLable(y1_valid['shop_id'].tolist(), shop)
            d_train = xgb.DMatrix(X1_train.values, y1_train_new)
            d_valid = xgb.DMatrix(X1_valid.values, y1_valid_new)
            watchlist = [(d_train, 'train'), (d_valid, 'valid')]
            mdl = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=30,
                            verbose_eval=500)
            acc_list.append(mdl.best_score)
        return np.mean(acc_list)

    xgbBO = BayesianOptimization(xgbCV, {
        'max_depth': (1, 10),
        'eta': (0.01, 0.5)
    })
    xgbBO.maximize(init_points=5, n_iter=25)
    print('\033[1;32m' + str(xgbBO.res['max']) + '\033[0m')
    temp_dict = xgbBO.res['max']['max_params']
    temp_dict['max_depth'] = np.round(temp_dict['max_depth']).astype(int)
    # 纠正格式
    # min_child_weight [default = 1]  调大这个参数能够控制过拟合 范围：[0,正无穷)
    # colsample_bytree [default = 1]  在建立树时对特征随机采样的比例。缺省值为1  取值范围：（0,1]
    # max_depth [默认= 6] 树的最大深度。缺省值为6 树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合 通常取值：3-10（int）
    # gamma [默认0] 在树的叶节点上进行进一步分区所需的最小损失减少。算法越大，越保守。范围：[0，∞]
    # scale_pos_weight [默认= 0] 大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛
    return xgbBO.res['max']['max_params']


def get_lgb_param(X1_train, y1_train, X1_valid, y1_valid, good_shop_list):
    def lgbCV(learning_rate, max_depth):
        acc_list = []
        params = {
            'objective': 'binary',
            'learning_rate': max(min(learning_rate, 0.5), 0),  # 比0大比0.5小
            'verbose': -1,
            'metric': 'auc',
            'max_depth': int(round(max_depth)),
        }
        for shop in good_shop_list:
            y1_train_new = get_newLable(y1_train['shop_id'].tolist(), shop)
            y1_valid_new = get_newLable(y1_valid['shop_id'].tolist(), shop)
            lgb_model = lgb.train(params, lgb.Dataset(X1_train.values, label=y1_train_new.ravel()), 1500,
                                  lgb.Dataset(X1_valid.values, label=y1_valid_new.ravel()), verbose_eval=100,
                                  early_stopping_rounds=10)
            acc = lgb_model.best_score['valid_0']['auc']
            acc_list.append(acc)
        return np.mean(acc_list)

    lgbBO = BayesianOptimization(lgbCV, {'learning_rate': (0.01, 0.5),
                                         'max_depth': (3, 15)
                                         })
    lgbBO.maximize(init_points=5, n_iter=25)
    print('\033[1;32m' + str(lgbBO.res['max']) + '\033[0m')
    temp_dict = lgbBO.res['max']['max_params']
    temp_dict['max_depth'] = np.round(temp_dict['max_depth']).astype(int)
    # 纠正格式
    # min_child_weight [default = 1]  调大这个参数能够控制过拟合 范围：[0,正无穷)
    # colsample_bytree [default = 1]  在建立树时对特征随机采样的比例。缺省值为1  取值范围：（0,1]
    # max_depth [默认= 6] 树的最大深度。缺省值为6 树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合 通常取值：3-10（int）
    # gamma [默认0] 在树的叶节点上进行进一步分区所需的最小损失减少。算法越大，越保守。范围：[0，∞]
    # scale_pos_weight [默认= 0] 大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛
    return lgbBO.res['max']['max_params']


def yyq_split(y):
    label_dict = defaultdict(lambda: [])
    for index, label in enumerate(y):
        label_dict[label].append(index)
    # return [[np.array(label_dict[0] + label_dict[1] + label_dict[2]), np.array(label_dict[3])],
    #         [np.array(label_dict[3] + label_dict[1] + label_dict[2]), np.array(label_dict[0])],
    #         [np.array(label_dict[3] + label_dict[0] + label_dict[2]), np.array(label_dict[1])],
    #         [np.array(label_dict[3] + label_dict[1] + label_dict[0]), np.array(label_dict[2])]
    #         ]
    return np.array(label_dict[0] + label_dict[1] + label_dict[2]), np.array(label_dict[3])
    # [np.array(label_dict[3] + label_dict[1] + label_dict[2]), np.array(label_dict[0])],
    # [np.array(label_dict[3] + label_dict[0] + label_dict[2]), np.array(label_dict[1])],
    # [np.array(label_dict[3] + label_dict[1] + label_dict[0]), np.array(label_dict[2])]


def yyqq_split(y):
    label_dict = defaultdict(lambda: [])
    for index, label in enumerate(y):
        label_dict[label].append(index)
    return [[np.array(label_dict[0] + label_dict[1]), np.array(label_dict[2])]
            ]


def move_to_wifi1(line):
    i = 2
    if line['wifi1'] == 10086:
        while ((line['wifi%d' % i] == 10086) and (i < 9)):
            i += 1
        print('用wifi%d来替换wifi1' % i)
        line['wifi1'] = line['wifi%d' % i]
    return line


def vote(result_list):
    result = []
    if len(result_list) > 3:
        for index in range(len(result_list[0])):
            vote_dict = defaultdict(lambda: 0)
            vote_dict[result_list[0][index]] += 1
            vote_dict[result_list[1][index]] += 1
            vote_dict[result_list[2][index]] += 1
            vote_dict[result_list[3][index]] += 1
            if len(vote_dict) == 3:
                for sid, count in vote_dict.items():
                    if count == 2:
                        result.append(sid)
                        break
            if len(vote_dict) == 2:
                for sid, count in vote_dict.items():
                    if count == 2:
                        result.append(sid)
                        break
                    elif count == 3:
                        result.append(sid)
                        break
            if len(vote_dict) == 1:
                for sid, count in vote_dict.items():
                    result.append(sid)
                    break
            if len(vote_dict) == 4:
                for sid, count in vote_dict.items():
                    result.append(sid)
                    break
    return result


def yyq_score(pred, label):
    label_value = label['shop_id'].tolist()
    if (len(pred) != len(label)):
        print('长度不符')
        return 0
    right_count = 0
    for v1, v2 in zip(pred, label_value):
        if v1 == v2:
            right_count += 1
    return right_count / len(pred)


def Cal_mall_area(dataset):
    max_longitude = np.unique(dataset[['longitude_y']][dataset['longitude_y'] == np.max(dataset['longitude_y'])])[0]
    min_longitude = np.unique(dataset[['longitude_y']][dataset['longitude_y'] == np.min(dataset['longitude_y'])])[0]
    max_latitude = np.unique(dataset[['latitude_y']][dataset['latitude_y'] == np.max(dataset['latitude_y'])])[0]
    min_latitude = np.unique(dataset[['latitude_y']][dataset['latitude_y'] == np.min(dataset['latitude_y'])])[0]
    return (max_latitude + min_latitude) / 2, (max_longitude + min_longitude) / 2


def Cal_open_sorce(train, test):
    print('Cal open Source')
    df = pd.concat([train, test])
    df1 = df.reset_index(drop=True)
    l = []
    wifi_dict = {}
    for index, row in df1.iterrows():
        r = {}
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            r[i[0]] = int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]] = 1
            else:
                wifi_dict[i[0]] += 1
        l.append(r)
    delate_wifi = []
    for i in wifi_dict:
        if wifi_dict[i] < 10:
            delate_wifi.append(i)
    m = []
    for row in l:
        new = {}
        for n in row.keys():
            if n not in delate_wifi:
                new[n] = row[n]
        m.append(new)
    df1 = pd.concat([df1, pd.DataFrame(m)], axis=1)
    train_wifi_feat = df1[df1['shop_id'].notnull()]
    test_wifi_feat = df1[df1['shop_id'].isnull()]
    df_train = train_wifi_feat.drop(['shop_id', 'wifi_infos'], axis=1)
    df_test = test_wifi_feat.drop(['shop_id', 'wifi_infos'], axis=1)
    print('Done')
    return df_train, df_test


def get_newLable(original_list, shop):
    new_list = []
    for item in original_list:
        if item == shop:
            new_list.append(1)
        else:
            new_list.append(0)
    return pd.Series(new_list)


def GetFinalPred(DT, shop_list, shop_dict):
    final_result = []
    cow_length = DT.shape[1]
    for index, line in DT.iterrows():
        for i in range(cow_length):
            shop_name = shop_list[i]
            value = line[i]
            if value == 1:
                # 如果为正结果，正类计数+1
                shop_dict[shop_name] += 1
            else:
                # 如果为负结果，所有负类计数+1
                for shop in shop_list:
                    if shop != shop_name:
                        shop_dict[shop] += 1
        # 此row最终结果
        final_result.append(sorted(shop_dict, key=lambda x: shop_dict[x])[-1])
        # 重置字典
        for key in shop_dict.keys():
            shop_dict[key] = 0
    return final_result


def two_list_score(list1, list2):
    right_count = 0
    for x1, x2 in zip(list1, list2):
        if x1 == x2:
            right_count += 1
    return right_count / len(list1)


# def xgb_GetFinalPred(DT, shop_list, shop_dict,test_id):
#     final_result = []
#     cow_length = DT.shape[1]
#     DF=pd.DataFrame()
#     for index, line in DT.iterrows():
#         for i in range(cow_length):
#             shop_name = shop_list[i]
#             value = line[i]
#             # 如果为正结果，正类计数+1
#             shop_dict[shop_name] += value
#             # 如果为负结果，所有负类计数+1
#             for shop in shop_list:
#                 if shop != shop_name:
#                     shop_dict[shop] += (1 - value)
#         shop_dict['row_id'] = test_id[index]
#         # 此row最终结果
#         final_result.append(sorted(shop_dict, key=lambda x: shop_dict[x])[-1])
#         # 更新概率DF
#         temp_df = pd.DataFrame(shop_dict,index=[test_id[index]])
#         if index==0:
#             DF = temp_df
#         else:
#             DF = pd.DataFrame(pd.concat([DF,temp_df]))
#         # 重置字典
#         for key in shop_dict.keys():
#             shop_dict[key] = 0
#     return final_result,DF
def xgb_GetFinalPred(DT):
    clist = DT.columns.tolist()
    final = []

    def get_large(line):
        larShop = 0
        larValue = 0
        # 选每一行最大的
        for column in clist:
            if line[column] > larValue:
                larValue = line[column]
                larShop = column
        final.append(larShop)

    DT.apply(lambda x: get_large(x), axis=1)
    return final


def Cal_Manhattan(DT1, DT2, core_latitude, core_longitude):
    combine = [DT1, DT2]
    for dataset in combine:
        dataset['Manhattan'] = np.abs(dataset['latitude'] - core_latitude) + np.abs(
            dataset['longitude'] - core_longitude)


def Cal_Euler(DT1, DT2, core_latitude, core_longitude):
    combine = [DT1, DT2]
    for dataset in combine:
        dataset['Euler'] = np.sqrt(
            np.power(dataset['longitude'] - core_longitude, 2) + np.power(dataset['latitude'] - core_latitude, 2))


def Cal_Wifi_distance(line, printer_dict):
    each_result = []
    wifi_names = set(printer_dict.keys())
    for wifi in wifi_names:
        if np.isnan(line[wifi]):
            continue
        else:
            # print('%f' % line[wifi] + '-' + '%f' % printer_dict[wifi])
            each_result.append(np.power(line[wifi] - printer_dict[wifi], 2))
    return np.sqrt(np.sum(each_result))
