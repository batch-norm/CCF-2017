import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
import datetime
from collections import defaultdict
from model_utils import yyq_split, move_to_wifi1, vote, yyq_score, Cal_mall_area, Cal_open_sorce, get_newLable, \
    GetFinalPred, two_list_score, xgb_GetFinalPred, Cal_Manhattan, Cal_Euler, get_xgb_param
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

print('Load data...')
with open('../../Data/pre_test.pkl', 'rb') as f:
    test = pickle.load(f)
with open('../../Data/pre_train.pkl', 'rb') as f:
    train = pickle.load(f)
mall_count = 1
mall_list = np.unique(train['mall_id'])

# 把确定不会变得特征放里面
# TODO：一定注意，这里面的特征不能和Binary里的重合
# TODO：当前特征[开源特征，经纬度，双距离]
for mall in mall_list:
    print(mall_count)
    x_train = train[['shop_id', 'row_id', 'wifi_infos', 'longitude', 'latitude']][train['mall_id'] == mall]
    y_train = train[['shop_id']][train['mall_id'] == mall]
    x_test = test[['wifi_infos', 'row_id', 'longitude', 'latitude']][test['mall_id'] == mall]
    # -------------------取交集------------------------------------------------------------------------------
    print('取交集.....')
    train_all_wifi = []


    def fun1(line):
        wifi_list = line['wifi_infos'].split(';')
        for wifi in wifi_list:
            temp = wifi.split('|')
            train_all_wifi.append(temp[0])


    x_train.apply(lambda x: fun1(x), axis=1)
    train_all_wifi = np.unique(train_all_wifi)
    test_all_wifi = []


    def fun2(line):
        wifi_list = line['wifi_infos'].split(';')
        for wifi in wifi_list:
            temp = wifi.split('|')
            test_all_wifi.append(temp[0])


    x_test.apply(lambda x: fun2(x), axis=1)
    test_all_wifi = np.unique(test_all_wifi)

    jiaoji = set(train_all_wifi) & set(test_all_wifi)
    # -------------------删除非交集------------------------------------------------------------------------------
    print('删交集....')
    test_new_wifi = []
    wifi_count = {}


    def fun3(line):
        delete_count = 0
        wifi_list = line['wifi_infos'].split(';')
        num_wifi = len(wifi_list)
        str_ = ''
        for wifi in wifi_list:
            xixi = wifi.split('|')
            if xixi[0] in jiaoji:
                str_ = str_ + ';' + wifi
                if xixi[0] not in wifi_count:
                    wifi_count[xixi[0]] = 1
                else:
                    wifi_count[xixi[0]] += 1
            else:
                delete_count += 1
        if delete_count == num_wifi:
            str_ = ';b_123456|-999|false'
        str_ = str_[1:]
        test_new_wifi.append(str_)


    x_test.apply(lambda x: fun3(x), axis=1)

    x_test['wifi_infos'] = test_new_wifi

    train_new_wifi = []


    def fun4(line):
        delete_count = 0
        wifi_list = line['wifi_infos'].split(';')
        num_wifi = len(wifi_list)
        str_ = ''
        for wifi in wifi_list:
            xixi = wifi.split('|')
            if xixi[0] in jiaoji:
                str_ = str_ + ';' + wifi
            else:
                delete_count += 1
        if delete_count == num_wifi:
            str_ = ';b_123456|-999|false'
        str_ = str_[1:]
        train_new_wifi.append(str_)


    x_train.apply(lambda x: fun4(x), axis=1)
    x_train['wifi_infos'] = train_new_wifi
    # -------------------生成强度字典------------------------------------------------------------------------------
    print('生成平均强度字典.....')


    # 生成强度平均值字典 先收集所有值 然后求平均
    def wifiDict(line):
        wifi_list = line['wifi_infos'].split(';')
        for wifi in wifi_list:
            each_wifi = wifi.split('|')
            # 第一层key为wifi名  添加的值为该wifi的强度
            strength_wifi[each_wifi[0]].append(int(each_wifi[1]))


    strength_wifi = defaultdict(lambda: [])
    x_train.apply(lambda x: wifiDict(x), axis=1)
    # 全部遍历完以后 生成平均值
    for wifi, shop_dict in strength_wifi.items():
        strength_wifi[wifi] = np.mean(shop_dict)
    print('生成magic特征......')
    def vote_value(x, y):
        # x是目前的强度值  y是历史平均值
        if np.isnan(x) or x == -999 or x == -9999:
            return 0
        else:
            return 1 / (1 + np.power(np.e, np.abs(x - y) / 15))
    combine = [x_train, x_test]
    for dataset in combine:
        # 用来保存每一条的最强关联wifi
        GY = []
        def CalWifiInfo(line):
            # 每一行的字典 key为shop value为累加的投票值
            vote_dict = defaultdict(lambda: 0)
            temp_wifi_list = line['wifi_infos'].split(';')
            for each_wifi in temp_wifi_list:
                xixi = each_wifi.split('|')
                # each_wifi[0]为wifi名 each_wifi[1]为wifi强度
                # 对于这个wifi对应的所有shop进行值投票
                for shop in strength_wifi[xixi[0]]:
                    vote_dict[shop] += vote_value(float(xixi[1]), strength_wifi[xixi[0]][shop])
            # 得到每行的投票字典
            vote_dict = sorted(vote_dict.items(), key=lambda h: h[1])
            # 字典[值]是个(key,value)list
            if len(vote_dict) == 0:
                GY.append(np.nan)
            else:
                largest_value = vote_dict[-1][1]
                if largest_value != 0:
                    GY.append(int(vote_dict[-1][0]))
                else:
                    GY.append(np.nan)
        get_wifi_info = lambda x: CalWifiInfo(x)
        dataset.apply(get_wifi_info, axis=1)
        dataset['super_magic_feature'] = GY
    ###################
    # 生成wifi指纹
    ###################
    # 先生成特征 全置为0
    for wifi in jiaoji:
        x_train['%s_printer' % wifi] = 0
        x_test['%s_printer' % wifi] = 0
    combine = [x_train, x_test]
    for dataset in combine:
        for index___, line in dataset.iterrows():
            wifi_list = line['wifi_infos'].split(';')
            for wifi in wifi_list:
                xixi = wifi.split('|')
                # xixi[0]是wifi名
                # 如果有这个wifi 就把printer置1
                dataset['%s_printer' % xixi[0]] = 1
    # 最后把b_123456_printer删除
    x_train = x_train.drop('b_123456_printer', axis=1)
    x_test = x_test.drop('b_123456_printer', axis=1)
    # -------------------生成当前wifi序列和历史wifi序列的欧拉距离------------------------------------------------------------------------------
    print('生成wifi欧拉距离.....')
    combine = [x_train, x_test]
    wifi_Euler_distance = []


    def Wifi_Euler(line):
        wifi_list = line['wifi_infos'].split(';')
        each_result = []
        for each_wifi in wifi_list:
            wifi = each_wifi.split('|')
            # wifi[0]为wifi名 wifi[1]为强度
            each_result.append(np.power(float(wifi[1]) - strength_wifi[wifi[0]], 2))
        if np.sum(each_result) == 0:
            wifi_Euler_distance.append(np.nan)
        else:
            wifi_Euler_distance.append(np.sqrt(np.sum(each_result)))


    x_train.apply(lambda x: Wifi_Euler(x), axis=1)
    x_train['wifi_Euler'] = wifi_Euler_distance
    wifi_Euler_distance = []
    x_test.apply(lambda x: Wifi_Euler(x), axis=1)
    x_test['wifi_Euler'] = wifi_Euler_distance

    # -------------------生成双距离------------------------------------------------------------------------------
    print('双距离......')
    core_latitude, core_longitude = Cal_mall_area(train[['longitude_y', 'latitude_y']][train['mall_id'] == mall])
    # 曼哈顿距离
    # TODO: 重置参考点
    Cal_Manhattan(x_train, x_test, core_latitude, core_longitude)
    # 欧氏距离
    Cal_Euler(x_train, x_test, core_latitude, core_longitude)
    # 经纬度聚类
    print("cluster")
    num_shop = len(np.unique(y_train['shop_id']))
    real_shop = train[['longitude_y', 'latitude_y']][train['mall_id'] == mall]
    real_location = []
    for long, lat in zip(real_shop['longitude_y'], real_shop['latitude_y']):
        real_location.append([long, lat])
    train_location = []
    for long, lat in zip(x_train['longitude'], x_train['latitude']):
        train_location.append([long, lat])
    test_location = []
    for long, lat in zip(x_test['longitude'], x_test['latitude']):
        test_location.append([long, lat])
    kmeans = KMeans(n_clusters=num_shop, random_state=1024).fit(np.array(train_location))
    #
    x_train['cluster'] = kmeans.predict(np.array(train_location))
    x_test['cluster'] = kmeans.predict(np.array(test_location))
    x_train = pd.get_dummies(data=x_train, columns=['cluster'], dummy_na=True)
    x_test = pd.get_dummies(data=x_test, columns=['cluster'], dummy_na=True)
    delete_list = []
    for column in x_train.columns:
        if column not in x_test.columns:
            delete_list.append(column)
    delete_list.remove('shop_id')
    x_train = x_train.drop(delete_list,axis=1)
    # -------------------添加开源特征------------------------------------------------------------------------------
    print('开源特征......')
    x_train, x_test = Cal_open_sorce(x_train, x_test)
    # 最后保存
    mall_count += 1
    with open('OpenFeat/train_%d.pkl' % mall, 'wb') as f:
        pickle.dump(x_train, f, -1)
    with open('OpenFeat/test_%d.pkl' % mall, 'wb') as f:
        pickle.dump(x_test, f, -1)
