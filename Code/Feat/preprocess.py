import pickle
import re

import numpy as np
import pandas as pd
import pickle
from feat_utils import sort_wifi, get_mode

####################
# Load Data
####################
print('Load Data.....')
train_shop = pd.read_csv('../../Data/shop_info.csv')
train_user = pd.read_csv('../../Data/behavior.csv')
test = pd.read_csv('../../Data/test.csv')
train = pd.merge(train_user, train_shop, how='left', on='shop_id')
print('Done.....')
print(train.head(5))

####################
# 删除公共wifi
####################
with open('../../Data/public_wifi_list_train.pkl', 'rb') as f:
    train_public_list = pickle.load(f)
with open('../../Data/public_wifi_list_test.pkl', 'rb') as f:
    test_public_list = pickle.load(f)
#----------------train--------------------------
train_new_wifi_infos = []
for index, line in train.iterrows():
    delete_count = 0
    print(index)
    wifi_list = line['wifi_infos'].split(';')
    num_wifi = len(wifi_list)
    str = ''
    for wifi in wifi_list:
        xixi = wifi.split('|')
        if xixi[0] not in train_public_list:
            str = str + ';' + wifi
        else:
            delete_count += 1
    if delete_count == num_wifi:
        str = 'b_123456|-999|false;b_123456|-999|false'
    str = str[1:]
    print(str)

    train_new_wifi_infos.append(str)
train['wifi_infos'] = train_new_wifi_infos

# ----------------test--------------------------
test_new_wifi_infos = []
for index, line in test.iterrows():
    delete_count = 0
    print(index)
    wifi_list = line['wifi_infos'].split(';')
    num_wifi = len(wifi_list)
    str = ''
    for wifi in wifi_list:
        xixi = wifi.split('|')
        if xixi[0] not in test_public_list:
            str = str + ';' + wifi
        else:
            delete_count += 1
    if delete_count == num_wifi:
        str = 'b_123456|-999|false;b_123456|-999|false'
    str = str[1:]
    print(str)
    test_new_wifi_infos.append(str)
test['wifi_infos'] = test_new_wifi_infos
####################
# wifi 取交集
####################
print('取交集......')
train_all_wifi = []

for index, line in train.iterrows():
    wifi_list = line['wifi_infos'].split(';')
    for wifi in wifi_list:
        temp = wifi.split('|')
        train_all_wifi.append(temp[0])
train_all_wifi = np.unique(train_all_wifi)

test_all_wifi = []
for index_, line in test.iterrows():
    wifi_list = line['wifi_infos'].split(';')
    for wifi in wifi_list:
        temp = wifi.split('|')
        test_all_wifi.append(temp[0])
test_all_wifi = np.unique(test_all_wifi)

jiaoji = set(train_all_wifi) & set(test_all_wifi)

test_new_wifi = []
for index__, line in test.iterrows():
    delete_count = 0
    print(index__)
    wifi_list = line['wifi_infos'].split(';')
    num_wifi = len(wifi_list)
    str = ''
    for wifi in wifi_list:
        xixi = wifi.split('|')
        if xixi[0] in jiaoji:
            str = str + ';' + wifi
        else:
            delete_count += 1
    if delete_count == num_wifi:
        str = ';b_123456|-999|false'
    str = str[1:]
    test_new_wifi.append(str)
test['wifi_infos'] = test_new_wifi

train_new_wifi = []
for index__, line in train.iterrows():
    delete_count = 0
    print(index__)
    wifi_list = line['wifi_infos'].split(';')
    num_wifi = len(wifi_list)
    str = ''
    for wifi in wifi_list:
        xixi = wifi.split('|')
        if xixi[0] in jiaoji:
            str = str + ';' + wifi
        else:
            delete_count += 1
    if delete_count == num_wifi:
        str = ';b_123456|-999|false'
    str = str[1:]
    train_new_wifi.append(str)
train['wifi_infos'] = train_new_wifi
print('Done')

####################
# Manage wifi
####################
print('Manage Wifi....')
sort = lambda line: sort_wifi(line)
train['wifi'] = train['wifi_infos'].apply(sort)
test['wifi'] = test['wifi_infos'].apply(sort)

combine = [train, test]
for dataset in combine:
    for i in range(1, 10):
        dataset['wifi%d' % i] = np.nan
        dataset['wifi%d_value' % i] = -999

####################
# Object to Int
####################
print('Object to Int')
train['shop_id'] = train['shop_id'].str.replace(r's_', '').astype(int)
train['category_id'] = train['category_id'].str.replace(r'c_', '').astype(int)
train['user_id'] = train['user_id'].str.replace(r'u_', '').astype(int)
train['mall_id'] = train['mall_id'].str.replace(r'm_', '').astype(int)
test['user_id'] = test['user_id'].str.replace(r'u_', '').astype(int)
test['mall_id'] = test['mall_id'].str.replace(r'm_', '').astype(int)
print('Done')

####################
# Tme
####################
print('mange Time...')
combine = [train, test]
for dataset in combine:
    temp = pd.DatetimeIndex(dataset['time_stamp'])
    dataset['month'] = temp.month
    dataset['day'] = temp.day
    dataset['hour'] = temp.hour  # 变换格式
    dataset['minute'] = temp.minute  # 时间最小粒度为分钟
    dataset['date'] = temp.date
    dataset['dayofweek'] = pd.DatetimeIndex(dataset.date).dayofweek  # 提取出星期几这个特征

    # train['dateDays']=(train.date-train.date[0]).astype('timedelta64[D]') #计算两个日期差几天
    dataset = dataset.drop(['date', 'time_stamp'], axis=1)

print('Done')
###################
# wifi count
##################
print('manage wificount....')


def wificount(wifi):
    wifi = wifi.split(';')
    return len(wifi)


count_wifi = lambda line: wificount(line)
combine = [train, test]
for dataset in combine:
    dataset['wifi_count'] = dataset['wifi_infos'].apply(count_wifi)
print('done')
###################
# For cv
##################
combine = [train, test]
for dateset in combine:
    dateset.loc[dateset['day'] < 8, 'cv'] = 0
    dateset.loc[(dateset['day'] >= 8) & (dateset['day'] < 15), 'cv'] = 1
    dateset.loc[(dateset['day'] >= 15) & (dateset['day'] < 22), 'cv'] = 2
    dateset.loc[(dateset['day'] >= 22) & (dateset['day'] < 32), 'cv'] = 3
    dateset['cv'] = dateset['cv'].astype(int)
# ###################
# # 用户购买力
# ##################
train_ind = set(train['user_id'])
test_ind = set(test['user_id'])
act_ind = train_ind & test_ind
train_price = train[['user_id', 'price']].groupby(['user_id'], as_index=False).mean()
train_price.columns = ['user_id', 'ind_price']
train_price['ind_price'] = train_price['ind_price'].astype(int)
train_price = train_price[train_price.user_id.isin(act_ind)]
train = train.merge(train_price, on='user_id', how='left')
test = test.merge(train_price, on='user_id', how='left')
#
#
# ###################
# # 用户常去商店类别
# ##################
ind_cat = train[['user_id', 'category_id']].groupby(['user_id'], as_index=False).agg(get_mode)
ind_cat.columns = ['user_id', 'ind_shop_cat']
ind_cat = ind_cat[ind_cat.user_id.isin(act_ind)]
train = train.merge(ind_cat, on='user_id', how='left')
test = test.merge(ind_cat, on='user_id', how='left')

###################
# 饭点
##################
combine = [train, test]
for dataset in combine:
    dataset['Eating'] = 0
    dataset.loc[
        (dataset['hour'] == 11) | (dataset['hour'] == 12) | (dataset['hour'] == 13) | (dataset['hour'] == 17) | (
            dataset['hour'] == 18) | (dataset['hour'] == 19), 'Eating'] = 1

train['latitude'] = train['latitude_x']
train['longitude'] = train['longitude_x']
combine = [train, test]
for dataset in combine:
    # dataset['ind_price'] = dataset['ind_price'].fillna(-99)
    # dataset['ind_shop_cat'] = dataset['ind_shop_cat'].fillna(-999)
    for i in range(1, 10):
        dataset['wifi%d' % i] = dataset['wifi%d' % i].fillna(10086)
    for i in range(1, 10):
        dataset['wifi%d' % i] = dataset['wifi%d' % i].astype(int)
train['latitude'] = round(train['latitude'], 5)
train['longitude'] = round(train['longitude'], 5)
test['latitude'] = round(test['latitude'], 5)
test['longitude'] = round(test['longitude'], 5)
train['row_id'] = range(1, train.shape[0] + 1)
print('Done')
print("Save data...")
with open('../../Data/pre_train.pkl', 'wb') as f:
    pickle.dump(train, f, -1)
with open('../../Data/pre_test.pkl', 'wb') as f:
    pickle.dump(test, f, -1)
print("Done")
