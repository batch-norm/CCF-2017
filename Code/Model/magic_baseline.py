import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
print('Load Data.....')
train_shop = pd.read_csv('../../Data/shop_info.csv')
train_user = pd.read_csv('../../Data/behavior.csv')
test = pd.read_csv('../../Data/test.csv')
train = pd.merge(train_user, train_shop, how='left', on='shop_id')
print('Done.....')

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
# 建立字典
####################
print('建立字典...')
def wifiDict(line):
    wifi_list = line['wifi_infos'].split(';')
    for wifi in wifi_list:
        each_wifi = wifi.split('|')
        # 第一层key为wifi名  第二层key为shop_id 添加的值为该wifi的强度
        strength_wifi[each_wifi[0]][line['shop_id']].append(int(each_wifi[1]))


def dd():
    return defaultdict(list)


strength_wifi = defaultdict(dd)
train.apply(lambda x: wifiDict(x), axis=1)
# 全部遍历完以后 生成平均值
for wifi, shop_dict in strength_wifi.items():
    for shop, value_list in shop_dict.items():
        # 如果值列表里没数据 平均值设为-9999
        if len(strength_wifi[wifi][shop]) == 0:
            strength_wifi[wifi][shop] = -9999
        else:
            strength_wifi[wifi][shop] = np.mean(strength_wifi[wifi][shop])
print('Done')
####################
# 生成结果
####################
print('生成结果')
submission = pd.DataFrame({
        "row_id": test['row_id'],
    })
x_test = test[['wifi_infos']]


def vote_value(x, y):
    # x是目前的强度值  y是历史平均值
    if np.isnan(x) or x == -999 or x == -9999:
        return 0
    else:
        return 1 / (1 + np.power(np.e, np.abs(x - y) / 20))
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
    largest_value = vote_dict[-1][1]
    if largest_value != 0:
        GY.append(vote_dict[-1][0])
    else:
        # 随便填的
        GY.append('s_166522')

get_wifi_info = lambda x: CalWifiInfo(x)
test.apply(get_wifi_info, axis=1)
test['shop_id'] = GY

test[['row_id','shop_id']].to_csv('magic_baseline_submission.csv',index=False)
print('Done')


