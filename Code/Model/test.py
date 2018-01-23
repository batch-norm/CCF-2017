import threading
from collections import defaultdict

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from model_utils import yyq_split, move_to_wifi1, vote, yyq_score, Cal_mall_area, Cal_open_sorce, get_newLable, \
    GetFinalPred, two_list_score, xgb_GetFinalPred, Cal_Manhattan, Cal_Euler, get_xgb_param
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


# def dd():
#     return defaultdict(list)
# with open('wifi_dict/615_wifi_dict.pkl', 'rb') as f:
#     houhouDict = pickle.load(f)
# print(houhouDict)
# # with open('../../Data/pre_train.pkl', 'rb') as f:
#     train = pickle.load(f)
#
# shop_list = np.unique(train['shop_id']).tolist()
# for shop in shop_list:
#     day_list = np.unique(train[['day']][train['shop_id'] == shop]).tolist()
#     state = 1
#     for day in day_list:
#         if day > 26:
#             state = 0
#             break
#     if len(day_list) < 15:
#         state = 0
#     if state == 1:
#         mall = np.unique(train[['mall_id']][train['shop_id'] == shop])[0]
#         print(str(mall) + ':' + str(shop) + ':' + str(day_list))

randn = np.random.randn(8,5)
alist = np.arange(1,9,1)
blist = list('ABCDE')
df = pd.DataFrame(randn,index=alist,columns=blist)
print(df.shape[1])
df = pd.get_dummies(data=df, columns=['A'], dummy_na=True)
print(df)
print(df.shape[1])
# wifi_sum = []
# wifi_average = []
# def get_sum(line):
#     wifi_sum.append(line['A']+line['B'])
#     wifi_average.append((line['A']+line['B'])/2)
# wifi_sum_ = lambda x:get_sum(x)
# df.apply(wifi_sum_,axis=1)
# df['sum'] = wifi_sum
# df['average'] = wifi_average
# print(df)

dict1 = {
    'h':1,
    'e':2,
    'l':3
}
print(set(dict1.keys()))