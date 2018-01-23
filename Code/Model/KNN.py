import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
import datetime
from collections import defaultdict
from model_utils import yyqq_split, move_to_wifi1, vote, yyq_score, Cal_mall_area,Cal_open_sorce
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

print('Load data...')
with open('../../Data/pre_test.pkl', 'rb') as f:
    test = pickle.load(f)
with open('../../Data/pre_train.pkl', 'rb') as f:
    train = pickle.load(f)
# def wificount(wifi):
#     wifi = wifi.split(';')
#     wifiID = []
#     for each in wifi:
#         each = each.split('|')
#         each[0] = int(each[0].replace(r'b_', ''))
#         if int(each[1])>-40:
#             wifiID.append(each[0])
#     return np.sum(np.unique(wifiID))
#
#
# count_wifi = lambda line: wificount(line)
# combine = [train, test]
# for dataset in combine:
#     dataset['wifi_map'] = dataset['wifi_infos'].apply(count_wifi)

train['latitude'] = train['latitude_x']
train['longitude'] = train['longitude_x']
# mall_id train和test一样
mall_list = np.unique(train['mall_id'])
submission = pd.DataFrame({
    "row_id": test['row_id'],
    "shop_id": np.zeros(len(test['row_id']))
})

Final_Final_score = []
# 填补空值：
combine = [train, test]
for dataset in combine:
    for i in range(1, 10):
        dataset['wifi%d' % i] = dataset['wifi%d' % i].fillna(10086)
    for i in range(1, 10):
        dataset['wifi%d' % i] = dataset['wifi%d' % i].astype(int)

##################
# hyperParameter
#################
# 调参
# print('tuning hyperParameters.....')
# turned_parameter = [
#     {'n_estimators': [1800], 'max_depth': [5], 'oob_score': ['False'], 'max_features': ['sqrt', 'log2']}]
# # CV generator
# ss = ShuffleSplit(n_splits=4, test_size=0.25, random_state=0)
# skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=2017)
# # Grid Search
# clf = GridSearchCV(RandomForestClassifier(), turned_parameter, cv=skf)
# clf.fit(X_train, Y_train)
# print('Best Parameters: ' + str(clf.best_params_))
train['latitude'] = round(train['latitude'], 5)
train['longitude'] = round(train['longitude'], 5)
test['latitude'] = round(test['latitude'], 5)
test['longitude'] = round(test['longitude'], 5)
print('Done')

# 应该把只有一个的label单独拿出来处理
all_score = []
mall_count = 1
# 添加其他特征

for mall in mall_list:

    print('共有%d个mall,当前处理第%d个' % (len(mall_list), mall_count))
    # 用什么模型
    model_method = 'xgb'
    # 前两个为固定，后面为自己加的特征
    x_train = train[['row_id','wifi1']][train['mall_id'] == mall]
    y_train = train[['shop_id']][train['mall_id'] == mall]
    cat = train[['category_id']][train['mall_id'] == mall]
    print('该mall有%d个店' % len(np.unique(y_train)))

    test_id = test['row_id'][test['mall_id'] == mall]
    x_test = test[['row_id','wifi1']][test['mall_id'] == mall]
    with open('OpenFeat/train_%d.pkl' % mall, 'rb') as f:
        open_train = pickle.load(f)
    with open('OpenFeat/test_%d.pkl' % mall, 'rb') as f:
        open_test = pickle.load(f)
    x_train = pd.merge(x_train, open_train, how='left', on='row_id')
    x_test = pd.merge(x_test, open_test, how='left', on='row_id')
    x_train = x_train.drop('row_id', axis=1)
    x_test = x_test.drop('row_id', axis=1)
    #--------------------------------------------------#
    #x_train,x_test = Cal_open_sorce(x_train,x_test)
    # 当前mall的信息
    print("当前mall:%d" % mall)
    num_label = len(np.unique(y_train['shop_id']))
    num_cat = len(np.unique(cat))
    print('样本量:%d' % x_train.shape[0])
    # 映射
    original_dict = y_train['shop_id'].tolist()
    shopid_dict = dict()
    for index, s in enumerate(np.unique(original_dict)):
        shopid_dict[s] = index
    sid = lambda line: shopid_dict[line]
    y_train['shop_id'] = y_train['shop_id'].apply(sid)
    # 本次mall的结果
    result = pd.DataFrame({
        "row_id": test_id,
        "shop_id": np.zeros(len(test_id))
    })
    # cv
    y = train[['cv']][train['mall_id'] == mall]
    score_list = []
    result_list = []
    # score_list
    knn_score = []
    rf_score = []
    xgb_score = []
    svc_score = []
    lgb_score = []
    dt_score = []
    pred = 0
    lgb_run = 0
    # 是否跑全部方法
    pipline = False
    for i, (train_index, test_index) in enumerate(yyqq_split(y['cv'].tolist())):

        # DT
        # ----------------------------------------------------------------------------------------------------#
        if model_method == 'dt':
            X1_train, X1_valid = x_train.iloc[train_index], x_train.iloc[test_index]
            y1_train, y1_valid = y_train.iloc[train_index], y_train.iloc[test_index]
            DT = DecisionTreeClassifier()
            DT.fit(X1_train, y1_train.values.ravel())
            # print('knn第%d折得分:%f'%(i+1,acc))
            acc = DT.score(X1_valid, y1_valid)
            dt_score.append(acc)
            score_list.append(acc)
            pred = DT.predict(x_test)
            result_list.append(pred)
        # ----------------------------------------------------------------------------------------------------#
        if model_method == 'knn':
            X1_train, X1_valid = x_train.iloc[train_index], x_train.iloc[test_index]
            y1_train, y1_valid = y_train.iloc[train_index], y_train.iloc[test_index]
            knn = KNeighborsClassifier(n_neighbors=5, weights='distance', leaf_size=15)
            knn.fit(X1_train, y1_train.values.ravel())
            # print('knn第%d折得分:%f'%(i+1,acc))
            acc = knn.score(X1_valid, y1_valid)
            knn_score.append(acc)
            score_list.append(acc)
            pred = knn.predict(x_test)
            result_list.append(pred)
        # ----------------------------------------------------------------------------------------------------#
        # RF
        if model_method == 'rf' or pipline == True:
            X1_train, X1_valid = x_train.iloc[train_index], x_train.iloc[test_index]
            y1_train, y1_valid = y_train.iloc[train_index], y_train.iloc[test_index]
            RF = RandomForestClassifier(n_estimators=1500)
            RF.fit(X1_train, y1_train.values.ravel())
            acc = RF.score(X1_valid, y1_valid)
            # print('RF第%d折得分:%f' % (i + 1, acc))
            rf_score.append(acc)
            score_list.append(acc)
            pred = RF.predict(x_test)
            result_list.append(pred)
        # ----------------------------------------------------------------------------------------------------#
        # SVC
        if model_method == 'qiji':
            X1_train, X1_valid = x_train.iloc[train_index], x_train.iloc[test_index]
            y1_train, y1_valid = y_train.iloc[train_index], y_train.iloc[test_index]
            svm = SVC()
            svm.fit(X1_train, y1_train.values.ravel())
            acc = svm.score(X1_valid, y1_valid)
            # print('RF第%d折得分:%f' % (i + 1, acc))
            svc_score.append(acc)
            score_list.append(acc)
            # pred = RF.predict(x_test)
            # result_list.append(pred)
        # xgboost
        if model_method == 'xgb':
            # X1_train, X1_valid = x_train.iloc[train_index], x_train.iloc[test_index]
            # y1_train, y1_valid = y_train.iloc[train_index], y_train.iloc[test_index]
            # xgbtest = xgb.DMatrix(x_test.values)
            d_train = xgb.DMatrix(x_train.values, y_train.values)
            d_test = xgb.DMatrix(x_test.values)

            watchlist = [(d_train, 'train')]
            params = {
                'objective': 'multi:softmax',
                'max_depth': 5,
                'eval_metric': 'merror',
                'eta': 0.1,
                'silent': 1,
                'num_class': num_label,
                'seed': 0,
                'num_leaves': 10
            }
            mdl = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=50,
                            verbose_eval=100)
            pred = mdl.predict(d_test)
            xgb_score.append(mdl.best_score)
            score_list.append(mdl.best_score)
            result_list.append(pred)
            print('xgb得分:%f' % ( 1 - mdl.best_score))
        # lgb
        if model_method == 'lgb' or pipline == True:
            start = datetime.datetime.now()
            print('LGB..........................')
            if lgb_run == 0:
                xD_train = x_train.values
                yD_train = y_train.values
                lgb_run += 1
            X1_train, X1_eval = xD_train[train_index], xD_train[test_index]
            y1_train, y1_eval = yD_train[train_index], yD_train[test_index]

            params = {'metric': 'multi_error',
                      'learning_rate': 0.01,
                      'max_depth': 4,
                      'max_bin': 8,
                      'objective': 'multiclass',
                      'num_class': num_label,
                      'feature_fraction': 0.8,
                      'bagging_fraction': 0.9,
                      'bagging_freq': 10,
                      'min_data': 500,
                      'verbose': -1
                      }
            lgb_model = lgb.train(params, lgb.Dataset(X1_train, label=y1_train.ravel()), 2000,
                                  lgb.Dataset(X1_eval, label=y1_eval.ravel()),
                                  early_stopping_rounds=150, verbose_eval=150)
            acc = 1 - lgb_model.best_score['valid_0']['multi_error']
            lgb_score.append(acc)
            score_list.append(acc)
            end = datetime.datetime.now()
            print('每一折时间:' + str(end - start))
            pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration)
            print(pred)
    if pipline == True:
        print('RF得分为：%f' % np.mean(rf_score))
        print('Lgb得分为:%f' % np.mean(svc_score))
        # print('xgboost得分为：%f' % np.mean(xgb_score))
    else:
        print('-----------------------------------------------------------------------得分为:%f' % np.mean(score_list))
    mall_count += 1
    # 产出结果
    result['shop_id'] = pred
    final = []
    # # 反映射
    shopid_dict_reverse = dict()
    for index, s in enumerate(np.unique(original_dict)):
        shopid_dict_reverse[index] = s
    def hihi(line):
        final.append(shopid_dict_reverse[line['shop_id']])
    result[['shop_id']].apply(lambda line: hihi(line),axis=1)
    result['shop_id'] = final
    submission.update(result)
    all_score.append(np.mean(score_list) * x_train.shape[0])
submission['row_id'] = submission['row_id'].astype(int)
submission['shop_id'] = submission['shop_id'].astype(int)
sort = lambda line: 's_' + str(line)
submission['shop_id'] = submission['shop_id'].apply(sort)
submission.to_csv('rf_submission.csv', index=False)
print('----------')
print('CV:' + str(np.sum(all_score) / train.shape[0]))
