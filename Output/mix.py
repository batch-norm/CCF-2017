import numpy as np
import pandas as pd
import pickle

# 先得到mall列表
with open('../Data/pre_test.pkl', 'rb') as f:
    test = pickle.load(f)
mall_list = np.unique(test['mall_id']).tolist()
submission = pd.DataFrame({
        "row_id": test['row_id'],
        "shop_id": np.zeros(len(test['row_id']))
    })
# 每个mall
for mall in mall_list:
    print(mall)
    test_id = test['row_id'][test['mall_id'] == mall]
    result = pd.DataFrame({
        "row_id": test_id,
        "shop_id": np.zeros(len(test_id))
    })
    with open('lgb_7/%d_probability_lgb.pkl' % mall, 'rb') as f:
        mall_1 = pickle.load(f)
    with open('xgb_15/%d_probability_xgb.pkl' % mall, 'rb') as f:
        mall_2 = pickle.load(f)
    # final_mall 是融合的结果文件
    final_mall = mall_1.copy()
    column_list = mall_1.columns.tolist()
    for column in column_list:
        # 这里可以设置比例 默认1：1
        final_mall[column] = mall_1[column] + mall_2[column]
    final_shop = []
    def vote(line):
        larShop = 0
        larValue = 0
        for column in column_list:
            if line[column]>larValue:
                larValue = line[column]
                larShop = column
        final_shop.append(int(larShop))

    final_mall.apply(lambda x:vote(x),axis=1)
    result['shop_id'] = final_shop
    submission.update(result)

submission['row_id'] = submission['row_id'].astype(int)
submission['shop_id'] = submission['shop_id'].astype(int)
sort = lambda line: 's_' + str(line)
submission['shop_id'] = submission['shop_id'].apply(sort)
submission.to_csv('mix_submission.csv', index=False)