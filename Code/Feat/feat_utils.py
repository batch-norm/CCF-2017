import re
import numpy as np
from collections import defaultdict
def sort_wifi(line):
    wifi = line.split(';')
    length = len(wifi)
    p = re.compile(r'\d+')
    final_result = []
    value_result = []
    temp_value = []
    for index in range(length):
        result = p.findall(wifi[index])
        temp_value.append(result[1])
    quickSort(temp_value,wifi,0,length-1)
    # 取前n个
    if length < 9:
        for i in range(length):
            name = p.findall(wifi[i])
            final_result.append(name[0])
            value_result.append(name[1])
        for i in range(length,9):
            final_result.append(10086)
            value_result.append(-999)
        return [final_result,value_result]
    else:
        for i in range(9):
            name = p.findall(wifi[i])
            final_result.append(name[0])
            value_result.append(name[1])
        return [final_result,value_result]



def quickSort(num,wifi,l,r):
    if l >= r:  # 如果只有一个数字时，结束递归
        return
    flag = l
    for i in range(l + 1, r + 1):  # 默认以第一个数字作为基准数，从第二个数开始比较，生成索引时要注意右部的值
        if num[flag] > num[i]:
            temp = wifi[i]
            tmp = num[i]
            del num[i]
            del wifi[i]
            wifi.insert(flag,temp)
            num.insert(flag, tmp)
    quickSort(num,wifi, l, flag - 1)  # 将基准数前后部分分别递归排序
    quickSort(num, wifi,flag + 1, r)
def count_nanWifi(dataset,wifi_list):
    wifi = dataset[wifi_list]
    count = 0
    for line in wifi.values:
        for item in line:
            if item == 10086:
                count += 1
    return count
def Cal_mall_area(dataset):
    max_longitude = np.unique(dataset[['longitude_y']][dataset['longitude_y'] == np.max(dataset['longitude_y'])])[0]
    min_longitude = np.unique(dataset[['longitude_y']][dataset['longitude_y'] == np.min(dataset['longitude_y'])])[0]
    max_latitude = np.unique(dataset[['latitude_y']][dataset['latitude_y'] == np.max(dataset['latitude_y'])])[0]
    min_latitude = np.unique(dataset[['latitude_y']][dataset['latitude_y'] == np.min(dataset['latitude_y'])])[0]
    return (max_longitude-min_longitude)*(max_latitude-min_latitude)


# 个人最愿意去的店铺种类    _mode众数填充   没后缀未填充
def get_mode(arr):
    dic = defaultdict(lambda: 0)
    for str in arr:
        dic[str] += 1
    return max(dic, key=dic.get)



