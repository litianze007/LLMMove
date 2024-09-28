import pandas as pd
import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
from models.LLMNational import LLMNational

def readTrain():
    trainPath = './datasets/mapogu-4k-ce7-fd6_row_id.csv'
    # 读取CSV文件
    data = pd.read_csv(trainPath)

    # 一组数据
    Row_ID_reviews = {}
    for _, row in data.iterrows():
        Row_ID = row['row_id']
        comment = row['comment_content']
        point = row['comment_point']

        if Row_ID not in Row_ID_reviews:
            Row_ID_reviews[Row_ID] = [] 
        Row_ID_reviews[Row_ID].append({
            'comment': comment,
            'point': point
        })

    return Row_ID_reviews

if __name__ == '__main__':

    # 加载模型
    model = LLMNational()

    # 数据提取
    data = readTrain()
    results = model.run(data)
