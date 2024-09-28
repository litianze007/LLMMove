import openai
import random
import os
import json
from tqdm import tqdm
import itertools
import pandas as pd
import re
import math

# # api调用机制
# openai.api_key  = ""
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
# )
# @retry(wait=wait_random_exponential(min=60, max=120), stop=stop_after_attempt(6))
# def openaiAPIcall(**kwargs):
#     try:
#         # 直接调用 openai.ChatCompletion.create 方法
#         return openai.ChatCompletion.create(**kwargs)
#     except openai.error.OpenAIError as e:
#         print(f"API 调用错误: {e}")
#         raise  # 重新抛出错误以触发重试机制



class LLMNational():
    def analyze_sentiment(self, review):
        # 创建 prompt，结合正面和负面评论以及国籍
        prompt = (
            f"Please carefully analyze the following review:\n\n"
            f"\"{review}\"\n\n"
            "Provide a sentiment score on a scale strictly between -1 (very negative) and 1 (very positive). "
            "The score must be a real number within this range, and should reflect the overall tone and sentiment of the review. "
            "If the review contains multiple aspects or points, prioritize those that have the greatest impact on the overall sentiment. "
            "Please ensure the score accurately reflects the most significant elements of the review, while ignoring minor or irrelevant details."
        )
        # 构建消息列表
        messages = [{"role": "user", "content": prompt}]
        # 调用 API
        response = openaiAPIcall(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0,
        )
        # 获取并解析响应内容
        res_content = response.choices[0].message['content'].strip()
            # 使用正则表达式提取情感分数
        match = re.search(r"[-+]?\d*\.\d+|\d+", res_content)
        if match:
            sentiment_score = float(match.group())
        else:
            raise ValueError("Unable to parse sentiment score from the response")
        return sentiment_score

    def run(self, data):
        self.Row_ID_reviews = data
        output_path = './datasets/mapogu-4k-ce7-fd6_row_id_score.json'

        # Step 1: 读取现有的 JSON 文件，并将已有的 row_id 存储到一个集合中
        existing_row_ids = set()
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    existing_data = json.loads(line)
                    existing_row_ids.add(existing_data['row_id'])
        except FileNotFoundError:
            print(f"No existing file found, creating new file at {output_path}.")
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse existing file {output_path}. Continuing without existing data.")

        # Step 2: 打开文件进行追加写入
        with open(output_path, 'a') as f:  # 'a' 模式表示追加
            for row_id, reviews in tqdm(self.Row_ID_reviews.items()):
                if row_id in existing_row_ids:
                    print(f"Row ID {row_id} already processed. Skipping.")
                    continue  # 如果 row_id 已存在，跳过此条记录

                for review in reviews:
                    # 检查 review['comment'] 是否为 NaN
                    if isinstance(review['comment'], float) and math.isnan(review['comment']):
                        print("Warning: Encountered NaN in comment, skipping this review.")
                        continue  # 跳过此条评论

                    # 获取评论文本
                    review_text = review['comment']

                    try:
                        # 获取情感评分
                        sentiment_score = self.analyze_sentiment(review_text)
                    except ValueError as e:
                        print(f"Error analyzing sentiment for row_id {row_id}: {e}")
                        continue  # 如果情感分析失败，跳过此条评论

                    # 检查 'point' 是否存在且不是 NaN
                    reviewer_score = review['point'] if not (isinstance(review['point'], float) and math.isnan(review['point'])) else None

                    # 存储结果
                    result = {
                        'row_id': row_id,
                        'sentiment_score': sentiment_score,
                        'Reviewer_Score': reviewer_score
                    }

                    # 将结果以 JSON 格式按行写入文件
                    f.write(json.dumps(result) + '\n')

        print(f"Sentiment analysis results saved to {output_path}")