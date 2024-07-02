#coding=utf-8
import random
from openai import OpenAI
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import pandas as pd
api_key = "sk-ySdVtr8lBSE43UKN3cC5041b6133414dBfAd2e10A2Ee1748"
client = OpenAI(base_url='https://api.chatgptid.net/v1', api_key=api_key)
@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(6))
def askChatGPT(prompt):
    if len(prompt)<1450:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {"role": "user",
                 "content": prompt}
            ]
        )
        message=response.choices[0].message.content
        print(message)
    else:
        message='超出限制'
        print('\n'+message)
    return message.replace("\n", "")
df = pd.read_excel("data0 .xlsx")

# prompts=('针对目标有一评论，请判断评论对目标的立场，回答的形式为立场：[支持/反对/中立]。\n目标：'+df['target']+'\n评论：'+df['text']).tolist()
# prompts=('请判断评论"'+df['text']+'"对"'+df['target']+'"事件的立场是什么，直接回答“支持”、“反对”或“中立”。').tolist()
# prompts=('针对目标有一评论，请判断评论对目标的立场，回答的形式为立场：[支持/反对/中立]'+'。\n背景：'+df['简约背景']+'\n目标：'+df['target']+'\n评论：'+df['text']).tolist()
# prompts=('针对目标有一评论，请判断评论对目标的立场，回答的形式为立场：'+df['全部立场标签']+'。'+'\n目标：'+df['target']+'\n评论：'+df['text']).tolist()
# prompts=('针对目标有一评论，请判断评论对目标的立场，回答的形式为立场：'+df['全部立场标签']+'。\n背景：'+df['简约背景']+'\n目标：'+df['target']+'\n评论：'+df['text']).tolist()
# prompts=('针对目标有一评论，请判断评论b对目标的立场，回答的形式为立场：[支持/反对/中立]。\n目标：'+df['target']+'\n评论a：'+df['示例']+'\n立场a：'+df['示例立场']+'\n评论b：'+df['text']).tolist()
# prompts=('针对目标b有一评论，请判断评论b对目标b的立场，回答的形式为立场：[支持/反对/中立]。\n目标a：'+df['示例目标']+'\n评论a：'+df['示例2']+'\n目标b：'+df['target']+'\n评论b：'+df['text']).tolist()
prompts=('针对目标有一评论，请判断评论b对目标的立场，回答的形式为立场：[支持/反对/中立]。\n目标：'+df['target']+'\n评论a：'+df['示例']+'\n立场a：让我们一步一步来思考。'+df['思维链']+'\n评论b：'+df['text']+'\n立场：让我们一步一来思考。').tolist()

with open('out.txt', 'a', encoding='utf-8') as file:

    # 按行将内容写入目标 txt 文件
    for line in prompts:
        file.write(askChatGPT(line)+"\n")
