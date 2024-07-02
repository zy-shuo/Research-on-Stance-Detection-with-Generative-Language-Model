from transformers import AutoTokenizer, AutoModel
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("/home/user/imported_models/chatglm2-6b/huggingface/THUDM/chatglm2-6b/", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/user/imported_models/chatglm2-6b/huggingface/THUDM/chatglm2-6b/", trust_remote_code=True).half().cuda()
model = model.eval()
def askglm(prompt):
    response, history = model.chat(tokenizer, prompt, history=[])
    print(response)    
    return response.replace("\n", "")

df = pd.read_excel("data0 .xlsx")
# prompts=('针对目标有一评论，请判断评论对目标的立场，不必解释原因，回答的形式为立场：[支持/反对/中立]。\n目标：'+df['target']+'\n评论：'+df['text']).tolist()
# prompts=('请判断评论"'+df['text']+'"对"'+df['target']+'"事件的立场是什么，不必解释原因，直接回答“支持”、“反对”或“中立”。').tolist()
# prompts=('针对目标有一评论，请判断评论对目标的立场，不必解释原因，回答的形式为立场：[支持/反对/中立]'+'。\n背景：'+df['简约背景']+'\n目标：'+df['target']+'\n评论：'+df['text']).tolist()
# prompts=('针对目标有一评论，请判断评论对目标的立场，回答的形式为立场：'+df['全部立场标签']+'。'+'\n目标：'+df['target']+'\n评论：'+df['text']).tolist()
# prompts=('针对目标有一评论，请判断评论对目标的立场，回答的形式为立场：'+df['全部立场标签']+'。\n背景：'+df['简约背景']+'\n目标：'+df['target']+'\n评论：'+df['text']).tolist()
# prompts=('针对目标有一评论，请判断评论b对目标的立场，不必解释原因，回答的形式为立场：[支持/反对/中立]。\n目标：'+df['target']+'\n评论a：'+df['示例']+'\n立场a：'+df['示例立场']+'\n评论b：'+df['text']).tolist()
# prompts=('针对目标有一评论，请判断评论b对目标的立场，回答的形式为立场：[支持/反对/中立]。\n目标：'+df['target']+'\n评论a：'+df['示例']+'\n立场a：让我们一步一步来思考。'+df['思维链']+'\n评论b：'+df['text']+'\n立场：让我们一步一来思考。').tolist()
prompts=('针对目标b有一评论，请判断评论b对目标b的立场，回答的形式为立场：[支持/反对/中立]。\n目标a：'+df['示例目标']+'\n评论a：'+df['示例2']+'\n目标b：'+df['target']+'\n评论b：'+df['text']).tolist()


with open('out.txt', 'a', encoding='utf-8') as file:
    # 按行将内容写入目标 txt 文件
    for line in prompts:
        file.write(askglm(line)+"\n")