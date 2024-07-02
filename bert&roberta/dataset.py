import torch

import pandas as pd

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer,model_name):
        self.data = data
        self.tokenizer=tokenizer
        self.model_name=model_name

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        label = self.data[index]["label"]
        if self.model_name=='bert':
            text = self.data[index]["text"]+' '+'目标:'+self.data[index]["target"]
            # text = self.data[index]["text"]+' '+'目标:'+self.data[index]["target"]+'[SEP]'+self.data[index]["全部立场标签"]
            # text = self.data[index]["text"]+' '+'目标:'+self.data[index]["target"]+'[SEP]'+'背景:'+self.data[index]["简约背景"]
            # text = self.data[index]["text"]+' '+'目标:'+self.data[index]["target"]+'[SEP]'+'背景:'+self.data[index]["简约背景"]+' '+self.data[index]["全部立场标签"]
        else:
            text = self.data[index]["text"] + ' ' + '目标:' + self.data[index]["target"]
            # text = self.data[index]["text"]+' '+'目标:'+self.data[index]["target"]+'[SEP]'+self.data[index]["全部立场标签"]
            # text = self.data[index]["text"]+' '+'目标:'+self.data[index]["target"]+'[SEP]'+'背景:'+self.data[index]["简约背景"]
            # text = self.data[index]["text"]+' '+'目标:'+self.data[index]["target"]+'[SEP]'+'背景:'+self.data[index]["简约背景"]+' '+self.data[index]["全部立场标签"]

        return text, label

    def collate_fn(self,batch):
        texts = [i[0] for i in batch]
        labels = [i[1] for i in batch]
        # 编码
        text_encoded = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=500,
                                       return_tensors='pt',
                                       return_length=True)
        # input_ids:编码之后的数字
        # attention_mask:是补零的位置是0,其他位置是1
        input_ids = text_encoded['input_ids'].to(device)
        attention_mask = text_encoded['attention_mask'].to(device)
        token_type_ids = text_encoded['token_type_ids'].to(device)
        labels = torch.LongTensor(labels).to(device)


        return input_ids, attention_mask, token_type_ids, labels
