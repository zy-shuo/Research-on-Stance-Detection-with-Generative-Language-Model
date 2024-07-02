import json
import torch

import models
from dataset import Dataset
from transformers import AdamW
from transformers.optimization import get_scheduler
from models import BertModel
from transformers import BertTokenizer

from params import get_config
def read_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def train(model,dataset,collate_fn,train_batch_size,num_epoch,save_file):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=train_batch_size,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=True)
    #定义优化器
    optimizer = AdamW(model.parameters(), lr=4e-5)
    #定义loss函数
    criterion = torch.nn.CrossEntropyLoss()
    #定义学习率调节器
    scheduler = get_scheduler(name='linear',
                              num_warmup_steps=25,
                              num_training_steps=40,
                              optimizer=optimizer)
    #模型切换到训练模式
    model.train()
    for j in range(num_epoch):
        acc_num=0
        num=0
        #按批次遍历训练集中的数据
        for i, (input_ids, attention_mask, token_type_ids,
                labels) in enumerate(loader):

            #模型计算
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

            #计算loss并使用梯度下降法优化模型参数
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            pre = out.argmax(dim=1)
            acc_num+=(pre == labels).sum().item()
            num+=len(labels)
        #输出各项数据的情况，便于观察
        scheduler.step()

        accuracy =  acc_num/num
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(j, loss.item(), lr, accuracy)
    torch.save(model.state_dict(), save_file)

if __name__ == '__main__':
    args=get_config()

    data=read_json(args.train_file)
    bert_tokenizer= BertTokenizer.from_pretrained(args.pretrained_weights)
    if args.model_name=="bert":
        model=models.BertModel(args.num_class,args.pretrained_weights)
    else:
        model = models.RobertaModel(args.num_class, args.pretrained_weights)
    model.to(args.device)
    dataset=Dataset(data,bert_tokenizer,args.model_name)
    collate_fn=dataset.collate_fn
    train(model,dataset,collate_fn=collate_fn,train_batch_size=args.train_batch_size,num_epoch=args.num_epoch,save_file=args.save_file)