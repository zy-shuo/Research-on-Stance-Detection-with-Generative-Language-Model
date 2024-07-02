from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import json
import torch
from params import get_config
import models
from dataset import Dataset
from transformers import AdamW
from transformers.optimization import get_scheduler
from models import BertModel
from transformers import BertTokenizer

def read_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def test(model,dataset,collate_fn,test_batch_size):
    # 定义测试数据集加载器
    pre = []
    loader_test = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=test_batch_size,
                                              collate_fn=collate_fn,
                                              shuffle=False,
                                              drop_last=False)

    # 下游任务模型切换到运行模式
    model.eval()

    #     print(len(loader_test))
    # 按批次遍历测试集中的数据
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):
        # 计算
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        out = out.argmax(dim=1)
        out = out.tolist()
        pre += out
    print(len(pre))

    labels = [i[1] for i in dataset]
    acc = accuracy_score(labels, pre)
    acc_nums = accuracy_score(labels, pre, normalize=False)
    print('acc:', acc, 'num:', acc_nums)  # 0.8 12
    # micro f1-score
    micro_f1 = f1_score(labels, pre, labels=[0, 1, 2], average='micro')
    print('micro_f1:', micro_f1)
    micro_f1_0 = f1_score(labels, pre, labels=[0], average='micro')
    print('micro_f1_0:', micro_f1_0)
    micro_f1_1 = f1_score(labels, pre, labels=[1], average='micro')
    print('micro_f1_1:', micro_f1_1)
    micro_f1_2 = f1_score(labels, pre, labels=[2], average='micro')
    print('micro_f1_2:', micro_f1_2)
    macro_f1 = f1_score(labels, pre, labels=[0, 1, 2], average='macro')
    print('macro_f1:', macro_f1)

    macro_f1_1 = f1_score(labels[0:100], pre[0:100], labels=[0, 1, 2], average='macro')
    print('恶意殴打他人者的妻女被网暴macro_f1:', macro_f1_1)
    macro_f1_2 = f1_score(labels[100:200], pre[100:200], labels=[0, 1, 2], average='macro')
    print('女子不让6岁男童上女厕所遭痛骂macro_f1:', macro_f1_2)
    macro_f1_3 = f1_score(labels[200:300], pre[200:300], labels=[0, 1, 2], average='macro')
    print('警方通告胡鑫宇为自杀macro_f1:', macro_f1_3)
    macro_f1_4 = f1_score(labels[300:400], pre[300:400], labels=[0, 1, 2], average='macro')
    print('满江红起诉大Vmacro_f1:', macro_f1_4)
    macro_f1_5 = f1_score(labels[400:500], pre[400:500], labels=[0, 1, 2], average='macro')
    print('泼水节女生选择原谅对方macro_f1:', macro_f1_5)
    print('**************************')
    micro_f1 = f1_score(labels[0:100], pre[0:100], labels=[0, 1, 2], average='micro')
    print('恶意殴打他人者的妻女被网暴micro_f1:', micro_f1)
    micro_f1 = f1_score(labels[100:200], pre[100:200], labels=[0, 1, 2], average='micro')
    print('女子不让6岁男童上女厕所遭痛骂micro_f1:', micro_f1)
    micro_f1 = f1_score(labels[200:300], pre[200:300], labels=[0, 1, 2], average='micro')
    print('警方通告胡鑫宇为自杀micro_f1:', micro_f1)
    micro_f1 = f1_score(labels[300:400], pre[300:400], labels=[0, 1, 2], average='micro')
    print('满江红起诉大Vmicro_f1:', micro_f1)
    micro_f1 = f1_score(labels[400:500], pre[400:500], labels=[0, 1, 2], average='micro')
    print('泼水节女生选择原谅对方micro_f1:', micro_f1)

if __name__ == '__main__':
    args=get_config()

    data=read_json(args.test_file)
    bert_tokenizer= BertTokenizer.from_pretrained(args.pretrained_weights)
    if args.model_name=="bert":
        model=models.BertModel(args.num_class,args.pretrained_weights)
    else:
        model = models.RobertaModel(args.num_class, args.pretrained_weights)
    model.load_state_dict(torch.load(args.save_file))
    model.to(args.device)
    dataset=Dataset(data,bert_tokenizer,args.model_name)
    collate_fn=dataset.collate_fn
    test(model,dataset,collate_fn=collate_fn,test_batch_size=args.test_batch_size)