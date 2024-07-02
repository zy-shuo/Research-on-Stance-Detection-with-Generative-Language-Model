import sys
import json
from tqdm import tqdm
from typing import List

import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from sklearn.metrics import accuracy_score, f1_score


def read_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def train_one_epoch(model: T5ForConditionalGeneration, device, data_loader, epoch, optimizer, lr_scheduler):
    model.train()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)

    sum_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs.loss, outputs.logits  # logits.shape = torch.Size([batch size, 2, vocab size])
        pred_labels = torch.max(logits, dim=-1).indices

        ground_truth_labels = torch.cat([ground_truth_labels, labels[:, 0]])
        predicted_labels = torch.cat([predicted_labels, pred_labels[:, 0]])

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())

        loss.backward()

        sum_loss += loss.detach()
        avg_loss = sum_loss.item() / (step + 1)

        data_loader.desc = "[train epoch {}] lr: {:.5f}, loss: {:.3f}, acc: {:.3f}".format(
            epoch, optimizer.param_groups[0]["lr"], avg_loss, accuracy
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


@torch.no_grad()
def validate(model: T5ForConditionalGeneration, device, data_loader, label_id_list: List, epoch):
    model.eval()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2)
        pred_labels = out[:, 1]

        ground_truth_labels = torch.cat([ground_truth_labels, labels[:, 0]])
        predicted_labels = torch.cat([predicted_labels, pred_labels])

        # 一般来说，训练过的 T5 模型在做分类任务时，生成的标签不会出现标签外的单词，具体请看原论文。以下代码只是证明一下确实没有生成标签外的单词
        for pred in pred_labels:
            if pred not in label_id_list:
                print(f"The predicted label is not in label_id_list, its index is {pred}")

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        macro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='macro')
        micro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='micro')
        weighted_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='weighted')

        data_loader.desc = "[valid epoch {}] acc: {:.3f}, macro_f1: {:.3f}, micro_f1: {:.3f}, weighted_f1: {:.3f}".format(
            epoch, accuracy, macro_f1, micro_f1, weighted_f1
        )

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1
    }


@torch.no_grad()
def test(model: T5ForConditionalGeneration, tokenizer: T5Tokenizer,device, data_loader, label_id_list: List):
    model.eval()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2)
        pred_labels = out[:, 1]

        # 一般来说，训练过的 T5 模型在做分类任务时，生成的标签不会出现标签外的单词，具体请看原论文。以下代码只是证明一下确实没有生成标签外的单词
        for pred in pred_labels:
            if pred not in label_id_list:
                print(f"The predicted label is not in label_id_list, its index is {pred}")

        ground_truth_labels = torch.cat([ground_truth_labels, labels[:, 0]])
        predicted_labels = torch.cat([predicted_labels, pred_labels])

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        acc_nums = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist(), normalize=False)
        # print('acc:', accuracy, 'num:', acc_nums)
        macro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='macro')
        # print('macro_f1:', macro_f1)
        micro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='micro')
        # print('micro_f1:', micro_f1)
        weighted_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='weighted')
        # print('weighted_f1:', weighted_f1)

        micro_f1_0 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), labels=[9011], average='micro')
        # print('micro_f1_favor:', micro_f1_0)
        micro_f1_1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), labels=[259], average='micro')
        # print('micro_f1_against:', micro_f1_1)
        micro_f1_2 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), labels=[59006], average='micro')
        # print('micro_f1_neutral:', micro_f1_2)
        data_loader.desc = "[test] acc: {:.3f}, macro_f1: {:.3f}, micro_f1: {:.3f}, weighted_f1: {:.3f}".format(
            accuracy, macro_f1, micro_f1, weighted_f1
        )
    macro_f1_1 = f1_score(ground_truth_labels.tolist()[0:100], predicted_labels.tolist()[0:100], labels=[9011, 259, 59006],
                          average='macro')
    # print('恶意殴打他人者的妻女被网暴macro_f1:', macro_f1_1)
    macro_f1_2 = f1_score(ground_truth_labels.tolist()[100:200], predicted_labels.tolist()[100:200], labels=[9011, 259, 59006],
                          average='macro')
    # print('女子不让6岁男童上女厕所遭痛骂macro_f1:', macro_f1_2)
    macro_f1_3 = f1_score(ground_truth_labels.tolist()[200:300], predicted_labels.tolist()[200:300], labels=[9011, 259, 59006],
                          average='macro')
    # print('警方通告胡鑫宇为自杀macro_f1:', macro_f1_3)
    macro_f1_4 = f1_score(ground_truth_labels.tolist()[300:400], predicted_labels.tolist()[300:400], labels=[9011, 259, 59006],
                          average='macro')
    # print('满江红起诉大Vmacro_f1:', macro_f1_4)
    macro_f1_5 = f1_score(ground_truth_labels.tolist()[400:500], predicted_labels.tolist()[400:500], labels=[9011, 259, 59006],
                          average='macro')
    # print('泼水节女生选择原谅对方macro_f1:', macro_f1_5)
    # print('**************************')
    micro_f1_a = f1_score(ground_truth_labels.tolist()[0:100], predicted_labels.tolist()[0:100], labels=[9011, 259, 59006],
                          average='micro')
    # print('恶意殴打他人者的妻女被网暴micro_f1:', micro_f1)
    micro_f1_b = f1_score(ground_truth_labels.tolist()[100:200], predicted_labels.tolist()[100:200], labels=[9011, 259, 59006],
                          average='micro')
    # print('女子不让6岁男童上女厕所遭痛骂micro_f1:', micro_f1)
    micro_f1_c = f1_score(ground_truth_labels.tolist()[200:300], predicted_labels.tolist()[200:300], labels=[9011, 259, 59006],
                          average='micro')
    # print('警方通告胡鑫宇为自杀micro_f1:', micro_f1)
    micro_f1_d = f1_score(ground_truth_labels.tolist()[300:400], predicted_labels.tolist()[300:400], labels=[9011, 259, 59006],
                          average='micro')
    # print('满江红起诉大Vmicro_f1:', micro_f1)
    micro_f1_e = f1_score(ground_truth_labels.tolist()[400:500], predicted_labels.tolist()[400:500], labels=[9011, 259, 59006],
                          average='micro')
    # print('泼水节女生选择原谅对方micro_f1:', micro_f1)
    # print(predicted_labels[0:100])
    # print(ground_truth_labels[0:100])
    # print(tokenizer.decode(predicted_labels))
    # print(tokenizer.decode(ground_truth_labels[0:100]))
    # print(ground_truth_labels.shape)
    return {
        'accuracy': accuracy,
        'acc_nums':acc_nums,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'micro_f1_favor':micro_f1_0,
        'micro_f1_against': micro_f1_1,
        'micro_f1_neutral': micro_f1_2,
        '恶意殴打他人者的妻女被网暴macro_f1': macro_f1_1,
        '女子不让6岁男童上女厕所遭痛骂macro_f1': macro_f1_2,
        '警方通告胡鑫宇为自杀macro_f1': macro_f1_3,
        '满江红起诉大Vmacro_f1': macro_f1_4,
        '泼水节女生选择原谅对方macro_f1': macro_f1_5,
        '恶意殴打他人者的妻女被网暴micro_f1': micro_f1_a,
        '女子不让6岁男童上女厕所遭痛骂micro_f1': micro_f1_b,
        '警方通告胡鑫宇为自杀micro_f1': micro_f1_c,
        '满江红起诉大Vmicro_f1': micro_f1_d,
        '泼水节女生选择原谅对方micro_f1': micro_f1_e

    }