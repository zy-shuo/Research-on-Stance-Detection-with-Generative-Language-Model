import torch

from transformers import BertModel

class BertModel(torch.nn.Module):
    def __init__(self,num_class,pretrained_weights):
        super().__init__()
        self.num_class=num_class
        self.pretrained_weights=pretrained_weights
        self.bert = BertModel.from_pretrained(self.pretrained_weights)
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        n_layers = 12
        n_layers_freeze = 8
        n_layers_ft = n_layers - n_layers_freeze
        for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        self.fc = torch.nn.Linear(in_features=768, out_features=self.num_class)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用预训练模型抽取数据特征
        #         with torch.no_grad():
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        pooled_output = out.pooler_output
        # 对抽取的特征只取第一个字的结果做分类即可
        #         out = self.fc(out.last_hidden_state[:, 0])
        out = self.fc(pooled_output)
        out = out.softmax(dim=1)

        return out


class RobertaModel(torch.nn.Module):
    def __init__(self,num_class,pretrained_weights):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_weights)
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        n_layers = 12
        n_layers_freeze = 6
        n_layers_ft = n_layers - n_layers_freeze
        for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        self.fc = torch.nn.Linear(in_features=768, out_features=num_class)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用预训练模型抽取数据特征
        #         with torch.no_grad():
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        pooled_output = out.pooler_output

        out = self.fc(pooled_output)
        out = out.softmax(dim=1)

        return out