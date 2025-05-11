import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    NOTE: Stole and modify from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    """

    def __init__(self, temperature, attn_dropout=0.):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout

    def forward(self, q, k, mask=None):
        r"""
        Parameters:
            q: multi-head query matrix
            k: multi-head key matrix
            mask: mask matrix
        Shape:
            q: [n_head, mask_num, embedding_dim]
            k: [n_head, concept_num, embedding_dim]
        Return: attention score of all queries
        """
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))  # [n_head, mask_num, concept_num]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # pay attention to add training=self.training!
        attn = F.dropout(F.softmax(attn, dim=0), self.dropout, training=self.training)  # pay attention that dim=-1 is not as good as dim=0!
        return attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    NOTE: Stole and modify from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
    """

    def __init__(self, n_head, concept_num, hidden_dim, input_dim, d_k, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.concept_num = concept_num
        self.d_k = d_k
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, input_dim, bias=True)
        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        # inferred latent graph, used for saving and visualization
        self.init_params()
    
    def init_params(self):
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.kaiming_normal_(self.w_qs.weight)
        nn.init.kaiming_normal_(self.w_ks.weight)
        nn.init.constant_(self.fc.bias, 0)


    def forward(self, query, key, mask=None):
        r"""
        Parameters:
            query: answered state for a student batch  截止目前的学生回答状态
            key: concept embedding matrix  各个知识点的表征
            mask: mask matrix
        Shape:
            query: [mask_num, hidden_dim]
            key: [concept_num, embedding_dim]
        Return:
            graphs: n_head types of inferred graphs
        """
        d_k, n_head = self.d_k, self.n_head
        len_q, len_k = query.size(0), key.size(0)

        # Pass through the pre-attention projection: lq x (n_head *dk)
        # Separate different heads: lq x n_head x dk
        query = self.fc(query)  # [mask_num, input_dim]
        q = self.w_qs(query).view(len_q, n_head, d_k)
        k = self.w_ks(key).view(len_k, n_head, d_k)

        # Transpose for attention dot product: n_head x lq/lk x dk
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        attn_score = self.attention(q, k, mask=mask)  # [n_head, mask_num, concept_num]
        # 表示当前状态下，学生对各个知识点的重视程度
        return attn_score

class PosLinear(nn.Linear):
    """
    权重保持为正的全连接层
    """
    def forward(self, input):
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class EraseAddGate(nn.Module):
    """
    Erase & Add Gate module
    表示学生经过当前学习的遗忘与习得后的知识状态
    """

    def __init__(self, n_head, feature_dim, concept_num, bias=True):
        super(EraseAddGate, self).__init__()
        self.head = n_head
        self.feature_dim = feature_dim    # 等价于hidden_dim
        self.concept_num = concept_num
        # 对多头注意力机制进行合并后的运算结果
        self.attention_fc = nn.Linear(self.feature_dim * self.head, self.feature_dim, bias=False)
        # erase gate
        # self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        # add gate
        # self.add = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.attention_fc.weight)
        # nn.init.kaiming_normal_(self.erase.weight)
        # nn.init.kaiming_normal_(self.add.weight)
        # nn.init.constant_(self.attention_fc.bias, 0)
        # nn.init.constant_(self.erase.bias, 0)
        # nn.init.constant_(self.add.bias, 0)


    def forward(self, ht, change_knowledge, attention, alpha):
        r"""
        Params:
            ht: 初始情况
            change_knowledge: input feature matrix 当前各学生当前在各个知识点上的隐状态向量变化
            attention: attention of each student on each concept at present 学生在当前对于各个知识点的关注与重视
            alpha: alpha of each student at present
        Shape:
            ht: [batch_size, concept_num, hidden_dim]
            change_knowledge: [batch_size, concept_num, hidden_dim]
            attention: [n_head, batch_size, concept_num]
            alpha: [batch_size]
            res: [batch_size, concept_num, hidden_dim]
        Return:
            res: returned feature matrix with old information erased and new information added
        """
        # erase_knowledge = self.erase(x)  # [batch_size, concept_num, hidden_dim]
        # add_knowledge = self.add(x)  # [batch_size, concept_num, hidden_dim]
        # change_knowledge = add_knowledge - erase_knowledge  # [batch_size, concept_num, hidden_dim]
        #print('1', torch.isnan(ht).any())
        #print('2', torch.isnan(change_knowledge).any())
        real_change = change_knowledge.unsqueeze(0).repeat(self.head,1,1,1) * attention.unsqueeze(-1).repeat(1,1,1,self.feature_dim)  
        # [n_head, batch_size, concept_num, hidden_dim]
        #print('3', torch.isnan(real_change).any())
        real_change = torch.cat(torch.split(real_change, [1]*self.head, dim=0), dim=-1)[0]  
        # [batch_size, concept_num, n_head*hidden_dim]
        #print('4', torch.isnan(real_change).any())
        #print(real_change.shape)
        # real_change = F.relu(self.attention_fc(real_change))    # [batch_size, concept_num, hidden_dim]
        #print('5', torch.isnan(real_change).any())
        # print(x.shape)
        # print(real_change.shape)
        # print(alpha.shape)
        # 保证有一部分是变化的
        res = ht + real_change + real_change * alpha.unsqueeze(-1).repeat(1,self.concept_num).unsqueeze(-1).repeat(1,1,self.feature_dim)
        # [batch_size, concept_num, hidden_dim]
        #print('6', torch.isnan(res).any())
        return res