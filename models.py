import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj

class RouteKT(nn.Module):

    def __init__(self, concept_num, question_num, student_num,
                 head, hidden_dim, embedding_dim, 
                 dropout=0.2, bias=True, has_cuda=False, device='cpu'):
        super(RouteKT, self).__init__()
        self.concept_num = concept_num
        self.exercise_num = question_num
        self.student_num = student_num
        self.head = head
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.dropout = dropout
        self.bias = bias 

        self.has_cuda = has_cuda
        self.device = device

        self.conv0 = GATConv(self.exercise_num + self.concept_num, self.embedding_dim, heads=1, 
                             concat=True, dropout=self.dropout)
        
        self.conv1 = GATConv(self.embedding_dim, self.embedding_dim, heads=self.head, 
                             concat=True, dropout=self.dropout)
        self.conv2 = GATConv(self.embedding_dim * self.head, self.embedding_dim, heads=self.head, 
                             concat=False, dropout=self.dropout)
        self.result = nn.Embedding(2+1, self.embedding_dim)    # extra index aims to process null value
        self.lstm_route = nn.LSTM(self.embedding_dim, self.embedding_dim, bias=self.bias, batch_first=True)
        
        self.fc0 = nn.Linear(self.embedding_dim * 3, self.embedding_dim, bias=self.bias)
        
        self.lstm_kt = nn.LSTM(self.embedding_dim, self.hidden_dim, bias=self.bias, batch_first=True)

        self.fc1 = nn.Linear(self.hidden_dim, self.concept_num, bias=self.bias)
        self.sigmoid = nn.Sigmoid()


    def forward(self, students, questions, features, features_len, routes, routes_len, answers, 
                whole_edge_index, whole_edge_attr, edge_index, edge_attr):
        r"""
        Parameters:
            students: student index matrix
            questions: question index matrix
            features: indicator index matrix
            features_len: length of indicator matrix
            routes: route index matrix
            routes_len: route of indicator matrix
            answers: answer matrix
            edge_index: the edge info of graph
            edge_attr: the edge value of graph
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            students: [batch_size, seq_len]
            questions: [batch_size, seq_len]
            features: [batch_size, seq_len, indicator_max_len]
            features_len: [batch_size, seq_len]
            routes: [batch_size, seq_len, route_max_len]
            routes_len: [batch_size, seq_len]
            answers: [batch_size, seq_len]
            edge_index: [2, edge_num]
            edge_attr: [edge_num, 1]
            pred_res: [batch_size, seq_len - 1]
            attn_res: [edge_num, head]
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
            attn_1: the head attention value for each edge (layer-1)
            attn_2: the head attention value for each edge (layer-2)
            state: each student's mastery on each concept
            batch_size: int
        """
        batch_size, seq_len = questions.shape
        route_max_len = routes.shape[-1]
        
        W = torch.arange(start=0, end=self.exercise_num + self.concept_num, step=1).long().to(features.device)
        W = F.one_hot(W, num_classes=self.exercise_num + self.concept_num).float()    
        # [exercise_num + concept_num, exercise_num + concept_num]
        W, (edge_0, attn_0) = self.conv0(W, whole_edge_index, return_attention_weights=True)
        # [exercise_num + concept_num, 1*embedding_dim] [2, whole_edge_num] [whole_edge_num, 1]
        W, edge_0, attn_0 = W, edge_0, attn_0[:,0]
        # [exercise_num + concept_num, embedding_dim] [2, whole_edge_num] [whole_edge_num]

        
        # deal with the attention value
        batch = torch.zeros((self.exercise_num + self.concept_num,)).long().to(features.device)
        attn_0 = to_dense_adj(edge_0, batch, attn_0, max_num_nodes=self.exercise_num + self.concept_num)[0]    
        # [exercise_num + concept_num, exercise_num + concept_num]
        attn_0 = attn_0[:self.exercise_num, self.exercise_num:]    # [exercise_num, concept_num]

        # split question and KC
        Q, X = W[:self.exercise_num], W[self.exercise_num:]    
        # [exercise_num, embedding_dim] [concept_num, embedding_dim]
        Q = torch.concat((Q, torch.zeros((1, self.embedding_dim)).to(Q.device)), dim=0)    # [exercise_num+1, embedding_dim]

        # input the graph data
        X = self.conv1(X, edge_index, edge_attr)    
        # [concept_num, head*embedding_dim]
        X = F.relu(X)
        X = self.conv2(X, edge_index, edge_attr)
        # [concept_num, embedding_dim]
        
        X = torch.concat((X, torch.zeros((1, self.embedding_dim)).to(X.device)), dim=0)
        # [concept_num+1, embedding_dim]

        # encode the promblem-solving route
        rt = X[routes]   # [batch_size, seq_len, route_max_len, embedding_dim]
        rt = rt.view(-1, route_max_len, self.embedding_dim)    # [batch_size*seq_len, route_max_len, embedding_dim]
        xt, (_, _) = self.lstm_route(rt)    # [batch_size*seq_len, route_max_len, embedding_dim]
        # xt = F.relu(self.fcr(xt))    # [batch_size*seq_len, route_max_len, embedding_dim]
        xt = xt.view(batch_size, seq_len, route_max_len, self.embedding_dim)
        # [batch_size, seq_len, route_max_len, embedding_dim]
        index = (routes_len-1).unsqueeze(2).unsqueeze(3).repeat(1, 1, route_max_len, self.embedding_dim)
        # [batch_size, seq_len, route_max_len, embedding_dim]

        index = torch.where(index < 0, 0, index)
        xt = torch.gather(xt, 2, index)[:,:,0,:]     # [batch_size, seq_len, embedding_dim]
        
        qt = Q[torch.where(questions < 0, self.exercise_num, questions).long()]    # [batch_size, seq_len, embedding_dim]
        r = self.result(torch.where(answers < 0, 2, answers).long())    # [batch_size, seq_len, embedding_dim]
        
        xt = torch.concat((qt, xt, r), dim=-1)    # [batch_size, seq_len, embedding_dim+embedding_dim+embedding_dim]
        xt = F.relu(self.fc0(xt))                 # [batch_size, seq_len, embedding_dim]

        ht, (_, _) = self.lstm_kt(xt)    # [batch_size, seq_len, hidden_dim]
        state = self.sigmoid(self.fc1(ht))    # [batch_size, seq_len, concept_num]
        
        pred_res = self._get_next_pred(attn_0, questions, routes, state)    # [batch_size, seq_len-1]
        
        return batch_size, attn_0, state, pred_res
    

    def _get_next_pred(self, attn_0, questions, routes, state):
        r"""
        Parameters:
            state: each student's mastery on each concept

        Shape:
            attn_0: [exercise_num, concept_num]
            state: [batch_size, seq_len, concept_num]
            questions: [batch_size, seq_len]
            routes: [batch_size, seq_len, route_max_len]
            pred: [batch_size, seq_len-1]
        Return:
            pred: predicted correct probability of each question answered at every next timestamp
        """
        
        new_questions = torch.where(questions < 0, 0, questions)    # [batch_size, seq_len]
        index = attn_0[new_questions[:,1:]]    # [batch_size, seq_len-1, concept_num]


        # calculate weighted mean based on the attention value 
        whole_index = index.sum(dim=-1)    # [batch_size, seq_len-1]
        whole_index = torch.where(whole_index > 0.0, whole_index, 1.0)
        pred = ((state[:,:-1,:] * index).sum(dim=-1)) / whole_index    # [batch_size, seq_len-1]
        
        return pred
