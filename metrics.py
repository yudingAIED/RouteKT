import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils import accuracy

# Calculate loss/AUC/ACC
class KTLoss(nn.Module):

    def __init__(self):
        super(KTLoss, self).__init__()

    def forward(self, pred_answers, real_answers):
        r"""
        Parameters:
            pred_answers: the correct probability of questions answered at the next timestamp
            real_answers: the real results(0 or 1) of questions answered at the next timestamp
        Shape:
            # the lengths of the two tensors are diffenrent!
            pred_answers: [batch_size, seq_len - 1]
            real_answers: [batch_size, seq_len]
        Return:
        """
        real_answers = real_answers[:, 1:]  # timestamp=1 ~ T
        # real_answers shape: [batch_size, seq_len - 1]
        # Here we can directly use nn.BCELoss, but this loss doesn't have ignore_index function
        answer_mask = torch.ne(real_answers, -1)
        pred_one, pred_zero = pred_answers, 1.0 - pred_answers  # [batch_size, seq_len - 1]

        # calculate auc and accuracy metrics
        try:
            y_true = real_answers[answer_mask].cpu().detach().numpy()
            y_pred = pred_one[answer_mask].cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred)  # may raise ValueError
            output = torch.cat((pred_zero[answer_mask].reshape(-1, 1), pred_one[answer_mask].reshape(-1, 1)), dim=1)
            label = real_answers[answer_mask].reshape(-1, 1)
            acc = accuracy(output, label)
            acc = float(acc.cpu().detach().numpy())
        except ValueError as e:
            auc, acc = -1, -1

        # calculate NLL loss
        """
        pred_one[answer_mask] = torch.log(pred_one[answer_mask])
        pred_zero[answer_mask] = torch.log(pred_zero[answer_mask])
        pred_answers = torch.cat((pred_zero.unsqueeze(dim=1), pred_one.unsqueeze(dim=1)), dim=1)
        # pred_answers shape: [batch_size, 2, seq_len - 1]
        """
        # print(pred_answers.shape)
        # print(pred_answers[answer_mask].shape)
        # print(pred_answers[answer_mask])
        # print(real_answers.shape)
        # print(real_answers[answer_mask].shape)
        # print(real_answers[answer_mask])
        loss_func = nn.BCELoss()  # ignore masked values in real_answers
        loss = loss_func(pred_answers[answer_mask], real_answers[answer_mask])
        
        return loss, auc, acc

# Calculate loss of the attention value from GAT
def graph_attention_loss(edge_index_attention, edge_attr_attention, edge, attn, concept_num, device):
    edge_index_attention, edge_attr_attention = edge_index_attention.to(device), edge_attr_attention.to(device)
    edge, attn = edge.to(device), attn.to(device)

    edge_index_attention = edge_index_attention[0] * concept_num + edge_index_attention[1]
    edge_index = edge[0] * concept_num + edge[1]

    edge_attention_value, edge_attention_indice = torch.sort(edge_index_attention)
    edge_value, edge_indice = torch.sort(edge_index)

    assert torch.equal(edge_attention_value, edge_value) == True

    loss_func = nn.MSELoss()
    loss = loss_func(edge_attr_attention[edge_attention_indice], attn[edge_indice])
    
    return loss

