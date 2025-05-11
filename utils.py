import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Calculate accuracy of prediction result and its corresponding label
# output: tensor, labels: tensor
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels.reshape(-1)).double()
    correct = correct.sum()
    return correct / len(labels)


# transform the graph data
def generate_graph(matrix):
    n = matrix.shape[0]
    edge_index, edge_attr = [[],[]], []
    for i in range(0,n):
        for j in range(0,n):
            if matrix[i][j] > 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_attr.append([matrix[i][j]])
    
    edge_index = np.array(edge_index, dtype=np.int64)
    edge_attr = np.array(edge_attr, dtype=np.float32)
    return edge_index, edge_attr


# calculate the ground truth attention value
def cal_graph_attention(matrix):
    n = matrix.shape[0]
    edge_index_attention, edge_attr_attention = [[],[]], []
    for i in range(0,n):
        for j in range(0,n):
            if matrix[i][j] > 0:
                edge_index_attention[0].append(i)
                edge_index_attention[1].append(j)
                whole_value = np.sum(matrix[i, :])
                edge_attr_attention.append(matrix[i][j]/whole_value)

    edge_index_attention = np.array(edge_index_attention, dtype=np.int64)
    edge_attr_attention = np.array(edge_attr_attention, dtype=np.float32)
    return edge_index_attention, edge_attr_attention
