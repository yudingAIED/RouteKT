import os
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from utils import generate_graph, cal_graph_attention

class KTDataset(Dataset):
    def __init__(self, students, questions, features, features_len, routes, routes_len, answers):
        super(KTDataset, self).__init__()
        self.students = students          
        self.questions = questions         
        self.features = features           
        self.features_len = features_len   
        self.routes = routes               
        self.routes_len = routes_len       
        self.answers = answers             

    def __getitem__(self, index):
        return self.students[index], self.questions[index], self.features[index], self.features_len[index], self.routes[index], self.routes_len[index], self.answers[index]

    def __len__(self):
        return len(self.questions)


def pad_collate(batch):
    (students, questions, features, features_len, routes, routes_len, answers) = zip(*batch)
    students = [torch.LongTensor(stu) for stu in students]
    questions = [torch.LongTensor(qt) for qt in questions]
    features = [torch.LongTensor(feat) for feat in features]
    features_len = [torch.LongTensor(feat_len) for feat_len in features_len]
    routes = [torch.LongTensor(rout) for rout in routes]
    routes_len = [torch.LongTensor(rout_len) for rout_len in routes_len]
    answers = [torch.FloatTensor(ans) for ans in answers]

    student_pad = pad_sequence(students, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    feature_len_pad = pad_sequence(features_len, batch_first=True, padding_value=-1)
    route_pad = pad_sequence(routes, batch_first=True, padding_value=-1)
    route_len_pad = pad_sequence(routes_len, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    return student_pad, question_pad, feature_pad, feature_len_pad, route_pad, route_len_pad, answer_pad


def read_KG(KG_path):
    df = pd.read_excel(KG_path, sheet_name='学习表现指标')
    origin_concept = set()
    indicator2id = dict()
    for index, row in df.iterrows():
        indicator2id[row['核心概念代码'] + '*' + row['学习表现指标']] = index
        origin_concept.add(row['核心概念代码'])
    return indicator2id, len(indicator2id.keys()), len(origin_concept)


def process_feature_data(s_knowledge, s_indicator, indicator2id):

    x = []
    L_knowledge = s_knowledge.split('!@#$%')
    L_indicator = s_indicator.split('!@#$%')

    for i in range(0,len(L_knowledge)):
        k = L_knowledge[i] + '*' + L_indicator[i]
        x.append(str(indicator2id[k]))
    
    return '#'.join(x)

def process_route_data(s, concept_num, indicator2id):

    x = []
    L = s.replace('!@#$%', '#').split('#')
    
    for i in range(0,len(L)):
        k = L[i]
        x.append(str(indicator2id[k]))
    
    return '#'.join(x)


def count_concept(x):
    return len(x.split('#'))


def process_data(s, max_len, concept_num):

    
    x = [concept_num] * max_len
    L = str(s).split('#')
    for i in range(0,len(L)):
        x[i] = int(L[i])
    x = np.array(x)
    
    return x



def my_load_dataset(file_path, dataset_file_path, KG_file_path, batch_size, train_ratio=0.7, val_ratio=0.2,
                    shuffle=True, use_cuda=True, device='cpu'):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        exercise_file_path: input file path of exercise data
        KG_path: input file path of KG data
        batch_size: the size of a student batch
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions)
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    """
    KG_path = os.path.join(file_path, KG_file_path)
    indicator2id, concept_num, origin_concept_num = read_KG(KG_path)

    dataset_path = os.path.join(file_path, dataset_file_path)
    df = pd.read_csv(dataset_path)
    df.loc[df['full_score'] == 'n.a.', 'full_score'] = 1
    df['score'], df['full_score'] = df['score'].astype(int), df['full_score'].astype(int)
    df['score'] = (df['score'] // df['full_score']).astype(int)
    df.loc[df['score'] > 1, 'score'] = 1

    
    attemp_question = df['question_id'].value_counts().mean()
    print('attemp_question: ', attemp_question)
    
    interaction_num = int(len(df))
    
    df['question_id'] = pd.factorize(df['question_id'])[0].astype(int)
    df['student_id'] = pd.factorize(df['student_id'])[0].astype(int)
    student_num = df['student_id'].max() + 1
    question_num = df['question_id'].max() + 1
    df = df.groupby('student_id').filter(lambda q: len(q) > 1).copy()
    

    
    df['indicator_code'] = df.apply(lambda row: process_feature_data(row["knowledge_code"], row["ability_code"], indicator2id), axis=1)
    df['route_code'] = df['extra_route'].apply(process_route_data, args=(concept_num, indicator2id))
    
    
    df['indicator_num'] = df['indicator_code'].apply(count_concept)
    df['route_num'] = df['route_code'].apply(count_concept)

    
    indicator_max_len = int(df['indicator_num'].max())
    route_max_len = int(df['route_num'].max())
    
    
    df['indicator_code'] = df['indicator_code'].apply(process_data, args=(indicator_max_len, concept_num))
    df['route_code'] = df['route_code'].apply(process_data, args=(route_max_len, concept_num))
    
    
    graph = np.zeros((concept_num, concept_num), dtype=int)
    graph_whole = np.zeros((question_num + concept_num, question_num + concept_num), dtype=int)
    question2route = dict()
    for _, row in df.iterrows():
        question2route[row['question_id']] = row['route_code']
    
    route_len_count = []
    for key, value in question2route.items():
        # print(value)
        # route = value.split('#')
        step = 0
        for i in range(0, value.shape[0]-1):
            if value[i] != concept_num and value[i+1] != concept_num:
                step += 1
                graph[int(value[i])][int(value[i+1])] += 1

                graph_whole[int(key)][int(value[i])+question_num] = 1
                graph_whole[int(key)][int(value[i+1])+question_num] = 1
        route_len_count.append(step+1)
    # print(route_len_count)
    # print(len(route_len_count))
    print("the average length of all problem-solving routes：", sum(route_len_count) / len(route_len_count))
    
    
    whole_edge_index, whole_edge_attr = generate_graph(graph_whole)
    whole_edge_index = torch.from_numpy(whole_edge_index)
    whole_edge_attr = torch.from_numpy(whole_edge_attr)

    edge_index, edge_attr = generate_graph(graph)
    edge_index = torch.from_numpy(edge_index)
    edge_attr = torch.from_numpy(edge_attr)

    
    edge_real, attn_real = cal_graph_attention(graph)
    edge_real = torch.from_numpy(edge_real)
    attn_real = torch.from_numpy(attn_real)

    feature_list = []
    feature_len_list = []
    route_list = []
    route_len_list = []
    student_list = []
    question_list = []
    answer_list = []
    seq_len_list = []

    def get_data(series):
        student_list.append(series['student_id'].tolist())
        question_list.append(series['question_id'].tolist())
        feature_list.append(series['indicator_code'].tolist())
        feature_len_list.append(series['indicator_num'].tolist())
        route_list.append(series['route_code'].tolist())
        route_len_list.append(series['route_num'].tolist())
        answer_list.append(series['score'].astype('float').tolist())
        seq_len_list.append(series['score'].shape[0])


    df.groupby('student_id').apply(get_data)
    max_seq_len = np.max(seq_len_list)
    mean_seq_len = np.mean(seq_len_list)
    print('max_seq_len: ', max_seq_len)
    print('mean_seq_len: ', mean_seq_len)
    
    print('student num: ', student_num)
    print('interaction_num: ', interaction_num)
    
    print('question_num: ', question_num)
    print('origin_concept_num: ', origin_concept_num)
    print('concept_num: ', concept_num)


    data_size = len(seq_len_list)

    
    kt_dataset = KTDataset(student_list, question_list, 
                           feature_list, feature_len_list, route_list, route_len_list, answer_list)
    train_size = int(train_ratio * data_size)
    val_size = int(val_ratio * data_size)
    test_size = data_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset,
                                                                             [train_size, val_size, test_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    

    return int(student_num), int(concept_num), int(question_num), \
        whole_edge_index, whole_edge_attr, edge_index, edge_attr, edge_real, attn_real, \
        train_data_loader, valid_data_loader, test_data_loader
