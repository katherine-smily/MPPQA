import numpy as np
import pandas as pd
import pickle
import csv
import os
import torch
import sys
import jsonlines
import argparse
import random
random.seed(42)
import itertools
import logging
from collections import defaultdict

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from utils import json_util
from transformers import PreTrainedTokenizer

import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_scatter import scatter_add

def generate_graph_data(triplets, num_rels):
    edges = np.array(triplets)
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))

    src = torch.tensor(src, dtype=torch.long).contiguous()
    dst = torch.tensor(dst, dtype=torch.long).contiguous()
    rel = torch.tensor(rel, dtype=torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel
    # data={"edge_index":edge_index,
    #       "entity":torch.from_numpy(uniq_entity),
    #       "edge_type":edge_type,
    #       "edge_norm":edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    #       }
    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)

    return data

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    # 把边进行独热编码 输入是tensor 和 编码长度
    one_hot = F.one_hot(edge_type, num_classes=2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

def read_examples_from_file(data, connection_type, link2id, linkType2id):
    max_node_num_per_process = 0
    guid_index = 1
    sent_examples = []
    pos_of_examples = []
    syn_of_examples = []
    example_size = []
    edges = []
    atten_edges = []
    conn_edges = []
    examples_added = 0
    sent_types = []  # sentence_type
    conn_edge_types = []

    tokenize_sents=[]
    # bio_tags=[]
    sent_tags=[] # 句子是否为答案的标签
    tokenidx2sentenceidxs_list=[]
    def get_tokenidx2sentenceidxs(sent_data):
        tokenidx2sentenceidx={}
        context_token_idx=0
        for idx, eachsentattr in enumerate(sent_data):
            if idx==0:
                continue
            for sentence_token_idx,token in enumerate(eachsentattr['tokenize_sent']):
                tokenidx2sentenceidx[sentence_token_idx+context_token_idx]=idx
            context_token_idx+=len(eachsentattr['tokenize_sent'])
        return tokenidx2sentenceidx
    
    for i,sent_data in enumerate(data['sent_data']):
        tokenidx2sentenceidxs_list.append(get_tokenidx2sentenceidxs(sent_data))
        if len(sent_data)<=1:
            continue
        if max_node_num_per_process < len(sent_data):
            max_node_num_per_process =  len(sent_data)
        # print(filetext)
        edge_index = []
        each_process_edge_type = []
        each_process_sen_type = []
        example = []
        pos_of_example = []
        syn_of_example = []
        tokenize_sent=[]
        # bio_tag=[]
        sent_tag=[]

        for idx, eachsentattr in enumerate(sent_data):
            # words = eachsentattr['sentence'].split()
            # words = [value for key, value in eachsentattr['id2token'].items()]
            words=eachsentattr['tokenize_sent']
            example.append(words)
            pos_of_example.append(eachsentattr["pos"])
            syn_of_example.append(eachsentattr['syntatic_triplets'])
            each_process_sen_type.append(eachsentattr['sen_type'])
            guid_index += 1
            relations = eachsentattr['relation_id']  # 即关联的关系
            relation_types = eachsentattr['rel_type']  # 关系的类型

            tokenize_sent.append(eachsentattr['tokenize_sent'])
            # bio_tag.append(eachsentattr['bio_tag'])
            sent_tag.append(eachsentattr['sent_tag'])

            for each_rel in relations:
                edge_index.append((idx, each_rel))

            for each_type in relation_types:
                each_process_edge_type.append(each_type)

        examples_added += len(sent_data)
    
        example_size.append(len(sent_data))
        conn_edges.append(edge_index)
        conn_edge_types.append(each_process_edge_type)
        sent_types.append([link2id[type_name] for type_name in each_process_sen_type])  # 等待

        tokenize_sents.append(tokenize_sent)
        # bio_tags.append(bio_tag)
        sent_tags.append(sent_tag)

        edges.append(list(itertools.combinations(range(0, len(sent_data)), 2)))  # 穷举了所有
        sent_examples.append(example)
        pos_of_examples.append(pos_of_example)
        syn_of_examples.append(syn_of_example)

    link_labels = []
    link_type_labels = []
    for ee, ec, et in zip(edges, conn_edges, conn_edge_types):
        each_label = []
        each_link_type_label = []
        type_idx = 0
        for e in ee:
            if e in ec:
                each_label.append(1)
                each_link_type_label.append(linkType2id[et[type_idx]])
                type_idx += 1
            else:
                each_label.append(0)
                each_link_type_label.append(linkType2id['none'])
        link_labels.append(each_label)
        link_type_labels.append(each_link_type_label)
    print("Examples to Graph Structure Stats :", len(edges), len(conn_edges),len(example_size),len(sent_examples), len(link_labels), len(link_type_labels))

    if connection_type == "complete":
        conn_edges = edges
    elif connection_type == "linear":
        linear_edges = []
        for each in edges:
            eachgraph_linear_edges = []
            for each_edge in each:
                # First node in edge is 1 diff from other node
                if each_edge[1]-each_edge[0] == 1:
                    eachgraph_linear_edges.append(each_edge)
            linear_edges.append(eachgraph_linear_edges)
        conn_edges = linear_edges

    return sent_examples, pos_of_examples, syn_of_examples, conn_edges, edges, link_labels, link_type_labels, sent_types, max_node_num_per_process,tokenize_sents,sent_tags,tokenidx2sentenceidxs_list
# sent_examples, pos_of_examples, syn_of_examples, conn_edges, edges, link_labels, link_type_labels, sent_types, max_node_num_per_process,tokenize_sents,sent_tags,tokenidx2sentenceidxs_list
def convert_examples_to_features(examples : List, pos_of_examples: List, syn_of_examples: List, max_seq_length: int,
                                 pos2id: dict, syn_rel2id: dict, tokenizer: PreTrainedTokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]",
                                 cls_token_segment_id=0, sep_token="[SEP]", sep_token_extra=False, pad_on_left=False,
                                 pad_token=0, pad_token_segment_id=0, pad_token_label_id=-100, sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):  # cls_token_segment_id=1 to check why it was 1
    all_features = []
    for each_process, each_pos_of_process, each_syn_of_process in zip(examples, pos_of_examples, syn_of_examples):  # 遍历每一个程序
        features = []

        for example, pos_of_example, syn_of_example in zip(each_process, each_pos_of_process, each_syn_of_process):
            # 每个example为一个句子
            tokens = []
            pos_of_tokens = []
            synid_of_tokens = []
            input_ids = []

            # if "roberta_1" not in model_type:  # update  ？？ 为什么要拒绝roberta_1?
            word_idx = 1  # 语法解析是从1开始，对单词进行编码。下标 （0， 1）默认为root语法关系
            for word, pos in zip(example, pos_of_example):
                word_tokens = tokenizer.tokenize(word)  # =tokenizer(word)
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    pos_of_tokens.extend([pos] * len(word_tokens))  # 如果单词被拆分为多个时候，词性也对应对应复制
                    synid_of_tokens.extend([word_idx] * len(word_tokens))  # 同理，如果单词
                word_idx += 1

            # Account for [CLS] and [SEP] with "- 2"
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                pos_of_tokens = pos_of_tokens[: (max_seq_length - special_tokens_count)]
                synid_of_tokens = synid_of_tokens[: (max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            pos_of_tokens += [sep_token]
            synid_of_tokens += [0]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                pos_of_tokens += [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)  # [ 0, 0, 0, 1 , 1, 1 ]
            if cls_token_at_end:
                tokens += [cls_token]
                pos_of_tokens += [cls_token]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                pos_of_tokens = [cls_token] + pos_of_tokens
                synid_of_tokens = [0] + synid_of_tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            """
                ID 化 
            """
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            pos_ids = [pos2id[pos_item] for pos_item in pos_of_tokens]
            syn_input_ids = synid_of_tokens

            # todo：存在advcl:relcl
            # for i,triplet in enumerate(syn_of_example):
            #     if triplet[1]=="advcl:relcl":
            #         syn_of_example[i][1]="advcl"
            syn_triplets = [[int(triplet[0]), syn_rel2id[triplet[1]], int(triplet[2])] for triplet in syn_of_example]


            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            # input_ids长度为max_seq_length
            input_ids += [pad_token] * padding_length
            pos_ids += [pad_token] * padding_length
            syn_input_ids += [0] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length

            # 过滤超过max_length的三元组进行删除
            filtered_syn_triplets = []
            for triplet in syn_triplets:
                if triplet[0] in syn_input_ids and triplet[2] in syn_input_ids:
                    filtered_syn_triplets.append(triplet)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(pos_ids) == max_seq_length
            assert len(syn_input_ids) == max_seq_length

            graph_data = generate_graph_data(filtered_syn_triplets, num_rels=len(syn_rel2id))
            # graph_data的key: ['edge_index', 'entity', 'edge_type', 'edge_norm']
            features.append([tokens, input_ids, input_mask, segment_ids, pos_ids, syn_input_ids, graph_data])
        all_features.append(features)
    return all_features

def oversample_positive_data(edges, labels):
    new_edges, new_labels = [], []
    for eedges, elabels in zip(edges, labels):
        sel_edges, sel_labels = [], []
        for e, l in zip(eedges, elabels):
            sel_edges.append(e)
            sel_labels.append(l)
            if l==1:  # If there is an edge n1n2 oversample that with n2n1 since procedural text no back direction
                sel_edges.append((e[1],e[0]))
                sel_labels.append(l)
                
        new_edges.append(sel_edges)
        new_labels.append(sel_labels)
        
    return new_edges, new_labels
    
def find_label_stats(labels):
    if labels==[]:
        return 0
    l = defaultdict(int)
    for each in labels:
        for e in each:
            l[e]+=1
            
    return l[0], l[1], round(l[1]/(l[0]+l[1]),2)

def prepare_data(examples,filepath, max_seq_len, tokenizer, window, is_oversample, graph_connection):
    link2id = json_util.load(os.path.join(filepath, 'link2id.json'))
    # {'none': 0, 'supplement': 1, 'sub_action': 2, 'next_action': 3}
    linkType2id = json_util.load(os.path.join(filepath, 'linkType2id.json'))
    # {'VERB': 0, 'ADP': 1, 'PROPN': 2, 'ADJ': 3, 
    pos2id = json_util.load(os.path.join(filepath, 'pos2id.json'))
    # {'nummod': 0, 'compound': 1, 'vocative': 2
    syn_rel2id = json_util.load(os.path.join(filepath, 'syn_rel2id.json'))

    sent_examples, pos_of_examples, syn_of_examples, conn_edges, edges, link_labels, link_type_labels, sent_types, max_node_num_per_process,tokenize_sents,sent_tags,tokenidx2sentenceidxs_list=\
        read_examples_from_file(examples, graph_connection,  link2id, linkType2id)
    
    features = convert_examples_to_features(sent_examples, pos_of_examples, syn_of_examples, max_seq_len,
                                 pos2id, syn_rel2id, tokenizer)

    if is_oversample:
        edges, link_labels = oversample_positive_data(edges, link_labels)
    # if window:
    #     edges, conn_edges, labels, link_type_labels = windowed_pairs_selection(edges, conn_edges, link_labels, link_type_labels, window)  # 按照windows进行切割

    _, _, edge_percent = find_label_stats(link_labels)

    # logger.info("Feature length Train: %d", len(features))
    return features, conn_edges, edges, link_labels, link_type_labels, sent_types, max_node_num_per_process,tokenize_sents,sent_tags,tokenidx2sentenceidxs_list,edge_percent

    # 这里也可以集成一下结果
    # 梳理下上面的数据流以及数据表示

# 集成到后面
# def convert_example_to_multispan_features(examples,tokenizer):
#     max_seq_length = min(512, tokenizer.model_max_length)
#     # input_ids有2058？，examples有448，这里context超过被truncate成多个input_ids
#     tokenized_examples = tokenizer(
#         examples["questions"],
#         examples["contexts"],
#         truncation="only_second",
#         max_length=max_seq_length,
#         # stride=data_args.doc_stride,
#         stride=128,
#         return_overflowing_tokens=True,
#         return_offsets_mapping=True,
#         # padding=False,
#         padding="max_length",
#         is_split_into_words=True,
#     )
#     word_id2sent_ids=examples["tokenidx2sentenceidxs"]
#     tokenized_examples["input_ids_idx_to_sent_idx_multispan"]=[]
    
#     if examples["ids"]==[]: # 对于空的test集
#         return tokenized_examples

#     sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
#     # sample_mapping对应id所在数据段，i对应input_ids的行数
#     for i, sample_index in enumerate(sample_mapping):
#         # Grab the sequence corresponding to that example (to know what is the context and what is the question).
#         sequence_ids = tokenized_examples.sequence_ids(i)
#         # Start token index of the current span in the text.
#         token_start_index = 0
#         while sequence_ids[token_start_index] != 1:
#             token_start_index += 1
#         word_ids=tokenized_examples.word_ids(i)

#         input_ids_idx_to_sent_idx=[]
#         none_cnt=0
#         for word_id in word_ids:
#             if word_id==None:
#                 none_cnt+=1
#                 input_ids_idx_to_sent_idx.append([sample_index,None])
#                 continue
#             if none_cnt<2:
#                 input_ids_idx_to_sent_idx.append([sample_index,0])
#                 continue
#             sent_idx=word_id2sent_ids[sample_index][word_id]
#             input_ids_idx_to_sent_idx.append([sample_index,sent_idx])

#         tokenized_examples["input_ids_idx_to_sent_idx_multispan"].append(input_ids_idx_to_sent_idx)
#     return tokenized_examples

# def prepare_data_graph(features, conn_edges, edges, labels,  sent_types,tokenize_sents,sent_tags,input_ids_idx_to_sent_idx):
#     # 这里可以集成原先的处理方式

#     pass