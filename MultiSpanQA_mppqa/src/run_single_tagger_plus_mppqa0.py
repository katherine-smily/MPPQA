import os
import sys
sys.path.append(".")
sys.path.append("./")
sys.path.append("../")

import logging
import collections
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss
import datasets
from datasets import load_dataset
import snoop
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import DataParallel
from multigrainedmodel import MultiGrainedAndSynticEnhancedModel
import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
    BertPreTrainedModel,
    BertModel,
    BertLayer
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from trainer import QuestionAnsweringTrainer
from eval_script import *

from utils import json_util
# from metric import *

from graph_util import *

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")
logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"
def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training is False or p == 0:
        return x
    dropout_mask = 1.0 / (1 - p) * torch.bernoulli((1 - p) * (x.new_zeros(x.size(0), x.size(2)) + 1))
    return dropout_mask.unsqueeze(1).expand_as(x) * x

def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3:  # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)
    
class AttentionScore(torch.nn.Module):
    """
    correlation_func = 1, sij = x1^Tx2
    correlation_func = 2, sij = (Wx1)D(Wx2)
    correlation_func = 3, sij = Relu(Wx1)DRelu(Wx2) #缩放点积注意力函数
    correlation_func = 4, sij = x1^TWx2
    correlation_func = 5, sij = Relu(Wx1)DRelu(Wx2)
    correlation_func = 6: sij = Relu(W1x1)^TRelu(W2x2)
    """

    def __init__(self, input_size, hidden_size, do_similarity=False, correlation_func=2):
        super(AttentionScore, self).__init__()
        self.correlation_func = correlation_func
        self.hidden_size = hidden_size

        if correlation_func == 2 or correlation_func == 3:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)
            if do_similarity:
                self.diagonal = torch.nn.Parameter(torch.ones(1, 1, 1) / (hidden_size ** 0.5), requires_grad=False)
            else:
                self.diagonal = torch.nn.Parameter(torch.ones(1, 1, hidden_size), requires_grad=True)

        if correlation_func == 4:
            self.linear = nn.Linear(input_size, input_size, bias=False)

        if correlation_func == 5:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)
        if correlation_func == 6:
            self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
            self.linear2 = nn.Linear(input_size, hidden_size, bias=False)
    # todo： 需要设置句子固定长度
    def forward(self, x1, x2):
        '''
        Input:
        x1: batch x word_num1 x dim
        x2: batch x word_num2 x dim
        Output:
        scores: batch x word_num1 x word_num2
        '''
        x1 = dropout(x1, p=my_dropout_p, training=self.training)
        x2 = dropout(x2, p=my_dropout_p, training=self.training)

        x1_rep = x1
        x2_rep = x2
        batch = x1_rep.size(0)
        word_num1 = x1_rep.size(1)
        word_num2 = x2_rep.size(1)
        dim = x1_rep.size(2)
        if self.correlation_func == 2 or self.correlation_func == 3:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)  # Wx1
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)  # Wx2
            if self.correlation_func == 3:
                x1_rep = F.relu(x1_rep)
                x2_rep = F.relu(x2_rep)
            x1_rep = x1_rep * self.diagonal.expand_as(x1_rep)
            # x1_rep is (Wx1)D or Relu(Wx1)D
            # x1_rep: batch x word_num1 x dim (corr=1) or hidden_size (corr=2,3)

        if self.correlation_func == 4:
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, dim)  # Wx2

        if self.correlation_func == 5:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)  # Wx1
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)  # Wx2
            x1_rep = F.relu(x1_rep)
            x2_rep = F.relu(x2_rep)

        if self.correlation_func == 6:
            x1_rep = self.linear1(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)
            x2_rep = self.linear2(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)
            x1_rep = F.relu(x1_rep)
            x2_rep = F.relu(x2_rep)

        scores = x1_rep.bmm(x2_rep.transpose(1, 2))
        return scores

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def set_seq_dropout(option):  # option = True or False
    global do_seq_dropout
    do_seq_dropout = option

def set_my_dropout_prob(p):  # p between 0 to 1
    global my_dropout_p
    my_dropout_p = p

# todo
class TaggerPlusForMultiSpanQA(BertPreTrainedModel):
    def __init__(self, config, span_lambda,args,sent_model):
        super().__init__(config, span_lambda)
        self.device_procedure = args.device
        self.sent_encode_model=DataParallel(MultiGrainedAndSynticEnhancedModel(config, args).to(self.device_procedure))#.cuda()
        self.sent_encode_model.module.load_state_dict(sent_model.module.state_dict(),strict=False)
        self.sent_encode_model=self.sent_encode_model.module
        self.sent_encode_model.zero_grad()
        # for name, child in self.sent_encode_model.module.named_children():
        #     if name in ['bert']:
        #         continue
        #     for param in child.parameters():
        #         param.requires_grad = False
        self.input_size=config.hidden_size+ args.pos_embedding_size
        self.attention_hidden_size=args.pos_embedding_size   # todo: 这里可以考虑要不要把隐藏层改成其它size
        self.similarity_attention=True
        self.correlation_func=2 # todo： 后期可以尝试不同的func类型
        self.max_sent_num=args.max_sent_num
        self.max_sent_token_num=args.max_sent_token_num

        set_seq_dropout(args.do_seq_dropout)
        set_my_dropout_prob(args.hidden_dropout_prob)

        self.loss_sent_tag = BCEFocalLoss()
        # torch.nn.BCELoss()
        self.loss_token_tag= BCEFocalLoss()
        # torch.nn.BCELoss()
        self.l1loss=torch.nn.L1Loss(reduction='sum')

        self.span_lambda = span_lambda
        self.sent_tag_lambda=args.sent_tag_lambda
        self.token_tag_lambda=args.token_tag_lambda
        self.boundary_lambda=args.boundary_lambda

        self.label2id= config.label2id
        self.num_labels = config.num_labels # （BIO）对应三个label
        self.max_spans = 21
        self.max_pred_spans = 30
        self.H = config.hidden_size # 隐藏层大小

        self.dense = nn.Linear(self.H, self.H)
        # self.bert = BertModel(config) #编码的模型
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent_linear=nn.Linear(self.input_size, 50)
        self.classifier = nn.Linear(self.H+50, config.num_labels)
        self.num_span_outputs = nn.Sequential(nn.Linear(self.H, 64),nn.ReLU(),nn.Linear(64, 1))
        

        config.num_attention_heads=6 # for span encoder
        intermediate_size=1024
        self.span_encoder = BertLayer(config) # spanencoder部分，输入为embedding（bertattention、bertintermediate、bertoutput）
        
        self.scoring = AttentionScore(self.input_size, self.attention_hidden_size, do_similarity=self.similarity_attention,
                                correlation_func=self.correlation_func)  # scores: batch x word_num1 x word_num2
        # self.token_scoring = CrossAttention(self.H,self.attention_hidden_size,self.input_size,self.attention_hidden_size)
        # self.token_scoring=CrossAttentionScore(self.H,self.attention_hidden_size,self.input_size)
        # self.init_weights()
        self.add_att_weight=args.add_att_weight
        self.add_boundary_loss=args.add_boundary_loss
        self.add_sent_tag_loss=args.add_sent_tag_loss
        self.att_init_weight=args.att_init_weight

        self.init_weights()

    def forward(
        self,
        data,
        # input_ids=None,
        # attention_mask=None,
        # token_type_ids=None,
        # labels=None,  #groundtruth的spans，labelsid？
        # num_span=None,  # groundtruth的nums_span数量
    ):
        # print(data)
        input_ids=torch.tensor(data["input_ids"],device=self.device)
        attention_mask=torch.tensor(data["attention_mask"],device=self.device)
        token_type_ids=torch.tensor(data["token_type_ids"],device=self.device)
        labels=torch.tensor(data["labels"],device=self.device)  #groundtruth的spans，labelsid？
        num_span=torch.tensor(data["num_span"],device=self.device)  # groundtruth的nums_span数量

        # num_nodes=data.num_nodes

        batch_num_nodes=[]
        for d in data["edge_index"]:
            batch_num_nodes.append(len(d))
        
        batch_size=len(batch_num_nodes)
        sent_tag_loss=0

        if self.add_sent_tag_loss:
            # self.sent_encode_model.module.eval()
            sen_repr_gru_syn = self.sent_encode_model(data)[-1]["sen_repr_gru_syn"]
            # self.sent_encode_model.module.train()
            # repr=sent_model_output[-1]
            x_sen_repr_gru_syn=torch.zeros(batch_size,self.max_sent_num,sen_repr_gru_syn.size(1),device=self.device_procedure)
            begin_idx=0
            for i,each_batch_num_nodes in enumerate(batch_num_nodes):
                tmp_each_sent_repr=sen_repr_gru_syn[begin_idx:begin_idx+each_batch_num_nodes]
                # tmp_each_sent_repr=repr['sen_repr_gru_syn'][i]
                begin_idx+=each_batch_num_nodes
                # if tmp_each_sent_repr.size(0)<self.max_sent_num:
                #     zero_repr=torch.zeros(self.max_sent_num-tmp_each_sent_repr.size(0),tmp_each_sent_repr.size(1)).to(tmp_each_sent_repr)
                #     tmp_each_sent_repr=torch.cat([tmp_each_sent_repr,zero_repr],dim=0)
                # else:
                #     tmp_each_sent_repr=tmp_each_sent_repr[:self.max_sent_num]
                # x_sen_repr_gru_syn[i]=tmp_each_sent_repr

                each_sent_repr=torch.zeros(self.max_sent_num,tmp_each_sent_repr.size(1),device=self.device_procedure) #.to(tmp_each_sent_repr)
                each_sent_repr[:min(tmp_each_sent_repr.size(0),self.max_sent_num),:]=tmp_each_sent_repr
                x_sen_repr_gru_syn[i]=each_sent_repr

            # 读取question作为x2
            x_question_repr=x_sen_repr_gru_syn[:,0,:]
            x_question_repr=x_question_repr.reshape(x_sen_repr_gru_syn.size(0),1,-1)
            # scores: batch x word_num1 x word_num2 b,1,512
            att_score=self.scoring(x_question_repr,x_sen_repr_gru_syn)

            # 读取label
            y_sent_tag=torch.zeros(batch_size,self.max_sent_num,dtype=torch.float,device=self.device_procedure) #.to(self.device_procedure)
            # with snoop:
            batch_sent_tag=data["sent_tag"]
                # print(batch_sent_tag)
            begin_idx=0
            for i,each_batch_num_nodes in enumerate(batch_num_nodes):
                # tmp_each_sent_tag=batch_sent_tag[begin_idx:begin_idx+each_batch_num_nodes]
                begin_idx+=each_batch_num_nodes
                tmp_each_sent_tag=torch.tensor(batch_sent_tag[i],dtype=torch.float,device=self.device_procedure) #.to(self.device_procedure)
                # if tmp_each_sent_tag.size(0)<self.max_sent_num:
                #     zero_tag=torch.zeros(self.max_sent_num-tmp_each_sent_tag.size(0),dtype=torch.float).to(tmp_each_sent_tag)
                #     tmp_each_sent_tag=torch.cat([tmp_each_sent_tag,zero_tag])
                # else:
                #     # 处理一下超过max_sent_num的情况
                #     tmp_each_sent_tag=tmp_each_sent_tag[:self.max_sent_num]
                #     pass

                # y_sent_tag[i]=tmp_each_sent_tag
                each_sent_tag=torch.zeros(self.max_sent_num,dtype=torch.float,device=self.device_procedure) #.to(tmp_each_sent_tag)
                each_sent_tag[:min(self.max_sent_num,tmp_each_sent_tag.size(0))]=tmp_each_sent_tag
                y_sent_tag[i]=each_sent_tag


            # 计算的句子attention score
            # att_score[:,:,0]=0.  # 把question的score mask掉 0.0013
            att_score_mask=torch.ones(att_score.size(),device=self.device_procedure)
            att_score_mask[:,:,0]=0
            att_score=att_score*att_score_mask #.to(self.device_procedure)
            att_score=torch.nn.functional.softmax(att_score,dim=-1) #.to(self.device_procedure)
            
            # att_score_concat=att_score.reshape(-1,att_score.size(2)) #.to(self.device_procedure)
            # y_sent_tag_concat=y_sent_tag.reshape(-1,y_sent_tag.size(1)) #.to(self.device_procedure)
            att_score=att_score.reshape(-1,att_score.size(2)) #.to(self.device_procedure)
            y_sent_tag=y_sent_tag.reshape(-1,y_sent_tag.size(1)) #.to(self.device_procedure)

            sent_att_score_weight=torch.unsqueeze(att_score,2).repeat(1,1,x_sen_repr_gru_syn.size(2))
            x_sen_repr_gru_syn_att=x_sen_repr_gru_syn*sent_att_score_weight*x_sen_repr_gru_syn.size(1) # (4,512,768)

            # 定义句子层级问答lossx
            sent_tag_loss = self.loss_sent_tag(att_score,y_sent_tag)
            sent_tag_loss *=self.sent_tag_lambda

        outputs = self.sent_encode_model.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]  # 对应其他输入字的输出
        pooled_output = outputs[1]  #对应CLS的输出
        # sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)  # spans的预测

        token_sent_matrix=torch.zeros(x_sen_repr_gru_syn_att.size(0),sequence_output.size(1),x_sen_repr_gru_syn_att.size(2),device=self.device_procedure)
        for b in range(batch_size):
            for id_idx in range(sequence_output.size(1)):
                _,token_to_sen_idx=data["input_ids_idx_to_sent_idx"][b][id_idx]
                if token_to_sen_idx==None or token_to_sen_idx==0:
                    continue
                token_sent_matrix[b,id_idx,:]=x_sen_repr_gru_syn_att[b,token_to_sen_idx]

        token_sent_matrix=self.sent_linear(token_sent_matrix)
        token_sent_sequence_output=torch.cat([sequence_output,token_sent_matrix],dim=-1)
        token_sent_sequence_output=self.dropout(token_sent_sequence_output)
        logits = self.classifier(token_sent_sequence_output)  # spans的预测

        # gather and pool span hidden
        B = pooled_output.size(0) # batch_size
        pred_spans = torch.zeros((B, self.max_pred_spans, self.H)).to(logits)
        pred_spans[:,0,:] = pooled_output # init the cls token use the bert cls token  将CLS加入到span encoder中
        span_mask = torch.zeros((B, self.max_pred_spans)).to(logits)  # 转tensor类型
        pred_labels = torch.argmax(logits, dim=-1)

        for b in range(B): # 遍历每个batch
            s_pred_labels = pred_labels[b]  # 每个batch的 tokens对应的BIO label
            s_sequence_output = sequence_output[b] # 对应的每个batch的tokens的 embedding
            indexes = [[]]  # 存放预测的多个spans的下标，
            flag=False
            for i in range(len(s_pred_labels)):
                if s_pred_labels[i] == self.label2id['B']: # B
                    indexes.append([i])
                    flag=True
                if s_pred_labels[i] == self.label2id['I'] and flag: # I
                    indexes[-1].append(i)
                if s_pred_labels[i] == self.label2id['O']: # O
                    flag=False
            indexes = indexes[:self.max_pred_spans]

            for i,index in enumerate(indexes):
                if i == 0:
                    span_mask[b,i] = 1
                    continue
                s_span = s_sequence_output[index[0]:index[-1]+1,:]  # 输出每个span对应的embeddings
                s_span = torch.mean(s_span, dim=0) # mean pooling，求平均数，进行池化，获得spans的embedding
                pred_spans[b,i,:] = s_span
                span_mask[b,i] = 1

        # encode span
        span_mask = span_mask[:,None,None,:] # extend for attention
        span_x = self.span_encoder(pred_spans, span_mask)[0]
        pooled_span_cls = span_x[:,0]   # spanencoder输出的CLS编码
        pooled_span_cls = torch.tanh(self.dense(pooled_span_cls))

        num_span_logits = self.num_span_outputs(pooled_span_cls) # 两层线性层进行预测

        outputs = (logits, num_span_logits, ) + outputs[:]   # 拼接spans、num以及初始encode的output
        if labels is not None: # for train
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)

            loss_boundary=0.
            boundary_loss=0.
            if self.add_boundary_loss:
                active_logits_B = logits[:,:,0].view(batch_size,-1)
                # active_logits_B = active_logits[:,0].view(-1)
                active_logits_B_softmax=torch.nn.functional.softmax(active_logits_B, dim=-1)
                
                # for i in range(active_logits_B_softmax.size(0)):
                    # loss_boundary+=torch.pow(torch.pow(torch.log(torch.sum(active_logits_B_softmax[i]*active_logits_B_softmax[i])),2),0.5)
                boundary_log=torch.log(torch.sum(active_logits_B_softmax*active_logits_B_softmax,dim=-1))
                boundary_zeros=torch.zeros(boundary_log.size(),device=self.device_procedure)
                loss_boundary=self.l1loss(boundary_log,boundary_zeros)
                loss_boundary*=self.boundary_lambda
                boundary_loss=loss_boundary

            # num_span regression
            loss_mse = MSELoss()
            num_span=num_span.type(torch.float) / self.max_spans
            num_span_loss = loss_mse(num_span_logits.view(-1), num_span.view(-1))
            num_span_loss *= self.span_lambda
            # structure classification

            loss = loss + num_span_loss + sent_tag_loss+boundary_loss

            outputs = (loss, ) + outputs

        return outputs


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=r"/root/MultiSpanQA_mppqa/bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        # default=r"/root/MultiSpanQA_mppqa/bert-base-uncased",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(default=False)
    structure_lambda: float= field(default=0.02)
    span_lambda: float= field(default=1)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The dir of the dataset to use."}
    )
    train_file: Optional[str] = field(
        default='train.json', metadata={"help": "The dir of the dataset to use."}
    )
    question_column_name: Optional[str] = field(
        default='question', metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    context_column_name: Optional[str] = field(
        default='context', metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default='label', metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_num_span: int = field(
        default=None,
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    doc_stride: int = field(
        default=128,
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    save_embeds: bool = field(
        default=False,
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
    )
    max_eval_samples: Optional[int] = field(
        default=None,
    )
    max_predict_samples: Optional[int] = field(
        default=None,
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    # procedure args
    datafile: str = field(
        default="data/mppqa/",
    )
    domain: str = field(
        default="wikihow",
    )
    # num_epochs: int = field(
    #     default=200,
    # )
    # model: str = field(
    #     default="bert-base-uncased",
    #     metadata={"help": "google/electra-small-discriminator, bert-base-uncased"},
    # )
    # batch_size: int = field(
    #     default="4",
    # )
    max_seq_len: int = field(
        default=80,
        metadata={"help": "max_seq_len of sentences,决定了构建语法树的大小,"},
    )
    num_classes: int = field(
        default=2,
    )
    num_node_classes: int = field(
        default=4,
    )
    num_link_classes: int = field(
        default=4,
    )
    # adam_epsilon: float = field(
    #     default=1e-8,
    # )
    # max_grad_norm: float = field(
    #     default=1.0,
    # )
    custom_model: str = field(
        default="multi_grained_syntatic_enhanced",
    )
    window_size: int = field(
        default=4,
    )
    do_oversample_data: bool = field(
        default=False,
    )
    use_pretrained_weights: bool = field(
        default=False,
    )
    gnn_layer1_out_dim: int = field(
        default=128,
    )
    gnn_layer2_out_dim: int = field(
        default=64,
    )
    dropout: int = field(
        default=0.2,
    )
    graph_connection_type: str = field(
        default="complete",
    )
    rgcn_n_bases: int = field(
        default=6,
    )
    rgcn_dropout: float = field(
        default=0.0,
    )
    pos_embedding_size: int = field(
        default=128,
    )
    dist_embeding_size: int = field(
        default=50,
    )
    head_num: int = field(
        default=4,
    )
    att_hidden_size: int = field(
        default=256,
    )
    att_dropout: float = field(
        default=0.0,
    )
    layer_num: int = field(
        default=3,
    )
    threshold: float = field(
        default=0.2,
    )
    negative_loss_weight: float = field(
        default=0.2,
    )
    max_sent_num: int = field(
        default=512,
    )
    max_sent_token_num:int=field(
        default=64,
    )
    top_k_num: int = field(
        default=3,
    )
    hidden_dropout_prob: float = field(
        default=0.2,
    )
    do_seq_dropout: bool = field(
        default=True,
    )
    add_att_weight:bool=field(
        default=True
    )
    sent_tag_lambda:float=field(
        default=0.02
    )
    boundary_lambda:float=field(
        default=0.02
    )
    token_tag_lambda:float=field(
        default=1
    )
    add_boundary_loss:bool=field(
        default=True
    )
    add_sent_tag_loss:bool=field(
        default=True
    )
    eval_best_model:bool=field(
        default=False
    )
    syn_model_path:str=field(
        default="best_model"
    )
    seed_procedure:int=field(
        default=42
    )
    att_init_weight:float=field(
        default=1.0
    )


def main():
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    args = data_args
    args.n_gpu = torch.cuda.device_count()
    
    # set_seed(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.max_node_num=0
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # log_level = training_args.get_process_log_level()
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {'train': os.path.join(data_args.data_dir, "train.json"),
                  'validation':os.path.join(data_args.data_dir, "valid.json")}
    if training_args.do_predict:
                  data_files['test'] = os.path.join(data_args.data_dir, "test.json")
    raw_datasets = load_dataset('json', field='data', data_files=data_files)

    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features

    question_column_name = data_args.question_column_name
    context_column_name = data_args.context_column_name
    label_column_name = data_args.label_column_name


    label_list = ["B", "I", "O"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        b_to_i_label.append(label_list.index(label.replace("B", "I")))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            # is_split_into_words=True # 是否需要
        )
    
    syn_rel2id = json_util.load(os.path.join(args.datafile, 'syn_rel2id.json'))
    senType2id = json_util.load(os.path.join(args.datafile, 'link2id.json'))
    pos2id = json_util.load(os.path.join(args.datafile, 'pos2id.json'))


    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length"

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    # 这里是数据的读取
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        features, conn_edges, edges, link_labels, link_type_labels, \
            sent_types, max_node_num_per_process,tokenize_sents,sent_tags, \
                tokenidx2sentenceidxs_list,edge_percent = \
            prepare_data(examples,filepath=r"//root/MultiSpanQA_mppqa/data/mppqa_multispan_3",
                        max_seq_len=80,tokenizer=tokenizer,window=args.window_size,is_oversample=args.do_oversample_data,
                        graph_connection=args.graph_connection_type)
        
        args.edge_percent = edge_percent
        if max_node_num_per_process > args.max_node_num:
            args.max_node_num = max_node_num_per_process

        tokenized_examples = tokenizer(
            examples[question_column_name],
            examples[context_column_name],
            truncation="only_second",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=padding,
            is_split_into_words=True,
        )
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["labels"] = []
        tokenized_examples["num_span"] = []
        tokenized_examples["example_id"] = []
        tokenized_examples["word_ids"] = []
        tokenized_examples["sequence_ids"] = []
        #  x, pos_ids, syn_graph_data,input_masks,segment_ids,senType,sen_lengths,num_nodes
        # tokenized_examples["sent_data"]=[]  # 添加图与sent数据

        tokenized_examples["x"]=[]
        tokenized_examples["senType"]=[]
        tokenized_examples["input_masks"]=[]
        tokenized_examples["segment_ids"]=[]
        tokenized_examples["pos_ids"]=[]
        tokenized_examples["syn_input_ids"]=[]
        
        # tokenized_examples["syn_graph_data"]=[]

        tokenized_examples["edge_index"]=[]
        tokenized_examples["edge_norm"]=[]
        tokenized_examples["edge_type"]=[]
        tokenized_examples["entity"]=[]
        tokenized_examples["neg_index"]=[]
        tokenized_examples["sen_lengths"]=[]
        # tokenized_examples["tokenize_sent"]=[]
        tokenized_examples["sent_tag"]=[]
        tokenized_examples["input_ids_idx_to_sent_idx"]=[]

        word_id2sent_ids=tokenidx2sentenceidxs_list
        for i, sample_index in enumerate(sample_mapping):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            word_ids=tokenized_examples.word_ids(i)


            label = examples[label_column_name][sample_index]
            word_ids = tokenized_examples.word_ids(i)
            previous_word_idx = None
            label_ids = [-100] * token_start_index

            for word_idx in word_ids[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            # with snoop:
            input_ids_idx_to_sent_idx=[]
            none_cnt=0
            is_out=-1
            for word_id in word_ids:
                if word_id==None:
                    none_cnt+=1
                    input_ids_idx_to_sent_idx.append([sample_index,None])
                    continue
                if none_cnt<2:
                    input_ids_idx_to_sent_idx.append([sample_index,0])
                    continue
                if word_id not in word_id2sent_ids[sample_index].keys():
                    is_out=word_id
                    sent_idx=input_ids_idx_to_sent_idx[-1][1]
                else:
                    sent_idx=word_id2sent_ids[sample_index][word_id]
                input_ids_idx_to_sent_idx.append([sample_index,sent_idx])
            # if is_out!=-1:
            #     print(f"word_ids:{is_out}, word_id2sent_ids[sample_index]:{word_id2sent_ids[sample_index]}")

            tokenized_examples["labels"].append(label_ids)
            tokenized_examples["num_span"].append(float(label_ids.count(0))) # count num of B as num_spans
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["word_ids"].append(word_ids)
            tokenized_examples["sequence_ids"].append(sequence_ids)

            eachgraph_features=features[sample_index]
            # eachgraph_connedges=conn_edges[sample_index]
            # eachgraph_alledges=edges[sample_index]
            eachgraph_labels=link_labels[sample_index]
            eachgraph_senType=sent_types[sample_index]
            eachtokenize_sent=tokenize_sents[sample_index]
            eachsent_tag=sent_tags[sample_index]

            token_ids = [e[1] for e in eachgraph_features]
            input_masks = [e[2] for e in eachgraph_features]
            segment_ids = [e[3] for e in eachgraph_features]
            pos_ids = [e[4] for e in eachgraph_features]
            syn_input_ids = [e[5] for e in eachgraph_features]
            syn_graph_data = [e[6] for e in eachgraph_features]
            neg_index = [idx for idx, e in enumerate(eachgraph_labels) if e == 0]
            sen_lengths = [np.sum(mask_arr) for mask_arr in input_masks]
            # edge_distance = [e[1]-e[0] for e in eachgraph_alledges]

            # 考虑是否全部转变成tensor
            each_feature = torch.tensor(token_ids, dtype=torch.long).to("cuda")
            each_pos_feature = torch.tensor(pos_ids, dtype=torch.long).to("cuda")
            each_syn_feature = torch.tensor(syn_input_ids, dtype=torch.long).to("cuda")


            tokenized_examples["x"].append(each_feature)
            tokenized_examples["senType"].append(eachgraph_senType)
            # if eachgraph_senType==None:
            #     print("none data!")
            tokenized_examples["input_masks"].append(input_masks)
            tokenized_examples["segment_ids"].append(segment_ids)
            tokenized_examples["pos_ids"].append(each_pos_feature)
            tokenized_examples["syn_input_ids"].append(each_syn_feature)
            
            # tokenized_examples["syn_graph_data"].append(syn_graph_data)
            edge_index=[]
            edge_norm=[]
            edge_type=[]
            entity=[]
            for each_syn_graph_data in syn_graph_data:
                edge_index.append(each_syn_graph_data.edge_index)
                edge_norm.append(each_syn_graph_data.edge_norm)
                edge_type.append(each_syn_graph_data.edge_type)
                entity.append(each_syn_graph_data.entity)
                # edge_index.append(torch.tensor(each_syn_graph_data["edge_index"],device=device))
                # edge_norm.append(torch.tensor(each_syn_graph_data["edge_norm"],device=device))
                # edge_type.append(torch.tensor(each_syn_graph_data["edge_type"],device=device))
                # entity.append(torch.tensor(each_syn_graph_data["entity"],device=device))
            tokenized_examples["edge_index"].append(edge_index)
            tokenized_examples["edge_norm"].append(edge_norm)
            tokenized_examples["edge_type"].append(edge_type)
            tokenized_examples["entity"].append(entity)
            tokenized_examples["neg_index"].append(neg_index)
            tokenized_examples["sen_lengths"].append(sen_lengths)
            # tokenized_examples["tokenize_sent"].append(eachtokenize_sent)
            tokenized_examples["sent_tag"].append(eachsent_tag)
            tokenized_examples["input_ids_idx_to_sent_idx"].append(input_ids_idx_to_sent_idx)
        
            # 在这个基础上补充sent的数据
        for keys,values in tokenized_examples.items():
            print(f"keys:{keys}, values len:{len(values)}\n")
        return tokenized_examples


    if training_args.do_train or data_args.save_embeds:
        train_examples = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_examples = train_examples.select(range(data_args.max_train_samples))
        # 补充对sent data的处理
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_examples.map(
                prepare_train_features,
                batched=True, # 将example分割为对应batch的字典
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )


    if training_args.do_eval:
        eval_examples = raw_datasets["validation"]
        # Validation Feature Creation
        if data_args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )


    if training_args.do_predict:
        predict_examples = raw_datasets["test"]
        # Predict Feature Creation
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    # data_collator = DataCollatorForTokenClassification(tokenizer)
            
    if training_args.do_train:
        tmp_train_dataset = train_dataset.remove_columns(["example_id","word_ids","sequence_ids"])
    if training_args.do_eval:
        tmp_eval_dataset = eval_dataset.remove_columns(["example_id","word_ids","sequence_ids"])

    # args.edge_percent = edge_percent
    # args.max_node_num = max_node_num
    args.num_syn_rel = len(syn_rel2id)
    args.num_pos_type = len(pos2id)

    model_to_load = os.path.join(args.syn_model_path, "senType", "best_model.pt")
    model_syn_whole=torch.load(model_to_load)

    model = TaggerPlusForMultiSpanQA.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        span_lambda=model_args.span_lambda,
        args=args,
        sent_model=model_syn_whole
    )

    # Run without Trainer

    import math
    import random

    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from huggingface_hub import Repository
    from transformers import (
        CONFIG_MAPPING,
        MODEL_MAPPING,
        AdamW,
        SchedulerType,
        get_scheduler,
    )

    accelerator = Accelerator()
    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    )
    data_collator_padding = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    )
    def collate_func(x):
        data={}
        for each_x in x:
            for key,value in each_x.items():
                if key not in data.keys():
                    data[key]=[value]
                else:
                    data[key].append(value)
        return data
    if training_args.do_train:
        # train_dataloader = DataLoader(
        #     tmp_train_dataset, shuffle=True, collate_fn=data_collator_padding, batch_size=training_args.per_device_train_batch_size
        # )
        train_dataloader = DataLoader(
            tmp_train_dataset, collate_fn=lambda x:collate_func(x),shuffle=True, batch_size=training_args.per_device_train_batch_size
        )
    else:
        train_dataloader=[]
    if training_args.do_eval:
        # eval_dataloader = DataLoader(
        #     tmp_eval_dataset, collate_fn=data_collator_padding, batch_size=training_args.per_device_eval_batch_size
        # )
        eval_dataloader = DataLoader(
            tmp_eval_dataset, collate_fn=lambda x:collate_func(x),batch_size=training_args.per_device_eval_batch_size
        )
    else:
        eval_dataloader=[]
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    # print(model)
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )

    current_path=os.path.dirname(__file__)
    model_dir = os.path.join(current_path,training_args.output_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "tagger_plus_model.pt")
    if training_args.do_train:
        # Train!
        total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        log_writer= SummaryWriter()
        count=0
        for epoch in range(int(training_args.num_train_epochs)):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(batch)
                loss = outputs[0]
                loss = loss / training_args.gradient_accumulation_steps
                if count%10==0:
                    log_writer.add_scalar(f'Loss/train ;{training_args.learning_rate}; {training_args.per_device_train_batch_size}',
                                          float(loss),count)
                    logger.info(f"epoch {epoch} ; batch {count} ; loss {loss}")
                count+=1
                    
                accelerator.backward(loss)
                if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
        # todo: 在valid集最高值时save model
        # save model
        torch.save(model.state_dict(),model_path)
        
    if not training_args.do_train:
        # load model
        model.load_state_dict(torch.load(model_path))
    
    # evaluate
    model.eval()
    all_p = []
    all_span_p = []
    all_struct_p = []
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            _, p, span_p, _, _ = model(batch)
            all_p.append(p.cpu().numpy())
            all_span_p.append(span_p.cpu().numpy())

    all_p = [i for x in all_p for i in x]
    all_span_p = np.concatenate(all_span_p)


    # Post processing
    features = eval_dataset
    examples = eval_examples
    if len(all_p) != len(features):
        raise ValueError(f"Got {len(all_p[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_confs = collections.OrderedDict()
    all_nums = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            sequence_ids = features[feature_index]['sequence_ids']
            word_ids = features[feature_index]['word_ids']
            confs = [np.max(l) for l in all_p[feature_index]]
            logits = [np.argmax(l) for l in all_p[feature_index]]
            labels = [id2label[l] for l in logits]
            nums = all_span_p[feature_index]
            prelim_predictions.append(
                {
                    "nums": nums,
                    "confs": confs,
                    "logits": logits,
                    "labels": labels,
                    "word_ids": word_ids,
                    "sequence_ids": sequence_ids
                }
            )

        previous_word_idx = -1
        ignored_index = [] # Some example tokens will be disappear after tokenization.
        valid_labels = []
        valid_confs = []
        valid_nums = sum(list(map(lambda x: x['nums'], prelim_predictions)))
        for x in prelim_predictions:
            confs = x['confs']
            labels = x['labels']
            word_ids = x['word_ids']
            sequence_ids = x['sequence_ids']

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            for word_idx,label,conf in list(zip(word_ids,labels,confs))[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    continue
                # We set the label for the first token of each word.
                elif word_idx > previous_word_idx:
                    ignored_index += range(previous_word_idx+1,word_idx)
                    valid_labels.append(label)
                    valid_confs.append(str(conf))
                    previous_word_idx = word_idx
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    continue

        context = example["context"]
        for i in ignored_index[::-1]:
            context = context[:i] + context[i+1:]
        assert len(context) == len(valid_labels) == len(valid_confs)
        
    
        # todo: 适用get_entities还是get_entities2?
        predict_entities = get_entities(valid_labels, context)
        predict_confs = get_entities(valid_labels, valid_confs)
        # predict_entities = get_entities_2(valid_labels, context)
        # predict_confs = get_entities_2(valid_labels, valid_confs)
        confidence = [x[0] for x in predict_confs]
        predictions = [x[0] for x in predict_entities]
        all_predictions[example["id"]] = predictions
        all_confs[example['id']] = confidence
        all_nums[example["id"]] = valid_nums


    # Evaluate on valid
    golds = read_gold(os.path.join(data_args.data_dir, "valid.json"))
    print(multi_span_evaluate(all_predictions, golds))
    # Span adjustment
    for key in all_predictions.keys():
        if len(all_predictions[key]) > math.ceil(all_nums[key]*21):
            confs = list(map(lambda x: max([float(y) for y in x.split()]), all_confs[key]))
            new_preds = sorted(zip(all_predictions[key],confs), key=lambda x: x[1], reverse=True)[:math.ceil(all_nums[key]*21)]
            new_preds = [x[0] for x in new_preds]
            all_predictions[key] = new_preds
    # Evaluate again
    print(multi_span_evaluate(all_predictions, golds))

if __name__ == "__main__":
    main()
