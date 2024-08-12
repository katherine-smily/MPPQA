import os
import sys
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
import transformers

from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorForTokenClassification,
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

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")
logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"


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

class TaggerPlusForMultiSpanQA(BertPreTrainedModel):
    def __init__(self, config, structure_lambda, span_lambda,major_lambda_CLS,major_lambda_span,num_majors,num_minors):
        super().__init__(config, structure_lambda, span_lambda,major_lambda_CLS,major_lambda_span)
        self.structure_lambda = structure_lambda
        self.span_lambda = span_lambda
        self.major_lambda_CLS=major_lambda_CLS
        self.major_lambda_span=major_lambda_span
        self.label2id= config.label2id
        self.num_labels = config.num_labels
        self.max_spans = 15
        self.max_pred_spans = 30
        self.H = config.hidden_size
        self.dense = nn.Linear(self.H, self.H)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.H, config.num_labels)
        # self.bilstm=nn.LSTM(input_size=self.H,hidden_size=self.H//2,bidirectional=True)
        # self.bilstm2linear_classifier=nn.Linear(self.H, config.num_labels)
        self.num_span_outputs = nn.Sequential(nn.Linear(self.H, 64),nn.ReLU(),nn.Linear(64, 1))
        self.structure_outputs = nn.Sequential(nn.Linear(self.H, 128),nn.ReLU(),nn.Linear(128, 6))
        self.major_spans_nums_per_batch=None
        self.golden_spans_each=None
        

        self.classifier_major_cls = nn.Sequential(nn.Linear(self.H, 128),nn.ReLU(),nn.Linear(128, num_majors))
        self.classifier_major_spans = nn.Sequential(nn.Linear(self.H, 128),nn.ReLU(),nn.Linear(128, num_majors))
        config.num_attention_heads=6 # for span encoder
        intermediate_size=1024
        self.span_encoder = BertLayer(config)
        self.num_majors=num_majors
        self.init_weights()
    # @snoop
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        num_span=None,
        structure=None,
        major_type=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]  # (batch_size, sequence_length, hidden_size) (4,512,768)
        pooled_output = outputs[1] # (batch_size, hidden_size)   # CLS?  (4,768)
        sequence_output = self.dropout(sequence_output)
        # logits_bilstm=self.bilstm(sequence_output)[0]  # 4,512,768
        # logits= self.bilstm2linear_classifier(logits_bilstm)  # 4,512,3
        logits = self.classifier(sequence_output)   # 线性预测spans结果的logit (batch_size, sequence_length，config.num_labels）(4,512,3)
        
        # logits_bilstm, (h_n, c_n)=self.span_bilstm_classifier(sequence_output)
        
        # 使用CLS对major type进行预测
        major_logits_CLS=self.classifier_major_cls(pooled_output)  # (batch_size, config.num_majors) (4,4)
        
        # 通过CLS预测major：
        # pred_major_CLS = torch.argmax(major_logits_CLS, dim=-1)   #(batch_size, config.num_majors）->(batch_size)
        
        
        # gather and pool span hidden
        # 在这里对logit进行处理？
        B = pooled_output.size(0) # batch_size 
        pred_spans = torch.zeros((B, self.max_pred_spans, self.H)).to(logits)   # 4,30,768
        pred_spans[:,0,:] = pooled_output # init the cls token use the bert cls token
        span_mask = torch.zeros((B, self.max_pred_spans)).to(logits)
        
        pred_labels = torch.argmax(logits, dim=-1)   #(batch_size, sequence_length）(4,512)
        
        major_spans_nums_per_batch=torch.zeros(B).to(logits) #(4)
        pred_major_spans_nums_per_batch=torch.zeros((B)).to(logits)  # (4)
        golden_spans_each=None
        
        pred_spans_each=None
        for b in range(B):
            s_pred_labels = pred_labels[b]
            s_sequence_output = sequence_output[b]    # 4*512*768->512*768
            indexes = [[]]
            # indexes_IO=get_indexes(s_pred_labels.cpu().numpy(),self.label2id)
            indexes_golden=[[]]
            
            flag=False
            for i in range(len(s_pred_labels)):
                if s_pred_labels[i] == self.label2id['B']: # B
                    indexes.append([i])
                    flag=True
                if s_pred_labels[i] == self.label2id['I'] and flag: # I
                    indexes[-1].append(i)
                if s_pred_labels[i] == self.label2id['O']: # O
                    flag=False
            indexes = indexes[:self.max_pred_spans]    # 每个batch BI对应的预测的token的indexs

            flag=False
            for i in range(len(labels[b])):
                if labels[b][i] == self.label2id['B']: # B
                    indexes_golden.append([i])
                    flag=True
                if labels[b][i] == self.label2id['I'] and flag: # I
                    indexes_golden[-1].append(i)
                if labels[b][i] == self.label2id['O']: # O
                    flag=False
            indexes_golden = indexes_golden    # 每个batch BI对应的预测的token的indexs

            for i,index in enumerate(indexes):
                if i == 0:
                    span_mask[b,i] = 1
                    continue
                span_mask[b,i] = 1
                if index==[]:
                    continue
                s_span = s_sequence_output[index[0]:index[-1]+1,:]   # 对应每个span的隐藏层表示？
                s_span = torch.mean(s_span, dim=0) # mean pooling  # 平均池化每个span的表示 ( hidden_size)
                pred_spans[b,i,:] = s_span    # 拼接池化的 
            
            # for i,index in enumerate(indexes_IO):
            #     if i == 0:
            #         continue
            #     if index==[]:
            #         continue
            #     s_span = s_sequence_output[index[0]:index[-1]+1,:]   # 对应每个span的隐藏层表示？
            #     s_span = torch.mean(s_span, dim=0) # mean pooling  # 平均池化每个span的表示 ( hidden_size)
                pred_major_spans_nums_per_batch[b]+=1
                # 这里需要把pred_span拆开，然后找到对应的标签
                if pred_spans_each==None:
                    pred_spans_each=s_span
                    pred_spans_each=pred_spans_each.unsqueeze(0)
                else:
                    pred_spans_each=torch.cat([pred_spans_each,s_span.unsqueeze(0)],dim=0)   
                # pred_spans_each.append(s_span)    # (k*hidden_size )
            
            
            
            if indexes_golden!=[[]]:
                for i,index_golden in enumerate(indexes_golden):
                    if index_golden==[]:
                        continue
                    s_span = s_sequence_output[index_golden[0]:index_golden[-1]+1,:]
                    s_span = torch.mean(s_span, dim=0)
                    major_spans_nums_per_batch[b]+=1
                    # 这里需要把pred_span拆开，然后找到对应的标签
                    if golden_spans_each==None:
                        golden_spans_each=s_span
                        golden_spans_each=golden_spans_each.unsqueeze(0)
                    else:
                        golden_spans_each=torch.cat([golden_spans_each,s_span.unsqueeze(0)],dim=0)   
                    # golden_spans_each.append(s_span)    # (k*hidden_size )
                self.golden_spans_each=torch.tensor(golden_spans_each).to(golden_spans_each)
                self.major_spans_nums_per_batch=torch.tensor(major_spans_nums_per_batch).to(major_spans_nums_per_batch)
                
        if self.golden_spans_each!=None:
            major_logits_span=self.classifier_major_spans(self.golden_spans_each)     # (batch_size*pred_spans_num, config.num_majors) (116,4)
        else:
            major_logits_span=torch.tensor([[]]).to(self.major_spans_nums_per_batch)
        # with snoop:
        major_logits_span_tuple=()
        cnt_begin=0
        cnt_end=0
        for idx in range(B):
            if idx>self.major_spans_nums_per_batch.size(0):
                break
            num=int(self.major_spans_nums_per_batch[idx])
            cnt_end+=num
            major_logits_span_tuple=major_logits_span_tuple+((major_logits_span[cnt_begin:cnt_end,:]),)
            cnt_begin+=num
        
        with torch.no_grad():
            pred_major_logits_span=None
            if pred_spans_each!=None:
                pred_major_logits_span=self.classifier_major_spans(pred_spans_each)
            
        pred_major_logits_span_tuple=()
        # if pred_major_logits_span!=None:
        cnt_begin=0
        cnt_end=0
        for idx in range(major_logits_CLS.size(0)):
            num=int(pred_major_spans_nums_per_batch[idx])
            if num==0:
                pred_major_logits_span_tuple=pred_major_logits_span_tuple+(torch.tensor([]).to(logits),)
                continue
            cnt_end+=num
            pred_major_logits_span_tuple=pred_major_logits_span_tuple+((torch.tensor(pred_major_logits_span[cnt_begin:cnt_end])),)
            cnt_begin+=num
        # major_logits_CLS2Span=()
        # for idx in range(major_logits_CLS.size(0)):
        #     for num in range(int(major_spans_nums_per_batch[idx])):
        #         major_logits_CLS2Span.append(major_logits_CLS[idx])
        
        # encode span
        span_mask = span_mask[:,None,None,:] # extend for attention(4,1,1,30)
        span_x = self.span_encoder(pred_spans, span_mask)[0]
        pooled_span_cls = span_x[:,0]
        pooled_span_cls = torch.tanh(self.dense(pooled_span_cls))

        num_span_logits = self.num_span_outputs(pooled_span_cls)  # (4,1) [[0.0244,,,]]
        structure_logits = self.structure_outputs(pooled_span_cls)
        # 0.6075,0.0244,0.0863,0.03,0.008
        outputs = (logits, num_span_logits,major_logits_CLS,major_logits_span_tuple,pred_major_logits_span_tuple, ) + outputs[:]           # 在这里对logit进行处理？
        if labels is not None: # for train
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)

            # num_span regression
            loss_mse = MSELoss()
            num_span=num_span.type(torch.float) / self.max_spans
            num_span_loss = loss_mse(num_span_logits.view(-1), num_span.view(-1))
            num_span_loss *= self.span_lambda
            # structure classification
            loss_focal = FocalLoss(gamma=0.5)
            structure_loss = loss_focal(structure_logits.view(-1, 6), structure.view(-1))
            structure_loss *= self.structure_lambda
            
            # major type loss cls
            loss_focal_major_CLS = FocalLoss(gamma=0.5)
            major_loss_CLS = loss_focal_major_CLS(major_logits_CLS.view(-1, self.num_majors), major_type.view(-1))
            major_loss_CLS*=self.major_lambda_CLS  #可以考虑调小一个量级
            
            # major type loss span   
            if self.golden_spans_each!=None:
                loss_focal_major_span = FocalLoss(gamma=0.5)
                major_type_span=[]
                for idx,mt in enumerate(major_type.view(-1)):
                    major_type_span.extend([mt]*int(self.major_spans_nums_per_batch[idx]))
                major_type_span=torch.tensor(major_type_span).to(major_type).squeeze()
                major_loss_span = loss_focal_major_span(major_logits_span.view(-1, self.num_majors),major_type_span.view(-1))
                major_loss_span*=self.major_lambda_span
            else:
                major_loss_span=0.
            
            loss = loss + num_span_loss + structure_loss+major_loss_CLS+major_loss_span  ##看一下各自比例

            outputs = (loss, ) + outputs

        return outputs

    def predict_span_type(self,predict_sequence_outputs):
        with torch.no_grad():
            predict_sequence_outputs=torch.tensor(predict_sequence_outputs).to(self.major_spans_nums_per_batch)
            pred_major_logits_span=self.classifier_major_spans(predict_sequence_outputs)
        return pred_major_logits_span
        
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
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
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(default=False)
    # structure_lambda: float= field(default=0.02)
    structure_lambda: float= field(default=0.00)
    span_lambda: float= field(default=1)
    major_lambda_CLS: float= field(default=0.02)
    minor_lambda_CLS: float= field(default=0.02)
    major_lambda_span: float= field(default=0.02)
    minor_lambda_span: float= field(default=0.02)
    major_span_ratio:float= field(default=0.1)
    minor_span_ratio:float= field(default=0.1)

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

# @snoop
def main():
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
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

    data_files = {'train': os.path.join(data_args.data_dir, data_args.train_file),
                  'validation':os.path.join(data_args.data_dir, "valid.json")}
    if training_args.do_predict:
                  data_files['test'] = os.path.join(data_args.data_dir, "test.json")
    raw_datasets = load_dataset('json', field='data', data_files=data_files)

    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features

    question_column_name = data_args.question_column_name
    context_column_name = data_args.context_column_name
    label_column_name = data_args.label_column_name

    # structure_list = ['Complex', 'Conjunction', 'Non-Redundant', 'Redundant', 'Share', '']
    structure_list = ['']
    structure_to_id = {l: i for i, l in enumerate(structure_list)}

    minor_type_list=["operator","object","tool","location","time","quantity","state"]
    minor_to_id = {l: i for i, l in enumerate(minor_type_list)}
    
    major_type_list=["entity","time","number","state"]
    major_to_id = {l: i for i, l in enumerate(major_type_list)}
    
    label_list = ["B", "I", "O"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        b_to_i_label.append(label_list.index(label.replace("B", "I")))

    # 这里补充config参数
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        # num_majors=len(major_type_list),
        # num_minor=len(minor_type_list),
        max_seq_length=512,
        # max_position_embeddings=1024,
        label2id=label2id,
        id2label=id2label,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"gpt2", "roberta","deberta","longformer"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            # max_position_embeddings=1024,
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
            # max_position_embeddings=1024,
            # add_prefix_space=True,
        )
    # config.max_position_embeddings=1024
    model = TaggerPlusForMultiSpanQA.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        structure_lambda=model_args.structure_lambda,
        span_lambda=model_args.span_lambda,
        major_lambda_CLS=model_args.major_lambda_CLS,
        major_lambda_span=model_args.major_lambda_span,
        num_majors=len(major_type_list),
        num_minors=len(minor_type_list),
    )

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
    padding = "max_length" if data_args.pad_to_max_length else False
    # padding = "max_length"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    # max_seq_length = 1024
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(  #151->625
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
        tokenized_examples["structure"] = []
        tokenized_examples["major_type"]=[]
        tokenized_examples["example_id"] = []
        tokenized_examples["word_ids"] = []
        tokenized_examples["sequence_ids"] = []

        for i, sample_index in enumerate(sample_mapping):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Start token index of the current span in the text.
            token_start_index = 0
            question_token_start_index=0
            while sequence_ids[question_token_start_index] != 0:   # 找到question的token start index
                question_token_start_index += 1
            while sequence_ids[token_start_index] != 1:   # 找到context的token start index
                token_start_index += 1

            label = examples[label_column_name][sample_index]
            word_ids = tokenized_examples.word_ids(i)
            previous_word_idx = None
            label_ids = [-100] * token_start_index    # 初始化label_ids? 长度为question？

            for word_idx in word_ids[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])   # 拼入context里的label
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            tokenized_examples["labels"].append(label_ids)
            tokenized_examples["num_span"].append(float(label_ids.count(0))) # count num of B as num_spans
            tokenized_examples["structure"].append(structure_to_id[examples['structure'][sample_index] if 'structure' in examples else ''])
            tokenized_examples["major_type"].append(major_to_id[examples['major_type'][sample_index] if 'major_type' in examples else ''])
            # tokenized_examples["minor_type"].append(minor_to_id[examples['minor_type'][sample_index] if 'minor_type' in examples else ''])
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["word_ids"].append(word_ids)
            tokenized_examples["sequence_ids"].append(sequence_ids)
            # tokenized_examples["question_range_sequence_idx"]=[question_token_start_index,token_start_index-2]
        return tokenized_examples


    if training_args.do_train or data_args.save_embeds:
        train_examples = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_examples = train_examples.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_examples.map(
                prepare_train_features,
                batched=True,
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
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # tmp_train_dataset = train_dataset.remove_columns(["example_id","word_ids","sequence_ids"])
    # tmp_eval_dataset = eval_dataset.remove_columns(["example_id","word_ids","sequence_ids"])

    if training_args.do_train:
        tmp_train_dataset = train_dataset.remove_columns(["example_id","word_ids","sequence_ids"])
    if training_args.do_eval:
        tmp_eval_dataset = eval_dataset.remove_columns(["example_id","word_ids","sequence_ids"])

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

    # train_dataloader = DataLoader(
    #     tmp_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    # )
    # eval_dataloader = DataLoader(
    #     tmp_eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
    # )

    if training_args.do_train:
        train_dataloader = DataLoader(
            tmp_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
        )
    else:
        train_dataloader=[]
    if training_args.do_eval:
        eval_dataloader = DataLoader(
            tmp_eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
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
    if training_args.do_train and training_args.do_eval:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

    if not training_args.do_train and training_args.do_eval:
        model, optimizer, eval_dataloader = accelerator.prepare(
            model, optimizer, eval_dataloader
        )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

    # 线性调整学习率
    lr_scheduler = get_scheduler( 
        name=training_args.lr_scheduler_type, 
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )

    # Train!
    current_path=os.path.dirname(__file__)
    model_dir = os.path.join(current_path,training_args.output_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "tagger_plus_model.pt")
    if training_args.do_train:
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
                outputs = model(**batch)  # 1,(4,512,3),(4,1),(4,4),([]),(4,29,4),(4,512,768),(4,768)
                loss = outputs[0]
                loss = loss / training_args.gradient_accumulation_steps
                if count%10==0:
                    log_writer.add_scalar(f'Loss/train ;{training_args.learning_rate}; {training_args.per_device_train_batch_size}',
                                          float(loss),count)
                    logger.info(f"epoch {epoch} ; batch {count} ; loss {loss}")
                count+=1
                accelerator.backward(loss)
                if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()  # 参数更新
                    lr_scheduler.step() # 学习率更新
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs_eval = model(**batch)  # 1,(4,512,3),(4,1),(4,4),([]),(4,29,4),(4,512,768),(4,768)
                    loss = outputs_eval[0]
                    loss = loss / training_args.gradient_accumulation_steps
                    if count%10==0:
                        log_writer.add_scalar(f'valid_Loss/train ;{training_args.learning_rate}; {training_args.per_device_train_batch_size}',
                                            float(loss),count)
                        logger.info(f"valid: epoch {epoch} ; batch {count} ; loss {loss}")
            if epoch%5==0:
                model_path_epoch = os.path.join(model_dir, "tagger_plus_model_epoch"+str(epoch)+".pt")
                torch.save(model.state_dict(),model_path_epoch)
        # save model
        torch.save(model.state_dict(),model_path)
        
    if not training_args.do_train:
        # load model
        model.load_state_dict(torch.load(model_path))
               
    # evaluate
    model.eval()
    all_p = []
    all_sequence_output=[]
    all_span_p = []
    all_struct_p = []
    all_major_CLS_p=[]
    all_major_span_p=()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            _, p, span_p,major_CLS_p,golden_major_span_p,major_span_p, sequence_output, _ = model(**batch)   # (loss,logits, num_span_logits,major_logits_CLS(batch*n),major_logits_span_tuple(batch*n*k),pred_major_logits_span_tuple,outputs[:])
            all_p.append(p.cpu().numpy())
            all_sequence_output.append(sequence_output.cpu().numpy())
            all_span_p.append(span_p.cpu().numpy())
            all_major_CLS_p.append(major_CLS_p.cpu().numpy())
            for idx in range(len(major_span_p)):
                all_major_span_p=all_major_span_p+(major_span_p[idx].cpu().numpy(),)
    all_p = [i for x in all_p for i in x]   # 拼接所有batch的sequence的BIO logit
    all_span_p = np.concatenate(all_span_p)  # 拼接所有batch的span数量的logit
    all_sequence_output=[i for x in all_sequence_output for i in x]
    all_major_CLS_p=np.concatenate(all_major_CLS_p)

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

    all_confs_major_span=collections.OrderedDict()
    all_prediction_major_span=collections.OrderedDict()
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
            confs = [np.max(l) for l in all_p[feature_index]]  # 置信度         
            logits = [np.argmax(l) for l in all_p[feature_index]]         
            labels = [id2label[l] for l in logits]
            nums = all_span_p[feature_index]
            
            logits_major_CLS=np.argmax(all_major_CLS_p[feature_index])  # (24,8,4)? (191,4)
            if len(all_major_span_p[feature_index])!=0:
                # print(f"all_major_CLS_p:{all_major_CLS_p},\nall_major_span_p:{len(all_major_span_p)}### {all_major_span_p[feature_index]},\nfeature_index:{feature_index},\nlogits_major_CLS:{logits_major_CLS}")
                confs_major_span=all_major_span_p[feature_index][:,logits_major_CLS]  # 会不会为空？
            else: 
                confs_major_span=[]
            prelim_predictions.append(
                {
                    "nums": nums,
                    "confs": confs,
                    "logits": logits,
                    "labels": labels,
                    "word_ids": word_ids,
                    "sequence_ids": sequence_ids,
                    "confs_major_span":confs_major_span,
                    "sequence_outputs":all_sequence_output[feature_index],
                    "logits_major_CLSs":logits_major_CLS
                }
            )
            

        previous_word_idx = -1
        ignored_index = [] # Some example tokens will be disappear after tokenization.
        valid_labels = []
        valid_confs = []
        valid_sequence_outputs=[]
        valid_type_confs=[]
        valid_nums = sum(list(map(lambda x: x['nums'], prelim_predictions)))
        valid_confs_major_span=[]
        valid_logits_major_CLSs=list(map(lambda x: x['logits_major_CLSs'], prelim_predictions))
        for x in prelim_predictions:
            confs = x['confs']
            labels = x['labels']
            word_ids = x['word_ids']
            sequence_ids = x['sequence_ids']
            confs_major_span=x['confs_major_span']
            sequence_outputs=x['sequence_outputs']
            valid_confs_major_span.extend(confs_major_span)
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            for word_idx,label,conf,sequence_output in list(zip(word_ids,labels,confs,sequence_outputs))[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    continue
                # We set the label for the first token of each word.
                elif word_idx > previous_word_idx:
                    ignored_index += range(previous_word_idx+1,word_idx)
                    valid_labels.append(label)
                    valid_confs.append(str(conf))   # 从sequence提取context的有效confs
                    valid_sequence_outputs.append(sequence_output)
                    previous_word_idx = word_idx
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    continue

        context = example["context"]
        for i in ignored_index[::-1]:
            context = context[:i] + context[i+1:]
        assert len(context) == len(valid_labels) == len(valid_confs)    

        # predict_entities = get_entities_2(valid_labels, context)
        # predict_confs = get_entities_2(valid_labels, valid_confs)
        predict_entities = get_entities(valid_labels, context)
        predict_confs = get_entities(valid_labels, valid_confs)   # 每个span的confs
        predict_sequence_outputs=get_sequence_outputs(valid_labels,valid_sequence_outputs)
        
        
        confidence = [x[0] for x in predict_confs]    # 补充增加type比例的conf

        predictions = [x[0] for x in predict_entities]
        
        #  = confidence.astype(valid_confs_major_span.dtype) * valid_confs_major_span * model_args.major_span_ratio
        
        all_predictions[example["id"]] = predictions
        all_prediction_major_span[example["id"]]=predictions
        # all_prediction_major_span[example["id"]]=predict_sequence_outputs
        
        all_confs[example['id']] = confidence
        all_nums[example["id"]] = valid_nums
        # if valid_confs_major_span!=[]:
        #     all_confs_major_span[example['id']] = valid_confs_major_span
        # else:
        #     all_confs_major_span[example['id']]=1
        
        if predict_sequence_outputs!=[]:
            pred_major_logits_span=model.predict_span_type(predict_sequence_outputs)
            pred_major_logits_span=pred_major_logits_span.cpu().numpy()
            tmp_pred_major_logits_span=[]
            for k in range(len(pred_major_logits_span)):
                pred_major_logit_softmax=[]
                for p in range(len(pred_major_logits_span[k])):
                    pred_major_logit_softmax.append(np.exp(pred_major_logits_span[k][p])/np.sum(pred_major_logits_span[k],axis=0))
                tmp_pred_major_logits_span.append(pred_major_logit_softmax)
            pred_major_logits=tmp_pred_major_logits_span
            pred_major_logits=[pred_major_logits_span[k][valid_logits_major_CLSs[0]] for k in range(len(pred_major_logits_span))]
            
            all_confs_major_span[example['id']] = pred_major_logits
        else:
            all_confs_major_span[example['id']]=1
    # Evaluate on valid
    golds = read_gold(os.path.join(data_args.data_dir, "valid.json"))
    print(multi_span_evaluate(all_predictions, golds))
    # Span adjustment
    # 尝试置信度低也进行筛选？
    for key in all_predictions.keys():
        if len(all_predictions[key]) > math.ceil(all_nums[key]*15):
            confs = list(map(lambda x: max([float(y) for y in x.split()]), all_confs[key])) # 置信度 1.33-0.7
            print(f"\nconfs:{confs}\nall_confs_major_span[key]:{all_confs_major_span[key]}\n") # 0.006,0.4,0.3
            # confidence_major_span=np.multiply(np.multiply(confs, all_confs_major_span[key]) , model_args.major_span_ratio)   
            confidence_major_span=np.multiply(confs, all_confs_major_span[key])
            new_preds = sorted(zip(all_predictions[key],confs), key=lambda x: x[1], reverse=True)[:math.ceil(all_nums[key]*15)]
            new_preds = [x[0] for x in new_preds]
            
            new_preds_major_span = sorted(zip(all_predictions[key],confidence_major_span), key=lambda x: x[1], reverse=True)[:math.ceil(all_nums[key]*15)]
            new_preds_major_span = [x[0] for x in new_preds_major_span]
            
            all_predictions[key] = new_preds
            all_prediction_major_span[key] = new_preds_major_span
    # Evaluate again 这里对评估函数再改写一下？
    print(multi_span_evaluate(all_predictions, golds))
    print(multi_span_evaluate(all_prediction_major_span, golds))

if __name__ == "__main__":
    main()
