# from transformers import AutoTokenizer, LlamaForCausalLM

# model = LlamaForCausalLM.from_pretrained("/home/cike/bihan/llama2")
# tokenizer = AutoTokenizer.from_pretrained("/home/cike/bihan/llama2")

# prompt = "Hey, are you conscious? Can you talk to me?"
# inputs = tokenizer(prompt, return_tensors="pt")

# # Generate
# generate_ids = model.generate(inputs.input_ids, max_length=30)
# tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


from typing import Dict, Any, List
import sys
sys.path.append("..")
sys.path.append(".")
from eval_script import evaluate_mppqa
# import json
# import re
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
# from transformers import DataCollatorForSeq2Seq
from transformers import LlamaForCausalLM, LlamaTokenizer
import multiprocessing as mp
import torch
from utils import save_dataset, read_msqa, read_quoref, set_seed, save_model, split_sequence
import argparse
import ast
import os
# 
device = torch.device("cuda:0")
if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",
                        default='7',
                        type=str)
    parser.add_argument("--model_name",
                        default=r'/home/cike/bihan/projects/llama2_mppqa/llama-2-7b-hf', # meta-llama/Llama-2-7b-hf
                        # default='microsoft/deberta-v3-base',
                        type=str)
    parser.add_argument("--dataset_name",
                        default='mppqa_multispan_3',
                        type=str)
    parser.add_argument("--dataset_split",
                        default='in_house',
                        type=str)
    parser.add_argument("--vanilla",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--only_eval",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--debug",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--results_save_path",
                        default='./results/',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir",
                        default='../outputs/',
                        type=str,
                        help="The output dreader2ctory whretriever the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=False,
                        type=ast.literal_eval,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--max_len",
                        default=512,
                        type=int)
    parser.add_argument("--lr",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = LlamaForCausalLM.from_pretrained(args.model_name,max_position_embeddings=2048,torch_dtype=torch.float16).cuda()
    # model = OPTModel.from_pretrained(args.model_name)

    model = model.cuda()
    # tokenizer = LlamaTokenizer.from_pretrained(args.model_name,padding_side='left')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    vanilla = args.vanilla
    force_answer = True
    if args.dataset_name == 'MultiSpanQA':
        force_answer = False
    only_eval = args.only_eval
    dim2 = 64
    debug = args.debug
    if args.model_name.endswith('/'):
        args.model_name = args.model_name[:-1]
    model_name_abb = args.model_name.split('/')[-1]
    config_name = f'{args.dataset_name}/{model_name_abb}/'

    parameter_name = f'lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}' \
                     f'_ga_{args.gradient_accumulation_steps}'
    output_model_path = f'./outputs/{config_name}/{parameter_name}/'
    path_save_result = f'./results/{config_name}/{parameter_name}/'

    data_path_base = f'./data/{args.dataset_name}/'
    data_path_train = f'{data_path_base}/train.json'
    data_path_dev = f'{data_path_base}/valid.json'
    data_path_test = f'{data_path_base}/test.json'

    os.makedirs(path_save_result, exist_ok=True)
    set_seed(args.seed)
    read_dataset = read_msqa

    # train_examples = read_dataset(data_path_train)
    dev_examples = read_dataset(data_path_dev)
    # test_examples = read_dataset(data_path_test)

    # if debug:
    #     train_examples = train_examples[:20]
    #     dev_examples = dev_examples[:20]
        # test_examples = test_examples[:20]

    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # prompt = f"Given a question and context, please extract answer from the context, ### Question: {dev_examples['question'][0]} ### Context: {dev_examples['context'][0]}"
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # # Generate
    # generate_ids = model.generate(inputs.input_ids, max_new_tokens=1000)
    # # b=tokenizer.decode(generate_ids, skip_special_tokens=True)
    # a=tokenizer.batch_decode(generate_ids, skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    # print(a)
    # # print(b)


    # process
    def preprocess(x):
        x['prompt'] = ['Given a question and context, Answer the question using context, and seperate the answer spans with "、". ### Question: ' + xi[0] + '### Context: ' + xi[1] +"### Answer:"
                 for xi in list(zip(x['question'],  x['context']))]
        # x['prompt'] = ['Given a question and context, Answer the question using context, and seperate the answer spans with "、". Question: ' + xi[0]   
        #          for xi in list(zip(x['question'],  x['context']))]
        return x

    # train_dataset=train_examples.map(lambda x: preprocess(x), batched=True)
    dev_dataset=dev_examples.map(lambda x: preprocess(x), batched=True)
    # test_dataset=test_examples.map(lambda x: preprocess(x), batched=True)



    # zeroshot
    def get_tokenized(x):
        # x["input_ids"]=tokenizer(x['prompt'], padding=True,truncation=True, return_tensors="pt")["input_ids"].tolist()
        # addition_input_ids=tokenizer("### Answer:", return_tensors="pt")["input_ids"].tolist()
        x["input_ids"]=tokenizer(x['prompt'], padding=True,truncation=True,max_length=1800, return_tensors="pt")["input_ids"].tolist()
        # x["input_ids"]=tokenizer(x['prompt'], return_tensors="pt")["input_ids"].tolist()
        return x

    dev_tokenized = dev_dataset.map(lambda x: get_tokenized(x), batched=True)
    def generate(x):
        # x['output'] = [model.generate(torch.tensor(x_ids).reshape(1, -1).to(device), 
        #                             max_new_tokens=100, 
        #                             do_sample=False)[0] for x_ids in x['input_ids']]
        x['output'] = [model.generate(torch.tensor(x_ids).reshape(1, -1).to(device), 
                            max_new_tokens=196,do_sample=True) for x_ids in x['input_ids']]
        return x
        
    dev_generated = dev_tokenized.map(lambda x: generate(x), batched=True, batch_size=8)
    
    def decode(x):
        # x['predicted_answers'] = [tokenizer.decode(torch.tensor(xo), skip_special_tokens=True).split("、") for xo in x['output']]
        x['predicted_answers'] = [tokenizer.batch_decode(torch.tensor(xo), skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)[0][len(x['prompt'][i]):].split("、") for i,xo in enumerate(x['output'])]
        return x

    dev_decoded = dev_generated.map(lambda x: decode(x), batched=True)
    
    print(evaluate_mppqa(dev_decoded))
    
    # fewshot




