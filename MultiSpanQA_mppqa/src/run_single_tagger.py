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
from torch.nn import CrossEntropyLoss, MSELoss
import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
    BertPreTrainedModel,
    BertModel,
    RobertaPreTrainedModel,
    RobertaModel,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from trainer import QuestionAnsweringTrainer
from eval_script import *

from os.path import normpath,join,dirname,exists
# from utils.path_util import from_project_root
Base_DIR=normpath(join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(0,Base_DIR)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")
logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"

project_root_url = normpath(join(dirname(__file__), '..'))
def from_project_root(rel_path, create=True):
    """ return system absolute path according to relative path, if path dirs not exists and create is True,
     required folders will be created

    Args:
        rel_path: relative path
        create: whether to create folds not exists

    Returns:
        str: absolute path

    """
    abs_path = normpath(join(project_root_url, rel_path))
    if create and not exists(dirname(abs_path)):
        os.makedirs(dirname(abs_path))
    return abs_path


class BertTaggerForMultiSpanQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits, ) + outputs[:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs


class RobertaTaggerForMultiSpanQA(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits, ) + outputs[:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs
    
# Todo: 直接改为single_predict的方式
# save_embeds为false
def postprocess_tagger_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    id2label,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
    save_embeds = False,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """

    # print(len(predictions),predictions)
    if len(predictions[0].shape) != 1: # Not CRF output
        if predictions[0].shape[-1] != 3:
            raise RuntimeError(f"`predictions` should be in shape of (max_seq_length, 3).")
        all_logits = predictions[0]
        all_hidden = predictions[1]
        all_labels = np.argmax(predictions[0], axis=2)

        if len(predictions[0]) != len(features):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")
    else:
        all_logits = predictions

    if -100 not in id2label.values():
        id2label[-100] = 'O'

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    # all_ids = []
    # all_valid_logits = []
    # all_valid_hidden = []

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

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
            logits = [l for l in all_logits[feature_index]]
            hidden = [l for l in all_hidden[feature_index]]
            labels = [id2label[l] for l in all_labels[feature_index]]
            prelim_predictions.append(
                {
                    "logits": logits,
                    "hidden": hidden,
                    "labels": labels,
                    "word_ids": word_ids,
                    "sequence_ids": sequence_ids
                }
            )

        previous_word_idx = -1
        ignored_index = []  # Some example tokens will disappear after tokenization.
        valid_labels = []
        # valid_logits = []
        # valid_hidden = []
        for x in prelim_predictions:
            logits = x['logits']
            hidden = x['hidden']
            labels = x['labels']
            word_ids = x['word_ids']
            sequence_ids = x['sequence_ids']

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            for word_idx, label, lo, hi in list(zip(word_ids,labels,logits,hidden))[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    continue
                # We set the label for the first token of each word.
                elif word_idx > previous_word_idx:
                    ignored_index += range(previous_word_idx+1, word_idx)
                    valid_labels.append(label)
                    # valid_logits.append(lo)
                    # valid_hidden.append(hi)
                    previous_word_idx = word_idx
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    continue

        context = example["context"]
        for i in ignored_index[::-1]:
            context = context[:i] + context[i+1:]
        assert len(context) == len(valid_labels)
        # 这里获取预测的entity(包含位置编号)，prediction是预测结果，all_prediction是所有example的预测结果
        predict_entities = get_entities(valid_labels, context)
        predictions = [x[0] for x in predict_entities]
        # 把allPrediction的json返回再处理就好了
        all_predictions[example["id"]] = predictions

    #     all_ids.append(example["id"])
    #     all_valid_logits.append(valid_logits)
    #     all_valid_hidden.append(valid_hidden)

    # all_valid_logits = np.array(all_valid_logits)
    # all_valid_hidden = np.array(all_valid_hidden)

    # If we have an output_dir, let's save all those dicts.
    
    return all_predictions
    # do not return file, return dict instead
    # if output_dir is not None:
    #     if not os.path.isdir(output_dir):
    #         raise EnvironmentError(f"{output_dir} is not a directory.")

    #     prediction_file = os.path.join(
    #         output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
    #     )

    #     logger.info(f"Saving predictions to {prediction_file}.")
    #     with open(prediction_file, "w") as writer:
    #         writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # return prediction_file
    
    
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="bert-base-uncased",metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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

# 传经常使用的parser参
def do_single_predict(
    model_name_or_path="bert-base-uncased",
    data_dir= "../data/MultiSpanQA_data",
    output_dir= "../output_tagger_021901",
    overwrite_output_dir=True,
    overwrite_cache=True,
    do_predict=True,
    max_seq_length=512,
    doc_stride=128
):
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    TrainingArguments.output_dir="../output_tagger_021901"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # args是dict，通过修改args来指定参数值
    # parser.parse_args(
    #     [       "--model_name_or_path" ,"bert-base-uncased", 
    #             "--data_dir", "../data/MultiSpanQA_data",
    #             "--output_dir", "../output_tagger_021901",
    #             "--overwrite_output_dir",
    #             "--overwrite_cache",
    #             "--do_predict",
    #             "--per_device_train_batch_size", "4",
    #             "--eval_accumulation_steps", "50",
    #             "--learning_rate", "3e-5",
    #             "--num_train_epochs", "3",
    #             "--max_seq_length",  "512",
    #             "--doc_stride", "128"
    #     ]
    # )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 通过os.system赋值参数
    
    
    # 不支持参数赋值
    # model_args["model_name_or_path"]=model_name_or_path
    # data_args["data_dir"]=data_dir
    # data_args["overwrite_cache"]=overwrite_cache
    # data_args["max_seq_length"]=max_seq_length
    # data_args["doc_stride"]=doc_stride
    # training_args["output_dir"]=output_dir
    # training_args["overwrite_output_dir"]=overwrite_output_dir
    # training_args["do_predict"]=do_predict
    
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
    data_files={}
    data_files['test'] = os.path.join(data_args.data_dir, "single_test.json")
    
    raw_datasets = load_dataset('json', field='data', data_files=data_files)

    question_column_name = data_args.question_column_name
    context_column_name = data_args.context_column_name
    # label_column_name = data_args.label_column_name

    # structure_list = ['Complex', 'Conjunction', 'Non-Redundant', 'Redundant', 'Share', '']
    # structure_to_id = {l: i for i, l in enumerate(structure_list)}

    label_list = ["B", "I", "O"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

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
        )

    if 'roberta' in model_args.model_name_or_path:
        model = RobertaTaggerForMultiSpanQA.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = BertTaggerForMultiSpanQA.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

# ignore `prepare_train_feature`

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
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

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        tokenized_examples["word_ids"] = []
        tokenized_examples["sequence_ids"] = []

        for i, sample_index in enumerate(sample_mapping):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            word_ids = tokenized_examples.word_ids(i)
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["word_ids"].append(word_ids)
            tokenized_examples["sequence_ids"].append(sequence_ids)

        return tokenized_examples
    
    # ignore do_eval


    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        all_prediction = postprocess_tagger_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            id2label=id2label,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
            save_embeds=data_args.save_embeds,
        )
        return all_prediction
    
    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        data_files=data_files,  # for quick evaluation
        train_dataset=None,
        eval_dataset=None,
        eval_examples=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=multi_span_evaluate_from_file,
    )
    
    # not do train:
    model.load_state_dict(torch.load(os.path.join(training_args.output_dir,'pytorch_model.bin')))
    trainer.model = model
    
    # if training_args.do_predict:
    predict_examples = raw_datasets["test"]
    test_column_names = raw_datasets["test"].column_names
    # Predict Feature Creation
    if data_args.max_predict_samples is not None:
        predict_examples = predict_examples.select(range(data_args.max_predict_samples))
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=test_column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )
    
    # if training_args.do_predict:
    logger.info("*** Predict ***")
    # ?predict_example是原来的，dataset是feature
    all_predict = trainer.get_all_predict(predict_dataset, predict_examples)
    
    
    return all_predict

# id为文章id，context为list，question为list
#
def get_single_test(id,context,question):
    single_test=collections.OrderedDict()
    single_test["data"]=[]
    single_test["data"].append({
        "id":id,
        "context":context,
        "question":question
    })
    # 待修改
    dir_rel_path="data/MultiSpanQA_data/"
    file_name="single_test.json"
    single_test_file=from_project_root(os.path.join(dir_rel_path, file_name))
    with open(single_test_file, "w",encoding="utf-8") as writer:
        writer.write(json.dumps(single_test, indent=4) + "\n")



def main():
    # os.system(“command”)执行相应的指令
    # 获得针对single_test.json文件的prediction结果
    all_predict = do_single_predict()
    print(all_predict)
    
if __name__=="__main__":
    main()