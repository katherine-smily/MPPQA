## sh
### roberta:
CUDA_VISIBLE_DEVICES=4 nohup python run_single_tagger_plus_roberta.py \
    --model_name_or_path roberta-base \
    --data_dir ../data/wiki_v4_2_json \
    --output_dir ../output_tagger_plus_roberta_080904 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --max_seq_length  512 \
    --doc_stride 128 \
    --pad_to_max_length true\
    >../logs_major/run_tagger_plus_roberta_080904.log 2>&1 &

### deberta:
CUDA_VISIBLE_DEVICES=4 nohup python run_single_tagger_plus_deberta.py \
    --model_name_or_path deberta-v3-base \
    --data_dir ../data/wiki_v4_2_json \
    --output_dir ../output_tagger_plus_deberta_080902 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 20 \
    --max_seq_length  512 \
    --doc_stride 128 \
    --pad_to_max_length true\
    >../logs_major/run_tagger_plus_deberta_080902.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python run_single_tagger_plus_deberta.py \
    --model_name_or_path deberta-v3-base \
    --data_dir ../data/wiki_v4_2_json \
    --output_dir ../output_tagger_plus_deberta_080902 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 20 \
    --max_seq_length  512 \
    --doc_stride 128 \
    --pad_to_max_length true\
    >../logs_major/run_tagger_plus_deberta_080902_eval.log 2>&1 &
### run_tagger_plus:
CUDA_VISIBLE_DEVICES=4 nohup python run_tagger_plus.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/wiki_v6_2_json \
    --output_dir ../output_tagger_plus_080504 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 8e-6 \
    --num_train_epochs 30 \
    --max_seq_length  512 \
    --doc_stride 128 \
    --pad_to_max_length true\
    >../logs_major/run_tagger_plus_080504.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python run_tagger_plus.py \
    --model_name_or_path deberta-v3-base \
    --data_dir ../data/wiki_v6_2_json \
    --output_dir ../output_tagger_plus_080505 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --max_seq_length  512 \
    --doc_stride 128 \
    --pad_to_max_length true\
    >../logs_major/run_tagger_plus_080505.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python run_single_tagger_plus_origin.py \
    --model_name_or_path deberta-v3-base \
    --data_dir ../data/wiki_v6_2_json \
    --output_dir ../output_tagger_plus_080902 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 2 \
    --eval_accumulation_steps 50 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --max_seq_length  512 \
    --doc_stride 128 \
    --pad_to_max_length true

CUDA_VISIBLE_DEVICES=2,5,7 nohup python run_tagger_plus.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/wiki_v5_2_json \
    --output_dir ../output_tagger_plus_080301 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_predict\
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 8 \
    --max_seq_length  512 \
    --doc_stride 128 \
    --pad_to_max_length true\
    >../logs_major/run_tagger_plus_080301.log 2>&1 &

CUDA_VISIBLE_DEVICES=5,7 python run_tagger_plus.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/wiki_v5_2_json \
    --output_dir ../output_tagger_plus_080401 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_seq_length  512 \
    --doc_stride 128 \
    --pad_to_max_length true\
    >../logs_major/run_tagger_plus_080401.log

### Fine-tune BERT tagger on MultiSpanQA
cd ~/bihan/projects/MultiSpanQA/src
CUDA_VISIBLE_DEVICES=4 nohup python run_tagger.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/MultiSpanQA_data \
    --output_dir ../output_tagger \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_tagger_021901.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python run_tagger.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/wiki_v4_2_json \
    --output_dir ../output_tagger_051801 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 2e-5 \
    --num_train_epochs  80\
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_tagger_051801.log 2>&1 &

+ 进行predict：
    + do_predict

### run_single_tagger.py
CUDA_VISIBLE_DEVICES=6 nohup python run_single_tagger.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/MultiSpanQA_data \
    --output_dir ../output_tagger_021901 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_predict \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_single_tagger_021901.log 2>&1 &

### run_tagger_plus.py
cd ~/bihan/projects/MultiSpanQA/src
CUDA_VISIBLE_DEVICES=6 nohup python run_tagger_plus.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/MultiSpanQA_data \
    --output_dir ../output_tagger_plus_022201 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_tagger_plus_022201.log 2>&1 &

### run_single_tagger_plus.py
CUDA_VISIBLE_DEVICES=5 nohup python run_single_tagger_plus.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/wiki_v4_json \
    --output_dir ../output_single_tagger_plus_080101 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_single_tagger_plus_080101.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_single_tagger_plus.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/wiki_v4_json \
    --output_dir ../output_single_tagger_plus_051304 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_single_tagger_plus_051304.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_single_tagger_plus.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/wiki_v4_json \
    --output_dir ../output_single_tagger_plus_051302 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_single_tagger_plus_051303.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_single_tagger_plus.py \
    --model_name_or_path bert-base-uncased \
    --data_dir ../data/wiki_v4_2_json \
    --output_dir ../output_single_tagger_plus_051307 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 8e-6 \
    --num_train_epochs 40 \
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_single_tagger_plus_051307.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_single_tagger_plus.py \
    --model_name_or_path roberta-base \
    --data_dir ../data/wiki_v4_2_json \
    --output_dir ../output_single_tagger_roberta_plus_051308 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 1.5e-5 \
    --num_train_epochs 20 \
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_single_tagger_roberta_plus_051308.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_single_tagger_plus.py \
    --model_name_or_path roberta-base \
    --data_dir ../data/wiki_v4_2_json \
    --output_dir ../output_single_tagger_roberta_plus_051501 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 400 \
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_single_tagger_roberta_plus_051501.log 2>&1 &

#### launch config:
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: 当前文件",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "cwd": "${workspaceFolder}",
            "args": [
                "--model_name_or_path" ,"bert-base-uncased", 
                "--data_dir", "../data/MultiSpanQA_data",
                "--output_dir", "../output",
                "--overwrite_output_dir",
                "--overwrite_cache",
                "--do_train",
                "--do_eval",
                "--per_device_train_batch_size", "4",
                "--eval_accumulation_steps", "50",
                "--learning_rate", "3e-5",
                "--num_train_epochs", "3",
                "--max_seq_length",  "512",
                "--doc_stride", "128"
            ],
            "env":{
                "CUDA_VISIBLE_DEVICES":"7"
        }
        }
    ]
}

### run_squad是single-span模型，可以不用考虑
### change MultiSpanQA to the format that can be trained on single-span model :
python generate_squad_format.py

### choose to fine-tune BERT on one of them (for example v1) using:
cd ~/bihan/projects/MultiSpanQA/src
CUDA_VISIBLE_DEVICES=5 nohup python run_squad.py \
    --model_name_or_path bert-base-uncased \
    --train_file ../data/MultiSpanQA_data/squad_train_softmax_v1.json \
    --validation_file ../data/MultiSpanQA_data/squad_valid.json \
    --output_dir ../output_squad \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --eval_accumulation_steps 50 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length  512 \
    --doc_stride 128 \
    >../logs/run_squad_021901.log 2>&1 &
