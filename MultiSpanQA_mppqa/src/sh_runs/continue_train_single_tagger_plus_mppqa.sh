CUDA_VISIBLE_DEVICES=0 nohup python run_single_tagger_plus_mppqa.py \
    --model_name_or_path /root/MultiSpanQA_mppqa/bert-base-uncased \
    --data_dir ../data/mppqa_multispan_3 \
    --output_dir ../output/multispan_mppqa/output_single_tagger_plus_020402 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 2 \
    --eval_accumulation_steps 50 \
    --learning_rate 8e-6 \
    --num_train_epochs 10 \
    --max_seq_length  512 \
    --doc_stride 128 \
    --window_size 0 \
    --max_sent_token_num 64 \
    --add_att_weight true \
    --span_lambda 1 \
    --sent_tag_lambda 0.02 \
    --boundary_lambda 0.02 \
    --token_tag_lambda 1 \
    --add_boundary_loss false \
    --eval_best_model false \
    --syn_model_path best_model \
    --att_init_weight 1 \
    --continue_training true \
    --continue_training_model_dir ../output/multispan_mppqa/output_single_tagger_plus_020401 \
    >../logs_v2/continue_mppqa_run_single_tagger_plus_020402.log 2>&1 &