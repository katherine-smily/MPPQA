CUDA_VISIBLE_DEVICES=4 nohup python get_sentences.py \
    --model_name_or_path /home/cike/bihan/projects/MultiSpanQA_mppqa/bert-base-uncased \
    --data_dir ../data/mppqa_multispan_3 \
    --output_dir ../output/multispan_mppqa/output_single_tagger_plus_020402 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_eval \
    --per_device_train_batch_size 2 \
    --eval_accumulation_steps 50 \
    --learning_rate 8e-6 \
    --num_train_epochs 1 \
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
    --sentence_threshold 0.0010 \
    >../logs/mppqa_get_sentences_041701.log 2>&1 &

## add
# 1. sentence_threshold 1/512->1/200=0.005
    # att_score 8*512