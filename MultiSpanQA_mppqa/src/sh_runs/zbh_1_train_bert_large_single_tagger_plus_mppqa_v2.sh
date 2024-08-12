CUDA_VISIBLE_DEVICES=0 nohup python run_single_tagger_plus_mppqa.py \
    --model_name_or_path /root/autodl-tmp/bert-large-uncased/bert-large-uncased \
    --data_dir ../data/mppqa_v2 \
    --output_dir ../output/mppqa_v2/output_single_tagger_plus_bert_large_050901 \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 1 \
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
    --num_attention_heads 8 \
    --syn_model_path /root/autodl-tmp/ProceduralKnowledgeExtra/best_model_bert_large \
    --att_init_weight 1 \
    >../logs_v2/mppqa_v2_run_single_tagger_plus_bert_large_050901.log 2>&1 &

# epoch 10