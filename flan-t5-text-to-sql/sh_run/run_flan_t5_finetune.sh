nohup python flan-t5-mppqa_finetune.py \
    --gpu 0 \
    --dataset_name mppqa_v2 \
    --model_name /root/flan-t5-small \
    --gradient_accumulation_steps 1 \
    --train_batch_size 1 \
    --epoch_num 6 \
 >logs/run_flan-t5_finetune_mppqa_v2_050501.log 2>&1 &