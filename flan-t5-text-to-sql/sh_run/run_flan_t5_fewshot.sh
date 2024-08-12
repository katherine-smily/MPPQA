nohup python flan-t5-mppqa_fewshot.py \
    --gpu 6 \
    --dataset_name mppqa_v2 \
    --shot 1\
    --model_name /home/cike/bihan/projects/flan-t5/flan-t5-base \
 >logs/run_flan-t5_oneshot_base_mppqa_v2_042601.log 2>&1 &