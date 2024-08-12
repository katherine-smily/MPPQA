nohup python flan-t5-mppqa.py \
    --gpu 4,6,7 \
    --dataset_name mppqa_v2 \
    --model_name /home/cike/bihan/projects/flan-t5/flan-t5-large \
 >logs/run_flan-t5_zeroshot_large_mppqa_v2_042301.log 2>&1 &


#  nohup python flan-t5-mppqa.py \
#     --gpu 0 \
#     --dataset_name mppqa_v2 \
#     --model_name /home/cike/bihan/projects/flan-t5/flan-t5-small \
#  >logs/run_flan-t5_zeroshot_small_mppqa_v2_042301.log 2>&1 &

# nohup python flan-t5-mppqa.py \
#     --gpu 1 \
#     --dataset_name mppqa_v2 \
#     --model_name /home/cike/bihan/projects/flan-t5/flan-t5-base \
#  >logs/run_flan-t5_zeroshot_base_mppqa_v2_042301.log 2>&1 &