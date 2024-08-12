# nohup python flan-t5-mppqa.py \
#     --gpu 0 \
#     --dataset_name mppqa_multispan_3 \
#     --model_name /home/cike/bihan/projects/flan-t5/flan-t5-large \
#  >logs/run_flan-t5_zeroshot_large_041001.log 2>&1 &


#  nohup python flan-t5-mppqa.py \
#     --gpu 1 \
#     --dataset_name mppqa_multispan_3 \
#     --model_name /home/cike/bihan/projects/flan-t5/flan-t5-small \
#  >logs/run_flan-t5_zeroshot_small_041001.log 2>&1 &

nohup python flan-t5-mppqa.py \
    --gpu 0,2 \
    --dataset_name mppqa_multispan_3 \
    --model_name /home/cike/bihan/projects/flan-t5/flan-t5-base \
 >logs/run_flan-t5_zeroshot_base_041001.log 2>&1 &