nohup python opt/opt_zeroshot.py \
    --gpu 7 \
    --dataset_name mppqa_v2 \
    --model_name /home/cike/bihan/projects/flan-t5-text-to-sql/opt/opt-350m \
 >logs/run_opt_zeroshot_350m_mppqa_v2_042601.log 2>&1 &


#  nohup python flan-t5-mppqa.py \
#     --gpu 1 \
#     --dataset_name mppqa_multispan_3 \
#     --model_name /home/cike/bihan/projects/flan-t5/flan-t5-small \
#  >logs/run_flan-t5_zeroshot_small_041001.log 2>&1 &

# nohup python flan-t5-mppqa.py \
#     --gpu 1 \
#     --dataset_name mppqa_multispan_3 \
#     --model_name /home/cike/bihan/projects/flan-t5/flan-t5-base \
#  >logs/run_flan-t5_zeroshot_base_041001.log 2>&1 &