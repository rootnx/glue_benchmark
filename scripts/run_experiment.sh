#!/bin/bash
#SBATCH -J baseline                           # 作业名
#SBATCH -o baseline.out                       # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4                     # 单任务使用的 CPU 核心数为 4
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
source ~/.bashrc

conda activate base
#!/bin/bash

python run.py \
    --task_name MED_NLI \
    --data_dir data/k-shot/MED_NLI/$aa-$i \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path /users10/cliu/medical_prompt/cliu/RoBERTa-large-PM-M3-Voc-hf \
    --few_shot_type prompt \
    --num_k $aa \
    --max_steps 1000 \
    --eval_steps 100 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --output_dir result/tmp_med_nli \
    --seed $i \
    --template "*cls**sent-_0*.*mask*.*+sentl_1**sep+* " \
    --mapping '{"contradiction":"No","entailment":"Yes","neutral":"maybe"}' \
    --num_sample 4 \
    --max_seq_len 512 \
    --show_predict \
    --result_file_dir ./predict_result/mednli/$aa-$i-original 