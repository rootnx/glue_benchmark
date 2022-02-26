#!/bin/bash
#SBATCH -J baseline                           # 作业名
#SBATCH -o baseline.out                       # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4                     # 单任务使用的 CPU 核心数为 4
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
source ~/.bashrc

# conda activate base
#!/bin/bash

python run.py \
    --task_name SST2 \
    --model_name_or_path BertForSST2 \
    --pretrained_model_name bert-base-uncased \
    --do_train \
    --do_eval \
    --data_dir data/original/SST-2 \
    --max_epochs 10 \
    --eval_steps 100 \
    --accu_steps 2 \
    --lr 3e-5 \
    --dropout 0.2 \
    --hidden_size 768 \
    --num_labels 2 \
    --output_dir result/ \
    --seed 10 \