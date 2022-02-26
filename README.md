# Glue Benchmark
目标：构建GLUE评测合集
1. 先构建一个任务, 完成训练和测试代码
2. 加入prompt tuning，跑实验看效果
3. 加入skip-connection

## 代码结构
做到 任务，数据，模型 互相解耦

训练和测试分开

run.py 主入口
src/ 放 dataset，models，utils
dataset.py 数据的dataset
models.py 模型类的定义
utils.py 提供一些辅助工具，比如时间戳，日志

## 添加新任务流程
1. 根据数据处理模块，在 src/dataset.py
2. 增加任务对应的trainer ，在src/trainer.py，记得注册name2trainer
3. 新增实验脚本，在 scripts/

## 实验
实验脚本在 scripts/ 目录


### 构建情感分类任务 SST-2
```
bash scripts/run_experiment_SST2.sh
```
