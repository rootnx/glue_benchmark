# prompt-skip-connection
目标：构建GLUE评测合集，并支持prompt
1. 先构建一个任务, 完成训练和测试代码
2. 加入prompt tuning，跑实验看效果
3. 加入skip-connection

代码结构
训练和测试分开
train.py 写trainer
eval.py 写测试函数
run.py 训练和测试

src/ 放 dataset，models，utils
dataset 数据的dataset
models 模型类的定义
utils 提供一些辅助工具
    时间戳

### 跑多次实验
log保存路径 由shell脚本指定 默认在models路径下
模型保存路径 由shell脚本指定
实验label 由shell脚本指定


## 构建情感分类任务 SST-2
