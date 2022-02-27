from .SST2 import *
from .SST2_PT import *


# 任务名到trainer的映射
name2trainer = {
    "SST2":Trainer_SST2,
    "SST2_PT":Trainer_SST2_PT,
}