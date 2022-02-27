from .BertForSST2 import *
from .BertPromptTuning import *

name2model = {
    "BertSST2":BertForSST2,
    "BertPromptTuningCLS":BertPromptTuningCLS
}