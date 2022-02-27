from dataclasses import dataclass,field
from transformers import HfArgumentParser
from typing import Callable, Dict, Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to model"}
    )
    pretrained_model_name: str = field(
        default="bert-base-uncased",
        metadata={"help":"Path to used pretrained model name or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    num_labels: Optional[int]= field(
        default = None,
        metadata={"help": "Used in classifion task"}
    )
    dropout: Optional[float] = field(
        default=0.2,
    )
    hidden_size: Optional[int] = field(
        default=768,
    )


@dataclass
class TrainArguments:
    """
    Arguments for training.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Specific the task name"}
    )
    # For logging
    tag: Optional[str] = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )
    log_dir: Optional[str] = field(
        default="log/log.txt",
        metadata={"help": "Path to save log file"}
    )
    do_train: Optional[bool] = field(
        default=True,
    )
    do_eval: Optional[bool] = field(
        default=True,
    )
    do_infer: Optional[bool] = field(
        default=False,
        metadata={"help": "do infer"}
    )
    do_test: Optional[bool] = field(
        default=False,
        metadata={"help": "test a model, need to specific a checkpoint"}
    )
    load_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "load a checkpoint"}
    )
    lr: Optional[float] = field(
        default="0.001",
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "data path"}
    )
    output_dir: Optional[str] = field(
        default="result",
    )
    max_epochs: Optional[int] = field(
        default=10,
    )
    eval_steps: Optional[int] = field(
        default=100,
    )
    accu_steps: Optional[int] = field(
        default=2,
        metadata={"help":"the num of gradient accumation"}
    )
    batch_size: Optional[int] = field(
        default=32,
    )
    lr: Optional[float] = field(
        default=3e-5,
    )
    seed: Optional[int] = field(
        default=10,
    )
    debug_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "Debug mode"}
    )


@dataclass
class SST2_Config_class:
    "the config of STS-B task"
    data_dir: str = "./data/original/SST-2"
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    max_epochs: int = 10
    test_steps: int = 100
    accu_steps: int = 2
    batch_size: int = 32
    lr: float = 3e-6
    dropout: float = 0.2
    hidden_size: int = 768

@dataclass
class PromptArguments:
    n_tokens: Optional[int] = field(
        default=None,
        metadata={"help":"the tokens used in prompt"}
    )


@dataclass
class Arguments(TrainArguments,PromptArguments,ModelArguments):
    pass

def get_parser():
    parser = HfArgumentParser(Arguments)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args()
    print(config)
    print(config.model_name_or_path)