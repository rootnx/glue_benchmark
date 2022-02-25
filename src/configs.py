from dataclasses import dataclass

@dataclass
class SST2_Config:
    "the config of STS-B task"
    data_dir: str
    model_name: str
    num_labels: int
    max_epochs: int
    test_steps: int
    accu_steps: int
    batch_size: int
    lr: float

