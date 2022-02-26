from dataclasses import dataclass

@dataclass
class SST2_Config:
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

