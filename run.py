from src.trainer import name2trainer
from src.dataset import name2dataset
from src.configs import get_parser



if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args()
    task_name = config.task_name

    # 根据任务获取 trainer
    Trainer = name2trainer[task_name]
    trainer = Trainer(config)
    trainer.run()