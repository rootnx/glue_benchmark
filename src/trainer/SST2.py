import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

from src.dataset import SST2_Dataset
from src.model import name2model
from src.utils import BasicLog


class Trainer_SST2:
    def __init__(self, config=None):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def collate_fn(self, batch):
        return ([s.text_a for s in batch], [s.label for s in batch])

    def eval(self, model, dataloader, tokenizer):
        model.eval()
        softmax = nn.Softmax(dim=1)
        predict_labels = []
        test_labels = []
        for batch in dataloader:
            inputs = batch[0]
            labels = batch[1]
            inputs = tokenizer(inputs, padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            logits = model(**inputs).cpu()
            logits = softmax(logits)
            predict_label = torch.argmax(logits, dim=1)
            predict_labels += [i.item() for i in predict_label]
            test_labels += labels

        # acc
        return accuracy_score(predict_labels, test_labels)

    def infer():
        raise NotImplementedError

    def train(self, model, train_dataloader, tokenizer, dev_dataloader=None):
        config = self.config
        logger = self.logger
        logger.info(config)
        max_epochs = config.max_epochs
        eval_steps = config.eval_steps
        accu_steps = config.accu_steps

        ce = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=config.lr)
        total_steps = int(len(train_dataloader)/accu_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.01*total_steps, num_training_steps=total_steps)

        # train
        model.train()
        best_result = {"best_result": 0, "best_steps": 0}
        steps = 0
        for epoch in range(max_epochs):
            all_loss = None
            for i, batch in enumerate(train_dataloader):
                # print(batch)
                inputs = batch[0]
                labels = torch.LongTensor(batch[1]).to(self.device)
                inputs = tokenizer(
                    inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)

                logits = model(**inputs)
                loss = ce(logits, labels)
                all_loss = loss if all_loss is None else loss+all_loss
                if steps % accu_steps == 0:
                    logger.info(
                        f"epoch:{epoch} step:{steps} loss:{all_loss.cpu().item()}")
                    all_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    all_loss = None
                if steps % eval_steps == 0:
                    acc = self.eval(model, dev_dataloader, tokenizer)
                    model.train()
                    logger.info(f"epoch:{epoch} step:{steps} acc:{acc}")
                    if acc > best_result["best_result"]:
                        best_result["best_result"] = acc
                        best_result["best_steps"] = steps
                steps += 1
        logger.info(
            f'Best_acc:{best_result["best_result"]} Best_steps:{best_result["best_steps"]}')

    def run(self, config=None):
        if config is None:
            config = self.config
        self.logger = BasicLog.get_basic_logger(config)

        data_dir = config.data_dir
        batch_size = config.batch_size
        if config.do_train:
            train_dataloader = DataLoader(SST2_Dataset(
                data_dir, "train"), batch_size=batch_size, collate_fn=self.collate_fn, shuffle=True)
        if config.do_eval:
            dev_dataloader = DataLoader(SST2_Dataset(
                data_dir, "dev"), batch_size=batch_size, collate_fn=self.collate_fn)
        if config.do_test:
            test_dataloader = DataLoader(SST2_Dataset(
                data_dir, "test"), batch_size=batch_size, collate_fn=self.collate_fn)

        # model
        Model = name2model[config.model_name_or_path]
        model = Model(config.pretrained_model_name).to(self.device)
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name)

        if config.do_train:
            self.train(model, train_dataloader, tokenizer, dev_dataloader)
        if config.do_test:
            self.eval()
        if config.do_infer:
            self.infer()
